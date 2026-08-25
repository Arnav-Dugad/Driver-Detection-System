/**
 * Optional in-browser scoring for a fusion model trained by `ml/train_fusion.py`
 * and exported by `ml/export_browser_model.py`.
 *
 * This is deliberately a side path. The deterministic engine in core.mjs stays
 * the default because it is explainable and needs no dataset to be trustworthy;
 * a learned model is only as good as the consented, subject-independent data
 * behind it. Loading one lets the two be compared in the live app without
 * adding a runtime dependency or sending anything to a server.
 */

import { clamp } from "./core.mjs";

const LEAF = -1;

/** Reject anything that is not the exact payload shape the exporter writes. */
export function isLearnedModel(model) {
  return Boolean(
    model &&
      model.format === "driver-fusion-forest" &&
      Array.isArray(model.features) &&
      Array.isArray(model.classes) &&
      Array.isArray(model.trees) &&
      model.trees.length > 0,
  );
}

/** Order a feature window into the vector the model was trained on. */
export function vectorize(model, features) {
  return model.features.map((name, index) => {
    const value = features?.[name];
    return typeof value === "number" && Number.isFinite(value)
      ? value
      : // Match the median imputation used during training.
        (model.imputer_medians?.[index] ?? 0);
  });
}

function walk(tree, vector) {
  let node = 0;
  // A tree cannot be deeper than its node count; the bound stops a malformed
  // export from spinning forever.
  for (let step = 0; step < tree.left.length; step += 1) {
    if (tree.left[node] === LEAF) break;
    node =
      vector[tree.feature[node]] <= tree.threshold[node]
        ? tree.left[node]
        : tree.right[node];
  }
  return tree.value[node];
}

/**
 * Average the per-class probabilities across the forest.
 * Returns null when the payload is not a usable model.
 */
export function predictProbabilities(model, features) {
  if (!isLearnedModel(model)) return null;
  const vector = vectorize(model, features);
  const totals = new Array(model.classes.length).fill(0);

  for (const tree of model.trees) {
    const leaf = walk(tree, vector);
    for (let index = 0; index < totals.length; index += 1) {
      totals[index] += leaf[index] ?? 0;
    }
  }

  const result = {};
  for (let index = 0; index < model.classes.length; index += 1) {
    result[model.classes[index]] = clamp(totals[index] / model.trees.length);
  }
  return result;
}

/** The most likely class and how confident the forest is about it. */
export function predictClass(model, features) {
  const probabilities = predictProbabilities(model, features);
  if (!probabilities) return null;
  const ranked = Object.entries(probabilities).sort((left, right) => right[1] - left[1]);
  return { label: ranked[0][0], confidence: ranked[0][1], probabilities };
}

/**
 * Blend a learned opinion into the deterministic score.
 *
 * The deterministic engine keeps the majority of the weight so a model trained
 * on a small classroom dataset can never override an obvious, explainable
 * signal such as a two-second eye closure.
 */
export function blendWithLearned(deterministicScore, model, features, weight = 0.35) {
  const prediction = predictClass(model, features);
  if (!prediction) return { score: deterministicScore, learned: null };

  const RISK_BY_LABEL = { alert: 8, focused: 8, distracted: 58, drowsy: 82 };
  const learnedScore = Object.entries(prediction.probabilities).reduce(
    (sum, [label, probability]) => sum + (RISK_BY_LABEL[label] ?? 40) * probability,
    0,
  );
  const share = clamp(weight) * prediction.confidence;
  return {
    score: Math.round(deterministicScore * (1 - share) + learnedScore * share),
    learned: prediction,
  };
}

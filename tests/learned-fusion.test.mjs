import assert from "node:assert/strict";
import test from "node:test";

import {
  blendWithLearned,
  isLearnedModel,
  predictClass,
  predictProbabilities,
  vectorize,
} from "../lib/detection/learned.mjs";

/**
 * A two-tree stump forest that splits on PERCLOS. Mirrors the array layout
 * ml/export_browser_model.py writes out of sklearn.
 */
const model = {
  format: "driver-fusion-forest",
  version: 1,
  features: ["perclos_60s", "ear_mean"],
  classes: ["alert", "drowsy"],
  imputer_medians: [0.05, 0.29],
  trees: [
    {
      left: [1, -1, -1],
      right: [2, -1, -1],
      feature: [0, -2, -2],
      threshold: [0.2, -2, -2],
      value: [
        [0.5, 0.5],
        [1, 0],
        [0, 1],
      ],
    },
    {
      left: [1, -1, -1],
      right: [2, -1, -1],
      feature: [0, -2, -2],
      threshold: [0.25, -2, -2],
      value: [
        [0.5, 0.5],
        [0.9, 0.1],
        [0.1, 0.9],
      ],
    },
  ],
};

test("only the exporter's own payload shape is accepted", () => {
  assert.equal(isLearnedModel(model), true);
  assert.equal(isLearnedModel(null), false);
  assert.equal(isLearnedModel({ format: "something-else", trees: [] }), false);
  assert.equal(isLearnedModel({ ...model, trees: [] }), false);
});

test("features are ordered as the model was trained and gaps are imputed", () => {
  assert.deepEqual(vectorize(model, { ear_mean: 0.3, perclos_60s: 0.4 }), [0.4, 0.3]);
  // A missing column falls back to the training median, exactly as sklearn did.
  assert.deepEqual(vectorize(model, { perclos_60s: 0.4 }), [0.4, 0.29]);
  assert.deepEqual(vectorize(model, {}), [0.05, 0.29]);
});

test("the forest separates an alert window from a drowsy one", () => {
  const alert = predictProbabilities(model, { perclos_60s: 0.02, ear_mean: 0.29 });
  assert.ok(alert.alert > alert.drowsy);

  const drowsy = predictProbabilities(model, { perclos_60s: 0.4, ear_mean: 0.12 });
  assert.ok(drowsy.drowsy > drowsy.alert);

  const ranked = predictClass(model, { perclos_60s: 0.4, ear_mean: 0.12 });
  assert.equal(ranked.label, "drowsy");
  assert.ok(ranked.confidence > 0.5);
});

test("probabilities across classes stay normalized", () => {
  const probabilities = predictProbabilities(model, { perclos_60s: 0.22 });
  const total = Object.values(probabilities).reduce((sum, value) => sum + value, 0);
  assert.ok(Math.abs(total - 1) < 1e-9);
});

test("a missing model leaves the deterministic score untouched", () => {
  const blended = blendWithLearned(64, null, { perclos_60s: 0.4 });
  assert.equal(blended.score, 64);
  assert.equal(blended.learned, null);
});

test("the learned opinion nudges the score without overriding it", () => {
  // A confident drowsy vote should raise a calm deterministic score, but the
  // explainable engine must keep the majority of the weight.
  const raised = blendWithLearned(30, model, { perclos_60s: 0.5 });
  assert.ok(raised.score > 30);
  assert.ok(raised.score < 60, `learned input must not dominate, got ${raised.score}`);

  // An obvious two-second closure must not be talked down into safety.
  const held = blendWithLearned(90, model, { perclos_60s: 0.01 });
  assert.ok(held.score > 60, `deterministic evidence must hold, got ${held.score}`);
});

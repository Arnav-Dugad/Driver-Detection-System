/**
 * Aggregates the live telemetry buffer into the exact feature window that
 * `ml/train_fusion.py` consumes.
 *
 * The browser and the training pipeline have to agree on this schema or the two
 * halves of the project silently drift apart, so FEATURE_COLUMNS is asserted
 * against the Python FEATURES list in tests/features.test.mjs.
 *
 * Everything here is numeric. No frame, image, or face template is ever part of
 * a feature window.
 */

import { calculatePerclos, clamp, longestClosure, mean } from "./core.mjs";

/** Must stay identical, and in order, to FEATURES in ml/train_fusion.py. */
export const FEATURE_COLUMNS = [
  "ear_mean",
  "ear_min",
  "perclos_60s",
  "longest_closure_ms",
  "blink_rate_per_min",
  "mar_mean",
  "mar_max",
  "yawns_10m",
  "yaw_deviation",
  "pitch_deviation",
  "gaze_deviation",
  "phone_ratio",
  "face_missing_ratio",
];

/** The columns `train_fusion.py` requires alongside the features themselves. */
export const LABEL_COLUMNS = ["subject_id", "label"];

const YAWN_WINDOW_MS = 10 * 60_000;

const round = (value, places = 5) => {
  if (!Number.isFinite(value)) return 0;
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
};

/**
 * Build one labeled-window feature vector.
 *
 * `samples` is an ascending buffer of
 * `{ time, ear, mar, closure, yaw, pitch, gaze, phoneVisible, faceFound }`.
 * Deviations are absolute distances from the driver's calibrated baseline, so
 * they mean the same thing across people and camera placements.
 */
export function buildFeatureWindow(
  samples,
  { baseline, now = null, windowMs = 60_000, blinkTimes = [], yawnTimes = [] } = {},
) {
  if (!Array.isArray(samples) || !samples.length) return null;

  const currentTime = now ?? samples[samples.length - 1].time;
  const window = samples.filter((sample) => currentTime - sample.time <= windowMs);
  if (!window.length) return null;

  const seen = window.filter((sample) => sample.faceFound !== false);
  const spanMinutes = Math.max(
    windowMs / 60_000 / 10,
    (currentTime - window[0].time) / 60_000,
  );

  const ears = seen.map((sample) => sample.ear).filter(Number.isFinite);
  const mars = seen.map((sample) => sample.mar).filter(Number.isFinite);

  const deviation = (key, fallback = 0) => {
    const base = baseline?.[key] ?? fallback;
    const values = seen
      .map((sample) => sample[key])
      .filter(Number.isFinite)
      .map((value) => Math.abs(value - base));
    return mean(values);
  };

  const blinksInWindow = blinkTimes.filter(
    (time) => currentTime - time <= windowMs,
  ).length;
  const yawnsInTenMinutes = yawnTimes.filter(
    (time) => currentTime - time <= YAWN_WINDOW_MS,
  ).length;

  return {
    ear_mean: round(mean(ears)),
    ear_min: round(ears.length ? Math.min(...ears) : 0),
    perclos_60s: round(calculatePerclos(window, currentTime, windowMs)),
    longest_closure_ms: round(longestClosure(window), 1),
    blink_rate_per_min: round(blinksInWindow / spanMinutes, 3),
    mar_mean: round(mean(mars)),
    mar_max: round(mars.length ? Math.max(...mars) : 0),
    yawns_10m: yawnsInTenMinutes,
    yaw_deviation: round(deviation("yaw")),
    pitch_deviation: round(deviation("pitch")),
    gaze_deviation: round(deviation("gaze", 0.5)),
    phone_ratio: round(
      clamp(window.filter((sample) => sample.phoneVisible).length / window.length),
    ),
    face_missing_ratio: round(
      clamp(window.filter((sample) => sample.faceFound === false).length / window.length),
    ),
  };
}

const escapeCsv = (value) => {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

/** Header row for a CSV that `train_fusion.py` can read without any reshaping. */
export function featureCsvHeader(extraColumns = []) {
  return [...LABEL_COLUMNS, ...FEATURE_COLUMNS, ...extraColumns].join(",");
}

/** Serialize one labeled window as a CSV row matching `featureCsvHeader`. */
export function featureCsvRow(features, { subjectId, label, extra = {} } = {}) {
  const extraColumns = Object.keys(extra);
  return [
    escapeCsv(subjectId),
    escapeCsv(label),
    ...FEATURE_COLUMNS.map((column) => escapeCsv(features?.[column] ?? 0)),
    ...extraColumns.map((column) => escapeCsv(extra[column])),
  ].join(",");
}

/** Assemble a complete CSV document from labeled windows. */
export function buildFeatureCsv(rows) {
  const extraColumns = Object.keys(rows[0]?.extra ?? {});
  return [
    featureCsvHeader(extraColumns),
    ...rows.map((row) =>
      featureCsvRow(row.features, {
        subjectId: row.subjectId,
        label: row.label,
        extra: row.extra ?? {},
      }),
    ),
  ].join("\n");
}

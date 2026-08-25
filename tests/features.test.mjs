import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  FEATURE_COLUMNS,
  buildFeatureCsv,
  buildFeatureWindow,
  featureCsvHeader,
} from "../lib/detection/features.mjs";

const pythonSource = await readFile(
  new URL("../ml/train_fusion.py", import.meta.url),
  "utf8",
);

/** Read the FEATURES list out of the training script itself. */
function pythonFeatureList() {
  const block = pythonSource.match(/^FEATURES = \[([\s\S]*?)\]/m);
  assert.ok(block, "ml/train_fusion.py must declare a FEATURES list");
  return [...block[1].matchAll(/"([^"]+)"/g)].map((match) => match[1]);
}

test("the browser emits exactly the schema the training pipeline expects", () => {
  // If this fails, the app and ml/train_fusion.py have drifted apart and any
  // exported CSV would be rejected or silently misaligned.
  assert.deepEqual(FEATURE_COLUMNS, pythonFeatureList());
});

test("the CSV header carries the grouping and label columns the pipeline requires", () => {
  const header = featureCsvHeader();
  const required = pythonSource.match(/^REQUIRED_COLUMNS = \[([\s\S]*?)\]/m);
  assert.ok(required, "ml/train_fusion.py must declare REQUIRED_COLUMNS");
  for (const column of [...required[1].matchAll(/"([^"]+)"/g)].map((match) => match[1])) {
    assert.ok(header.split(",").includes(column), `header is missing ${column}`);
  }
});

/** A minute of samples at 30 Hz with a scripted eye closure. */
function buildSamples({ now = 60_000, closedFrom = null, closedMs = 0 } = {}) {
  const samples = [];
  for (let time = now - 59_000; time <= now; time += 33) {
    const closed =
      closedFrom !== null && time >= closedFrom && time < closedFrom + closedMs;
    samples.push({
      time,
      ear: closed ? 0.05 : 0.29,
      mar: 0.11,
      closure: closed ? 1 : 0,
      yaw: 0.02,
      pitch: -0.01,
      gaze: 0.52,
      phoneVisible: false,
      faceFound: true,
    });
  }
  return samples;
}

test("a feature window reports every column with a usable value", () => {
  const features = buildFeatureWindow(buildSamples(), {
    baseline: { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0, gaze: 0.5 },
    now: 60_000,
  });
  for (const column of FEATURE_COLUMNS) {
    assert.ok(column in features, `missing ${column}`);
    assert.ok(Number.isFinite(features[column]), `${column} must be numeric`);
  }
  assert.ok(Math.abs(features.ear_mean - 0.29) < 0.01);
  assert.ok(Math.abs(features.gaze_deviation - 0.02) < 0.005);
  assert.equal(features.phone_ratio, 0);
});

test("a long closure moves PERCLOS and the longest-closure feature together", () => {
  const drowsy = buildFeatureWindow(
    buildSamples({ closedFrom: 30_000, closedMs: 12_000 }),
    { baseline: { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0, gaze: 0.5 }, now: 60_000 },
  );
  assert.ok(drowsy.perclos_60s > 0.15, `expected raised PERCLOS, got ${drowsy.perclos_60s}`);
  assert.ok(drowsy.longest_closure_ms > 11_000);
  assert.ok(drowsy.ear_min < 0.1);
});

test("blink rate and yawn count come from event times, not frames", () => {
  const now = 60_000;
  const features = buildFeatureWindow(buildSamples({ now }), {
    baseline: { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0, gaze: 0.5 },
    now,
    blinkTimes: Array.from({ length: 14 }, (_, index) => now - index * 4_000),
    // One yawn sits outside the ten-minute window and must not be counted.
    yawnTimes: [now - 1_000, now - 200_000, now - 900_000],
  });
  assert.ok(features.blink_rate_per_min > 10 && features.blink_rate_per_min < 20);
  assert.equal(features.yawns_10m, 2);
});

test("an empty buffer produces no window rather than a row of zeros", () => {
  assert.equal(buildFeatureWindow([], {}), null);
  assert.equal(buildFeatureWindow(null, {}), null);
});

test("exported CSV rows line up with the header", () => {
  const features = buildFeatureWindow(buildSamples(), {
    baseline: { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0, gaze: 0.5 },
    now: 60_000,
  });
  const csv = buildFeatureCsv([
    { subjectId: "p01", label: "alert", features },
    { subjectId: "p01", label: "drowsy", features },
  ]);
  const [header, ...rows] = csv.split("\n");
  assert.equal(rows.length, 2);
  for (const row of rows) {
    assert.equal(row.split(",").length, header.split(",").length);
  }
  assert.ok(header.startsWith("subject_id,label,ear_mean"));
});

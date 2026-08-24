import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCalibration,
  calculatePerclos,
  calculateRisk,
  classifySignals,
  formatDuration,
} from "../lib/detection/core.mjs";

test("ordinary blinks do not become critical alerts", () => {
  const result = calculateRisk({ eyeClosedMs: 220, perclos: 0.04 });
  assert.equal(result.state, "focused");
  assert.ok(result.score < 20);
});

test("persistent concurrent impairment escalates to danger", () => {
  const result = calculateRisk({
    eyeClosedMs: 2200,
    perclos: 0.34,
    recentYawns: 3,
    headAwayMs: 2400,
    gazeAwayMs: 2100,
    phoneVisible: true,
  });
  assert.equal(result.state, "danger");
  assert.ok(result.score >= 76);
  assert.ok(result.concurrent >= 3);
});

test("calibration uses robust medians instead of a single frame", () => {
  const samples = Array.from({ length: 40 }, (_, index) => ({
    ear: index === 0 ? 0.02 : 0.29 + (index % 3) * 0.002,
    mar: 0.11 + (index % 2) * 0.002,
    yaw: -0.01 + (index % 3) * 0.01,
    pitch: 0.43 + (index % 3) * 0.005,
    gaze: 0.49 + (index % 3) * 0.01,
  }));
  const baseline = buildCalibration(samples);
  assert.ok(baseline.ear > 0.28 && baseline.ear < 0.31);
  assert.ok(baseline.mar > 0.1 && baseline.mar < 0.13);
  assert.ok(baseline.gaze > 0.47 && baseline.gaze < 0.53);
});

test("PERCLOS only counts samples inside the rolling window", () => {
  const now = 100_000;
  const samples = [
    { time: now - 70_000, closed: true },
    { time: now - 5_000, closed: true },
    { time: now - 4_000, closed: false },
    { time: now - 3_000, closed: false },
  ];
  assert.equal(calculatePerclos(samples, now), 1 / 3);
});

test("sensitivity changes personalized thresholds predictably", () => {
  const signals = { ear: 0.21, mar: 0.3, yaw: 0.08, pitch: 0.44, gaze: 0.68 };
  const baseline = { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0.44, gaze: 0.5 };
  const low = classifySignals(signals, baseline, 0.25);
  const high = classifySignals(signals, baseline, 0.9);
  assert.equal(low.eyesClosed, false);
  assert.equal(high.eyesClosed, true);
  assert.ok(high.thresholds.gaze < low.thresholds.gaze);
});

test("durations are formatted for a glanceable cockpit", () => {
  assert.equal(formatDuration(7), "00:07");
  assert.equal(formatDuration(3672), "01:01:12");
});

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCalibration,
  calculatePerclos,
  calculateRisk,
  circadianRisk,
  classifyCameraQuality,
  classifySignals,
  decomposePose,
  detectYawn,
  estimateSignalConfidence,
  explainRisk,
  extractBlendshapeSignals,
  formatDuration,
  fuseClosure,
  longestClosure,
  updateBaseline,
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
    pitch: 0.01 + (index % 3) * 0.005,
    gaze: 0.49 + (index % 3) * 0.01,
  }));
  const baseline = buildCalibration(samples);
  assert.ok(baseline.ear > 0.28 && baseline.ear < 0.31);
  assert.ok(baseline.mar > 0.1 && baseline.mar < 0.13);
  assert.ok(baseline.gaze > 0.47 && baseline.gaze < 0.53);
});

test("sensitivity changes personalized thresholds predictably", () => {
  const signals = { ear: 0.21, mar: 0.3, yaw: 0.08, pitch: 0.01, gaze: 0.68 };
  const baseline = { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0, gaze: 0.5 };
  const low = classifySignals(signals, baseline, 0.25);
  const high = classifySignals(signals, baseline, 0.9);
  assert.equal(low.eyesClosed, false);
  assert.equal(high.eyesClosed, true);
  assert.ok(high.thresholds.gaze < low.thresholds.gaze);
});

test("camera quality separates dim scenes from a covered lens", () => {
  assert.equal(
    classifyCameraQuality({ brightness: 0.18, contrast: 0.12, darkPixelRatio: 0.7, faceFound: true }),
    "low-light",
  );
  assert.equal(
    classifyCameraQuality({ brightness: 0.03, contrast: 0.01, darkPixelRatio: 0.98, faceFound: false }),
    "obstructed",
  );
  assert.equal(
    classifyCameraQuality({ brightness: 0.48, contrast: 0.18, darkPixelRatio: 0.12, faceFound: true }),
    "clear",
  );
});

test("durations are formatted for a glanceable cockpit", () => {
  assert.equal(formatDuration(7), "00:07");
  assert.equal(formatDuration(3672), "01:01:12");
});

// --- Head pose from the facial transformation matrix -----------------------

/** Column-major 4x4 for a pure rotation about the Y (yaw) axis. */
const yawMatrix = (angle, scale = 1) => {
  const c = Math.cos(angle) * scale;
  const s = Math.sin(angle) * scale;
  return { data: [c, 0, -s, 0, 0, scale, 0, 0, s, 0, c, 0, 0, 0, 0, 1] };
};

/** Column-major 4x4 for a pure rotation about the X (pitch) axis. */
const pitchMatrix = (angle) => {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return { data: [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1] };
};

test("pose decomposition recovers true yaw and pitch angles", () => {
  const yawed = decomposePose(yawMatrix(0.5236));
  assert.ok(Math.abs(yawed.yaw - 0.5236) < 1e-6, "30 degrees of yaw");
  assert.ok(Math.abs(yawed.pitch) < 1e-6);
  assert.ok(Math.abs(yawed.roll) < 1e-6);

  const pitched = decomposePose(pitchMatrix(-0.3491));
  assert.ok(Math.abs(pitched.pitch + 0.3491) < 1e-6, "20 degrees of pitch");
  assert.ok(Math.abs(pitched.yaw) < 1e-6);
});

test("pose decomposition ignores how close the driver sits to the camera", () => {
  // The matrix carries face scale; a driver leaning in must not read as turning.
  const near = decomposePose(yawMatrix(0.4, 1));
  const far = decomposePose(yawMatrix(0.4, 3.5));
  assert.ok(Math.abs(near.yaw - far.yaw) < 1e-6);
});

test("pose decomposition rejects an absent or malformed matrix", () => {
  assert.equal(decomposePose(null), null);
  assert.equal(decomposePose({ data: [1, 0, 0] }), null);
});

// --- Blendshapes and fused eye closure -------------------------------------

const blendshapeSet = (overrides = {}) => ({
  categories: Object.entries({
    eyeBlinkLeft: 0.1,
    eyeBlinkRight: 0.1,
    jawOpen: 0.05,
    mouthPucker: 0.02,
    eyeLookOutLeft: 0.1,
    eyeLookOutRight: 0.1,
    eyeLookInLeft: 0.1,
    eyeLookInRight: 0.1,
    eyeLookUpLeft: 0.05,
    eyeLookUpRight: 0.05,
    eyeLookDownLeft: 0.05,
    eyeLookDownRight: 0.05,
    ...overrides,
  }).map(([categoryName, score]) => ({ categoryName, score })),
});

test("blendshape signals read closure and centre gaze on the shared axis", () => {
  const neutral = extractBlendshapeSignals(blendshapeSet());
  assert.ok(Math.abs(neutral.gaze - 0.5) < 1e-9, "balanced look-in/out is straight ahead");
  assert.ok(neutral.blink < 0.2);

  const shut = extractBlendshapeSignals(
    blendshapeSet({ eyeBlinkLeft: 0.95, eyeBlinkRight: 0.93 }),
  );
  assert.ok(shut.blink > 0.9);

  const glancing = extractBlendshapeSignals(
    blendshapeSet({ eyeLookOutLeft: 0.8, eyeLookOutRight: 0.8 }),
  );
  assert.ok(glancing.gaze > 0.6, "a sustained outward look moves off centre");
});

test("blendshape extraction degrades to null when the model omits them", () => {
  assert.equal(extractBlendshapeSignals(null), null);
  assert.equal(extractBlendshapeSignals({ categories: [] }), null);
  assert.equal(extractBlendshapeSignals({ categories: [{ categoryName: "jawOpen", score: 1 }] }), null);
});

test("fused closure falls back to geometry when no blendshape is present", () => {
  const result = fuseClosure({ ear: 0.145, earBaseline: 0.29, blink: null });
  assert.equal(result.source, "geometry");
  assert.ok(Math.abs(result.closure - 0.5) < 1e-9);
});

test("fused closure trusts the learned score when the head is turned away", () => {
  // Off-axis foreshortening collapses EAR, which alone would read as a closure.
  const facingForward = fuseClosure({
    ear: 0.06,
    earBaseline: 0.29,
    blink: 0.05,
    poseMagnitude: 0,
  });
  const turnedAway = fuseClosure({
    ear: 0.06,
    earBaseline: 0.29,
    blink: 0.05,
    poseMagnitude: 1.0,
  });
  assert.ok(turnedAway.closure < facingForward.closure);
  assert.ok(turnedAway.closure < 0.45, "an open eye seen off-axis is not a microsleep");
});

test("classification prefers the learned blink over a foreshortened EAR", () => {
  const baseline = { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0, gaze: 0.5 };
  const openButAngled = classifySignals(
    {
      ear: 0.09,
      mar: 0.1,
      yaw: 0.5,
      pitch: 0,
      gaze: 0.5,
      blendshapes: { blink: 0.04 },
    },
    baseline,
  );
  assert.equal(openButAngled.eyesClosed, false);
  assert.equal(openButAngled.closureSource, "fused");

  const genuinelyShut = classifySignals(
    { ear: 0.05, mar: 0.1, yaw: 0, pitch: 0, gaze: 0.5, blendshapes: { blink: 0.95 } },
    baseline,
  );
  assert.equal(genuinelyShut.eyesClosed, true);
});

test("head deviation is judged in degrees, not landmark ratios", () => {
  const baseline = { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0, gaze: 0.5 };
  const base = { ear: 0.29, mar: 0.1, pitch: 0, gaze: 0.5 };
  // ~9 degrees is an ordinary mirror check; ~29 degrees is off-road orientation.
  assert.equal(classifySignals({ ...base, yaw: 0.16 }, baseline).headAway, false);
  assert.equal(classifySignals({ ...base, yaw: 0.51 }, baseline).headAway, true);
});

// --- Time-weighted PERCLOS -------------------------------------------------

/** Build closure samples for `durationMs` at a given rate, starting at `start`. */
const closureRun = (start, durationMs, hz, closure) => {
  const step = 1000 / hz;
  const samples = [];
  for (let time = start; time < start + durationMs; time += step) {
    samples.push({ time, closure });
  }
  return samples;
};

test("PERCLOS measures time closed, not frames counted", () => {
  const now = 100_000;
  // Two seconds shut, then two seconds open, sampled evenly at 30 Hz.
  const samples = [
    ...closureRun(now - 4_000, 2_000, 30, 1),
    ...closureRun(now - 2_000, 2_000, 30, 0),
  ];
  const perclos = calculatePerclos(samples, now);
  assert.ok(Math.abs(perclos - 0.5) < 0.03, `expected ~0.5, received ${perclos}`);
});

test("PERCLOS stays stable when frames drop during the closure", () => {
  const now = 100_000;
  // Same two seconds shut and two seconds open, but the loop stalls to 5 Hz
  // while the eyes are closed - which is exactly when devices slow down.
  const dropped = [
    ...closureRun(now - 4_000, 2_000, 5, 1),
    ...closureRun(now - 2_000, 2_000, 30, 0),
  ];
  const timeWeighted = calculatePerclos(dropped, now);
  assert.ok(
    Math.abs(timeWeighted - 0.5) < 0.05,
    `time weighting should hold near 0.5, received ${timeWeighted}`,
  );

  // Counting samples instead would have understated the closure badly.
  const countedClosed = dropped.filter((sample) => sample.closure >= 0.8).length;
  const sampleCounted = countedClosed / dropped.length;
  assert.ok(
    sampleCounted < 0.25,
    `sample counting drops to ${sampleCounted}, which is the bias being removed`,
  );
});

test("PERCLOS applies the P80 cut and the rolling window", () => {
  const now = 100_000;
  // A drooping-but-open eye at 70% closed is not counted under P80.
  const drooping = closureRun(now - 3_000, 3_000, 30, 0.7);
  assert.equal(calculatePerclos(drooping, now), 0);

  const stale = closureRun(now - 90_000, 3_000, 30, 1);
  assert.equal(calculatePerclos(stale, now), 0, "samples outside the window are ignored");
});

test("PERCLOS still accepts the boolean sample form", () => {
  const now = 100_000;
  const samples = [
    { time: now - 70_000, closed: true },
    { time: now - 300, closed: true },
    { time: now - 200, closed: false },
    { time: now - 100, closed: false },
  ];
  const perclos = calculatePerclos(samples, now);
  assert.ok(perclos >= 0 && perclos <= 1);
});

test("longest closure finds the worst uninterrupted run", () => {
  const now = 100_000;
  const samples = [
    ...closureRun(now - 6_000, 400, 30, 1),
    ...closureRun(now - 5_600, 600, 30, 0),
    ...closureRun(now - 5_000, 1_800, 30, 1),
    ...closureRun(now - 3_200, 600, 30, 0),
  ];
  const longest = longestClosure(samples);
  assert.ok(longest > 1_600 && longest < 1_900, `received ${longest}`);
});

// --- Yawn versus speech ----------------------------------------------------

/** A steady, sustained wide opening: the shape of a real yawn. */
const yawnHistory = (start = 0) => {
  const samples = [];
  for (let index = 0; index < 90; index += 1) {
    const time = start + index * 33;
    const elapsed = index * 33;
    // Slow rise, long hold, slow fall.
    const mar = elapsed < 600 ? 0.1 + (elapsed / 600) * 0.4 : 0.5;
    samples.push({ time, mar, closure: 0.45 });
  }
  return samples;
};

/** Speech reaches the same peak but oscillates several times a second. */
const speechHistory = (start = 0) => {
  const samples = [];
  for (let index = 0; index < 90; index += 1) {
    const time = start + index * 33;
    // Stays above the opening threshold throughout, so only the oscillation
    // gate - not the duration gate - can reject it.
    const mar = 0.5 + Math.sin(index * 0.9) * 0.1;
    samples.push({ time, mar, closure: 0.05 });
  }
  return samples;
};

test("a sustained steady opening is recognized as a yawn", () => {
  const result = detectYawn(yawnHistory());
  assert.equal(result.active, true);
  assert.ok(result.confidence >= 0.5);
  assert.ok(result.durationMs > 1_200);
});

test("talking is not mistaken for a yawn despite the same mouth opening", () => {
  const result = detectYawn(speechHistory());
  assert.equal(result.active, false, "speech oscillates and must not read as fatigue");
  assert.ok(result.oscillation > 2.2);
});

test("a brief wide opening is too short to be a yawn", () => {
  const brief = yawnHistory().slice(0, 25);
  assert.equal(detectYawn(brief).active, false);
});

test("yawn detection needs a real buffer before it decides", () => {
  assert.equal(detectYawn([]).active, false);
  assert.equal(detectYawn(null).active, false);
});

// --- Confidence and its effect on risk -------------------------------------

test("confidence collapses when the frame cannot be read", () => {
  const good = estimateSignalConfidence({
    brightness: 0.45,
    contrast: 0.2,
    fps: 30,
    faceFound: true,
    blendshapeAvailable: true,
  });
  assert.ok(good > 0.85, `a clear frame should be trusted, received ${good}`);

  assert.equal(estimateSignalConfidence({ faceFound: false }), 0);
  assert.ok(estimateSignalConfidence({ brightness: 0.04, contrast: 0.01, fps: 30 }) < 0.2);
  assert.ok(estimateSignalConfidence({ brightness: 0.45, contrast: 0.2, fps: 4 }) < 0.35);
  assert.ok(
    estimateSignalConfidence({ brightness: 0.45, contrast: 0.2, fps: 30, yawMagnitude: 1.2 }) < 0.4,
    "an extreme head angle is not a reliable read",
  );
});

test("low confidence softens the risk claim without silencing it", () => {
  const inputs = { eyeClosedMs: 1_600, perclos: 0.3, gazeAwayMs: 1_800 };
  const confident = calculateRisk({ ...inputs, confidence: 1 });
  const unsure = calculateRisk({ ...inputs, confidence: 0.2 });
  assert.ok(unsure.score < confident.score);
  assert.ok(unsure.score > 0, "an unreadable frame still reports what evidence it has");
  assert.equal(unsure.confidence, 0.2);
});

test("risk defaults are unchanged when confidence is not supplied", () => {
  const explicit = calculateRisk({ eyeClosedMs: 1_500, confidence: 1, contextGain: 1 });
  const implicit = calculateRisk({ eyeClosedMs: 1_500 });
  assert.equal(explicit.score, implicit.score);
});

// --- Circadian context -----------------------------------------------------

test("circadian context is bounded and never manufactures an alert", () => {
  const trough = circadianRisk(4, 0);
  const midday = circadianRisk(11, 0);
  assert.ok(trough.multiplier > midday.multiplier);
  assert.ok(trough.multiplier <= 1.18, "context is capped so evidence still leads");
  assert.equal(midday.multiplier, 1);

  // Context alone must not push a calm driver into an alert state.
  const calm = calculateRisk({ eyeClosedMs: 0, perclos: 0.02, contextGain: trough.multiplier });
  assert.equal(calm.state, "focused");
});

test("circadian context names its factors and flags an overdue break", () => {
  const long = circadianRisk(15, 150);
  assert.ok(long.factors.some((factor) => factor.key === "afternoon"));
  assert.ok(long.factors.some((factor) => factor.key === "timeOnTask"));
  assert.equal(long.breakDue, true);
  assert.equal(circadianRisk(11, 20).breakDue, false);
});

// --- Counterfactual explanation --------------------------------------------

test("the explanation names the change that would lower the score most", () => {
  const assessment = calculateRisk({ eyeClosedMs: 1_900, perclos: 0.3, gazeAwayMs: 900 });
  const explanation = explainRisk(assessment);
  assert.equal(explanation.key, "eyes");
  assert.ok(explanation.to < explanation.from);
  assert.match(explanation.sentence, /^Risk would fall from \d+ to \d+ if /);
});

test("a calm driver gets no counterfactual to explain", () => {
  assert.equal(explainRisk(calculateRisk({ eyeClosedMs: 0, perclos: 0 })), null);
  assert.equal(explainRisk(null), null);
});

// --- Adaptive baseline -----------------------------------------------------

test("the baseline follows a confident posture shift", () => {
  const anchor = { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0, roll: 0, gaze: 0.5 };
  let baseline = { ...anchor };
  for (let index = 0; index < 200; index += 1) {
    baseline = updateBaseline(baseline, { ...anchor, yaw: 0.1 }, { anchor, confidence: 1 });
  }
  assert.ok(baseline.yaw > 0.02, `expected drift toward the new posture, got ${baseline.yaw}`);
});

test("baseline drift is clamped so it cannot normalize a drowsy face", () => {
  const anchor = { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0, roll: 0, gaze: 0.5 };
  let baseline = { ...anchor };
  // A driver whose eyes are half shut for minutes must not have that accepted.
  for (let index = 0; index < 2_000; index += 1) {
    baseline = updateBaseline(baseline, { ...anchor, ear: 0.12 }, { anchor, confidence: 1 });
  }
  assert.ok(baseline.ear >= anchor.ear * 0.85 - 1e-9, `clamped at 15%, got ${baseline.ear}`);
});

test("the baseline holds still when the system is unsure", () => {
  const anchor = { ear: 0.29, mar: 0.11, yaw: 0, pitch: 0, roll: 0, gaze: 0.5 };
  const held = updateBaseline(anchor, { ...anchor, ear: 0.15 }, { anchor, confidence: 0 });
  assert.equal(held.ear, anchor.ear);
});

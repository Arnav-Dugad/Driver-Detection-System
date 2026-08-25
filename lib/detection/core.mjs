/**
 * Pure, dependency-free signal processing used by the browser vision loop.
 * Keeping these functions separate makes the safety logic deterministic and testable.
 *
 * Angle convention: yaw, pitch, and roll are radians measured from the driver's
 * calibrated neutral pose. Positive yaw turns the head toward the driver's left,
 * positive pitch tips the chin down. Both are derived from MediaPipe's facial
 * transformation matrix when it is available, and approximated from landmark
 * geometry when it is not.
 */

export const DEFAULT_BASELINE = {
  ear: 0.285,
  mar: 0.12,
  yaw: 0,
  pitch: 0,
  roll: 0,
  gaze: 0.5,
};

export const clamp = (value, min = 0, max = 1) =>
  Math.min(max, Math.max(min, value));

export const mean = (values) =>
  values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;

export const median = (values) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
};

export const distance = (a, b) =>
  Math.hypot((a?.x ?? 0) - (b?.x ?? 0), (a?.y ?? 0) - (b?.y ?? 0));

const ramp = (value, start, end) => clamp((value - start) / (end - start));

/**
 * Classify a small grayscale camera sample. Requiring low contrast and a
 * missing face for obstruction avoids treating an ordinary dim cabin as a
 * covered lens.
 */
export function classifyCameraQuality({
  brightness = 1,
  contrast = 1,
  darkPixelRatio = 0,
  faceFound = false,
}) {
  const normalizedBrightness = clamp(brightness);
  const normalizedContrast = clamp(contrast);
  const normalizedDarkRatio = clamp(darkPixelRatio);

  if (
    !faceFound &&
    normalizedContrast < 0.035 &&
    (normalizedBrightness < 0.11 || normalizedDarkRatio > 0.86)
  ) {
    return "obstructed";
  }
  if (normalizedBrightness < 0.24 || normalizedDarkRatio > 0.64) {
    return "low-light";
  }
  return "clear";
}

const eyeAspectRatio = (landmarks, indices) => {
  const [left, upperLeft, upperRight, right, lowerRight, lowerLeft] = indices.map(
    (index) => landmarks[index],
  );
  const horizontal = 2 * distance(left, right);
  if (!horizontal) return 0;
  return (
    (distance(upperLeft, lowerLeft) + distance(upperRight, lowerRight)) /
    horizontal
  );
};

const irisRatio = (landmarks, irisIndices, outerIndex, innerIndex) => {
  const irisCenter = {
    x: mean(irisIndices.map((index) => landmarks[index]?.x ?? 0)),
    y: mean(irisIndices.map((index) => landmarks[index]?.y ?? 0)),
  };
  const outer = landmarks[outerIndex];
  const inner = landmarks[innerIndex];
  const width = distance(outer, inner);
  if (!width) return 0.5;
  return clamp(distance(outer, irisCenter) / width);
};

/**
 * Decompose MediaPipe's 4x4 facial transformation matrix into true Euler
 * angles. The matrix arrives column-major, so R[row][col] lives at
 * data[col * 4 + row]. Each basis column is normalized first because the
 * matrix also carries the face's scale, which would otherwise skew the angles.
 *
 * Rotation order is YXZ (yaw, then pitch, then roll), the order that keeps yaw
 * and pitch independent for an upright head.
 */
export function decomposePose(matrix) {
  const data = matrix?.data ?? matrix;
  if (!data || data.length < 16) return null;

  const columnLength = (column) =>
    Math.hypot(data[column * 4], data[column * 4 + 1], data[column * 4 + 2]) || 1;
  const scaleX = columnLength(0);
  const scaleY = columnLength(1);
  const scaleZ = columnLength(2);

  const m11 = data[0] / scaleX;
  const m21 = data[1] / scaleX;
  const m31 = data[2] / scaleX;
  const m22 = data[5] / scaleY;
  const m13 = data[8] / scaleZ;
  const m23 = data[9] / scaleZ;
  const m33 = data[10] / scaleZ;

  const pitch = Math.asin(-clamp(m23, -1, 1));
  // Near gimbal lock the yaw/roll split is arbitrary, so collapse roll to zero
  // and keep yaw meaningful rather than letting both values oscillate.
  const gimbalLocked = Math.abs(m23) > 0.9999;
  const yaw = gimbalLocked ? Math.atan2(-m31, m11) : Math.atan2(m13, m33);
  const roll = gimbalLocked ? 0 : Math.atan2(m21, m22);

  return { yaw, pitch, roll };
}

const blendshapeScore = (categories, name) =>
  categories?.find((category) => category.categoryName === name)?.score ?? null;

/**
 * Read the blendshape head that MediaPipe already runs alongside the landmarks.
 *
 * These scores are learned rather than geometric, so they stay usable where the
 * landmark ratios struggle: eyeglasses, partial occlusion, and off-axis pose.
 * Returns null when the model was built without a blendshape output.
 */
export function extractBlendshapeSignals(blendshapes) {
  const categories = blendshapes?.categories ?? blendshapes;
  if (!Array.isArray(categories) || !categories.length) return null;

  const blinkLeft = blendshapeScore(categories, "eyeBlinkLeft");
  const blinkRight = blendshapeScore(categories, "eyeBlinkRight");
  if (blinkLeft === null && blinkRight === null) return null;

  const blink = mean([blinkLeft, blinkRight].filter((value) => value !== null));
  const jawOpen = blendshapeScore(categories, "jawOpen") ?? 0;
  const mouthPucker = blendshapeScore(categories, "mouthPucker") ?? 0;

  // Horizontal gaze is the net of looking out versus in on each eye, mapped
  // onto the same 0..1 axis the iris-geometry gaze already uses (0.5 = ahead).
  const lookOut = mean([
    blendshapeScore(categories, "eyeLookOutLeft") ?? 0,
    blendshapeScore(categories, "eyeLookOutRight") ?? 0,
  ]);
  const lookIn = mean([
    blendshapeScore(categories, "eyeLookInLeft") ?? 0,
    blendshapeScore(categories, "eyeLookInRight") ?? 0,
  ]);
  const lookUp = mean([
    blendshapeScore(categories, "eyeLookUpLeft") ?? 0,
    blendshapeScore(categories, "eyeLookUpRight") ?? 0,
  ]);
  const lookDown = mean([
    blendshapeScore(categories, "eyeLookDownLeft") ?? 0,
    blendshapeScore(categories, "eyeLookDownRight") ?? 0,
  ]);

  return {
    blink: clamp(blink),
    blinkLeft: blinkLeft === null ? null : clamp(blinkLeft),
    blinkRight: blinkRight === null ? null : clamp(blinkRight),
    jawOpen: clamp(jawOpen),
    mouthPucker: clamp(mouthPucker),
    gaze: clamp(0.5 + (lookOut - lookIn) / 2),
    gazeVertical: clamp(0.5 + (lookDown - lookUp) / 2),
    // A wide asymmetry means one eye is occluded or mistracked.
    asymmetry:
      blinkLeft === null || blinkRight === null ? 0 : Math.abs(blinkLeft - blinkRight),
  };
}

/**
 * Blend the geometric and learned reads of eye closure into one 0..1 fraction,
 * where 0 is a fully open calibrated eye and 1 is fully shut.
 *
 * EAR is precise head-on but distorts off-axis: pitching the chin down
 * foreshortens the eyelid gap and reads as a closure, while yaw compresses the
 * visible eye width and inflates EAR instead. The blendshape is coarser but is
 * trained through those projections, so it holds up where the ratio does not.
 * Weighting the two by how much each can be trusted right now beats either
 * alone, and the function still works when only EAR is available.
 */
export function fuseClosure({
  ear = 0,
  earBaseline = DEFAULT_BASELINE.ear,
  blink = null,
  poseMagnitude = 0,
}) {
  const safeBaseline = Math.max(0.001, earBaseline);
  const geometric = clamp(1 - ear / safeBaseline);
  if (blink === null || blink === undefined) {
    return { closure: geometric, source: "geometry", geometric, learned: null };
  }

  const learned = clamp(blink);
  // Landmark projection error grows from roughly 15 degrees off-axis and the
  // ratio is unusable by about 40.
  const geometricTrust = clamp(1 - ramp(Math.abs(poseMagnitude), 0.26, 0.7));
  // The learned score is most authoritative at its extremes, where the model is
  // reporting a confidently open or confidently shut eye rather than a midpoint.
  const learnedCertainty = clamp(Math.abs(learned - 0.5) * 2);
  const geometricWeight =
    clamp(0.25 + geometricTrust * 0.45) * (1 - learnedCertainty * 0.45);
  return {
    closure: clamp(geometric * geometricWeight + learned * (1 - geometricWeight)),
    source: "fused",
    geometric,
    learned,
  };
}

/**
 * Derive interpretable geometry from MediaPipe's 478 face landmarks.
 * Outputs are normalized, so they are largely independent of camera resolution.
 *
 * When the facial transformation matrix is supplied, yaw and pitch come from a
 * true rotation decomposition. Otherwise they are approximated from landmark
 * ratios and converted into the same radian scale so every downstream
 * threshold keeps one meaning.
 */
export function extractFaceSignals(landmarks, { matrix = null, blendshapes = null } = {}) {
  if (!landmarks || landmarks.length < 468) return null;

  const leftEar = eyeAspectRatio(landmarks, [362, 385, 387, 263, 373, 380]);
  const rightEar = eyeAspectRatio(landmarks, [33, 160, 158, 133, 153, 144]);
  const mouthWidth = distance(landmarks[78], landmarks[308]);
  const mar = mouthWidth
    ? (distance(landmarks[13], landmarks[14]) +
        distance(landmarks[82], landmarks[312])) /
      (2 * mouthWidth)
    : 0;

  let gaze = 0.5;
  if (landmarks.length >= 478) {
    const leftGaze = irisRatio(landmarks, [473, 474, 475, 476, 477], 362, 263);
    const rightGaze = irisRatio(landmarks, [468, 469, 470, 471, 472], 33, 133);
    gaze = mean([leftGaze, rightGaze]);
  }

  const pose = decomposePose(matrix);
  let yaw;
  let pitch;
  let roll = 0;
  let poseSource;

  if (pose) {
    ({ yaw, pitch, roll } = pose);
    poseSource = "matrix";
  } else {
    // Fallback: nose offset against cheek width approximates sin(yaw), and the
    // nose's height between the eye line and the chin approximates pitch.
    const cheekWidth = distance(landmarks[234], landmarks[454]) || 1;
    const faceCenterX = ((landmarks[234]?.x ?? 0) + (landmarks[454]?.x ?? 0)) / 2;
    const yawRatio = ((landmarks[1]?.x ?? faceCenterX) - faceCenterX) / cheekWidth;
    const eyeMidY = mean([landmarks[33]?.y ?? 0, landmarks[263]?.y ?? 0]);
    const chinY = landmarks[152]?.y ?? eyeMidY + 0.2;
    const pitchRatio =
      ((landmarks[1]?.y ?? eyeMidY) - eyeMidY) / Math.max(0.001, chinY - eyeMidY);
    yaw = Math.asin(clamp(yawRatio * 2, -1, 1));
    // The neutral nose sits about 44% of the way from the eye line to the chin;
    // the full ratio range spans roughly +/- 40 degrees.
    pitch = clamp((pitchRatio - 0.44) * 1.4, -1.2, 1.2);
    poseSource = "geometry";
  }

  const learned = extractBlendshapeSignals(blendshapes);
  const ear = mean([leftEar, rightEar]);
  const poseMagnitude = Math.max(Math.abs(yaw), Math.abs(pitch));

  return {
    ear,
    leftEar,
    rightEar,
    mar,
    yaw,
    pitch,
    roll,
    gaze,
    poseSource,
    blendshapes: learned,
    // Fused closure needs a baseline, so the neutral default is used here and
    // the live loop recomputes it against the driver's calibrated value.
    closure: fuseClosure({
      ear,
      earBaseline: DEFAULT_BASELINE.ear,
      blink: learned?.blink ?? null,
      poseMagnitude,
    }).closure,
  };
}

export function classifySignals(signals, baseline = DEFAULT_BASELINE, sensitivity = 0.62) {
  const adjustedSensitivity = clamp(sensitivity);
  const eyeThreshold = baseline.ear * (0.67 + adjustedSensitivity * 0.12);
  const yawnThreshold = Math.max(0.32, baseline.mar * (2.55 - adjustedSensitivity * 0.35));
  // Sustained head deviation past roughly 17 degrees of yaw or 15 degrees of
  // pitch is the range driver-monitoring work treats as off-road orientation.
  const yawThreshold = 0.36 - adjustedSensitivity * 0.1;
  const pitchThreshold = 0.32 - adjustedSensitivity * 0.09;
  const gazeThreshold = 0.23 - adjustedSensitivity * 0.055;

  const baselineYaw = baseline.yaw ?? 0;
  const baselinePitch = baseline.pitch ?? 0;
  const poseMagnitude = Math.max(
    Math.abs(signals.yaw - baselineYaw),
    Math.abs(signals.pitch - baselinePitch),
  );

  const closure = fuseClosure({
    ear: signals.ear,
    earBaseline: baseline.ear,
    blink: signals.blendshapes?.blink ?? null,
    poseMagnitude,
  });

  // The learned blink score and the geometric threshold have to agree on what
  // "closed" means, so the fused fraction is compared against the same relative
  // cut the EAR threshold represents.
  const closureCut = clamp(1 - eyeThreshold / Math.max(0.001, baseline.ear));

  return {
    eyesClosed: closure.closure > closureCut,
    closure: closure.closure,
    closureSource: closure.source,
    yawning: signals.mar > yawnThreshold,
    headAway:
      Math.abs(signals.yaw - baselineYaw) > yawThreshold ||
      Math.abs(signals.pitch - baselinePitch) > pitchThreshold,
    gazeAway: Math.abs(signals.gaze - (baseline.gaze ?? 0.5)) > gazeThreshold,
    thresholds: {
      eye: eyeThreshold,
      closure: closureCut,
      yawn: yawnThreshold,
      yaw: yawThreshold,
      pitch: pitchThreshold,
      gaze: gazeThreshold,
    },
  };
}

/**
 * Separate a yawn from speech using the shape of the opening rather than its
 * depth alone. A yawn opens slowly, holds wide for over a second, and stays
 * steady while held. Speech reaches similar peak openness but oscillates at
 * roughly 2-5 Hz and never holds, which is why a bare MAR threshold
 * misclassifies conversation as fatigue.
 *
 * `history` is an ascending list of `{ time, mar, closure }` samples.
 */
export function detectYawn(history, { threshold = 0.35, now = null } = {}) {
  const idle = { active: false, confidence: 0, durationMs: 0, oscillation: 0 };
  if (!Array.isArray(history) || history.length < 4) return idle;

  const latest = history[history.length - 1];
  const currentTime = now ?? latest.time;
  if (latest.mar <= threshold) return idle;

  // Walk back to the start of the current continuous opening.
  let startIndex = history.length - 1;
  while (startIndex > 0 && history[startIndex - 1].mar > threshold) {
    startIndex -= 1;
  }
  const plateau = history.slice(startIndex);
  const durationMs = currentTime - plateau[0].time;
  if (durationMs < 1_200) return { ...idle, durationMs };

  // Count direction changes across the hold. Speech reverses constantly; a held
  // yawn drifts in one direction and reverses only a couple of times.
  let reversals = 0;
  let previousDirection = 0;
  for (let index = 1; index < plateau.length; index += 1) {
    const delta = plateau[index].mar - plateau[index - 1].mar;
    if (Math.abs(delta) < 0.004) continue;
    const direction = Math.sign(delta);
    if (previousDirection && direction !== previousDirection) reversals += 1;
    previousDirection = direction;
  }
  const seconds = Math.max(0.001, durationMs / 1000);
  const oscillation = reversals / seconds;
  if (oscillation > 2.2) return { ...idle, durationMs, oscillation };

  // A genuine yawn also narrows the eyes, so closure adds confidence without
  // being required — some drivers yawn with their eyes open.
  const closureSupport = clamp(mean(plateau.map((sample) => sample.closure ?? 0)) / 0.4);
  const holdScore = ramp(durationMs, 1_200, 2_600);
  const steadyScore = clamp(1 - oscillation / 2.2);
  const depthScore = ramp(Math.max(...plateau.map((sample) => sample.mar)), threshold, threshold + 0.2);

  const confidence = clamp(
    holdScore * 0.34 + steadyScore * 0.3 + depthScore * 0.22 + closureSupport * 0.14,
  );
  return { active: confidence >= 0.5, confidence, durationMs, oscillation };
}

/**
 * How much the current frame's measurements can be trusted, on 0..1.
 *
 * Reporting a confident risk score from a frame the system cannot actually read
 * is worse than admitting the gap, so this value both attenuates risk and is
 * shown to the driver directly.
 */
export function estimateSignalConfidence({
  brightness = 0.5,
  contrast = 0.2,
  fps = 30,
  yawMagnitude = 0,
  pitchMagnitude = 0,
  faceFound = true,
  blendshapeAvailable = false,
  asymmetry = 0,
} = {}) {
  if (!faceFound) return 0;

  // Too dark to resolve an eyelid, or so blown out that the lids wash together.
  const lighting = ramp(brightness, 0.1, 0.26) * (1 - ramp(brightness, 0.86, 0.98));
  // A flat frame carries no usable texture even when it is bright.
  const texture = ramp(contrast, 0.02, 0.08);
  // Below roughly 12 fps an ordinary blink can fall between two frames.
  const rate = ramp(fps, 8, 18);
  // Landmark error grows sharply past about 35 degrees off-axis.
  const pose = 1 - ramp(Math.max(Math.abs(yawMagnitude), Math.abs(pitchMagnitude)), 0.61, 1.05);
  // A large left/right disagreement means one eye is occluded or mistracked.
  const symmetry = 1 - ramp(asymmetry, 0.35, 0.7);

  const parts = [lighting, texture, rate, pose, symmetry];
  const worst = Math.min(...parts);
  // The weakest input should dominate without a single soft signal zeroing the
  // whole estimate, so the floor is blended with the average.
  const base = worst * 0.7 + mean(parts) * 0.3;
  return clamp(base * (blendshapeAvailable ? 1 : 0.9));
}

const CIRCADIAN_LABELS = {
  night: "early-morning circadian low",
  afternoon: "afternoon circadian dip",
  timeOnTask: "time at the wheel",
};

/**
 * Bounded context gain from time of day and time on task.
 *
 * Crash and lapse risk is well documented as elevated through the early-morning
 * circadian trough and again in the mid-afternoon dip, and it rises with hours
 * of continuous driving. This nudges the same evidence toward an earlier alert
 * rather than inventing risk on its own, and every factor is reported so the
 * driver can see exactly why.
 */
export function circadianRisk(localHour = 12, minutesOnTask = 0) {
  const hour = ((localHour % 24) + 24) % 24;
  const factors = [];

  // Trough centered near 04:00, tapering out by roughly 06:30.
  const nightDistance = Math.min(Math.abs(hour - 4), Math.abs(hour + 24 - 4), Math.abs(hour - 28));
  const night = clamp(1 - nightDistance / 3.5);
  if (night > 0.05) factors.push({ key: "night", weight: night, label: CIRCADIAN_LABELS.night });

  // Secondary dip centered near 15:00.
  const afternoon = clamp(1 - Math.abs(hour - 15) / 2.2);
  if (afternoon > 0.05) {
    factors.push({ key: "afternoon", weight: afternoon, label: CIRCADIAN_LABELS.afternoon });
  }

  // Continuous driving: negligible for the first hour, saturating near three.
  const onTask = ramp(minutesOnTask, 60, 180);
  if (onTask > 0.05) {
    factors.push({ key: "timeOnTask", weight: onTask, label: CIRCADIAN_LABELS.timeOnTask });
  }

  const combined = clamp(Math.max(night, afternoon) * 0.65 + onTask * 0.35);
  return {
    // Capped at +18% so context can never manufacture an alert by itself.
    multiplier: 1 + combined * 0.18,
    combined,
    factors: factors.sort((left, right) => right.weight - left.weight),
    breakDue: minutesOnTask >= 120,
  };
}

/**
 * Explainable, temporal risk fusion. A single blink should never trigger danger;
 * concurrent and persistent signals receive additional weight.
 */
export function calculateRisk({
  eyeClosedMs = 0,
  perclos = 0,
  yawnActive = false,
  recentYawns = 0,
  headAwayMs = 0,
  gazeAwayMs = 0,
  phoneVisible = false,
  faceMissingMs = 0,
  sensitivity = 0.62,
  confidence = 1,
  contextGain = 1,
}) {
  const gain = 0.88 + clamp(sensitivity) * 0.32;
  const components = {
    eyes: 38 * ramp(eyeClosedMs, 420, 1900),
    perclos: 18 * ramp(perclos, 0.12, 0.38),
    yawn: (yawnActive ? 8 : 0) + 9 * ramp(recentYawns, 1, 4),
    head: 13 * ramp(headAwayMs, 700, 3600),
    gaze: 11 * ramp(gazeAwayMs, 650, 3000),
    phone: phoneVisible ? 28 : 0,
    missing: 14 * ramp(faceMissingMs, 1400, 5200),
  };

  const concurrent = [
    eyeClosedMs > 800,
    perclos > 0.2,
    headAwayMs > 1200,
    gazeAwayMs > 1200,
    phoneVisible,
  ].filter(Boolean).length;
  const synergy = concurrent >= 3 ? 14 : concurrent === 2 ? 7 : 0;

  // A poorly seen frame should soften the claim without silencing it entirely,
  // so confidence scales the score across a floor rather than multiplying to zero.
  const trust = 0.4 + clamp(confidence) * 0.6;
  const raw = Object.values(components).reduce((sum, value) => sum + value, 0) * gain + synergy;
  const score = Math.round(clamp(raw * trust * clamp(contextGain, 1, 1.25), 0, 100));

  const state =
    score >= 76 ? "danger" : score >= 54 ? "warning" : score >= 28 ? "caution" : "focused";
  const primary = Object.entries(components).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "eyes";

  return {
    score,
    state,
    primary,
    components,
    concurrent,
    confidence: clamp(confidence),
    contextGain: clamp(contextGain, 1, 1.25),
    // Retained so an explanation can be rebuilt without re-deriving the inputs.
    scale: gain * trust * clamp(contextGain, 1, 1.25),
    synergy,
  };
}

const COMPONENT_LABELS = {
  eyes: "your eyes stayed open",
  perclos: "your eye closure rate settled",
  yawn: "the yawning stopped",
  head: "your head faced forward",
  gaze: "your gaze returned to the road",
  phone: "the phone were out of sight",
  missing: "your face were fully visible",
};

/**
 * Counterfactual explanation: what the score would become if the single largest
 * contributor were removed. Answers "why did it alert?" in the driver's own
 * terms instead of exposing a weight vector.
 */
export function explainRisk(assessment) {
  const { components, score, scale = 1, synergy = 0 } = assessment ?? {};
  if (!components || !score) return null;

  const ranked = Object.entries(components)
    .filter(([, value]) => value > 0.5)
    .sort((left, right) => right[1] - left[1]);
  if (!ranked.length) return null;

  const [key, value] = ranked[0];
  const total = Object.values(components).reduce((sum, item) => sum + item, 0);
  const withoutTop = Math.round(clamp((total - value) * scale + synergy * scale, 0, 100));
  if (withoutTop >= score) return null;

  return {
    key,
    label: COMPONENT_LABELS[key] ?? key,
    from: score,
    to: withoutTop,
    delta: score - withoutTop,
    sentence: `Risk would fall from ${score} to ${withoutTop} if ${COMPONENT_LABELS[key] ?? key}.`,
    ranked: ranked.map(([name, weight]) => ({ key: name, weight })),
  };
}

export function buildCalibration(samples) {
  if (!samples.length) return DEFAULT_BASELINE;
  const earValues = samples.map((sample) => sample.ear).filter((value) => value > 0.12);
  const sortedEar = [...earValues].sort((a, b) => a - b);
  const openEyeValues = sortedEar.slice(Math.floor(sortedEar.length * 0.35));
  return {
    ear: clamp(median(openEyeValues) || DEFAULT_BASELINE.ear, 0.18, 0.42),
    mar: clamp(median(samples.map((sample) => sample.mar)), 0.04, 0.25),
    yaw: median(samples.map((sample) => sample.yaw ?? 0)),
    pitch: median(samples.map((sample) => sample.pitch ?? 0)),
    roll: median(samples.map((sample) => sample.roll ?? 0)),
    gaze: clamp(median(samples.map((sample) => sample.gaze)), 0.25, 0.75),
  };
}

const DRIFT_LIMIT = 0.15;

/**
 * Slowly re-center the baseline on the driver's current posture.
 *
 * Over a long session people slide down the seat and shift the mirror, which
 * walks the calibrated neutral away from reality and produces false alerts.
 * The update runs only while the driver is confidently focused, and the result
 * is clamped to +/-15% of the original calibration so the system can never
 * adapt its way into accepting a genuinely drowsy face as normal.
 */
export function updateBaseline(current, sample, { anchor, confidence = 1, rate = 0.02 } = {}) {
  if (!current || !sample) return current;
  const origin = anchor ?? current;
  const step = clamp(rate) * clamp(confidence);
  if (step <= 0) return current;

  const drift = (key, min, max) => {
    const observed = sample[key];
    if (typeof observed !== "number" || Number.isNaN(observed)) return current[key];
    const blended = current[key] * (1 - step) + observed * step;
    const base = origin[key] ?? 0;
    const span = Math.abs(base) * DRIFT_LIMIT || DRIFT_LIMIT * 0.35;
    return clamp(blended, Math.max(min, base - span), Math.min(max, base + span));
  };

  return {
    ...current,
    ear: drift("ear", 0.18, 0.42),
    mar: drift("mar", 0.04, 0.25),
    yaw: drift("yaw", -0.6, 0.6),
    pitch: drift("pitch", -0.6, 0.6),
    roll: drift("roll", -0.6, 0.6),
    gaze: drift("gaze", 0.25, 0.75),
  };
}

// A stalled loop must not let one sample claim credit for the whole gap.
const MAX_SAMPLE_GAP_MS = 400;

/**
 * Time-weighted PERCLOS on the P80 definition: the proportion of *time* the
 * eyes are at least 80% closed across the rolling window.
 *
 * Counting samples instead of time biases the measure by frame rate, and frame
 * rate drops hardest during exactly the long closures that matter most, so a
 * sample-counted PERCLOS understates fatigue on slower devices. Each sample is
 * weighted by the wall-clock interval it represents.
 *
 * Samples are `{ time, closure }` where closure is a 0..1 fraction; the older
 * `{ time, closed }` boolean form is still accepted.
 */
export function calculatePerclos(samples, now, windowMs = 60_000, closedAt = 0.8) {
  if (!Array.isArray(samples) || !samples.length) return 0;
  const withinWindow = samples.filter((sample) => now - sample.time <= windowMs);
  if (!withinWindow.length) return 0;
  if (withinWindow.length === 1) {
    return closureFraction(withinWindow[0]) >= closedAt ? 1 : 0;
  }

  let closedMs = 0;
  let totalMs = 0;
  for (let index = 1; index < withinWindow.length; index += 1) {
    const span = Math.min(
      MAX_SAMPLE_GAP_MS,
      Math.max(0, withinWindow[index].time - withinWindow[index - 1].time),
    );
    if (!span) continue;
    totalMs += span;
    // The interval is attributed to the state it ended in.
    if (closureFraction(withinWindow[index]) >= closedAt) closedMs += span;
  }
  if (!totalMs) return 0;
  return closedMs / totalMs;
}

function closureFraction(sample) {
  if (typeof sample.closure === "number") return clamp(sample.closure);
  return sample.closed ? 1 : 0;
}

/** Longest uninterrupted closure in the buffer, in milliseconds. */
export function longestClosure(samples, closedAt = 0.8) {
  if (!Array.isArray(samples) || samples.length < 2) return 0;
  let longest = 0;
  let runStart = null;
  for (const sample of samples) {
    if (closureFraction(sample) >= closedAt) {
      if (runStart === null) runStart = sample.time;
      longest = Math.max(longest, sample.time - runStart);
    } else {
      runStart = null;
    }
  }
  return longest;
}

export function formatDuration(totalSeconds) {
  const safeSeconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const seconds = safeSeconds % 60;
  return hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

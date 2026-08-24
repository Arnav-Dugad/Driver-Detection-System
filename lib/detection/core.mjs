/**
 * Pure, dependency-free signal processing used by the browser vision loop.
 * Keeping these functions separate makes the safety logic deterministic and testable.
 */

export const DEFAULT_BASELINE = {
  ear: 0.285,
  mar: 0.12,
  yaw: 0,
  pitch: 0.44,
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
 * Derive interpretable geometry from MediaPipe's 478 face landmarks.
 * Outputs are normalized, so they are largely independent of camera resolution.
 */
export function extractFaceSignals(landmarks) {
  if (!landmarks || landmarks.length < 468) return null;

  const leftEar = eyeAspectRatio(landmarks, [362, 385, 387, 263, 373, 380]);
  const rightEar = eyeAspectRatio(landmarks, [33, 160, 158, 133, 153, 144]);
  const mouthWidth = distance(landmarks[78], landmarks[308]);
  const mar = mouthWidth
    ? (distance(landmarks[13], landmarks[14]) +
        distance(landmarks[82], landmarks[312])) /
      (2 * mouthWidth)
    : 0;

  const cheekWidth = distance(landmarks[234], landmarks[454]) || 1;
  const faceCenterX = ((landmarks[234]?.x ?? 0) + (landmarks[454]?.x ?? 0)) / 2;
  const yaw = ((landmarks[1]?.x ?? faceCenterX) - faceCenterX) / cheekWidth;

  const eyeMidY = mean([landmarks[33]?.y ?? 0, landmarks[263]?.y ?? 0]);
  const chinY = landmarks[152]?.y ?? eyeMidY + 0.2;
  const pitchDenominator = Math.max(0.001, chinY - eyeMidY);
  const pitch = ((landmarks[1]?.y ?? eyeMidY) - eyeMidY) / pitchDenominator;

  let gaze = 0.5;
  if (landmarks.length >= 478) {
    const leftGaze = irisRatio(landmarks, [473, 474, 475, 476, 477], 362, 263);
    const rightGaze = irisRatio(landmarks, [468, 469, 470, 471, 472], 33, 133);
    gaze = mean([leftGaze, rightGaze]);
  }

  return {
    ear: mean([leftEar, rightEar]),
    leftEar,
    rightEar,
    mar,
    yaw,
    pitch,
    gaze,
  };
}

export function classifySignals(signals, baseline = DEFAULT_BASELINE, sensitivity = 0.62) {
  const adjustedSensitivity = clamp(sensitivity);
  const eyeThreshold = baseline.ear * (0.67 + adjustedSensitivity * 0.12);
  const yawnThreshold = Math.max(0.32, baseline.mar * (2.55 - adjustedSensitivity * 0.35));
  const yawThreshold = 0.105 - adjustedSensitivity * 0.025;
  const pitchThreshold = 0.15 - adjustedSensitivity * 0.035;
  const gazeThreshold = 0.23 - adjustedSensitivity * 0.055;

  return {
    eyesClosed: signals.ear < eyeThreshold,
    yawning: signals.mar > yawnThreshold,
    headAway:
      Math.abs(signals.yaw - baseline.yaw) > yawThreshold ||
      Math.abs(signals.pitch - baseline.pitch) > pitchThreshold,
    gazeAway: Math.abs(signals.gaze - baseline.gaze) > gazeThreshold,
    thresholds: {
      eye: eyeThreshold,
      yawn: yawnThreshold,
      yaw: yawThreshold,
      pitch: pitchThreshold,
      gaze: gazeThreshold,
    },
  };
}

const ramp = (value, start, end) => clamp((value - start) / (end - start));

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
  const score = Math.round(
    clamp(
      Object.values(components).reduce((sum, value) => sum + value, 0) * gain + synergy,
      0,
      100,
    ),
  );

  const state =
    score >= 76 ? "danger" : score >= 54 ? "warning" : score >= 28 ? "caution" : "focused";
  const primary = Object.entries(components).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "eyes";

  return { score, state, primary, components, concurrent };
}

export function buildCalibration(samples) {
  if (!samples.length) return DEFAULT_BASELINE;
  const earValues = samples.map((sample) => sample.ear).filter((value) => value > 0.12);
  const sortedEar = [...earValues].sort((a, b) => a - b);
  const openEyeValues = sortedEar.slice(Math.floor(sortedEar.length * 0.35));
  return {
    ear: clamp(median(openEyeValues) || DEFAULT_BASELINE.ear, 0.18, 0.42),
    mar: clamp(median(samples.map((sample) => sample.mar)), 0.04, 0.25),
    yaw: median(samples.map((sample) => sample.yaw)),
    pitch: median(samples.map((sample) => sample.pitch)),
    gaze: clamp(median(samples.map((sample) => sample.gaze)), 0.25, 0.75),
  };
}

export function calculatePerclos(samples, now, windowMs = 60_000) {
  const withinWindow = samples.filter((sample) => now - sample.time <= windowMs);
  if (!withinWindow.length) return 0;
  return withinWindow.filter((sample) => sample.closed).length / withinWindow.length;
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

/**
 * The live per-frame state machine from GuardianDashboard, without React,
 * timers, audio, or a camera.
 *
 * The dashboard's frame loop and this tracker step through the same sequence:
 * five-second calibration, signal classification, time-weighted PERCLOS, blink
 * and closure timing, head and gaze persistence, yawn morphology, risk fusion,
 * exponential smoothing, and bounded baseline drift. Having it as a pure
 * function of recorded frames is what lets `ml/replay_rule_engine.mjs` score the
 * deterministic engine on public datasets exactly as it behaves in the app, so
 * it can be compared with learned models on equal terms.
 *
 * Circadian context is deliberately off by default here: an offline replay has
 * no meaningful wall-clock hour, and the term can only raise a score anyway.
 */

import {
  DEFAULT_BASELINE,
  buildCalibration,
  calculatePerclos,
  calculateRisk,
  classifySignals,
  detectYawn,
  estimateSignalConfidence,
  updateBaseline,
} from "./core.mjs";
import { buildFeatureWindow } from "./features.mjs";

const stateFor = (risk) =>
  risk >= 76 ? "danger" : risk >= 54 ? "warning" : risk >= 28 ? "caution" : "focused";

/**
 * @param {object} [options]
 * @param {number} [options.sensitivity] Same 0..1 setting as the app.
 * @param {number} [options.calibrationMs] Length of the calibration phase.
 * @param {number} [options.featureWindowMs] Window for buildFeatureWindow().
 * @param {object} [options.baseline] A stored personal baseline. When given,
 *   calibration is skipped, as for a driver who enrolled once while fresh.
 */
export function createTracker({
  sensitivity = 0.62,
  calibrationMs = 5_200,
  featureWindowMs = 60_000,
  baseline: enrolled = null,
} = {}) {
  let phase = enrolled ? "live" : "calibrating";
  let startedAt = null;
  let calibration = [];
  let baseline = enrolled ?? DEFAULT_BASELINE;
  let anchor = enrolled ?? DEFAULT_BASELINE;
  let closureSamples = [];
  let marHistory = [];
  let telemetry = [];
  let blinkTimes = [];
  let yawnTimes = [];
  let yawnFlag = false;
  let eyeClosedAt = 0;
  let headAwayAt = 0;
  let gazeAwayAt = 0;
  let faceMissingAt = 0;
  let smoothed = 0;
  let lastTime = null;

  const keep = (list, now, ms) => list.filter((item) => now - (item.time ?? item) <= ms);

  /**
   * Advance one frame.
   * `frame` = { time, faceFound, signals?, brightness?, contrast?, phoneVisible? }
   * where `signals` has the shape extractFaceSignals() returns.
   */
  function step(frame) {
    const now = frame.time;
    if (startedAt === null) startedAt = now;
    const fps = lastTime === null || now === lastTime ? 30 : 1000 / (now - lastTime);
    lastTime = now;
    const signals = frame.faceFound ? frame.signals : null;

    if (phase === "calibrating") {
      if (signals) calibration.push(signals);
      if (now - startedAt >= calibrationMs && calibration.length) {
        baseline = buildCalibration(calibration);
        anchor = baseline;
        phase = "live";
      }
      return { phase, risk: 0, state: "focused" };
    }

    if (!signals) {
      if (!faceMissingAt) faceMissingAt = now;
      const assessment = calculateRisk({ faceMissingMs: now - faceMissingAt, sensitivity });
      smoothed = smoothed * 0.84 + assessment.score * 0.16;
      telemetry.push({
        time: now, ear: 0, mar: 0, closure: 0, yaw: 0, pitch: 0, gaze: 0.5,
        risk: Math.round(smoothed), phoneVisible: false, faceFound: false,
      });
      telemetry = keep(telemetry, now, featureWindowMs);
      const risk = Math.round(smoothed);
      // As in the app, a missing face alone tops out at "warning".
      const state = risk >= 54 ? "warning" : risk >= 28 ? "caution" : "focused";
      return { phase, risk, state, primary: "missing", faceFound: false };
    }
    faceMissingAt = 0;

    const classified = classifySignals(signals, baseline, sensitivity);
    const confidence = estimateSignalConfidence({
      brightness: frame.brightness ?? 0.5,
      contrast: frame.contrast ?? 0.2,
      fps,
      yawMagnitude: Math.abs(signals.yaw - baseline.yaw),
      pitchMagnitude: Math.abs(signals.pitch - baseline.pitch),
      faceFound: true,
      blendshapeAvailable: Boolean(signals.blendshapes),
      asymmetry: signals.blendshapes?.asymmetry ?? 0,
    });

    closureSamples.push({ time: now, closure: classified.closure });
    closureSamples = keep(closureSamples, now, 60_000);
    marHistory.push({ time: now, mar: signals.mar, closure: classified.closure });
    marHistory = keep(marHistory, now, 8_000);
    const perclos = calculatePerclos(closureSamples, now);

    let blink = false;
    if (classified.eyesClosed) {
      if (!eyeClosedAt) eyeClosedAt = now;
    } else if (eyeClosedAt) {
      const closure = now - eyeClosedAt;
      if (closure > 70 && closure < 750) {
        blink = true;
        blinkTimes.push(now);
      }
      eyeClosedAt = 0;
    }
    blinkTimes = keep(blinkTimes, now, 60_000);
    headAwayAt = classified.headAway ? headAwayAt || now : 0;
    gazeAwayAt = classified.gazeAway ? gazeAwayAt || now : 0;

    const yawn = detectYawn(marHistory, { threshold: classified.thresholds.yawn, now });
    let yawnStarted = false;
    if (yawn.active && !yawnFlag) {
      yawnFlag = true;
      yawnStarted = true;
      yawnTimes.push(now);
    } else if (!yawn.active) {
      yawnFlag = false;
    }
    yawnTimes = keep(yawnTimes, now, 10 * 60_000);

    const eyeClosedMs = eyeClosedAt ? now - eyeClosedAt : 0;
    const headAwayMs = headAwayAt ? now - headAwayAt : 0;
    const gazeAwayMs = gazeAwayAt ? now - gazeAwayAt : 0;
    const assessment = calculateRisk({
      eyeClosedMs,
      perclos,
      yawnActive: yawn.active,
      recentYawns: yawnTimes.length,
      headAwayMs,
      gazeAwayMs,
      phoneVisible: Boolean(frame.phoneVisible),
      sensitivity,
      confidence,
    });
    smoothed = smoothed * 0.78 + assessment.score * 0.22;
    const risk = Math.round(smoothed);
    const state = stateFor(risk);

    if (state === "focused" && confidence > 0.7 && !classified.eyesClosed) {
      baseline = updateBaseline(baseline, signals, { anchor, confidence });
    }

    telemetry.push({
      time: now,
      ear: signals.ear,
      mar: signals.mar,
      closure: classified.closure,
      yaw: signals.yaw,
      pitch: signals.pitch,
      gaze: signals.gaze,
      risk,
      phoneVisible: Boolean(frame.phoneVisible),
      faceFound: true,
    });
    telemetry = keep(telemetry, now, featureWindowMs);

    return {
      phase,
      risk,
      rawRisk: assessment.score,
      state,
      primary: assessment.primary,
      perclos,
      closure: classified.closure,
      eyesClosed: classified.eyesClosed,
      eyeClosedMs,
      headAway: classified.headAway,
      gazeAway: classified.gazeAway,
      yawnActive: yawn.active,
      yawnStarted,
      blink,
      confidence,
      faceFound: true,
    };
  }

  /** The labeled-window features the research recorder would write now. */
  function features(now = lastTime) {
    return buildFeatureWindow(telemetry, {
      baseline: anchor,
      now,
      windowMs: featureWindowMs,
      blinkTimes,
      yawnTimes,
    });
  }

  return {
    step,
    features,
    get phase() {
      return phase;
    },
    get baseline() {
      return baseline;
    },
  };
}

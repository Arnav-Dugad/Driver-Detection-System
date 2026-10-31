/**
 * Score recorded per-frame signals with the app's deterministic engine.
 *
 *   node ml/replay_rule_engine.mjs <input.json> <output.json>
 *
 * The input is written by ml/datasets.py: a list of clips, each with columnar
 * per-frame arrays. Every clip gets a fresh tracker, so it calibrates on its own
 * first five seconds exactly as a new session in the app would. The output keeps
 * the per-frame risk trace and the feature window the research recorder would
 * have written at the end of the clip.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { createTracker } from "../lib/detection/tracker.mjs";

const [input, output, windowArg] = process.argv.slice(2);
if (!input || !output) {
  console.error("usage: node ml/replay_rule_engine.mjs <input.json> <output.json> [windowMs]");
  process.exit(2);
}
const featureWindowMs = Number(windowArg ?? 30_000);
const clips = JSON.parse(readFileSync(input, "utf8"));

const results = clips.map((clip) => {
  // A clip may carry an enrolled baseline; otherwise it calibrates on its start.
  const tracker = createTracker({ featureWindowMs, baseline: clip.baseline ?? null });
  const risk = [];
  const rawRisk = [];
  const closure = [];
  const eyesClosed = [];
  const yawnActive = [];
  const blinks = [];
  const yawns = [];
  const n = clip.time_ms.length;
  for (let index = 0; index < n; index += 1) {
    const faceFound = Boolean(clip.face_found[index]);
    const blink = clip.blink?.[index];
    const result = tracker.step({
      time: clip.time_ms[index],
      faceFound,
      brightness: clip.brightness?.[index],
      contrast: clip.contrast?.[index],
      signals: faceFound
        ? {
            ear: clip.ear[index],
            mar: clip.mar[index],
            yaw: clip.yaw[index],
            pitch: clip.pitch[index],
            roll: clip.roll[index],
            gaze: clip.gaze[index],
            blendshapes:
              blink === null || blink === undefined || Number.isNaN(blink)
                ? null
                : { blink, asymmetry: clip.asymmetry?.[index] ?? 0 },
          }
        : null,
    });
    risk.push(result.risk ?? 0);
    rawRisk.push(result.rawRisk ?? result.risk ?? 0);
    closure.push(result.closure ?? null);
    eyesClosed.push(result.eyesClosed ? 1 : 0);
    yawnActive.push(result.yawnActive ? 1 : 0);
    if (result.blink) blinks.push(index);
    if (result.yawnStarted) yawns.push(index);
  }
  return {
    id: clip.id,
    calibrated: tracker.phase === "live",
    baseline: tracker.baseline,
    risk,
    rawRisk,
    closure,
    eyesClosed,
    yawnActive,
    blinks,
    yawns,
    features: tracker.features(),
  };
});

writeFileSync(output, JSON.stringify(results));
console.log(`replayed ${results.length} clips with the deterministic engine`);

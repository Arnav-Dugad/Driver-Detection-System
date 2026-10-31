import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { isEyeModel, predictClosed, prepareEyeModel, standardizeCrop } from "../lib/detection/eyeNet.mjs";
import {
  SEQUENCE_CHANNELS,
  createSequenceBuffer,
  isSequenceModel,
  predictSequence,
  prepareSequenceModel,
  sequenceFrame,
  sequenceRisk,
} from "../lib/detection/sequence.mjs";
import { createTracker } from "../lib/detection/tracker.mjs";

const load = (name) =>
  JSON.parse(readFileSync(new URL(`../public/models/${name}`, import.meta.url), "utf8"));

const eyeModel = load("eye-state-cnn.json");
const sequenceModel = load("drowsiness-gru.json");

test("the shipped eye CNN reproduces PyTorch outputs", () => {
  assert.ok(isEyeModel(eyeModel));
  const prepared = prepareEyeModel(eyeModel);
  assert.ok(eyeModel.test_vectors.length >= 3);
  for (const vector of eyeModel.test_vectors) {
    const p = predictClosed(prepared, Float32Array.from(vector.pixels));
    assert.ok(Math.abs(p - vector.p_closed) < 2e-3, `JS ${p} vs PyTorch ${vector.p_closed}`);
  }
});

test("eye CNN rejects malformed payloads and inputs", () => {
  assert.equal(prepareEyeModel({ format: "something-else" }), null);
  assert.equal(predictClosed(prepareEyeModel(eyeModel), new Float32Array(10)), null);
});

test("crop standardization removes brightness and contrast", () => {
  const crop = Float32Array.from({ length: 1024 }, (_, i) => (i % 32) * 4);
  const brighter = crop.map((value) => value * 1.5 + 40);
  const a = standardizeCrop(crop);
  const b = standardizeCrop(brighter);
  // Not identical because of the +8 stabiliser, but the same pattern.
  assert.ok(Math.abs(a[100] - b[100]) < 0.1);
});

test("the shipped GRU reproduces PyTorch outputs", () => {
  assert.ok(isSequenceModel(sequenceModel));
  assert.deepEqual(sequenceModel.channels, SEQUENCE_CHANNELS);
  const prepared = prepareSequenceModel(sequenceModel);
  for (const vector of sequenceModel.test_vectors) {
    const probabilities = predictSequence(prepared, vector.frames);
    sequenceModel.classes.forEach((name, index) => {
      assert.ok(
        Math.abs(probabilities[name] - vector.probabilities[index]) < 1e-3,
        `${name}: JS ${probabilities[name]} vs PyTorch ${vector.probabilities[index]}`,
      );
    });
  }
});

test("sequence frames follow the training channel order", () => {
  const vector = sequenceFrame(
    {
      ear: 0.3,
      mar: 0.2,
      yaw: 0.1,
      pitch: -0.1,
      roll: 0,
      gaze: 0.55,
      blendshapes: { blink: 0.2, squint: 0.3, wide: 0.1, jawOpen: 0.05, gaze: 0.6, browInnerUp: 0.4 },
    },
    { ear: 0.3, mar: 0.12, yaw: 0, pitch: 0, roll: 0, gaze: 0.5 },
    0.1,
  );
  assert.equal(vector.length, SEQUENCE_CHANNELS.length);
  assert.ok(Math.abs(vector[SEQUENCE_CHANNELS.indexOf("ear_rel")] - 1) < 1e-6);
  assert.ok(Math.abs(vector[SEQUENCE_CHANNELS.indexOf("mar_rel")] - 0.08) < 1e-6);
  assert.equal(vector[SEQUENCE_CHANNELS.indexOf("face_found")], 1);
  assert.equal(sequenceFrame(null, {}, 0)[SEQUENCE_CHANNELS.indexOf("face_found")], 0);
});

test("the sequence buffer resamples a variable frame rate onto 15 Hz", () => {
  const buffer = createSequenceBuffer({ seconds: 2 });
  let time = 0;
  while (time < 2_500) {
    buffer.push(time, new Float32Array(SEQUENCE_CHANNELS.length));
    time += 20 + (time % 3) * 15; // 20-50 ms jitter
  }
  assert.equal(buffer.length, 30);
  assert.ok(buffer.full);
});

test("sequence risk weights low vigilance halfway", () => {
  assert.equal(sequenceRisk({ alert: 1, low_vigilant: 0, drowsy: 0 }), 0);
  assert.equal(sequenceRisk({ alert: 0, low_vigilant: 1, drowsy: 0 }), 50);
  assert.equal(sequenceRisk({ alert: 0, low_vigilant: 0, drowsy: 1 }), 100);
});

test("the offline tracker calibrates, then scores a long closure as risk", () => {
  const tracker = createTracker();
  const open = { ear: 0.3, mar: 0.12, yaw: 0, pitch: 0, roll: 0, gaze: 0.5, blendshapes: { blink: 0.05, asymmetry: 0 } };
  const shut = { ...open, ear: 0.05, blendshapes: { blink: 0.95, asymmetry: 0 } };
  let result;
  for (let i = 0; i < 90; i += 1) result = tracker.step({ time: i * 66, faceFound: true, signals: open });
  assert.equal(tracker.phase, "live");
  const calm = result.risk;
  for (let i = 90; i < 150; i += 1) result = tracker.step({ time: i * 66, faceFound: true, signals: shut });
  assert.ok(result.eyeClosedMs > 3_000);
  assert.ok(result.risk > calm + 25, `risk ${result.risk} after a 4 s closure`);
  const features = tracker.features();
  assert.ok(features.perclos_60s > 0.3);
});

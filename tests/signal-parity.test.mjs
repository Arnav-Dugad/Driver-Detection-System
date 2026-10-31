import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { extractFaceSignals } from "../lib/detection/core.mjs";
import { eyeCropTransform } from "../lib/detection/eyeCrop.mjs";

// Real MediaPipe frames recorded by ml/make_parity_fixture.py, with the values
// ml/signals.py computed from them. The offline experiments are only valid for
// the app if both implementations agree.
const frames = JSON.parse(
  readFileSync(new URL("./fixtures/signal-parity.json", import.meta.url), "utf8"),
);

const close = (actual, expected, label) =>
  assert.ok(
    Math.abs(actual - expected) < 1e-4,
    `${label}: JavaScript ${actual} vs Python ${expected}`,
  );

test("the fixture holds real frames", () => {
  assert.ok(frames.length >= 3);
  assert.equal(frames[0].landmarks.length, 478);
});

test("JavaScript signals match the Python port on recorded MediaPipe output", () => {
  for (const frame of frames) {
    const signals = extractFaceSignals(
      frame.landmarks.map(([x, y]) => ({ x, y })),
      {
        matrix: { data: frame.matrix },
        blendshapes: frame.blendshapes,
        aspect: frame.width / frame.height,
      },
    );
    const e = frame.expected;
    close(signals.ear, e.ear, "ear");
    close(signals.leftEar, e.left_ear, "leftEar");
    close(signals.rightEar, e.right_ear, "rightEar");
    close(signals.mar, e.mar, "mar");
    close(signals.gaze, e.gaze, "gaze");
    close(signals.yaw, e.yaw, "yaw");
    close(signals.pitch, e.pitch, "pitch");
    close(signals.roll, e.roll, "roll");
    close(signals.closure, e.closure, "closure");
    close(signals.blendshapes.blink, e.blink, "blink");
    close(signals.blendshapes.jawOpen, e.jaw_open, "jawOpen");
    close(signals.blendshapes.gaze, e.bs_gaze, "blendshape gaze");
  }
});

test("aspect correction puts EAR in the physiological range on a non-square frame", () => {
  // The fixture clip is a portrait phone video (9:16), where uncorrected EAR
  // is deflated; a landscape webcam inflates it by the same factor.
  const frame = frames[0];
  const aspect = frame.width / frame.height;
  const landmarks = frame.landmarks.map(([x, y]) => ({ x, y }));
  const corrected = extractFaceSignals(landmarks, { aspect });
  const uncorrected = extractFaceSignals(landmarks);
  assert.ok(corrected.ear > 0.2 && corrected.ear < 0.4, `corrected EAR ${corrected.ear}`);
  const distortion = uncorrected.ear / corrected.ear;
  assert.ok(Math.abs(distortion - aspect) < 0.1 * aspect, `distortion ${distortion} vs aspect ${aspect}`);
});

test("eye crop transforms match the training crop", () => {
  for (const frame of frames) {
    const landmarks = frame.landmarks.map(([x, y]) => ({ x, y }));
    for (const side of ["left", "right"]) {
      const [a, b, c, d, e, f] = eyeCropTransform(landmarks, side, frame.width, frame.height);
      const [[pa, pc, pe], [pb, pd, pf]] = frame.crops[side];
      for (const [js, py] of [[a, pa], [b, pb], [c, pc], [d, pd], [e, pe], [f, pf]]) {
        assert.ok(Math.abs(js - py) < 1e-3, `${side} crop transform ${js} vs ${py}`);
      }
    }
  }
});

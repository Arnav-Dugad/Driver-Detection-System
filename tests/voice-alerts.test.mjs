import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../app/GuardianDashboard.tsx", import.meta.url),
  "utf8",
);

test("ships six multilingual warning catalogs", () => {
  for (const locale of ["en-IN", "hi-IN", "kn-IN", "mr-IN", "ta-IN", "te-IN"]) {
    assert.match(source, new RegExp(`"${locale}": \\{`));
  }
  assert.match(source, /id="voice-language"/);
  assert.match(source, /Kannada/);
  assert.match(source, /ಕನ್ನಡ/);
});

test("uses contextual, rate-limited voice guidance", () => {
  for (const alertKind of [
    "sessionStart",
    "calibration",
    "eyes",
    "perclos",
    "yawn",
    "repeatedYawn",
    "gaze",
    "head",
    "phone",
    "missing",
    "warning",
    "danger",
    "recovery",
  ]) {
    assert.match(source, new RegExp(`${alertKind}:`));
  }
  assert.match(source, /selectNaturalVoice/);
  assert.match(source, /resolveVoice/);
  assert.match(source, /\["hi-IN", "en-IN"\]/);
  assert.match(source, /VOICE_ALERTS\[resolution\.language\]\[kind\]/);
  assert.match(source, /speechSynthesis\.getVoices/);
  assert.match(source, /speechSynthesis\.resume/);
  assert.match(source, /VOICE_COOLDOWNS/);
  assert.match(source, /speechSynthesis\.cancel/);
});

test("starts in precision mode and keeps the camera surface clean", () => {
  assert.match(source, /performance: "precision"/);
  assert.doesNotMatch(source, /MULTIMODAL FUSION CORE/);
  assert.doesNotMatch(source, /scan-beam|hud-corner|vision-telemetry|canvasRef/);
});

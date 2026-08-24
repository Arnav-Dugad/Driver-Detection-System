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
    "lowLight",
    "obstruction",
    "warning",
    "danger",
    "recovery",
  ]) {
    assert.match(source, new RegExp(`${alertKind}:`));
  }
  assert.match(source, /selectNaturalVoice/);
  assert.match(source, /VOICE_ALERTS\[language\]\[kind\]/);
  assert.match(source, /utterance\.lang = naturalVoice\?\.lang \|\| language/);
  assert.match(source, /speechSynthesis\.getVoices/);
  assert.doesNotMatch(source, /resolveVoice|speechSynthesis\.resume/);
  assert.doesNotMatch(source, /voices\.find\(\(voice\) => voice\.default\)|voice: voices\[0\]/);
  assert.match(source, /VOICE_COOLDOWNS/);
  assert.match(source, /speechSynthesis\.cancel/);
});

test("starts in precision mode with a restrained face tracker", () => {
  assert.match(source, /performance: "precision"/);
  assert.match(source, /className="face-tracker"/);
  assert.match(source, /index \+= 6/);
  assert.doesNotMatch(source, /MULTIMODAL FUSION CORE/);
  assert.doesNotMatch(source, /scan-beam|hud-corner|vision-telemetry/);
});

test("includes persistent dark mode and sustained camera-quality warnings", () => {
  assert.match(source, /driver-detection-theme/);
  assert.match(source, /document\.documentElement\.dataset\.theme/);
  assert.match(source, /measureCameraFrame/);
  assert.match(source, /classifyCameraQuality/);
  assert.match(source, /qualityCandidateRef\.current\.count >= 2/);
  assert.match(source, /soundAlert\(false, false, flag\)/);
});

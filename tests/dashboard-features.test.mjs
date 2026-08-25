import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(path, import.meta.url), "utf8");

const dashboard = await read("../app/GuardianDashboard.tsx");
const driveMode = await read("../app/DriveMode.tsx");
const hooks = await read("../app/hooks.ts");
const styles = await read("../app/globals.css");
const worker = await read("../public/sw.js");
const manifest = JSON.parse(await read("../public/manifest.webmanifest"));

test("the vision loop consumes the blendshapes and pose matrix it asks for", () => {
  // Both outputs were being computed by MediaPipe and thrown away.
  assert.match(dashboard, /outputFaceBlendshapes: true/);
  assert.match(dashboard, /outputFacialTransformationMatrixes: true/);
  assert.match(dashboard, /facialTransformationMatrixes\?\.\[0\]/);
  assert.match(dashboard, /faceBlendshapes\?\.\[0\]/);
});

test("risk is gated by measurement confidence and time-of-day context", () => {
  assert.match(dashboard, /estimateSignalConfidence\(/);
  assert.match(dashboard, /circadianRisk\(/);
  assert.match(dashboard, /confidence,\s*\n\s*contextGain: context\.multiplier/);
});

test("yawns are judged by shape, not by a bare mouth threshold", () => {
  assert.match(dashboard, /detectYawn\(marHistoryRef\.current/);
  assert.doesNotMatch(dashboard, /yawnActive: classified\.yawning/);
});

test("PERCLOS is fed continuous closure rather than a boolean", () => {
  assert.match(dashboard, /closureSamplesRef\.current\.push\(\{ time: now, closure: classified\.closure \}\)/);
  assert.doesNotMatch(dashboard, /eyeSamplesRef/);
});

test("the baseline adapts only while the driver is confidently focused", () => {
  assert.match(dashboard, /smoothedState === "focused" && confidence > 0\.7/);
  assert.match(dashboard, /anchor: baselineAnchorRef\.current/);
});

test("telemetry is published at a readable rate, not at frame rate", () => {
  assert.match(dashboard, /TELEMETRY_INTERVAL_MS/);
  assert.match(dashboard, /now - lastTelemetryAtRef\.current >= TELEMETRY_INTERVAL_MS/);
});

test("performance profiles govern capture, cadence, and draw rate", () => {
  assert.match(dashboard, /PERFORMANCE_PROFILES/);
  for (const key of ["eco", "balanced", "precision"]) {
    assert.ok(dashboard.includes(key + ": { width:"), "missing the " + key + " profile");
  }
  assert.match(dashboard, /profile\.phoneCadenceMs/);
  assert.match(dashboard, /profile\.minFrameMs/);
  assert.match(dashboard, /profile\.overlayEvery/);
});

test("drive mode is a separate glanceable screen with no video preview", () => {
  assert.match(dashboard, /<DriveMode/);
  assert.match(driveMode, /drive-risk/);
  assert.match(driveMode, /End session/);
  // A driver must never be shown a video of themselves.
  assert.doesNotMatch(driveMode, /<video/);
  assert.match(styles, /\.drive-video-host/);
});

test("the camera stream survives entering and leaving drive mode", () => {
  // Drive mode renders a different subtree, so React mounts a fresh <video>.
  // A plain ref would leave it with no srcObject and stall the render loop.
  assert.match(dashboard, /const attachVideo = useCallback/);
  assert.match(dashboard, /element\.srcObject = streamRef\.current/);
  assert.doesNotMatch(dashboard, /<video ref=\{videoRef\}/);
});

test("the session survives a phone screen timeout", () => {
  assert.match(hooks, /wakeLock/);
  assert.match(hooks, /visibilitychange/);
  assert.match(dashboard, /useWakeLock\(active && phase !== "demo"\)/);
});

test("a backgrounded tab is reported rather than silently stopping", () => {
  assert.match(dashboard, /usePageVisible\(\)/);
  assert.match(dashboard, /Monitoring paused/);
});

test("alerts can be snoozed and can be felt", () => {
  assert.match(dashboard, /snoozeUntilRef\.current/);
  assert.match(dashboard, /!snoozed &&/);
  assert.match(dashboard, /vibrate\(/);
});

test("the counterfactual explanation reaches the interface", () => {
  assert.match(dashboard, /explainRisk\(assessment\)/);
  assert.match(dashboard, /counterfactual\.sentence/);
});

test("the local session journal is finally rendered", () => {
  assert.match(dashboard, /<SessionHistory/);
  assert.match(dashboard, /readSessions\(\)/);
});

test("research capture is opt-in and numeric only", () => {
  assert.match(dashboard, /research: false/, "recording must default to off");
  assert.match(dashboard, /settingsRef\.current\.research &&/);
  assert.match(dashboard, /buildFeatureWindow\(/);
  assert.match(dashboard, /No image, video, or face template is ever captured/);
});

test("the app is installable and works offline", () => {
  assert.equal(manifest.display, "standalone");
  assert.ok(manifest.icons.length > 0);
  assert.match(dashboard, /serviceWorker/);
  assert.match(worker, /\/models\//);
  assert.match(worker, /\/wasm\//);
  assert.match(worker, /cacheFirst/);
});

test("the mobile layout is corrected for real phones", () => {
  // The URL bar makes 100vh taller than the visible viewport.
  assert.doesNotMatch(styles, /min-height: 100vh/);
  // The canvas must crop exactly like the video it sits on top of.
  assert.match(styles, /\.face-tracker \{[^}]*object-fit: cover/s);
  // Risk outranks the camera preview on a phone.
  assert.match(styles, /\.right-rail \{ display: contents; \}/);
  assert.match(styles, /\.risk-panel \{ order: 1; \}/);
});

test("screen readers are told when the risk state changes", () => {
  assert.match(dashboard, /aria-live="polite"/);
  assert.match(styles, /\.sr-live/);
});

test("keyboard shortcuts drive the whole session", () => {
  const marker = "useKeyboardShortcuts({";
  const at = dashboard.indexOf(marker);
  assert.ok(at > -1, "the dashboard should register a shortcut map");
  const block = dashboard.slice(at, at + 800);
  for (const key of ["space", "c", "d", "f", "m", "s", "v", "h", "escape"]) {
    assert.ok(block.includes(key + ":"), "missing the " + key + " shortcut");
  }
});

import assert from "node:assert/strict";
import { access, stat } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server renders the finished driver-monitoring cockpit", async () => {
  await access(new URL("../dist/server/index.js", import.meta.url));
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Driver Drowsiness &amp; Distraction Detection System/);
  assert.match(html, /See fatigue before it becomes a decision/);
  assert.match(html, /Start private monitoring/);
  assert.match(html, /Zero frames retained/);
  assert.match(html, /research prototype/i);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|Your site is taking shape/i);
});

test("ships the on-device models and both WASM variants", async () => {
  const assets = [
    ["../public/models/face_landmarker.task", 3_000_000],
    ["../public/models/efficientdet_lite0.tflite", 10_000_000],
    ["../public/wasm/vision_wasm_internal.wasm", 8_000_000],
    ["../public/wasm/vision_wasm_nosimd_internal.wasm", 8_000_000],
  ];
  for (const [path, minimumSize] of assets) {
    const info = await stat(new URL(path, import.meta.url));
    assert.ok(info.size > minimumSize, `${path} should be a complete model asset`);
  }
});

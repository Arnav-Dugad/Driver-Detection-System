import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

test("GitHub Pages export has correct project paths and offline models", async () => {
  const output = new URL("../dist/client/", import.meta.url);
  const html = await readFile(new URL("index.html", output), "utf8");
  assert.match(html, /<title>Driver Drowsiness &amp; Distraction Detection System<\/title>/);
  assert.match(html, /\/Driver-Detection-System\/_next\//);
  assert.match(html, /https:\/\/arnav-dugad\.github\.io\/Driver-Detection-System\/og\.png/);

  await Promise.all([
    access(new URL("_next/", output)),
    access(new URL("models/face_landmarker.task", output)),
    access(new URL("models/efficientdet_lite0.tflite", output)),
    access(new URL("wasm/vision_wasm_internal.wasm", output)),
    access(new URL(".nojekyll", output)),
  ]);

  const chunksDirectory = new URL("_next/static/chunks/", output);
  const dashboardChunk = (await readdir(chunksDirectory)).find((name) =>
    name.startsWith("GuardianDashboard-"),
  );
  assert.ok(dashboardChunk, "the driver dashboard browser bundle should exist");
  const dashboardSource = await readFile(new URL(dashboardChunk, chunksDirectory), "utf8");
  assert.match(dashboardSource, /\/Driver-Detection-System/);
  assert.match(dashboardSource, /models\/face_landmarker\.task/);
  assert.match(dashboardSource, /models\/efficientdet_lite0\.tflite/);
  assert.match(dashboardSource, /\/wasm/);
});

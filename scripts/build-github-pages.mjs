import { spawnSync } from "node:child_process";
import { access, readFile, rename, rm } from "node:fs/promises";
import { resolve } from "node:path";
import process from "node:process";

const root = process.cwd();
const vinextCli = resolve(root, "node_modules", "vinext", "dist", "cli.js");
const environment = {
  ...process.env,
  GITHUB_PAGES: "true",
  NEXT_PUBLIC_BASE_PATH: "/Driver-Detection-System",
  WRANGLER_LOG_PATH: ".wrangler/wrangler.log",
};

const build = spawnSync(process.execPath, [vinextCli, "build"], {
  cwd: root,
  env: environment,
  stdio: "inherit",
});

const clientDirectory = resolve(root, "dist", "client");
const indexFile = resolve(clientDirectory, "index.html");
const prerenderManifest = resolve(root, "dist", "server", "vinext-prerender.json");

let exportCompleted = false;
try {
  const manifest = JSON.parse(await readFile(prerenderManifest, "utf8"));
  await access(indexFile);
  exportCompleted = manifest.routes?.some(
    (route) => route.route === "/" && route.status === "rendered",
  );
} catch {
  exportCompleted = false;
}

if (build.error || (!exportCompleted && build.status !== 0)) {
  if (build.error) console.error(build.error.message);
  process.exit(build.status || 1);
}

// Vinext keeps assetPrefix in the disk path. GitHub Pages already mounts this
// artifact below /Driver-Detection-System, so flatten the generated _next tree.
const nestedPrefixDirectory = resolve(clientDirectory, "Driver-Detection-System");
const nestedNextDirectory = resolve(nestedPrefixDirectory, "_next");
const nextDirectory = resolve(clientDirectory, "_next");
try {
  await access(nestedNextDirectory);
  await rm(nextDirectory, { recursive: true, force: true });
  await rename(nestedNextDirectory, nextDirectory);
  await rm(nestedPrefixDirectory, { recursive: true, force: true });
} catch {
  await access(nextDirectory);
}

await Promise.all([
  access(resolve(clientDirectory, "models", "face_landmarker.task")),
  access(resolve(clientDirectory, "models", "efficientdet_lite0.tflite")),
  access(resolve(clientDirectory, "wasm", "vision_wasm_internal.wasm")),
]);

console.log("GitHub Pages static export is ready in dist/client.");

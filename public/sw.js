/*
 * Offline shell and model cache.
 *
 * The whole point of this project is that inference happens on the device, so
 * it should not need a network at all once it has run once. The app shell is
 * precached on install; the ~10 MB of model and WASM assets are cached the
 * first time they are actually fetched, which keeps installation fast and
 * still leaves the app fully functional offline afterwards.
 */

const VERSION = "aegis-v1";
const SHELL_CACHE = `${VERSION}-shell`;
const ASSET_CACHE = `${VERSION}-assets`;

// Resolved against the scope so the same file works at a domain root and under
// a GitHub Pages project path.
const scoped = (path) => new URL(path, self.registration.scope).toString();

const SHELL = ["", "manifest.webmanifest", "favicon.svg"];

// Immutable, content-addressed, or very large: always prefer the cached copy.
const isDurableAsset = (url) =>
  url.pathname.includes("/models/") ||
  url.pathname.includes("/wasm/") ||
  url.pathname.includes("/_next/static/");

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL.map(scoped)))
      // A shell entry that 404s must not block activation.
      .catch(() => undefined)
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys.filter((key) => !key.startsWith(VERSION)).map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("message", (event) => {
  if (event.data === "skip-waiting") void self.skipWaiting();
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) cache.put(request, response.clone());
  return response;
}

async function networkFirst(request, cacheName, fallback) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
  } catch (error) {
    const cached = (await cache.match(request)) || (fallback && (await cache.match(fallback)));
    if (cached) return cached;
    throw error;
  }
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (request.mode === "navigate") {
    event.respondWith(networkFirst(request, SHELL_CACHE, scoped("")));
    return;
  }

  if (isDurableAsset(url)) {
    event.respondWith(cacheFirst(request, ASSET_CACHE));
    return;
  }

  event.respondWith(networkFirst(request, SHELL_CACHE));
});

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type WakeLockSentinel = { released: boolean; release(): Promise<void> };

/**
 * Hold the screen awake for the duration of a session.
 *
 * Without this a phone sleeps roughly thirty seconds in, the render loop stops,
 * and monitoring dies silently while the interface still claims to be running.
 * The lock is dropped by the browser whenever the tab is hidden, so it is
 * re-acquired on every return to the foreground.
 */
export function useWakeLock(active: boolean) {
  const sentinelRef = useRef<WakeLockSentinel | null>(null);
  const [held, setHeld] = useState(false);

  const release = useCallback(async () => {
    const sentinel = sentinelRef.current;
    sentinelRef.current = null;
    if (sentinel && !sentinel.released) {
      try {
        await sentinel.release();
      } catch {
        // A lock the browser already reclaimed is not an error worth surfacing.
      }
    }
  }, []);

  useEffect(() => {
    if (!active) {
      void release();
      return;
    }

    const navigatorWithLock = navigator as Navigator & {
      wakeLock?: { request(type: "screen"): Promise<WakeLockSentinel> };
    };
    if (!navigatorWithLock.wakeLock) return;

    let cancelled = false;
    const acquire = async () => {
      if (cancelled || document.visibilityState !== "visible") return;
      try {
        sentinelRef.current = await navigatorWithLock.wakeLock!.request("screen");
        setHeld(true);
      } catch {
        // Denied by policy or battery saver; the session still runs.
        setHeld(false);
      }
    };

    const onVisibility = () => {
      if (document.visibilityState === "visible") void acquire();
    };

    void acquire();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVisibility);
      void release();
    };
  }, [active, release]);

  return active && held;
}

/**
 * Track whether the page is actually on screen.
 *
 * A backgrounded tab stops requestAnimationFrame, so without this the session
 * clock keeps counting while nothing is being detected.
 */
export function usePageVisible() {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const update = () => setVisible(document.visibilityState === "visible");
    update();
    document.addEventListener("visibilitychange", update);
    return () => document.removeEventListener("visibilitychange", update);
  }, []);

  return visible;
}

/**
 * Single-key shortcuts, ignored while the driver is typing in a control.
 */
export function useKeyboardShortcuts(bindings: Record<string, () => void>) {
  const bindingsRef = useRef(bindings);
  useEffect(() => {
    bindingsRef.current = bindings;
  }, [bindings]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || target?.isContentEditable) {
        return;
      }
      const key = event.key === " " ? "space" : event.key.toLowerCase();
      const handler = bindingsRef.current[key];
      if (!handler) return;
      event.preventDefault();
      handler();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}

/** Short haptic patterns for the states worth feeling through a steering wheel. */
export function vibrate(pattern: number | number[]) {
  if (typeof navigator === "undefined" || !("vibrate" in navigator)) return false;
  try {
    return navigator.vibrate(pattern);
  } catch {
    return false;
  }
}

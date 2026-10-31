/** Type surface for the offline/live rule-engine tracker. Runtime lives in tracker.mjs. */

import type { Baseline, FaceSignals } from "./core.d.mts";
import type { FeatureWindow } from "./features.d.mts";

export type TrackerFrame = {
  time: number;
  faceFound: boolean;
  signals?: FaceSignals | null;
  brightness?: number;
  contrast?: number;
  phoneVisible?: boolean;
};
export type TrackerResult = {
  phase: "calibrating" | "live";
  risk: number;
  state: "focused" | "caution" | "warning" | "danger";
  rawRisk?: number;
  primary?: string;
  perclos?: number;
  closure?: number;
  eyesClosed?: boolean;
  eyeClosedMs?: number;
  yawnActive?: boolean;
  blink?: boolean;
  confidence?: number;
  faceFound?: boolean;
};
export type Tracker = {
  step(frame: TrackerFrame): TrackerResult;
  features(now?: number): FeatureWindow | null;
  readonly phase: "calibrating" | "live";
  readonly baseline: Baseline;
};
export declare function createTracker(options?: {
  sensitivity?: number;
  calibrationMs?: number;
  featureWindowMs?: number;
  baseline?: Baseline | null;
}): Tracker;

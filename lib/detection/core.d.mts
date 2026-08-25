/** Type surface for the pure detection core. Runtime lives in core.mjs. */

export type Baseline = {
  ear: number;
  mar: number;
  yaw: number;
  pitch: number;
  roll: number;
  gaze: number;
};

export type BlendshapeSignals = {
  blink: number;
  blinkLeft: number | null;
  blinkRight: number | null;
  jawOpen: number;
  mouthPucker: number;
  gaze: number;
  gazeVertical: number;
  asymmetry: number;
};

export type FaceSignals = {
  ear: number;
  leftEar: number;
  rightEar: number;
  mar: number;
  yaw: number;
  pitch: number;
  roll: number;
  gaze: number;
  poseSource: "matrix" | "geometry";
  blendshapes: BlendshapeSignals | null;
  closure: number;
};

export type ClosureSample = { time: number; closure?: number; closed?: boolean };

export type RiskComponents = {
  eyes: number;
  perclos: number;
  yawn: number;
  head: number;
  gaze: number;
  phone: number;
  missing: number;
};

export type RiskAssessment = {
  score: number;
  state: "focused" | "caution" | "warning" | "danger";
  primary: string;
  components: RiskComponents;
  concurrent: number;
  confidence: number;
  contextGain: number;
  scale: number;
  synergy: number;
};

export type Explanation = {
  key: string;
  label: string;
  from: number;
  to: number;
  delta: number;
  sentence: string;
  ranked: Array<{ key: string; weight: number }>;
};

export type CircadianContext = {
  multiplier: number;
  combined: number;
  factors: Array<{ key: string; weight: number; label: string }>;
  breakDue: boolean;
};

export type YawnVerdict = {
  active: boolean;
  confidence: number;
  durationMs: number;
  oscillation: number;
};

export type Classification = {
  eyesClosed: boolean;
  closure: number;
  closureSource: "geometry" | "fused";
  yawning: boolean;
  headAway: boolean;
  gazeAway: boolean;
  thresholds: {
    eye: number;
    closure: number;
    yawn: number;
    yaw: number;
    pitch: number;
    gaze: number;
  };
};

export declare const DEFAULT_BASELINE: Baseline;
export declare function clamp(value: number, min?: number, max?: number): number;
export declare function mean(values: number[]): number;
export declare function median(values: number[]): number;
export declare function distance(
  a: { x?: number; y?: number } | undefined,
  b: { x?: number; y?: number } | undefined,
): number;

export declare function classifyCameraQuality(input: {
  brightness?: number;
  contrast?: number;
  darkPixelRatio?: number;
  faceFound?: boolean;
}): "clear" | "low-light" | "obstructed";

export declare function decomposePose(
  matrix: { data: number[] | Float32Array } | number[] | Float32Array | null | undefined,
): { yaw: number; pitch: number; roll: number } | null;

export declare function extractBlendshapeSignals(
  blendshapes:
    | { categories?: Array<{ categoryName: string; score: number }> }
    | Array<{ categoryName: string; score: number }>
    | null
    | undefined,
): BlendshapeSignals | null;

export declare function fuseClosure(input: {
  ear?: number;
  earBaseline?: number;
  blink?: number | null;
  poseMagnitude?: number;
}): { closure: number; source: "geometry" | "fused"; geometric: number; learned: number | null };

export declare function extractFaceSignals(
  landmarks: Array<{ x: number; y: number; z?: number }> | undefined,
  options?: { matrix?: unknown; blendshapes?: unknown },
): FaceSignals | null;

export declare function classifySignals(
  signals: { ear: number; mar: number; yaw: number; pitch: number; gaze: number; blendshapes?: { blink?: number } | null },
  baseline?: Partial<Baseline>,
  sensitivity?: number,
): Classification;

export declare function detectYawn(
  history: Array<{ time: number; mar: number; closure?: number }> | null | undefined,
  options?: { threshold?: number; now?: number | null },
): YawnVerdict;

export declare function estimateSignalConfidence(input?: {
  brightness?: number;
  contrast?: number;
  fps?: number;
  yawMagnitude?: number;
  pitchMagnitude?: number;
  faceFound?: boolean;
  blendshapeAvailable?: boolean;
  asymmetry?: number;
}): number;

export declare function circadianRisk(
  localHour?: number,
  minutesOnTask?: number,
): CircadianContext;

export declare function calculateRisk(input: {
  eyeClosedMs?: number;
  perclos?: number;
  yawnActive?: boolean;
  recentYawns?: number;
  headAwayMs?: number;
  gazeAwayMs?: number;
  phoneVisible?: boolean;
  faceMissingMs?: number;
  sensitivity?: number;
  confidence?: number;
  contextGain?: number;
}): RiskAssessment;

export declare function explainRisk(
  assessment: RiskAssessment | null | undefined,
): Explanation | null;

export declare function buildCalibration(
  samples: Array<Partial<Baseline> & Record<string, unknown>>,
): Baseline;

export declare function updateBaseline(
  current: Baseline,
  sample: Partial<Baseline>,
  options?: { anchor?: Baseline; confidence?: number; rate?: number },
): Baseline;

export declare function calculatePerclos(
  samples: ClosureSample[],
  now: number,
  windowMs?: number,
  closedAt?: number,
): number;

export declare function longestClosure(samples: ClosureSample[], closedAt?: number): number;

export declare function formatDuration(totalSeconds: number): string;

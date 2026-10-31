/** Type surface for the temporal model runtime. Runtime lives in sequence.mjs. */

import type { Baseline, FaceSignals } from "./core.d.mts";

export declare const SEQUENCE_CHANNELS: string[];
export declare const SEQUENCE_HZ: number;
export type PreparedSequenceModel = { classes: string[]; hidden: number; temperature: number };
export type SequenceBuffer = {
  push(time: number, vector: Float32Array): void;
  readonly full: boolean;
  readonly length: number;
  frames(): Float32Array[];
  reset(): void;
};
export declare function sequenceFrame(
  signals: FaceSignals | null,
  baseline: Partial<Baseline>,
  closure: number | null,
): Float32Array;
export declare function createSequenceBuffer(options?: { seconds?: number; hz?: number }): SequenceBuffer;
export declare function isSequenceModel(model: unknown): boolean;
export declare function prepareSequenceModel(model: unknown): PreparedSequenceModel | null;
export declare function predictSequence(
  prepared: PreparedSequenceModel | null,
  frames: ArrayLike<ArrayLike<number>>,
): Record<string, number> | null;
export declare function sequenceRisk(probabilities: Record<string, number> | null): number | null;

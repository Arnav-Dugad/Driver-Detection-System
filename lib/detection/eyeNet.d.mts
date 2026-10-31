/** Type surface for the eye-state CNN runtime. Runtime lives in eyeNet.mjs. */

export declare const EYE_INPUT_SIZE: number;
export type PreparedEyeModel = { layers: unknown[]; head: unknown };
export declare function isEyeModel(model: unknown): boolean;
export declare function prepareEyeModel(model: unknown): PreparedEyeModel | null;
export declare function standardizeCrop(pixels: ArrayLike<number>): Float32Array;
export declare function predictClosed(
  prepared: PreparedEyeModel | null,
  pixels: ArrayLike<number> | null,
): number | null;

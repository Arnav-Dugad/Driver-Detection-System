/** Type surface for landmark-aligned eye crops. Runtime lives in eyeCrop.mjs. */

type Point = { x: number; y: number };
export declare const EYE_CROP_SCALE: number;
export declare const EYE_CORNERS: { left: [number, number]; right: [number, number] };
export declare function eyeCropTransform(
  landmarks: Point[],
  side: "left" | "right",
  width: number,
  height: number,
): [number, number, number, number, number, number] | null;
export declare function cropEyes(
  source: CanvasImageSource,
  landmarks: Point[],
  width: number,
  height: number,
  canvas: HTMLCanvasElement,
): { left: Float32Array; right: Float32Array } | null;

/**
 * Landmark-aligned 32x32 grayscale eye crops for the eye-state CNN.
 *
 * Mirrors eye_crop_transform() and to_gray() in ml/signals.py: a square crop
 * centred between the eye corners, rotated so the corners are level, 1.6x the
 * corner-to-corner width on a side. Training and inference therefore see the
 * same framing of the eye whatever the head roll or camera resolution.
 */

import { EYE_INPUT_SIZE } from "./eyeNet.mjs";

export const EYE_CROP_SCALE = 1.6;
// Image-left corner first for both eyes, so the angle is near zero upright.
export const EYE_CORNERS = { left: [362, 263], right: [33, 133] };

/**
 * Affine map from image pixels to crop pixels, as canvas setTransform()
 * arguments [a, b, c, d, e, f]. Returns null for a degenerate eye.
 */
export function eyeCropTransform(landmarks, side, width, height) {
  const [inner, outer] = EYE_CORNERS[side];
  const a = landmarks?.[inner];
  const b = landmarks?.[outer];
  if (!a || !b) return null;
  const ax = a.x * width;
  const ay = a.y * height;
  const bx = b.x * width;
  const by = b.y * height;
  const eyeWidth = Math.hypot(bx - ax, by - ay);
  if (eyeWidth < 2) return null;
  const cx = (ax + bx) / 2;
  const cy = (ay + by) / 2;
  const angle = Math.atan2(by - ay, bx - ax);
  const scale = EYE_INPUT_SIZE / (eyeWidth * EYE_CROP_SCALE);
  const cos = Math.cos(-angle) * scale;
  const sin = Math.sin(-angle) * scale;
  const half = EYE_INPUT_SIZE / 2;
  return [cos, sin, -sin, cos, half - (cos * cx - sin * cy), half - (sin * cx + cos * cy)];
}

/**
 * Cut both eye crops out of a video frame. `canvas` is a reusable 32x32
 * scratch canvas. Returns { left, right } as Float32Array luma (0..255) or
 * null when either eye cannot be framed.
 */
export function cropEyes(source, landmarks, width, height, canvas) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  canvas.width = EYE_INPUT_SIZE;
  canvas.height = EYE_INPUT_SIZE;
  const crops = {};
  for (const side of ["left", "right"]) {
    const transform = eyeCropTransform(landmarks, side, width, height);
    if (!transform) return null;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, EYE_INPUT_SIZE, EYE_INPUT_SIZE);
    context.imageSmoothingEnabled = true;
    context.setTransform(...transform);
    context.drawImage(source, 0, 0, width, height);
    const { data } = context.getImageData(0, 0, EYE_INPUT_SIZE, EYE_INPUT_SIZE);
    const gray = new Float32Array(EYE_INPUT_SIZE * EYE_INPUT_SIZE);
    for (let i = 0, p = 0; i < gray.length; i += 1, p += 4) {
      gray[i] = data[p] * 0.299 + data[p + 1] * 0.587 + data[p + 2] * 0.114;
    }
    crops[side] = gray;
  }
  return crops;
}

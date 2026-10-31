"""Python port of the per-frame signal maths in lib/detection/core.mjs.

The browser computes eye, mouth, gaze, and pose signals from MediaPipe output
in JavaScript. Offline experiments have to compute exactly the same numbers or
a model trained here would be scored on different inputs in the app, so every
function below mirrors its JavaScript namesake line for line.
tests/signal-parity.test.mjs runs both implementations on the same recorded
MediaPipe frames and fails if they disagree.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

DEFAULT_BASELINE = {"ear": 0.285, "mar": 0.12, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "gaze": 0.5}

# Eye landmark groups, in the order p1..p6 of the Soukupova-Cech EAR formula.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]

# The 52 ARKit-style blendshape names MediaPipe emits, kept so that per-frame
# exports have a stable column order.
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def ramp(value: float, start: float, end: float) -> float:
    return clamp((value - start) / (end - start))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if len(values) else 0.0


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(landmarks: np.ndarray, indices: Sequence[int]) -> float:
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in indices)
    horizontal = distance(p1, p4)
    if not horizontal:
        return 0.0
    return (distance(p2, p6) + distance(p3, p5)) / (2 * horizontal)


def iris_ratio(landmarks: np.ndarray, iris: Sequence[int], outer: int, inner: int) -> float:
    center = (mean([landmarks[i][0] for i in iris]), mean([landmarks[i][1] for i in iris]))
    width = distance(landmarks[outer], landmarks[inner])
    if not width:
        return 0.5
    return clamp(distance(landmarks[outer], center) / width)


def decompose_pose(data: Sequence[float] | None):
    """`data` is the 4x4 matrix flattened column-major, as MediaPipe JS exposes it."""
    if data is None or len(data) < 16:
        return None

    def column_length(column: int) -> float:
        return math.hypot(data[column * 4], data[column * 4 + 1], data[column * 4 + 2]) or 1.0

    scale_x, scale_y, scale_z = column_length(0), column_length(1), column_length(2)
    m11 = data[0] / scale_x
    m21 = data[1] / scale_x
    m31 = data[2] / scale_x
    m22 = data[5] / scale_y
    m13 = data[8] / scale_z
    m23 = data[9] / scale_z
    m33 = data[10] / scale_z

    pitch = math.asin(-clamp(m23, -1, 1))
    gimbal_locked = abs(m23) > 0.9999
    yaw = math.atan2(-m31, m11) if gimbal_locked else math.atan2(m13, m33)
    roll = 0.0 if gimbal_locked else math.atan2(m21, m22)
    return yaw, pitch, roll


def blendshape_signals(scores: dict[str, float] | None):
    if not scores:
        return None
    blink_left = scores.get("eyeBlinkLeft")
    blink_right = scores.get("eyeBlinkRight")
    if blink_left is None and blink_right is None:
        return None
    blink = mean([v for v in (blink_left, blink_right) if v is not None])
    look_out = mean([scores.get("eyeLookOutLeft", 0.0), scores.get("eyeLookOutRight", 0.0)])
    look_in = mean([scores.get("eyeLookInLeft", 0.0), scores.get("eyeLookInRight", 0.0)])
    look_up = mean([scores.get("eyeLookUpLeft", 0.0), scores.get("eyeLookUpRight", 0.0)])
    look_down = mean([scores.get("eyeLookDownLeft", 0.0), scores.get("eyeLookDownRight", 0.0)])
    return {
        "blink": clamp(blink),
        "blink_left": None if blink_left is None else clamp(blink_left),
        "blink_right": None if blink_right is None else clamp(blink_right),
        "jaw_open": clamp(scores.get("jawOpen", 0.0)),
        "bs_gaze": clamp(0.5 + (look_out - look_in) / 2),
        "bs_gaze_vertical": clamp(0.5 + (look_down - look_up) / 2),
        "asymmetry": 0.0
        if blink_left is None or blink_right is None
        else abs(blink_left - blink_right),
    }


def fuse_closure(ear: float, ear_baseline: float, blink: float | None, pose_magnitude: float):
    safe_baseline = max(0.001, ear_baseline)
    geometric = clamp(1 - ear / safe_baseline)
    if blink is None:
        return geometric
    learned = clamp(blink)
    geometric_trust = clamp(1 - ramp(abs(pose_magnitude), 0.26, 0.7))
    learned_certainty = clamp(abs(learned - 0.5) * 2)
    geometric_weight = clamp(0.25 + geometric_trust * 0.45) * (1 - learned_certainty * 0.45)
    return clamp(geometric * geometric_weight + learned * (1 - geometric_weight))


def face_signals(
    landmarks: np.ndarray, matrix: Sequence[float] | None, scores: dict | None, aspect: float = 1.0
):
    """Mirror of extractFaceSignals(). `landmarks` is an (N, 2+) array of normalized x, y.

    `aspect` is frame width / height; x is rescaled so ratios use square pixels.
    """
    if landmarks is None or len(landmarks) < 468:
        return None
    if aspect != 1:
        landmarks = np.array(landmarks, dtype=np.float64, copy=True)
        landmarks[:, 0] *= aspect
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
    mouth_width = distance(landmarks[78], landmarks[308])
    mar = (
        (distance(landmarks[13], landmarks[14]) + distance(landmarks[82], landmarks[312]))
        / (2 * mouth_width)
        if mouth_width
        else 0.0
    )
    gaze = 0.5
    if len(landmarks) >= 478:
        gaze = mean([
            iris_ratio(landmarks, LEFT_IRIS, 362, 263),
            iris_ratio(landmarks, RIGHT_IRIS, 33, 133),
        ])

    pose = decompose_pose(matrix)
    if pose:
        yaw, pitch, roll = pose
    else:
        cheek_width = distance(landmarks[234], landmarks[454]) or 1.0
        face_center_x = (landmarks[234][0] + landmarks[454][0]) / 2
        yaw_ratio = (landmarks[1][0] - face_center_x) / cheek_width
        eye_mid_y = mean([landmarks[33][1], landmarks[263][1]])
        chin_y = landmarks[152][1]
        pitch_ratio = (landmarks[1][1] - eye_mid_y) / max(0.001, chin_y - eye_mid_y)
        yaw = math.asin(clamp(yaw_ratio * 2, -1, 1))
        pitch = clamp((pitch_ratio - 0.44) * 1.4, -1.2, 1.2)
        roll = 0.0

    learned = blendshape_signals(scores)
    ear = mean([left_ear, right_ear])
    pose_magnitude = max(abs(yaw), abs(pitch))
    out = {
        "ear": ear,
        "left_ear": left_ear,
        "right_ear": right_ear,
        "mar": mar,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "gaze": gaze,
        "closure": fuse_closure(
            ear, DEFAULT_BASELINE["ear"], learned["blink"] if learned else None, pose_magnitude
        ),
    }
    if learned:
        out.update({k: v for k, v in learned.items() if v is not None})
    return out


# --- Eye crops for the eye-state CNN ---------------------------------------
# lib/detection/eyeCrop.mjs reproduces this crop on a canvas. The crop is
# square, centred on the eye corners' midpoint, rotated so the corners are
# level, and EYE_CROP_SCALE times the corner-to-corner width on each side.

EYE_CORNERS = {"left": (362, 263), "right": (33, 133)}  # image-left corner first
EYE_CROP_SCALE = 1.6
EYE_CROP_SIZE = 32


def eye_crop_transform(landmarks: np.ndarray, side: str, width: int, height: int):
    """Return the 2x3 affine matrix mapping image pixels to a crop of EYE_CROP_SIZE."""
    inner, outer = EYE_CORNERS[side]
    ax, ay = landmarks[inner][0] * width, landmarks[inner][1] * height
    bx, by = landmarks[outer][0] * width, landmarks[outer][1] * height
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    eye_width = math.hypot(bx - ax, by - ay)
    if eye_width < 2:
        return None
    # Both corner pairs are listed image-left first, so the angle is near zero
    # for an upright head and the crop only undoes head roll.
    angle = math.atan2(by - ay, bx - ax)
    side_px = eye_width * EYE_CROP_SCALE
    scale = EYE_CROP_SIZE / side_px
    cos, sin = math.cos(-angle) * scale, math.sin(-angle) * scale
    # Rotate about the eye centre, scale, then move the centre to the crop centre.
    half = EYE_CROP_SIZE / 2
    return np.array([
        [cos, -sin, half - (cos * cx - sin * cy)],
        [sin, cos, half - (sin * cx + cos * cy)],
    ], dtype=np.float32)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma, the same weights eyeCrop.mjs applies to canvas pixels."""
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)

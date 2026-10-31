"""Per-frame input channels for the temporal drowsiness models.

lib/detection/sequence.mjs builds the same vector live in the browser; both
read the channel list from SEQUENCE_CHANNELS so the GRU sees identical inputs
in training and in the app (tests/sequence-net.test.mjs checks the order).

Brightness and contrast are deliberately not channels. In RLDD the alert and
drowsy videos were often filmed at different times of day, so lighting is
correlated with the label; a model allowed to see it can score well by
recognising the room rather than the driver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals import fuse_closure

SEQUENCE_CHANNELS = [
    "closure",        # fused EAR + blendshape closure, 0 open .. 1 shut
    "ear_rel",        # EAR / calibrated open-eye EAR
    "blink",          # learned eyeBlink score
    "squint",         # eyeSquint, drooping lids
    "wide",           # eyeWide, effortful eye opening
    "mar_rel",        # mouth aspect ratio minus calibrated
    "jaw_open",       # learned jawOpen score
    "yaw_rel",        # radians from calibrated neutral
    "pitch_rel",
    "roll_rel",
    "gaze_rel",       # iris ratio minus calibrated
    "gaze_bs",        # blendshape horizontal gaze minus centre
    "brow_inner_up",  # raised inner brows, effortful wakefulness
    "face_found",
]

SEQUENCE_HZ = 15


def sequence_matrix(frame: pd.DataFrame, baseline: dict) -> np.ndarray:
    """(T, len(SEQUENCE_CHANNELS)) float32; rows without a face are zero but for face_found."""
    found = frame["face_found"].to_numpy(bool)
    out = np.zeros((len(frame), len(SEQUENCE_CHANNELS)), dtype=np.float32)
    if not found.any():
        return out
    rows = frame[found]

    def column(name, default=0.0):
        return rows[name].fillna(default).to_numpy(np.float64) if name in rows else np.full(len(rows), default)

    ear = column("ear")
    blink = column("blink", np.nan)
    yaw = column("yaw") - baseline["yaw"]
    pitch = column("pitch") - baseline["pitch"]
    closure = np.array([
        fuse_closure(e, baseline["ear"], None if np.isnan(b) else b, max(abs(y), abs(p)))
        for e, b, y, p in zip(ear, blink, yaw, pitch)
    ])
    values = {
        "closure": closure,
        "ear_rel": ear / max(0.001, baseline["ear"]),
        "blink": np.nan_to_num(blink),
        "squint": (column("bs_eyeSquintLeft") + column("bs_eyeSquintRight")) / 2,
        "wide": (column("bs_eyeWideLeft") + column("bs_eyeWideRight")) / 2,
        "mar_rel": column("mar") - baseline["mar"],
        "jaw_open": column("jaw_open"),
        "yaw_rel": yaw,
        "pitch_rel": pitch,
        "roll_rel": column("roll") - baseline["roll"],
        "gaze_rel": column("gaze") - baseline["gaze"],
        "gaze_bs": column("bs_gaze", 0.5) - 0.5,
        "brow_inner_up": column("bs_browInnerUp"),
        "face_found": np.ones(len(rows)),
    }
    for index, name in enumerate(SEQUENCE_CHANNELS):
        out[found, index] = values[name]
    return out

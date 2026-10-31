"""Record real MediaPipe outputs and the Python signal values for the parity test.

    python ml/make_parity_fixture.py

Writes tests/fixtures/signal-parity.json: a few frames of raw landmarks, pose
matrix, and blendshapes from a public RLDD clip, plus what ml/signals.py
computes from them and the matching eye-crop transforms. The JavaScript test
recomputes everything with lib/detection and fails on any disagreement.
"""

from __future__ import annotations

import json

import numpy as np

from common import ROOT
from extract_signals import frames, landmarker, output_size, probe, url_for
from signals import eye_crop_transform, face_signals
import mediapipe as mp

CLIP = "UTA-RLDD/Fold1_part1/04/10.mp4"


def main() -> None:
    url = url_for(CLIP)
    width, height, _, _ = probe(url)
    size = output_size(width, height)
    detector = landmarker()
    records = []
    for index, rgb in enumerate(frames(url, size, 120, 3, 15)):
        result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                                           int(index * 1000 / 15))
        if not result.face_landmarks or index % 9:
            continue
        points = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]])
        matrix = np.asarray(result.facial_transformation_matrixes[0]).flatten(order="F").tolist()
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        h, w = rgb.shape[:2]
        signals = face_signals(points, matrix, scores, aspect=w / h)
        records.append({
            "width": w, "height": h,
            "landmarks": points[:, :2].round(7).tolist(),
            "matrix": [round(v, 7) for v in matrix],
            "blendshapes": [{"categoryName": k, "score": round(v, 7)} for k, v in scores.items()],
            "expected": {k: v for k, v in signals.items() if isinstance(v, float)},
            "crops": {side: eye_crop_transform(points, side, w, h).round(6).tolist() for side in ("left", "right")},
        })
    detector.close()
    target = ROOT / "tests" / "fixtures" / "signal-parity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records))
    print(f"wrote {len(records)} frames to {target}")


if __name__ == "__main__":
    main()

"""Generate clearly synthetic data for testing the training pipeline.

Never report performance on this file as real-world model accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("ml/data/synthetic_demo.csv"))
    parser.add_argument("--subjects", type=int, default=24)
    parser.add_argument("--rows-per-subject", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def clipped(rng: np.random.Generator, center: float, spread: float, low: float, high: float) -> float:
    return float(np.clip(rng.normal(center, spread), low, high))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    labels = ["focused", "caution", "drowsy", "distracted"]

    for subject_index in range(args.subjects):
        subject = f"synthetic_subject_{subject_index + 1:02d}"
        eye_offset = rng.normal(0, 0.012)
        mouth_offset = rng.normal(0, 0.008)
        for _ in range(args.rows_per_subject):
            label = str(rng.choice(labels, p=[0.46, 0.2, 0.18, 0.16]))
            drowsy = label == "drowsy"
            distracted = label == "distracted"
            caution = label == "caution"
            rows.append(
                {
                    "subject_id": subject,
                    "label": label,
                    "ear_mean": clipped(rng, 0.285 + eye_offset - (0.09 if drowsy else 0.025 if caution else 0), 0.018, 0.08, 0.42),
                    "ear_min": clipped(rng, 0.24 + eye_offset - (0.13 if drowsy else 0.04 if caution else 0), 0.025, 0.04, 0.38),
                    "perclos_60s": clipped(rng, 0.05 + (0.3 if drowsy else 0.1 if caution else 0), 0.045, 0, 0.75),
                    "longest_closure_ms": clipped(rng, 260 + (1450 if drowsy else 350 if caution else 0), 170, 60, 4500),
                    "blink_rate_per_min": clipped(rng, 16 + (8 if drowsy else 2 if caution else 0), 3.5, 2, 45),
                    "mar_mean": clipped(rng, 0.11 + mouth_offset + (0.12 if drowsy else 0), 0.025, 0.03, 0.5),
                    "mar_max": clipped(rng, 0.22 + mouth_offset + (0.28 if drowsy else 0.07 if caution else 0), 0.05, 0.05, 0.8),
                    "yawns_10m": int(np.clip(rng.poisson(3.2 if drowsy else 0.8 if caution else 0.2), 0, 12)),
                    "yaw_deviation": clipped(rng, 0.025 + (0.14 if distracted else 0.045 if caution else 0), 0.025, 0, 0.4),
                    "pitch_deviation": clipped(rng, 0.03 + (0.13 if distracted else 0.04 if caution else 0), 0.025, 0, 0.4),
                    "gaze_deviation": clipped(rng, 0.05 + (0.3 if distracted else 0.1 if caution else 0), 0.055, 0, 0.7),
                    "phone_ratio": clipped(rng, 0.005 + (0.5 if distracted else 0.025 if caution else 0), 0.05, 0, 1),
                    "face_missing_ratio": clipped(rng, 0.01 + (0.18 if distracted else 0.03 if caution else 0), 0.035, 0, 1),
                    "data_origin": "SYNTHETIC_PIPELINE_TEST_ONLY",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} synthetic rows to {args.output.resolve()}")
    print("Do not report metrics from synthetic data as real-world accuracy.")


if __name__ == "__main__":
    main()

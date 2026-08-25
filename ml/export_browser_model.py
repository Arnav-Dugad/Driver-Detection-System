"""Export a trained fusion model into a compact JSON the browser can score.

The deterministic engine in `lib/detection/core.mjs` stays the default and the
fallback. This exists so a model trained on real, consented, subject-independent
data can be compared against it in the live app without adding an ONNX runtime
or breaking the no-cloud, no-API-key guarantee.

Run `train_fusion.py` first, then:

    python ml/export_browser_model.py --model ml/artifacts/driver_fusion.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from train_fusion import FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a fusion model for browser inference.")
    parser.add_argument("--model", type=Path, default=Path("ml/artifacts/driver_fusion.joblib"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/driver_fusion.json"))
    parser.add_argument(
        "--max-trees",
        type=int,
        default=60,
        help="Subsample the forest to keep the browser payload small.",
    )
    parser.add_argument(
        "--decimals", type=int, default=4, help="Rounding applied to thresholds."
    )
    return parser.parse_args()


def export_tree(tree, decimals: int) -> dict[str, list]:
    """Flatten one sklearn tree into parallel arrays."""
    inner = tree.tree_
    # Leaf rows hold class counts; normalize them into probabilities once here
    # so the browser only has to walk the tree.
    values = inner.value.reshape(inner.value.shape[0], -1)
    totals = values.sum(axis=1, keepdims=True)
    probabilities = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    return {
        "left": inner.children_left.tolist(),
        "right": inner.children_right.tolist(),
        "feature": inner.feature.tolist(),
        "threshold": np.round(inner.threshold, decimals).tolist(),
        "value": np.round(probabilities, decimals).tolist(),
    }


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"No model at {args.model}. Run train_fusion.py first.")

    pipeline = joblib.load(args.model)
    forest = pipeline.named_steps["model"]
    imputer = pipeline.named_steps["prepare"].named_transformers_["numeric"]

    estimators = list(forest.estimators_)
    if args.max_trees and len(estimators) > args.max_trees:
        # An evenly spaced subsample keeps the ensemble representative.
        indices = np.linspace(0, len(estimators) - 1, args.max_trees).astype(int)
        estimators = [estimators[index] for index in indices]

    payload = {
        "format": "driver-fusion-forest",
        "version": 1,
        "features": FEATURES,
        "classes": [str(label) for label in forest.classes_],
        # The browser must impute exactly the way training did.
        "imputer_medians": np.round(imputer.statistics_, args.decimals).tolist(),
        "trees": [export_tree(estimator, args.decimals) for estimator in estimators],
        "note": (
            "Exported for optional in-browser scoring. Valid only for the dataset "
            "and protocol it was trained on."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(f"Exported {len(payload['trees'])} trees to {args.output} ({size_kb:.0f} KB)")
    if size_kb > 2048:
        print("Warning: over 2 MB. Lower --max-trees or retrain with a smaller max_depth.")


if __name__ == "__main__":
    main()

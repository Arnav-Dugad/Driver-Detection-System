"""Shared helpers for the experiment scripts: paths, data loading, metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ml" / "data"
REPORTS = ROOT / "ml" / "reports"
FIGURES = REPORTS / "figures"
PUBLIC_MODELS = ROOT / "public" / "models"
LAB = ROOT / "public" / "results"

for directory in (REPORTS, FIGURES, LAB):
    directory.mkdir(parents=True, exist_ok=True)


def load_signals(dataset: str) -> pd.DataFrame:
    files = sorted((DATA / "signals" / dataset).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No extracted signals for {dataset}; run ml/extract_signals.py {dataset}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def calibrate(frame: pd.DataFrame, seconds: float = 5.2) -> dict:
    """Python twin of buildCalibration() over the first `seconds` of face frames."""
    start = frame["time_ms"].iloc[0]
    head = frame[(frame["time_ms"] - start <= seconds * 1000) & frame["face_found"]]
    if head.empty:
        head = frame[frame["face_found"]]
    ears = np.sort(head["ear"][head["ear"] > 0.12].to_numpy())
    open_eyes = ears[int(len(ears) * 0.35):]
    return {
        "ear": float(np.clip(np.median(open_eyes) if len(open_eyes) else 0.285, 0.18, 0.42)),
        "mar": float(np.clip(head["mar"].median(), 0.04, 0.25)),
        "yaw": float(head["yaw"].median()),
        "pitch": float(head["pitch"].median()),
        "roll": float(head["roll"].median()),
        "gaze": float(np.clip(head["gaze"].median(), 0.25, 0.75)),
    }


def bootstrap_ci(values_by_group: dict, statistic, repeats: int = 2000, seed: int = 7):
    """Percentile CI from resampling *subjects* with replacement, never frames.

    Frames from one person are strongly correlated, so resampling frames would
    produce intervals far narrower than the evidence supports.
    """
    rng = np.random.default_rng(seed)
    groups = list(values_by_group)
    estimates = []
    for _ in range(repeats):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        estimates.append(statistic([values_by_group[g] for g in chosen]))
    low, high = np.nanpercentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def expected_calibration_error(probabilities: np.ndarray, truth: np.ndarray, bins: int = 10):
    """Top-label ECE with equal-width confidence bins (Guo et al., 2017)."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    curve = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            gap = abs(correct[mask].mean() - confidence[mask].mean())
            error += mask.mean() * gap
            curve.append({"bin": float((low + high) / 2), "accuracy": float(correct[mask].mean()),
                          "confidence": float(confidence[mask].mean()), "count": int(mask.sum())})
    return float(error), curve


def write_json(path: Path, payload, compact: bool = False) -> None:
    """Reports are indented for reading; shipped models are compact for download size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (json.dumps(payload, separators=(",", ":"), default=_jsonable) if compact
            else json.dumps(payload, indent=2, default=_jsonable))
    path.write_text(text, encoding="utf-8")


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def plot_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
    })
    return plt


# One palette for every figure in the report and deck.
COLORS = {
    "accent": "#2563eb",
    "muted": "#94a3b8",
    "dark": "#0f172a",
    "good": "#16a34a",
    "warn": "#d97706",
    "bad": "#dc2626",
    "series": ["#2563eb", "#0d9488", "#d97706", "#9333ea", "#dc2626", "#64748b"],
}

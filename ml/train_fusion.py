"""Train and evaluate a subject-independent driver-state fusion model.

This script works on aggregated numeric windows, never raw face images. It keeps
every participant inside exactly one split to prevent identity leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline


FEATURES = [
    "ear_mean",
    "ear_min",
    "perclos_60s",
    "longest_closure_ms",
    "blink_rate_per_min",
    "mar_mean",
    "mar_max",
    "yawns_10m",
    "yaw_deviation",
    "pitch_deviation",
    "gaze_deviation",
    "phone_ratio",
    "face_missing_ratio",
]

REQUIRED_COLUMNS = ["subject_id", "label", *FEATURES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Aegis temporal fusion with participant-safe splits."
    )
    parser.add_argument("--data", type=Path, required=True, help="Labeled CSV file")
    parser.add_argument(
        "--output", type=Path, default=Path("ml/artifacts"), help="Artifact directory"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller search for a fast classroom smoke test",
    )
    return parser.parse_args()


def validate_data(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame["subject_id"].nunique() < 4:
        raise ValueError("Use at least four participants for a subject-independent split.")
    if frame["label"].nunique() < 2:
        raise ValueError("The label column must contain at least two driver states.")
    if frame.empty:
        raise ValueError("The input CSV is empty.")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)
    validate_data(frame)

    x = frame[FEATURES]
    y = frame["label"].astype(str)
    groups = frame["subject_id"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_index, test_index = next(splitter.split(x, y, groups))
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    train_groups = groups.iloc[train_index]

    train_subjects = sorted(set(train_groups))
    test_subjects = sorted(set(groups.iloc[test_index]))
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        raise RuntimeError(f"Subject leakage detected: {sorted(overlap)}")

    preprocessing = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = RandomForestClassifier(
        random_state=args.seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    pipeline = Pipeline([("prepare", preprocessing), ("model", classifier)])

    parameter_grid = (
        {
            "model__n_estimators": [120],
            "model__max_depth": [12],
            "model__min_samples_leaf": [2],
            "model__max_features": ["sqrt"],
        }
        if args.quick
        else {
            "model__n_estimators": [250, 500],
            "model__max_depth": [10, 18, None],
            "model__min_samples_leaf": [2, 5],
            "model__max_features": ["sqrt", 0.7],
        }
    )
    folds = min(5, len(train_subjects))
    if folds < 2:
        raise ValueError("The training split needs at least two participants.")
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=folds),
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train, groups=train_groups)

    prediction = search.best_estimator_.predict(x_test)
    probabilities = search.best_estimator_.predict_proba(x_test)
    classes = list(search.best_estimator_.named_steps["model"].classes_)

    report: dict[str, object] = {
        "dataset": str(args.data),
        "random_seed": args.seed,
        "rows": int(len(frame)),
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "subject_overlap": [],
        "classes": classes,
        "best_parameters": search.best_params_,
        "cross_validation_macro_f1": float(search.best_score_),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "test_macro_f1": float(f1_score(y_test, prediction, average="macro")),
        "classification_report": classification_report(
            y_test, prediction, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=classes).tolist(),
        "warning": "Results are valid only for the supplied labeled dataset and protocol.",
    }

    if len(classes) == 2 and len(set(y_test)) == 2:
        positive = classes[1]
        binary_truth = (y_test.to_numpy() == positive).astype(int)
        report["test_roc_auc"] = float(roc_auc_score(binary_truth, probabilities[:, 1]))
    elif len(set(y_test)) == len(classes):
        report["test_roc_auc_ovr_macro"] = float(
            roc_auc_score(y_test, probabilities, labels=classes, multi_class="ovr", average="macro")
        )

    importance = permutation_importance(
        search.best_estimator_,
        x_test,
        y_test,
        scoring="f1_macro",
        n_repeats=8 if args.quick else 20,
        random_state=args.seed,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, args.output / "aegis_fusion.joblib")
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    importance_frame.to_csv(args.output / "feature_importance.csv", index=False)
    prediction_frame = frame.iloc[test_index][["subject_id", "label"]].copy()
    prediction_frame["prediction"] = prediction
    for index, class_name in enumerate(classes):
        prediction_frame[f"probability_{class_name}"] = probabilities[:, index]
    prediction_frame.to_csv(args.output / "test_predictions.csv", index=False)

    print("Training complete")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Held-out subjects: {len(test_subjects)}")
    print(f"Macro F1: {report['test_macro_f1']:.4f}")
    print(f"Balanced accuracy: {report['test_balanced_accuracy']:.4f}")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()

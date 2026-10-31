"""Drowsiness classification on UTA-RLDD: rule engine vs classical vs temporal models.

Input: per-frame signals from ml/extract_signals.py rldd (6 clips of 36 s per
video, 15 Hz). Each clip calibrates on its first 5.2 s exactly as a new app
session does; the next 30 s are classified as alert, low_vigilant, or drowsy.

Protocol (Ghoddoosian et al., CVPRW 2019): the dataset's own five folds of 12
participants. Every model is trained on four folds and scored on the fifth, so
each prediction is for a person the model has never seen. Video accuracy
averages a video's clip probabilities, matching the paper's headline metric.

    python ml/train_drowsiness.py           # full run
    python ml/train_drowsiness.py --quick   # fewer epochs, smoke test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (COLORS, DATA, FIGURES, LAB, PUBLIC_MODELS, REPORTS, ROOT, bootstrap_ci, calibrate,
                    expected_calibration_error, load_signals, plot_style, write_json)
from sequence_features import SEQUENCE_CHANNELS, SEQUENCE_HZ, sequence_matrix

torch.set_num_threads(8)
CLASSES = ["alert", "low_vigilant", "drowsy"]
CALIBRATION_S = 5.2
WINDOW_S = 30
T = int(WINDOW_S * SEQUENCE_HZ)
PUBLISHED = {"HM-LSTM (Ghoddoosian et al.)": 0.652, "LSTM (Ghoddoosian et al.)": 0.614,
             "Human judges (Ghoddoosian et al.)": 0.578}
CHANNEL_GROUPS = {
    "eyes": ["closure", "ear_rel", "blink", "squint", "wide"],
    "mouth": ["mar_rel", "jaw_open"],
    "head": ["yaw_rel", "pitch_rel", "roll_rel"],
    "gaze_brow": ["gaze_rel", "gaze_bs", "brow_inner_up"],
}


# --- clips --------------------------------------------------------------------

def build_clips(protocol: str = "session"):
    """Clips plus their per-frame sequences and rule-engine replay records.

    protocol "session": each clip calibrates on its own first 5.2 s, as every
    new app session does today. If the driver is already drowsy, that baseline
    is a drowsy face.
    protocol "enrolled_all": each person calibrates once, on the first 5.2 s of
    their alert video, and the whole baseline (eyes, mouth, pose, gaze) is
    reused for all their clips. RLDD's states are separate recordings, often
    with the camera moved, so a carried-over pose baseline partly measures
    "same camera setup as enrolment": this protocol is kept as the confounded
    reference, not as a result.
    protocol "enrolled": only the person-specific eye and mouth reference (EAR,
    MAR) is carried over; pose and gaze still calibrate per clip. This is the
    controlled test of whether a fresh baseline helps.
    In both enrolled protocols the enrolment clip is left out of evaluation.
    """
    frame = load_signals("rldd")
    enrolled = {}
    if protocol.startswith("enrolled"):
        for subject, part in frame[(frame["label"] == "alert") & (frame["segment"] == 0)].groupby("subject_id"):
            enrolled[subject] = calibrate(part.sort_values("time_ms"), CALIBRATION_S)
    clips, sequences, replay = [], [], []
    for (session, segment), clip in frame.groupby(["session", "segment"], sort=True):
        clip = clip.sort_values("time_ms")
        start = clip["time_ms"].iloc[0]
        if clip["face_found"].mean() < 0.5:
            continue  # unusable footage is excluded and counted, not imputed
        subject = clip["subject_id"].iloc[0]
        baseline = calibrate(clip, CALIBRATION_S)
        if protocol.startswith("enrolled"):
            if subject not in enrolled or (clip["label"].iloc[0] == "alert" and segment == 0):
                continue
            baseline = (enrolled[subject] if protocol == "enrolled_all"
                        else {**baseline, "ear": enrolled[subject]["ear"], "mar": enrolled[subject]["mar"]})
        body = clip[clip["time_ms"] - start > CALIBRATION_S * 1000].head(T)
        matrix = sequence_matrix(body, baseline)
        if len(matrix) < T:
            matrix = np.vstack([matrix, np.zeros((T - len(matrix), matrix.shape[1]), np.float32)])
        sequences.append(matrix)
        clips.append({"id": f"{session}#{segment}", "video": session, "subject_id": clip["subject_id"].iloc[0],
                      "fold": clip["fold"].iloc[0], "label": clip["label"].iloc[0],
                      "face_found": float(body["face_found"].mean())})
        cols = ["time_ms", "face_found", "ear", "mar", "yaw", "pitch", "roll", "gaze", "blink",
                "asymmetry", "brightness", "contrast"]
        part = clip.head(int(CALIBRATION_S * SEQUENCE_HZ) + T)
        record = {"id": clips[-1]["id"], "baseline": baseline if protocol.startswith("enrolled") else None}
        for column in cols:
            values = part[column].astype(float) if column in part else pd.Series(np.nan, index=part.index)
            record[column] = [None if np.isnan(v) else round(float(v), 6) for v in values]
        replay.append(record)
    return pd.DataFrame(clips), np.stack(sequences), replay


def run_rule_engine(replay, protocol="session"):
    """Score every clip with the app's own JavaScript engine, unchanged."""
    source = DATA / f"rldd_replay_input_{protocol}.json"
    target = DATA / f"rldd_replay_output_{protocol}.json"
    source.write_text(json.dumps(replay))
    subprocess.run(["node", str(ROOT / "ml" / "replay_rule_engine.mjs"), str(source), str(target),
                    str(WINDOW_S * 1000)], check=True)
    return {r["id"]: r for r in json.loads(target.read_text())}


def window_features(clips, sequences, replayed):
    """Hand-crafted 30 s window features: the app's 13 plus sequence statistics."""
    rows = []
    channel = {name: i for i, name in enumerate(SEQUENCE_CHANNELS)}
    for clip, seq in zip(clips.itertuples(), sequences):
        seen = seq[seq[:, channel["face_found"]] > 0]
        if not len(seen):
            seen = seq
        r = replayed[clip.id]
        risk = np.array(r["risk"][-T:], dtype=float)
        row = dict(r["features"] or {})
        row.update({
            "risk_mean": risk.mean(), "risk_p90": np.percentile(risk, 90), "risk_max": risk.max(),
            "closure_mean": seen[:, channel["closure"]].mean(), "closure_std": seen[:, channel["closure"]].std(),
            "closure_p90": np.percentile(seen[:, channel["closure"]], 90),
            "blink_mean": seen[:, channel["blink"]].mean(),
            "squint_mean": seen[:, channel["squint"]].mean(), "wide_mean": seen[:, channel["wide"]].mean(),
            "brow_mean": seen[:, channel["brow_inner_up"]].mean(),
            "jaw_max": seen[:, channel["jaw_open"]].max(),
            "pitch_std": seen[:, channel["pitch_rel"]].std(), "yaw_std": seen[:, channel["yaw_rel"]].std(),
            "roll_std": seen[:, channel["roll_rel"]].std(),
            "head_motion": np.abs(np.diff(seen[:, channel["pitch_rel"]])).mean() if len(seen) > 1 else 0,
            "blinks_detected": len(r["blinks"]), "yawns_detected": len(r["yawns"]),
        })
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


FEATURE_GROUPS = {
    "eyes": ["ear_mean", "ear_min", "perclos_60s", "longest_closure_ms", "blink_rate_per_min", "closure_mean",
             "closure_std", "closure_p90", "blink_mean", "squint_mean", "wide_mean", "blinks_detected"],
    "mouth": ["mar_mean", "mar_max", "yawns_10m", "jaw_max", "yawns_detected"],
    "head": ["yaw_deviation", "pitch_deviation", "pitch_std", "yaw_std", "roll_std", "head_motion"],
    "gaze_brow": ["gaze_deviation", "brow_mean"],
}


# --- temporal networks -----------------------------------------------------------

class GRUNet(nn.Module):
    def __init__(self, inputs, hidden=32):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(2 * hidden, len(CLASSES))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(self.drop(torch.cat([out.mean(1), out.max(1).values], 1)))


class TCN(nn.Module):
    def __init__(self, inputs, width=32):
        super().__init__()
        self.inp = nn.Conv1d(inputs, width, 1)
        self.blocks = nn.ModuleList(nn.Conv1d(width, width, 5, dilation=d, padding=2 * d) for d in (1, 2, 4, 8))
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(2 * width, len(CLASSES))

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for block in self.blocks:
            h = h + F.relu(block(h))
        return self.head(self.drop(torch.cat([h.mean(2), h.max(2).values], 1)))


def fit_network(kind, x_train, y_train, x_val, y_val, epochs, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GRUNet(x_train.shape[2]) if kind == "gru" else TCN(x_train.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    weights = torch.tensor(len(y_train) / (3 * np.bincount(y_train, minlength=3)), dtype=torch.float32)
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train).long()
    best, best_state, best_epoch = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(xt))
        for start in range(0, len(xt), 32):
            index = order[start:start + 32]
            # Random 20-30 s crops: the model must not depend on a clip's exact start.
            length = int(rng.integers(int(T * 0.66), T + 1))
            offset = int(rng.integers(0, T - length + 1))
            batch = xt[index, offset:offset + length]
            batch = batch + 0.02 * torch.randn_like(batch)
            loss = F.cross_entropy(model(batch), yt[index], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(torch.from_numpy(x_val)), torch.from_numpy(y_val).long()).item()
        if val_loss < best:
            best, best_epoch = val_loss, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


@torch.no_grad()
def logits_of(model, x):
    return model(torch.from_numpy(x)).numpy()


def fit_temperature(logits, y):
    """One scalar T minimising validation NLL (Guo et al., 2017)."""
    grid = np.exp(np.linspace(np.log(0.3), np.log(5), 120))
    nll = [F.cross_entropy(torch.from_numpy(logits / t), torch.from_numpy(y).long()).item() for t in grid]
    return float(grid[int(np.argmin(nll))])


def softmax(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# --- metrics --------------------------------------------------------------------

def evaluate(name, probabilities, clips, y):
    prediction = probabilities.argmax(1)
    videos = clips.assign(**{f"p{i}": probabilities[:, i] for i in range(3)}).groupby("video")
    video_prob = videos[[f"p{i}" for i in range(3)]].mean()
    video_truth = videos["label"].first().map(CLASSES.index)
    video_pred = video_prob.to_numpy().argmax(1)
    subjects = clips["subject_id"].to_numpy()
    by_subject = {s: (y[subjects == s], prediction[subjects == s]) for s in np.unique(subjects)}
    video_subject = videos["subject_id"].first().to_numpy()
    by_subject_video = {s: (video_truth.to_numpy()[video_subject == s], video_pred[video_subject == s])
                        for s in np.unique(video_subject)}
    pooled_accuracy = lambda parts: accuracy_score(np.concatenate([p[0] for p in parts]),
                                                   np.concatenate([p[1] for p in parts]))
    binary = np.isin(y, [0, 2])
    ece, reliability = expected_calibration_error(probabilities, y)
    fold_accuracy = [float(accuracy_score(y[clips["fold"] == f], prediction[clips["fold"] == f]))
                     for f in sorted(clips["fold"].unique())]
    return {
        "name": name,
        "clip_accuracy": float(accuracy_score(y, prediction)),
        "clip_accuracy_ci95": bootstrap_ci(by_subject, pooled_accuracy),
        "clip_macro_f1": float(f1_score(y, prediction, average="macro")),
        "video_accuracy": float(accuracy_score(video_truth, video_pred)),
        "video_accuracy_ci95": bootstrap_ci(by_subject_video, pooled_accuracy),
        "alert_vs_drowsy_auc": float(roc_auc_score(y[binary] == 2, probabilities[binary, 2])),
        "ece": ece,
        "reliability": reliability,
        "fold_accuracy": fold_accuracy,
        "confusion_clip": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
        "confusion_video": confusion_matrix(video_truth, video_pred, labels=[0, 1, 2]).tolist(),
    }


def mcnemar(a_correct, b_correct):
    """Exact McNemar test on paired correctness."""
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}


# --- main -----------------------------------------------------------------------

TABULAR_MODELS = ("rule_engine", "logistic_regression", "random_forest", "gradient_boosting")


def run_protocol(protocol, epochs, models, ablate=False, started=None):
    """Five-fold, subject-independent out-of-fold probabilities for `models`."""
    started = started or time.time()
    clips, sequences, replay = build_clips(protocol)
    replayed = run_rule_engine(replay, protocol)
    features = window_features(clips, sequences, replayed)
    y = clips["label"].map(CLASSES.index).to_numpy()
    print(f"[{protocol}] {len(clips)} clips from {clips['video'].nunique()} videos, "
          f"{clips['subject_id'].nunique()} people", flush=True)

    feature_names = sum(FEATURE_GROUPS.values(), [])
    factories = {
        "rule_engine": (["risk_mean", "risk_p90", "risk_max"],
                        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
        "logistic_regression": (feature_names,
                                lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.3, class_weight="balanced"))),
        "random_forest": (feature_names,
                          lambda: RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=8, random_state=0)),
        "gradient_boosting": (feature_names,
                              lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, class_weight="balanced", random_state=0)),
    }
    networks = [m for m in ("tcn", "gru") if m in models]
    oof = {name: np.zeros((len(clips), 3)) for name in [*models, *(["gru_uncalibrated"] if "gru" in models else [])]}
    groups = [*FEATURE_GROUPS, *[f"without_{g}" for g in FEATURE_GROUPS]]
    ablation = {k: np.zeros((len(clips), 3)) for k in groups} if ablate else {}
    temperatures, best_epochs, importances = [], [], []

    for fold in sorted(clips["fold"].unique()):
        test = (clips["fold"] == fold).to_numpy()
        train = ~test
        for name in [m for m in models if m in factories]:
            columns, factory = factories[name]
            model = factory().fit(features.loc[train, columns], y[train])
            oof[name][test] = model.predict_proba(features.loc[test, columns])
            if name == "gradient_boosting" and ablate:
                importance = permutation_importance(model, features.loc[test, columns], y[test],
                                                    scoring="f1_macro", n_repeats=8, random_state=0, n_jobs=8)
                importances.append(importance.importances_mean)
        for group, columns in (FEATURE_GROUPS.items() if ablate else []):
            for key, use in ((group, columns), (f"without_{group}", [c for c in feature_names if c not in columns])):
                model = factories["gradient_boosting"][1]().fit(features.loc[train, use], y[train])
                ablation[key][test] = model.predict_proba(features.loc[test, use])

        # Sequences: normalise on training frames with a face, hold out 20% of
        # training *people* for early stopping and temperature fitting.
        if networks:
            train_subjects = clips.loc[train, "subject_id"].unique()
            val_subjects = np.random.default_rng(int(fold[-1])).choice(
                train_subjects, max(2, len(train_subjects) // 5), replace=False)
            val = train & clips["subject_id"].isin(val_subjects).to_numpy()
            fit = train & ~val
            seen = sequences[fit][sequences[fit][:, :, -1] > 0]
            mean, std = seen.mean(0), seen.std(0) + 1e-3
            mean[-1], std[-1] = 0.0, 1.0
            normalise = lambda s: ((s - mean) / std).astype(np.float32)  # noqa: E731
            for kind in networks:
                model, best_epoch = fit_network(kind, normalise(sequences[fit]), y[fit], normalise(sequences[val]),
                                                y[val], epochs, seed=int(fold[-1]))
                temperature = fit_temperature(logits_of(model, normalise(sequences[val])), y[val])
                test_logits = logits_of(model, normalise(sequences[test]))
                oof[kind][test] = softmax(test_logits / temperature)
                if kind == "gru":
                    oof["gru_uncalibrated"][test] = softmax(test_logits)
                    temperatures.append(temperature)
                    best_epochs.append(best_epoch)
        print(f"[{protocol}] {fold} done ({time.time() - started:.0f}s)", flush=True)

    importance = (pd.Series(np.mean(importances, 0), index=feature_names).sort_values(ascending=False)
                  if importances else pd.Series(dtype=float))
    return {"clips": clips, "sequences": sequences, "replayed": replayed, "y": y, "oof": oof,
            "ablation": ablation, "temperatures": temperatures, "best_epochs": best_epochs,
            "importance": importance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    epochs = 6 if args.quick else 30
    started = time.time()

    main_run = run_protocol("session", epochs, [*TABULAR_MODELS, "tcn", "gru"], ablate=True, started=started)
    clips, y, oof = main_run["clips"], main_run["y"], main_run["oof"]
    results = {name: evaluate(name, p, clips, y) for name, p in oof.items()}
    ablations = {name: evaluate(name, p, clips, y) for name, p in main_run["ablation"].items()}
    best = max((n for n in results if n != "gru_uncalibrated"), key=lambda n: results[n]["video_accuracy"])
    tests = {f"{best}_vs_rule_engine": mcnemar(oof[best].argmax(1) == y, oof["rule_engine"].argmax(1) == y)}
    if best != "gru":
        tests["gru_vs_rule_engine"] = mcnemar(oof["gru"].argmax(1) == y, oof["rule_engine"].argmax(1) == y)
    tests["gradient_boosting_vs_logistic_regression"] = mcnemar(oof["gradient_boosting"].argmax(1) == y,
                                                                 oof["logistic_regression"].argmax(1) == y)

    # Calibration protocols: one personal calibration while fresh, reused.
    # "enrolled_all" carries the whole baseline and is confounded by camera
    # placement; "enrolled" carries only eyes and mouth and is the real test.
    protocol_models = ["rule_engine", "logistic_regression", "gradient_boosting", "gru"]
    protocols = {}
    for name in ("enrolled_all", "enrolled"):
        run = run_protocol(name, epochs, protocol_models, started=started)
        same = clips["id"].isin(run["clips"]["id"]).to_numpy()
        subset = clips[same].reset_index(drop=True)
        order = subset["id"].map({cid: i for i, cid in enumerate(run["clips"]["id"])}).to_numpy()
        protocols[name] = {
            "clips": int(len(run["clips"])),
            "results": {m: evaluate(m, p, run["clips"], run["y"]) for m, p in run["oof"].items()},
            "session_same_clips": {m: evaluate(m, oof[m][same], subset, y[same]) for m in run["oof"]},
            "mcnemar_vs_session": {m: mcnemar(run["oof"][m][order].argmax(1) == y[same], oof[m][same].argmax(1) == y[same])
                                   for m in protocol_models},
        }

    # The app calibrates every session, so the shipped GRU is the session one.
    final = export_gru(main_run["sequences"], y, int(np.median(main_run["best_epochs"])),
                       float(np.mean(main_run["temperatures"])), "session")
    report = {
        "dataset": {"clips": int(len(clips)), "videos": int(clips["video"].nunique()),
                    "people": int(clips["subject_id"].nunique()), "clip_seconds": WINDOW_S,
                    "hz": SEQUENCE_HZ, "calibration_seconds": CALIBRATION_S,
                    "class_counts": clips["label"].value_counts().to_dict(),
                    "excluded_low_face_clips": int(6 * clips["video"].nunique() - len(clips))},
        "published_video_accuracy": PUBLISHED,
        "models": results,
        "best_model": best,
        "significance": tests,
        "ablation_gradient_boosting": ablations,
        "permutation_importance": main_run["importance"].round(4).to_dict(),
        "calibration_protocols": protocols,
        "gru": {"temperatures": main_run["temperatures"], "best_epochs": main_run["best_epochs"], **final},
    }
    write_json(REPORTS / "drowsiness_metrics.json", report)
    write_json(LAB / "drowsiness.json", {**report, "examples": example_traces(clips, main_run["replayed"], oof["gru"])})
    figures(report, oof, y, clips)
    print(f"done in {time.time() - started:.0f}s")
    for name, r in results.items():
        print(f"{name:22s} video {r['video_accuracy']:.3f}  clip {r['clip_accuracy']:.3f}  "
              f"F1 {r['clip_macro_f1']:.3f}  AUC(a/d) {r['alert_vs_drowsy_auc']:.3f}  ECE {r['ece']:.3f}")
    for protocol, block in protocols.items():
        for name, r in block["results"].items():
            base = block["session_same_clips"][name]
            print(f"{protocol:13s} {name:20s} video {r['video_accuracy']:.3f} (session {base['video_accuracy']:.3f})  "
                  f"AUC(a/d) {r['alert_vs_drowsy_auc']:.3f} (session {base['alert_vs_drowsy_auc']:.3f})")


def export_gru(sequences, y, epochs, temperature, protocol="session"):
    seen = sequences[sequences[:, :, -1] > 0]
    mean, std = seen.mean(0), seen.std(0) + 1e-3
    mean[-1], std[-1] = 0.0, 1.0
    x = ((sequences - mean) / std).astype(np.float32)
    # No held-out people remain, so train for the median best epoch found in CV.
    model, _ = fit_network("gru", x, y, x[:8], y[:8], epochs, seed=11)
    gru = model.gru
    payload = {
        "format": "drowsiness-gru",
        "version": 1,
        "classes": CLASSES,
        "channels": SEQUENCE_CHANNELS,
        "hz": SEQUENCE_HZ,
        "window_seconds": WINDOW_S,
        "normalize": {"mean": mean.round(6).tolist(), "std": std.round(6).tolist()},
        "gru": {"hidden": gru.hidden_size,
                "weight_ih": gru.weight_ih_l0.detach().numpy().round(6).tolist(),
                "weight_hh": gru.weight_hh_l0.detach().numpy().round(6).tolist(),
                "bias_ih": gru.bias_ih_l0.detach().numpy().round(6).tolist(),
                "bias_hh": gru.bias_hh_l0.detach().numpy().round(6).tolist()},
        "head": {"weight": model.head.weight.detach().numpy().round(6).tolist(),
                 "bias": model.head.bias.detach().numpy().round(6).tolist()},
        "temperature": round(temperature, 4),
        "calibration": protocol,
        "trained_on": "UTA-RLDD, all 60 participants (research use; cite Ghoddoosian et al. 2019)",
    }
    # Reference outputs for tests/sequence-net.test.mjs, on rounded weights.
    probe = sequences[[0, len(sequences) // 2]][:, :90]
    reference = GRUNet(len(SEQUENCE_CHANNELS), gru.hidden_size)
    with torch.no_grad():
        reference.gru.weight_ih_l0.copy_(torch.tensor(payload["gru"]["weight_ih"]))
        reference.gru.weight_hh_l0.copy_(torch.tensor(payload["gru"]["weight_hh"]))
        reference.gru.bias_ih_l0.copy_(torch.tensor(payload["gru"]["bias_ih"]))
        reference.gru.bias_hh_l0.copy_(torch.tensor(payload["gru"]["bias_hh"]))
        reference.head.weight.copy_(torch.tensor(payload["head"]["weight"]))
        reference.head.bias.copy_(torch.tensor(payload["head"]["bias"]))
    reference.eval()
    norm = ((probe - mean) / std).astype(np.float32)
    payload["test_vectors"] = [
        {"frames": seq.round(6).tolist(), "probabilities": softmax(logits_of(reference, norm[i:i + 1]) / temperature)[0].tolist()}
        for i, seq in enumerate(probe)
    ]
    write_json(PUBLIC_MODELS / "drowsiness-gru.json", payload, compact=True)
    parameters = sum(p.numel() for p in model.parameters())
    return {"parameters": int(parameters), "weights_kb": round(parameters * 4 / 1024, 1), "export_kb": round((PUBLIC_MODELS / "drowsiness-gru.json").stat().st_size / 1024, 1),
            "final_epochs": epochs, "temperature": temperature, "protocol": protocol}


def example_traces(clips, replayed, gru_probabilities):
    """One alert and one drowsy clip per Lab page chart: rule risk trace + GRU verdict."""
    out = []
    for label in ("alert", "drowsy"):
        candidates = clips.index[(clips["label"] == label) & (clips["face_found"] > 0.95)]
        index = candidates[len(candidates) // 2]
        risk = replayed[clips.loc[index, "id"]]["risk"][-T:]
        out.append({"label": label, "id": clips.loc[index, "id"], "risk": risk[::3],
                    "gru": dict(zip(CLASSES, gru_probabilities[index].round(3).tolist()))})
    return out


def figures(report, oof, y, clips):
    plt = plot_style()
    names = {"rule_engine": "Rule engine (app)", "logistic_regression": "Logistic regression",
             "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
             "tcn": "TCN", "gru": "GRU"}
    models = report["models"]
    order = sorted(names, key=lambda n: models[n]["video_accuracy"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, name in enumerate(order):
        r = models[name]
        low, high = r["video_accuracy_ci95"]
        color = COLORS["accent"] if name == report["best_model"] else COLORS["muted"]
        ax.barh(i, r["video_accuracy"], color=color, height=0.6)
        ax.errorbar(r["video_accuracy"], i, xerr=[[r["video_accuracy"] - low], [high - r["video_accuracy"]]],
                    color=COLORS["dark"], capsize=3, lw=1)
        ax.text(high + 0.01, i, f"{r['video_accuracy']:.1%}", va="center", fontsize=8)
    for label, value in PUBLISHED.items():
        ax.axvline(value, color=COLORS["warn"] if "HM" in label else COLORS["muted"], ls="--", lw=0.9)
        ax.text(value, len(order) - 0.35, label.split(" (")[0], rotation=90, fontsize=7, va="top", ha="right",
                color=COLORS["dark"])
    ax.axvline(1 / 3, color=COLORS["muted"], ls=":", lw=0.8)
    ax.set_yticks(range(len(order)), [names[n] for n in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("3-class video accuracy, held-out people (95% CI over people)")
    ax.set_title("UTA-RLDD, 60 people, official 5-fold split")
    fig.savefig(FIGURES / "rldd_model_comparison.png")
    plt.close(fig)

    best = report["best_model"]
    matrix = np.array(models[best]["confusion_video"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    normalised = matrix / matrix.sum(1, keepdims=True)
    ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]}\n{normalised[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if normalised[i, j] > 0.55 else COLORS["dark"])
    labels = ["Alert", "Low vig.", "Drowsy"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set(xlabel="Predicted", ylabel="True", title=f"{names[best]}: video confusion")
    ax.grid(False)
    fig.savefig(FIGURES / "rldd_confusion.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3.6))
    for key, label, color in (("gru_uncalibrated", "GRU before", COLORS["muted"]), ("gru", "GRU after temperature", COLORS["accent"])):
        ece, curve = expected_calibration_error(oof[key], y)
        ax.plot([c["confidence"] for c in curve], [c["accuracy"] for c in curve], "o-", color=color, ms=4,
                label=f"{label} (ECE {ece:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=COLORS["muted"], lw=0.8)
    ax.set(xlabel="Predicted confidence", ylabel="Observed accuracy", title="Reliability diagram")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(FIGURES / "rldd_calibration.png")
    plt.close(fig)

    ablation = report["ablation_gradient_boosting"]
    full = models["gradient_boosting"]["clip_accuracy"]
    groups = list(FEATURE_GROUPS)
    fig, ax = plt.subplots(figsize=(5.4, 3))
    ax.bar(np.arange(len(groups)) - 0.2, [ablation[g]["clip_accuracy"] for g in groups], 0.4,
           color=COLORS["accent"], label="Only this group")
    ax.bar(np.arange(len(groups)) + 0.2, [ablation[f"without_{g}"]["clip_accuracy"] for g in groups], 0.4,
           color=COLORS["muted"], label="All but this group")
    ax.axhline(full, color=COLORS["dark"], lw=1, ls="--")
    ax.text(len(groups) - 0.5, full + 0.005, f"all features {full:.1%}", ha="right", fontsize=8)
    ax.axhline(1 / 3, color=COLORS["muted"], lw=0.8, ls=":")
    ax.set_xticks(range(len(groups)), ["Eyes", "Mouth", "Head pose", "Gaze + brow"])
    ax.set(ylabel="Clip accuracy", title="Signal-group ablation (gradient boosting)")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(FIGURES / "rldd_ablation.png")
    plt.close(fig)

    importance = pd.Series(report["permutation_importance"]).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.barh(range(len(importance)), importance.values, color=COLORS["accent"])
    ax.set_yticks(range(len(importance)), importance.index, fontsize=8)
    ax.set(xlabel="Drop in macro F1 when shuffled", title="Top features (permutation importance)")
    fig.savefig(FIGURES / "rldd_importance.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

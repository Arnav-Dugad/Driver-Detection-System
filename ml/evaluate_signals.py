"""Frame- and event-level evaluation of the app's eye and yawn signals on DMD.

DMD annotates every face-camera frame with the driver's eye state and marks
each blink and yawn, so the browser's measurements can be checked directly:

  * closed-eye detection per frame: geometric EAR, the learned blink
    blendshape, the app's pose-weighted fusion of the two, the new eye CNN
    (out-of-fold, so never scored on a driver it trained on), and a stacked
    fusion of all three learned per driver-fold;
  * robustness as the head pitches down, where EAR is expected to fail;
  * the aspect-ratio bug: the same detector with and without the correction;
  * blink and yawn *events* from the unchanged JavaScript engine, replayed
    over each session, against the annotated intervals.

    python ml/evaluate_signals.py      (after ml/train_eye_cnn.py)
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from common import COLORS, DATA, FIGURES, LAB, REPORTS, ROOT, bootstrap_ci, calibrate, load_signals, plot_style, write_json
from signals import fuse_closure

# classifySignals() at the default sensitivity 0.62: eyes count as closed when
# fused closure exceeds 1 - (0.67 + 0.62 * 0.12).
CLOSURE_CUT = 1 - (0.67 + 0.62 * 0.12)
DMD_ASPECT = 1280 / 720


def frame_scores():
    data = load_signals("dmd")
    parts = []
    for session, frame in data.groupby("session"):
        frame = frame.sort_values("frame").copy()
        baseline = calibrate(frame)
        found = frame["face_found"].to_numpy(bool)
        ear = frame["ear"].to_numpy(float)
        blink = frame["blink"].to_numpy(float)
        pose = np.maximum(np.abs(frame["yaw"] - baseline["yaw"]), np.abs(frame["pitch"] - baseline["pitch"])).to_numpy()
        frame["score_ear"] = np.clip(1 - ear / baseline["ear"], 0, 1)
        frame["score_blendshape"] = blink
        frame["score_fused"] = [fuse_closure(e, baseline["ear"], b, p) if f else np.nan
                                for e, b, p, f in zip(ear, blink, pose, found)]
        # What the app computed before the aspect fix: EAR inflated by the
        # frame's aspect ratio and the calibration clamp at 0.42.
        ear_raw = ear * DMD_ASPECT
        base_raw = float(np.clip(baseline["ear"] * DMD_ASPECT, 0.18, 0.42))
        frame["score_ear_uncorrected"] = np.clip(1 - ear_raw / base_raw, 0, 1)
        frame["score_fused_uncorrected"] = [fuse_closure(e, base_raw, b, p) if f else np.nan
                                            for e, b, p, f in zip(ear_raw, blink, pose, found)]
        frame["pitch_rel"] = frame["pitch"] - baseline["pitch"]
        parts.append(frame)
    scored = pd.concat(parts, ignore_index=True)
    cnn = pd.read_parquet(DATA / "eye_cnn_oof.parquet")[["session", "frame", "cnn_eyenet_s_int8"]]
    scored = scored.merge(cnn.rename(columns={"cnn_eyenet_s_int8": "score_cnn"}), on=["session", "frame"], how="left")
    return scored


def stacked(labelled):
    """Logistic stacking of EAR, blendshape, and CNN, fitted per held-out driver fold."""
    columns = ["score_ear", "score_blendshape", "score_cnn"]
    x = labelled[columns].fillna(0).to_numpy()
    y = labelled["y"].to_numpy()
    out = np.zeros(len(labelled))
    coefficients = []
    for train, test in GroupKFold(5).split(x, y, labelled["subject_id"]):
        model = LogisticRegression(class_weight="balanced").fit(x[train], y[train])
        out[test] = model.predict_proba(x[test])[:, 1]
        coefficients.append(model.coef_[0].tolist() + [model.intercept_[0]])
    return out, np.mean(coefficients, axis=0)


def detector_metrics(labelled, column, subjects):
    score = labelled[column].fillna(0).to_numpy()
    y = labelled["y"].to_numpy()
    by_subject = {s: (y[subjects == s], score[subjects == s]) for s in np.unique(subjects)}

    def auc(parts):
        t = np.concatenate([p[0] for p in parts])
        s = np.concatenate([p[1] for p in parts])
        return roc_auc_score(t, s) if 0 < t.mean() < 1 else np.nan

    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "roc_auc_ci95": bootstrap_ci(by_subject, auc, repeats=1000),
        "best_f1": float(f1.max()),
        "per_driver_auc": {s: float(roc_auc_score(*v)) for s, v in by_subject.items() if 0 < v[0].mean() < 1},
    }


def tune_threshold(labelled):
    """Pick the fused-closure cut by leave-one-driver-out cross-validation.

    The cut is chosen on the other drivers and applied to the held-out one, so
    the reported F1 is what a new driver would get, not the in-sample optimum.
    """
    cuts = np.round(np.arange(0.2, 0.81, 0.02), 2)
    score = labelled["score_fused"].fillna(0).to_numpy()
    y = labelled["y"].to_numpy()
    drivers = labelled["subject_id"].to_numpy()
    decisions = np.zeros(len(y), dtype=bool)
    chosen = []
    for driver in np.unique(drivers):
        train = drivers != driver
        f1s = [f1_score(y[train], score[train] > cut) for cut in cuts]
        cut = float(cuts[int(np.argmax(f1s))])
        chosen.append(cut)
        decisions[~train] = score[~train] > cut
    curve = [{"cut": float(c), "f1": float(f1_score(y, score > c))} for c in cuts]
    return {"recommended_cut": float(np.median(chosen)), "fold_cuts": chosen,
            "cv_f1": float(f1_score(y, decisions)), "old_cut": CLOSURE_CUT, "curve": curve}, decisions


def events(data):
    """Replay the JavaScript engine over each DMD session and score its events."""
    clips = []
    for session, frame in data.groupby("session"):
        frame = frame.sort_values("frame")
        record = {"id": session}
        for column in ["time_ms", "face_found", "ear", "mar", "yaw", "pitch", "roll", "gaze", "blink",
                       "asymmetry", "brightness", "contrast"]:
            record[column] = [None if pd.isna(v) else float(v) for v in frame[column]]
        clips.append(record)
    source, target = DATA / "dmd_replay_input.json", DATA / "dmd_replay_output.json"
    source.write_text(json.dumps(clips))
    subprocess.run(["node", str(ROOT / "ml" / "replay_rule_engine.mjs"), str(source), str(target), "60000"], check=True)
    replayed = {r["id"]: r for r in json.loads(target.read_text())}

    totals = {"blink": {"tp": 0, "fp": 0, "fn": 0}, "yawn": {"tp": 0, "fp": 0, "fn": 0}}
    hours = 0.0
    frame_rule = []
    for session, frame in data.groupby("session"):
        frame = frame.sort_values("frame").reset_index(drop=True)
        r = replayed[session]
        hours += (frame["time_ms"].iloc[-1] - frame["time_ms"].iloc[0]) / 3.6e6
        frame_rule.append(pd.DataFrame({"session": session, "frame": frame["frame"], "rule_closed": r["eyesClosed"]}))
        for kind, label, predicted, tolerance in (("blink", "blink_label", r["blinks"], 8),
                                                  ("yawn", "yawn_label", r["yawns"], 30)):
            flags = frame[label].to_numpy(bool)
            starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
            ends = np.flatnonzero(flags & ~np.r_[flags[1:], False])
            intervals = list(zip(starts - tolerance, ends + tolerance))
            matched = set()
            for index in predicted:
                hit = next((i for i, (a, b) in enumerate(intervals) if a <= index <= b and i not in matched), None)
                if hit is None:
                    totals[kind]["fp"] += 1
                else:
                    matched.add(hit)
                    totals[kind]["tp"] += 1
            totals[kind]["fn"] += len(intervals) - len(matched)
    out = {}
    for kind, t in totals.items():
        precision = t["tp"] / max(1, t["tp"] + t["fp"])
        recall = t["tp"] / max(1, t["tp"] + t["fn"])
        out[kind] = {**t, "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / max(1e-9, precision + recall),
                     "false_alarms_per_hour": t["fp"] / hours}
    out["hours"] = hours
    return out, pd.concat(frame_rule)


def main() -> None:
    data = frame_scores()
    labelled = data[data["eyes_state"].isin(["open", "close"]) & data["face_found"]].copy()
    labelled["y"] = (labelled["eyes_state"] == "close").astype(int)
    labelled["score_stacked"], coefficients = stacked(labelled)
    subjects = labelled["subject_id"].to_numpy()

    detectors = {
        "score_ear": "EAR (geometry)",
        "score_blendshape": "Blink blendshape",
        "score_fused": "App fusion (EAR + blendshape)",
        "score_cnn": "EyeNet CNN (ours)",
        "score_stacked": "Stacked fusion (ours)",
    }
    results = {key: {"label": label, **detector_metrics(labelled, key, subjects)} for key, label in detectors.items()}

    # Robustness to looking down: AUC per pitch band relative to calibration.
    bands = [(-1, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 2)]
    robustness = []
    for low, high in bands:
        part = labelled[(labelled["pitch_rel"] > low) & (labelled["pitch_rel"] <= high)]
        if part["y"].nunique() < 2 or len(part) < 200:
            continue
        robustness.append({"band": f"{np.degrees(max(low, 0)):.0f}-{np.degrees(min(high, 0.6)):.0f} deg",
                           "frames": int(len(part)),
                           **{key: float(roc_auc_score(part["y"], part[key].fillna(0))) for key in detectors}})

    # The aspect-ratio bug, measured as the app's own closed-eye decision.
    truth = labelled["y"].to_numpy()
    aspect = {}
    for name, column in (("corrected", "score_fused"), ("before_fix", "score_fused_uncorrected"),
                         ("ear_corrected", "score_ear"), ("ear_before_fix", "score_ear_uncorrected")):
        decision = labelled[column].fillna(0).to_numpy() > CLOSURE_CUT
        aspect[name] = {"f1": float(f1_score(truth, decision)), "recall": float(decision[truth == 1].mean()),
                        "false_positive_rate": float(decision[truth == 0].mean())}

    # Held-out decisions: each driver judged with a cut chosen on the others.
    tuning, tuned = tune_threshold(labelled)
    aspect["corrected_tuned"] = {"f1": float(f1_score(truth, tuned)), "recall": float(tuned[truth == 1].mean()),
                                 "false_positive_rate": float(tuned[truth == 0].mean())}

    event_metrics, rule_frames = events(data)
    rule = labelled.merge(rule_frames, on=["session", "frame"])
    rule_decision = {"f1": float(f1_score(rule["y"], rule["rule_closed"])),
                     "recall": float(rule.loc[rule["y"] == 1, "rule_closed"].mean()),
                     "false_positive_rate": float(rule.loc[rule["y"] == 0, "rule_closed"].mean())}

    report = {
        "dataset": {"frames": int(len(data)), "labelled_frames": int(len(labelled)),
                    "drivers": int(data["subject_id"].nunique()), "sessions": int(data["session"].nunique()),
                    "closed_share": float(truth.mean()), "face_found": float(data["face_found"].mean())},
        "closed_eye_detectors": results,
        "stacked_coefficients": dict(zip(["ear", "blendshape", "cnn", "intercept"], np.round(coefficients, 4))),
        "pitch_robustness": robustness,
        "aspect_ratio_fix": aspect,
        "threshold_tuning": tuning,
        "rule_engine_eyes_closed": rule_decision,
        "events": event_metrics,
    }
    write_json(REPORTS / "signal_metrics.json", report)
    roc = {}
    for key in detectors:
        fpr, tpr, _ = roc_curve(truth, labelled[key].fillna(0))
        step = max(1, len(fpr) // 150)
        roc[key] = {"fpr": fpr[::step].round(4).tolist(), "tpr": tpr[::step].round(4).tolist()}
    write_json(LAB / "signals.json", {**report, "roc": roc})
    figures(report, labelled, detectors)
    for key, r in results.items():
        print(f"{r['label']:34s} AUC {r['roc_auc']:.3f} {r['roc_auc_ci95']}  best F1 {r['best_f1']:.3f}")
    print("aspect fix:", aspect)
    print("events:", event_metrics)


def figures(report, labelled, detectors):
    plt = plot_style()
    colors = [COLORS["muted"], COLORS["series"][1], COLORS["series"][2], COLORS["accent"], COLORS["series"][3]]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for (key, label), color in zip(detectors.items(), colors):
        fpr, tpr, _ = roc_curve(labelled["y"], labelled[key].fillna(0))
        ax.plot(fpr, tpr, color=color, lw=2.2 if "cnn" in key or "stacked" in key else 1.4,
                label=f"{label}  {report['closed_eye_detectors'][key]['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color=COLORS["muted"], lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Closed-eye detection on DMD (13 drivers)")
    ax.legend(title="ROC-AUC", loc="lower right", fontsize=7.5, title_fontsize=8, frameon=False)
    fig.savefig(FIGURES / "dmd_closed_eye_roc.png")
    plt.close(fig)

    rows = report["pitch_robustness"]
    if rows:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        for (key, label), color in zip(detectors.items(), colors):
            ax.plot([r["band"] for r in rows], [r[key] for r in rows], "o-", color=color, label=label, ms=4)
        ax.set(ylabel="ROC-AUC", xlabel="Head pitched down from calibration",
               title="Robustness as the driver looks down")
        ax.legend(fontsize=7, frameon=False)
        fig.savefig(FIGURES / "dmd_pitch_robustness.png")
        plt.close(fig)

    aspect = report["aspect_ratio_fix"]
    fig, ax = plt.subplots(figsize=(4.6, 3))
    names = ["Before fix\n(old cut)", "Fixed EAR\n(old cut)", "Fixed EAR\n(tuned cut)"]
    values = [aspect["before_fix"]["f1"], aspect["corrected"]["f1"], aspect["corrected_tuned"]["f1"]]
    bars = ax.bar(range(3), values, 0.55, color=[COLORS["muted"], COLORS["warn"], COLORS["accent"]])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), names, fontsize=8)
    ax.set(ylabel="Closed-eye F1, app fusion", ylim=(0.5, 1), title="Aspect fix, then a data-tuned threshold")
    fig.savefig(FIGURES / "dmd_aspect_fix.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

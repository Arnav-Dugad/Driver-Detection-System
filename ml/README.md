# Optional learned fusion experiment

The live browser app works without Python. This directory is for the formal machine-learning experiment in a college report.

## What it trains

`train_fusion.py` trains a class-balanced Random Forest on numeric temporal windows. It performs a subject-independent holdout, tunes only on the training participants with GroupKFold, and writes:

- `driver_fusion.joblib`
- `metrics.json`
- `feature_importance.csv`
- `test_predictions.csv`

## Getting the data out of the app

You no longer have to build the CSV by hand. In the app open
**Settings -> Research -> Record labeled windows** (off by default), pick a
label, run a session, then export. The browser writes exactly the schema below,
and `tests/features.test.mjs` asserts the two lists match by reading the
`FEATURES` list in `train_fusion.py` directly, so the app and this pipeline
cannot silently drift apart.

The recorder captures numeric aggregates only. No image, video frame, face
template, landmark set, or identity is written, and nothing is uploaded.

## CSV format

Every row is one labeled temporal window. Required columns:

```text
subject_id,label,ear_mean,ear_min,perclos_60s,longest_closure_ms,
blink_rate_per_min,mar_mean,mar_max,yawns_10m,yaw_deviation,
pitch_deviation,gaze_deviation,phone_ratio,face_missing_ratio
```

Use anonymous participant codes in `subject_id`. Suggested labels are `focused`, `caution`, `drowsy`, and `distracted`. Do not put adjacent windows from one participant into different splits; the script prevents this by splitting on `subject_id`.

## Setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r ml\requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Verify the pipeline with synthetic data

```powershell
python ml\generate_demo_data.py
python ml\train_fusion.py --data ml\data\synthetic_demo.csv --quick
```

Synthetic results only show that the code works. They are not evidence of real-world accuracy and must be labeled as synthetic in a presentation.

## Train on real labeled features

```powershell
python ml\train_fusion.py --data path\to\your_labeled_windows.csv --output ml\artifacts\real_experiment
```

Keep the raw consented research data outside Git. Commit only aggregate metrics and documentation unless every participant explicitly approved publication.

## Score a trained model in the browser

`export_browser_model.py` flattens the trained forest into a compact JSON that
`lib/detection/learned.mjs` can walk directly, with no ONNX runtime and no
server call:

```powershell
python ml\export_browser_model.py --model ml\artifacts\driver_fusion.joblib
```

The export subsamples the forest (`--max-trees`, default 60) to keep the payload
small enough to ship to a phone. It also carries the training medians so the
browser imputes missing features exactly the way training did.

The deterministic engine in `lib/detection/core.mjs` stays the default and the
fallback. A learned model can only nudge the blended score, never override
obvious explainable evidence such as a two-second eye closure - a model trained
on a classroom-sized dataset has not earned that authority.

## Recommended report tables

- participants and recording conditions
- label definitions and reviewer agreement
- participant-safe split counts
- per-class precision, recall, and F1
- macro F1 and balanced accuracy
- confusion matrix
- false alerts per hour
- alert latency for each event class
- subgroup results and known failures
- ablation comparison against EAR-only and no-calibration baselines
- deterministic engine versus learned fusion on the same held-out participants

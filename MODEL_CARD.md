# Model card — Driver Drowsiness & Distraction Detection System

## Intended use

The system is intended for education, prototyping, offline demonstrations, and consented research into driver-state monitoring. It is not intended to control a vehicle, diagnose a medical condition, identify a person, or replace sleep and responsible driving.

## Model components

- **MediaPipe Face Landmarker:** produces 478 normalized face and iris landmarks.
- **EfficientDet-Lite0:** detects the `cell phone` object category.
- **Temporal fusion:** deterministic, personalized signal-processing and risk-scoring logic in `lib/detection/core.mjs`.
- **Optional learned fusion:** a Random Forest training and evaluation pipeline in `ml/train_fusion.py`.

## Inputs and outputs

Input is a webcam video frame. Intermediate outputs are face landmarks, object detections, and normalized numeric geometry. The live output is a 0–100 attention risk score, a risk state, a primary contributing signal, and event flags.

No identity embedding is produced. No frame is intentionally persisted.

## Decision thresholds

Eye, mouth, head, and gaze thresholds are calculated relative to a five-second robust median baseline. Sensitivity adjusts these thresholds within bounded limits. Risk uses persistence ramps and a concurrence bonus, then exponential smoothing.

## Evaluation status

The repository includes deterministic logic tests and build tests. It does **not** include a claimed real-world accuracy number because no representative, consented, subject-independent evaluation dataset is bundled.

Before reporting accuracy, evaluate on people who never appear in training and report:

- per-class precision, recall, and F1;
- macro F1 and balanced accuracy;
- false positive alerts per hour;
- event detection latency;
- ROC-AUC where appropriate;
- results by lighting, glasses, camera, skin tone, and head angle;
- confidence intervals and the number of participants.

## Known limitations

- extreme low light, glare, occlusion, and large camera angles can break landmark tracking;
- glasses and sunglasses can reduce eye and iris accuracy;
- talking can resemble yawning;
- a visible phone does not prove interaction;
- facial geometry and behavior vary between people;
- browser performance and alert timing vary by hardware;
- demo mode is synthetic and must not be used as accuracy evidence;
- the system has not been certified under an automotive functional-safety standard.

## Fairness and privacy

Evaluation should include diverse participants and explicitly measure subgroup performance. Do not collect footage without informed consent. Prefer numeric features, minimize retention, encrypt research data, restrict access, and publish only aggregate results.

## Versioning

Any change to landmark indices, thresholds, temporal windows, model files, or alert behavior should update this card and be treated as a model change.

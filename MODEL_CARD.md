# Model card — Driver Drowsiness & Distraction Detection

## Intended use

The system is intended for education, prototyping, offline demonstrations, and consented research into driver-state monitoring. It is not intended to control a vehicle, diagnose a medical condition, identify a person, or replace sleep and responsible driving.

## Model components

- **MediaPipe Face Landmarker:** produces 478 normalized face and iris landmarks,
  52 face blendshape scores, and a 4x4 facial transformation matrix.
- **EfficientDet-Lite0:** detects the `cell phone` object category.
- **Temporal fusion:** deterministic, personalized signal-processing and risk-scoring logic in `lib/detection/core.mjs`.
- **EyeNet (trained here):** a 15,290-parameter CNN that classifies 32x32
  landmark-aligned eye crops as open or closed. BatchNorm folded, int8 weights
  (15 KB), plain-JavaScript inference in `lib/detection/eyeNet.mjs`. Trained on
  the DMD drowsiness subset (13 drivers).
- **Drowsiness GRU (trained here):** 32 hidden units over 30 s of 14 calibrated
  per-frame channels at 15 Hz; outputs alert / low-vigilant / drowsy,
  temperature-scaled. Trained on all 60 UTA-RLDD participants;
  `lib/detection/sequence.mjs`.
- **Optional recorder fusion:** the original Random Forest pipeline in
  `ml/train_fusion.py` for consented recordings, scored by
  `lib/detection/learned.mjs`. The deterministic engine remains the default and
  the fallback; learned models are shown beside it in the Model Lab.

## Signal derivation

- **Head pose** comes from a Euler decomposition (YXZ order) of the facial
  transformation matrix, with scale divided out so the angles do not change with
  how close the driver sits to the camera. When the matrix is unavailable, yaw and
  pitch are approximated from landmark ratios and converted to the same radian
  scale. Thresholds are angular: roughly 17 degrees of yaw or 15 degrees of pitch
  at default sensitivity.
- **Eye closure** fuses the geometric Eye Aspect Ratio with the learned
  `eyeBlinkLeft` / `eyeBlinkRight` blendshapes. EAR is precise head-on but
  distorts off-axis - pitching the chin down foreshortens the eyelid gap and
  reads as a closure, while yaw compresses the visible eye width and inflates
  EAR. The two are weighted by current pose reliability and by how far the
  learned score sits from its midpoint. Where no blendshape is available the
  system falls back to EAR alone.
- **PERCLOS** uses the P80 definition and is **time-weighted**: the proportion of
  wall-clock time the eyes are at least 80% closed across a rolling 60-second
  window. Counting samples instead biases the measure by frame rate, and frame
  rate drops hardest during exactly the long closures that matter most.
- **Yawns** require the morphology of a yawn rather than a mouth-opening
  threshold: a sustained opening held for at least 1.2 seconds with low
  oscillation. Speech reaches a comparable peak but oscillates at roughly
  2-5 Hz, which is what previously produced false yawn detections.
- **Confidence** is estimated per frame from brightness, contrast, frame rate,
  head angle, and left/right blink asymmetry. It attenuates the risk score
  across a floor rather than to zero, and is shown to the driver directly. A
  frame the system cannot read is reported as such instead of scored
  confidently.
- **Circadian context** applies a bounded multiplier (at most +18%) reflecting
  documented elevated-risk windows near 02:00-06:00 and 14:00-16:00 and time on
  task. It cannot move a calm driver into an alert state on its own, and every
  contributing factor is named in the interface.

## Inputs and outputs

Input is a webcam video frame. Intermediate outputs are face landmarks, object detections, and normalized numeric geometry. The live output is a 0–100 attention risk score, a risk state, a primary contributing signal, and event flags.

No identity embedding is produced. No frame is intentionally persisted.

## Decision thresholds

Eye, mouth, head, and gaze thresholds are calculated relative to a five-second robust median baseline. Sensitivity adjusts these thresholds within bounded limits. Risk uses persistence ramps and a concurrence bonus, then exponential smoothing.

The baseline adapts slowly during high-confidence, focused, face-present periods
so that posture drift over a long session does not accumulate into false alerts.
Adaptation is clamped to +/-15% of the originally calibrated values, so the
system cannot adapt its way into treating a genuinely drowsy face as normal.

## Evaluation

All results are for people absent from training. Full numbers, intervals and
figures: `ml/reports/*.json`, `ml/reports/figures/`, and the Model Lab (`/lab`).

### Eye state (DMD, 13 drivers, 65,290 labelled frames, 5-fold by driver)

| Model | Parameters | Accuracy | F1 closed (95% CI over drivers) | ROC-AUC |
| --- | --- | --- | --- | --- |
| Pixels + PCA + logistic regression | 65,601 | 97.2% | 0.900 (0.846-0.936) | 0.989 |
| EyeNet-S int8 (shipped) | 15,290 | 98.9% | 0.959 (0.934-0.977) | 0.998 |

Out of domain, MRL Eye (4,000 near-infrared images, 16 subjects, never used for
training): 92.6% accuracy, ROC-AUC 0.958.

### Closed-eye decision in the app

The aspect-ratio correction plus a closed-eye cut of 0.52, chosen by
leave-one-driver-out cross-validation, gives closed-eye F1 0.948 on held-out
drivers (0.918 before the fix, 0.803 with the fix and the old cut). Replayed
through the app's own engine: F1 0.956, false-positive rate 0.3%.

Events, replayed over 49 minutes of DMD: blinks 87.0% precision / 62.8% recall;
yawns 69.8% precision / 39.5% recall (most misses are yawns behind a hand).

### Drowsiness (UTA-RLDD, 60 people, official 5 folds)

Three-class video accuracy (alert / low vigilant / drowsy; chance 33%), 95%
intervals from resampling people:

| Model | Video accuracy | Alert vs drowsy ROC-AUC | ECE |
| --- | --- | --- | --- |
| Rule engine (app, replayed) | 47.8% (42.3-53.3) | 0.630 | 0.017 |
| Logistic regression | 53.3% (48.1-58.6) | 0.761 | 0.077 |
| GRU, temperature-scaled (shipped) | 52.2% (45.9-58.7) | 0.780 | 0.024 |
| Published HM-LSTM (whole videos) | 65.2% | - | - |

Logistic regression and the GRU beat the rule engine (exact McNemar p = 0.018
and 0.035). Each video contributes six 30-second clips, about a third of the
footage the published model read. Reusing an alert-state calibration looked
like a large gain until pose and gaze, which identify the recording setup, were
excluded; with only the eye and mouth reference reused, the gain disappeared,
so the app keeps per-session calibration.

## Known limitations

- extreme low light, glare, occlusion, and large camera angles can break landmark tracking;
- glasses and sunglasses can reduce eye and iris accuracy;
- talking can still resemble yawning under unusual speech patterns, though the
  duration and oscillation gates reject ordinary conversation;
- yawns covered by a hand are mostly missed;
- the stricter, data-tuned eye cut misses partial blinks, lowering blink-rate
  recall;
- the drowsiness labels are self-rated KSS states of whole recordings, not
  moment-to-moment ground truth;
- the public DMD subset has 13 drivers, so per-driver intervals are wide;
- a visible phone does not prove interaction;
- facial geometry and behavior vary between people;
- browser performance and alert timing vary by hardware;
- demo mode is synthetic and must not be used as accuracy evidence;
- the system has not been certified under an automotive functional-safety standard.

## Fairness and privacy

Evaluation should include diverse participants and explicitly measure subgroup performance. Do not collect footage without informed consent. Prefer numeric features, minimize retention, encrypt research data, restrict access, and publish only aggregate results.

## Optional research capture

The application can record labeled feature windows for training, but only when a
person explicitly enables it. It is off by default. What is written is one row
per second of the thirteen numeric aggregates listed in `ml/train_fusion.py`,
plus a randomly generated local participant tag used solely to keep each person
inside one train/test split. No image, video frame, face template, landmark set,
or identity is captured, and nothing is transmitted anywhere: the export is a
file the person downloads themselves.

## Dataset licences

UTA-RLDD (Ghoddoosian et al., CVPRW 2019) is used for research with citation.
DMD (Ortega et al., ECCV-W 2020) is CC BY-NC-ND 4.0: used for non-commercial
coursework only, and no frame or crop is redistributed. MRL Eye (Fusek, ISVC
2018) is used as a test set only.

## Versioning

Any change to landmark indices, thresholds, temporal windows, signal
definitions, model files, or alert behavior should update this card and be
treated as a model change. The PERCLOS definition changed from sample-counted to
time-weighted P80, and head pose changed from landmark ratios to a matrix
decomposition in radians; thresholds from earlier versions do not transfer.
Landmark ratios are now computed in square pixels (the `aspect` option), and the
closed-eye cut moved from 0.26 to 0.52 accordingly; EAR values from earlier
versions are not comparable.

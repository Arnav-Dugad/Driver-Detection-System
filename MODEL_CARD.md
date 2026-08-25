# Model card — Driver Drowsiness & Distraction Detection System

## Intended use

The system is intended for education, prototyping, offline demonstrations, and consented research into driver-state monitoring. It is not intended to control a vehicle, diagnose a medical condition, identify a person, or replace sleep and responsible driving.

## Model components

- **MediaPipe Face Landmarker:** produces 478 normalized face and iris landmarks,
  52 face blendshape scores, and a 4x4 facial transformation matrix.
- **EfficientDet-Lite0:** detects the `cell phone` object category.
- **Temporal fusion:** deterministic, personalized signal-processing and risk-scoring logic in `lib/detection/core.mjs`.
- **Optional learned fusion:** a Random Forest training and evaluation pipeline in
  `ml/train_fusion.py`, exportable for in-browser scoring through
  `ml/export_browser_model.py` and `lib/detection/learned.mjs`. The deterministic
  engine remains the default and the fallback.

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
- talking can still resemble yawning under unusual speech patterns, though the
  duration and oscillation gates reject ordinary conversation;
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

## Versioning

Any change to landmark indices, thresholds, temporal windows, signal
definitions, model files, or alert behavior should update this card and be
treated as a model change. The PERCLOS definition changed from sample-counted to
time-weighted P80, and head pose changed from landmark ratios to a matrix
decomposition in radians; thresholds from earlier versions do not transfer.

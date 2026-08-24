# Improvement roadmap

## Stage 1 — strong college submission

- Record consented non-driving sessions across at least 15–20 participants.
- Include normal blinks, reading, conversation, yawns, long closures, head turns, glasses, and multiple lighting conditions.
- Label events with two reviewers and report inter-rater agreement.
- Keep every participant in only one train/validation/test split.
- Measure macro F1, recall for the danger class, false alarms per hour, and alert latency.
- Compare the system against simple fixed-threshold EAR and single-frame baselines.

## Stage 2 — research-grade prototype

- Train a temporal model on sequences instead of only aggregated windows.
- Distill the learned fusion model to ONNX for browser inference.
- Add uncertainty calibration and show when the system does not know.
- Evaluate subgroup performance across lighting, glasses, skin tones, facial hair, and camera quality.
- Add low-light and infrared profiles.
- Run ablation studies to quantify the value of PERCLOS, gaze, head pose, yawns, and phone detection.

## Stage 3 — simulator study

- Use a driving simulator; do not induce sleepiness on public roads.
- Measure how early alerts occur before lane deviation or reaction-time failure.
- Test tone, voice, seat vibration, and escalating multimodal alerts.
- Record alert acceptance, annoyance, false alarm response, and recovery time.
- Obtain institutional ethics approval before studying people.

## Stage 4 — product engineering

- Move inference to a dedicated camera computer or vehicle-grade edge device.
- Add watchdog processes, health checks, thermal limits, and graceful degradation.
- Sign model files and releases.
- Add a PWA shell and offline asset cache.
- Design a vibration alert through a safe dedicated device.
- Add fleet analytics based on anonymous events, never raw faces.
- Commission accessibility, privacy, cybersecurity, and automotive safety reviews.

## High-impact ideas

- Driver-specific continual calibration that updates only during high-confidence focused periods.
- Uncertainty-aware risk fusion that reduces confidence under glare or partial occlusion.
- Circadian context using time of day and session duration without collecting identity.
- A “minimum risk maneuver” handoff to a simulator or robotic platform.
- Multilingual alert packs generated and reviewed by native speakers.
- Privacy-preserving federated learning with secure aggregation.
- Counterfactual explanations: “risk would fall from 68 to 31 if gaze returned forward.”
- A public benchmark dashboard that publishes failures as well as successes.

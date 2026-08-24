# College presentation guide

## A clean 6-minute demo

1. **Problem — 30 seconds:** explain that fatigue and distraction are temporal, not single-frame events.
2. **Architecture — 60 seconds:** show the flow in `docs/ARCHITECTURE.md`.
3. **Privacy — 30 seconds:** point out that models and WASM are bundled and frames stay in memory.
4. **Live demo — 2 minutes:** calibrate, blink normally, look aside briefly, then safely demonstrate a longer closure while stationary.
5. **Demo mode — 45 seconds:** show the critical phone scenario without depending on camera conditions.
6. **Evaluation — 60 seconds:** present subject-independent results from your own dataset; do not present synthetic metrics as accuracy.
7. **Limitations — 30 seconds:** mention lighting, camera position, glasses, and lack of automotive certification.
8. **Roadmap — 45 seconds:** show uncertainty modeling, ONNX temporal fusion, and simulator testing.

## Questions teachers often ask

### “Where is the machine learning?”

MediaPipe and EfficientDet are neural perception models. The system converts their output into normalized features, then uses personalized temporal fusion. The `ml/` directory adds a subject-aware supervised fusion pipeline for a formal experiment.

### “Why not classify each image?”

A normal blink and a microsleep can look identical in one frame. Duration, frequency, and concurrent signals are essential.

### “How did you avoid data leakage?”

Split by participant, not by random frames. Adjacent frames from the same person are highly correlated and must not appear in both training and testing.

### “What is innovative?”

The project combines personal calibration, landmark geometry, object awareness, rolling PERCLOS, temporal persistence, concurrence escalation, explainability, and entirely local inference in a consumer interface.

### “What are the limitations?”

It is camera-dependent, not certified, not evaluated for every population or condition, and can miss events or raise false alarms. Honest limitations strengthen the project.

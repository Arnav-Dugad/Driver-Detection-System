# Architecture

## Runtime data flow

```text
Webcam frame (memory only)
  ├─ MediaPipe Face Landmarker → 478 normalized face + iris points
  │    ├─ EAR → blink and microsleep duration
  │    ├─ MAR → yawn state
  │    ├─ iris ratio → gaze deviation
  │    └─ nose / eyes / cheeks / chin → head yaw and pitch
  └─ EfficientDet-Lite0 → phone presence
             ↓
       Personal baseline
             ↓
  Rolling temporal state (up to 60 seconds)
             ↓
  Explainable risk fusion + concurrency bonus
             ↓
  UI state, local alerts, numeric event journal
```

## Why hybrid intelligence

A frame classifier alone tends to confuse normal blinks with microsleeps and momentary glances with distraction. Aegis separates perception from temporal reasoning:

- neural models answer “where are the face landmarks and objects?”;
- normalized geometry answers “what is the driver doing?”;
- temporal fusion answers “has it persisted, repeated, or occurred with another risk?”

The result is inspectable and easy to evaluate. It also avoids sending high-bandwidth video to a server.

## Core signals

### Eye Aspect Ratio

For six ordered points around each eye:

```text
EAR = (distance(p2,p6) + distance(p3,p5)) / (2 × distance(p1,p4))
```

Both eyes are averaged. The threshold is relative to the driver’s calibrated open-eye EAR.

### Mouth Aspect Ratio

MAR compares vertical inner-lip distance with mouth width. It is evaluated over time because speech and a single open-mouth frame are not reliable yawns.

### PERCLOS

PERCLOS is the proportion of recent samples classified as eye-closed. Aegis uses a rolling 60-second window in the live interface.

### Head and gaze deviation

Yaw and pitch use normalized face geometry. Iris position is measured relative to the eye corners. Both compare against the current driver’s median calibration baseline.

## Risk fusion

Every signal has a persistence ramp. For example, an eye closure below ordinary blink duration contributes almost nothing, while a sustained closure rises quickly. Phone presence has a high immediate contribution. Multiple concurrent signals add a synergy bonus.

The displayed score is exponentially smoothed to avoid flicker. Alert cooldowns prevent repeated sound on every frame.

## Performance

- face landmarking runs on new video frames;
- object detection runs less frequently because it is more expensive;
- React state is updated from compact numeric telemetry, never image buffers;
- the canvas overlay is cleared and redrawn without retaining frames;
- precision, balanced, and eco modes change phone-detection cadence.

## Security and privacy boundary

The hosted server delivers static code and model assets. The browser owns camera permission and inference. No route accepts images, video, facial geometry, or telemetry. Only an explicit download creates a report file, and that report contains numeric session data.

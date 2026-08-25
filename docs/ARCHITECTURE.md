# Architecture

## Runtime data flow

```text
Webcam frame (memory only)
  ├─ MediaPipe Face Landmarker
  │    ├─ 478 face + iris points
  │    │    ├─ EAR ──────────────┐
  │    │    ├─ MAR → yawn shape  │
  │    │    └─ iris ratio → gaze │
  │    ├─ 52 blendshapes         ├─→ fused eye closure (0..1)
  │    │    ├─ eyeBlink L/R ─────┘
  │    │    ├─ jawOpen
  │    │    └─ eyeLook in/out/up/down
  │    └─ 4x4 transformation matrix → true yaw / pitch / roll
  └─ EfficientDet-Lite0 → phone presence
             ↓
  Personal baseline (calibrated, then slowly adapted within ±15%)
             ↓
  Rolling temporal state (time-weighted, up to 60 seconds)
             ↓
  Signal confidence  ──┐
  Circadian context  ──┼─→ explainable risk fusion + concurrency bonus
  Persistence ramps  ──┘
             ↓
  UI state, counterfactual explanation, local alerts,
  numeric event journal, numeric replay buffer
```

## Why hybrid intelligence

A frame classifier alone tends to confuse normal blinks with microsleeps and momentary glances with distraction. The system separates perception from temporal reasoning:

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

PERCLOS follows the P80 definition: the proportion of **time** the eyes are at
least 80% closed across a rolling 60-second window. Each sample is weighted by
the wall-clock interval it represents rather than counted equally.

This matters more than it sounds. Counting samples makes the measure a function
of frame rate, and frame rate degrades hardest during long closures - exactly
the events PERCLOS exists to catch - so a sample-counted PERCLOS systematically
understates fatigue on slower devices. A stalled loop is capped so one sample
cannot claim credit for a long gap.

### Yawns

A mouth-opening threshold fires on ordinary conversation. Yawns are instead
identified by shape: a sustained opening held for at least 1.2 seconds with low
oscillation, optionally corroborated by eye narrowing. Speech reaches a similar
peak opening but reverses direction several times a second and never holds.

### Head and gaze deviation

Yaw and pitch use normalized face geometry. Iris position is measured relative to the eye corners. Both compare against the current driver’s median calibration baseline.

## Risk fusion

Every signal has a persistence ramp. For example, an eye closure below ordinary blink duration contributes almost nothing, while a sustained closure rises quickly. Phone presence has a high immediate contribution. Multiple concurrent signals add a synergy bonus.

The displayed score is exponentially smoothed to avoid flicker. Alert cooldowns prevent repeated sound on every frame.

### Confidence and context

Two further terms shape the final score:

- **Signal confidence** (0..1) is derived from brightness, contrast, frame rate,
  head angle, and left/right blink asymmetry. It attenuates risk across a floor
  rather than to zero, and the interface reports it directly. Claiming a
  confident score from a frame the system cannot actually read is worse than
  admitting the gap.
- **Circadian context** applies a bounded multiplier of at most +18% from time of
  day and time on task. It nudges existing evidence toward an earlier alert; it
  cannot manufacture one.

### Counterfactual explanation

Because the fusion keeps its per-signal contributions, the interface can state
what would actually change the outcome: "Risk would fall from 68 to 31 if your
gaze returned to the road." This is the same arithmetic the score is built from,
re-run with the largest contributor removed.

## Performance

- face landmarking runs on new video frames;
- object detection runs less frequently because it is more expensive;
- React state is updated from compact numeric telemetry, never image buffers;
- the canvas overlay is cleared and redrawn without retaining frames;
- precision, balanced, and eco modes change capture resolution, landmark frame
  pacing, phone-detection cadence, and overlay redraw rate; precision is the
  default, and eco exists so a phone can monitor a long drive without thermal
  throttling;
- React state is published at about 10 Hz while the detection maths still runs on
  every frame, so the interface never re-renders at camera frame rate;
- the screen wake lock is held for the session and re-acquired whenever the page
  returns to the foreground, because a sleeping phone stops the render loop.

## Security and privacy boundary

The hosted server delivers static code and model assets. The browser owns camera permission and inference. No route accepts images, video, facial geometry, or telemetry. Only an explicit download creates a report file, and that report contains numeric session data.

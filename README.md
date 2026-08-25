# Driver Drowsiness & Distraction Detection System

Driver Drowsiness & Distraction Detection System is a privacy-first college machine-learning project. It runs in the browser, uses the webcam only after permission is granted, and processes every frame on the device.

There are no API keys, paid services, subscriptions, accounts, or cloud inference. The included model files and runtime are served from this project, so camera frames never need to leave the browser.

> **Safety notice:** This system is a research prototype, not certified automotive safety equipment. It must not be relied on to prevent a crash. Never interact with a laptop or phone while driving. If you feel sleepy, pull over safely and rest.

## What is included

**Detection**

- 478-point face and iris landmarks, 52 blendshapes, and a 3D pose matrix from MediaPipe
- eye closure fused from geometric EAR and the learned blink blendshapes, so glasses and head angle degrade it far less
- true head yaw, pitch, and roll from a matrix decomposition, independent of how close you sit to the camera
- time-weighted PERCLOS on the standard P80 definition, so the measure does not drift with frame rate
- yawn detection by shape and duration rather than mouth opening alone, which rejects ordinary conversation
- per-frame confidence estimation: when the system cannot see well, it says so instead of guessing
- circadian and time-on-task context, bounded so it can never manufacture an alert
- a baseline that adapts to posture drift, clamped so it can never normalize a drowsy face
- EfficientDet-Lite0 phone detection
- automatic low-light and obstructed-camera warnings
- multi-signal temporal risk fusion with concurrency escalation
- five-second personal calibration

**Interface**

- counterfactual explanations: "Risk would fall from 68 to 31 if your gaze returned to the road"
- a dedicated full-screen **Drive Mode** for a phone on a mount, with a screen wake lock so the session survives
- context-aware voice alerts for eye closure, PERCLOS, yawning, gaze, head pose, phone use, driver visibility, escalating risk, and recovery
- natural multilingual speech in English, Hindi, Kannada, Marathi, Tamil, and Telugu
- vibration alerts for a noisy cabin or a muted phone
- one-tap five-minute snooze after a false alarm
- a session history and trends view, including your own time-of-day risk pattern
- camera-hidden privacy display that keeps detection active
- installable as an app, and fully functional offline once the models are cached
- keyboard shortcuts throughout, and screen-reader announcements for risk changes
- synthetic demo mode for presentations without a camera
- light and dark modes, fullscreen camera, and a layout built for phones as well as desktops

**Research**

- privacy-safe JSON reports including a 60-second numeric replay buffer
- an opt-in labeled-window recorder that exports CSV straight into the training pipeline
- an optional subject-aware training, evaluation, and browser-export pipeline under `ml/`
- deterministic unit tests for every signal-processing and risk decision

## Quick start — Windows

### 1. Install Node.js

Install Node.js 22 or newer from [nodejs.org](https://nodejs.org/). Use the LTS installer and keep the default options.

Open a new PowerShell window after installation and check it:

```powershell
node --version
npm --version
```

### 2. Download the project

If Git is installed:

```powershell
git clone https://github.com/Arnav-Dugad/Driver-Detection-System.git
cd Driver-Detection-System
```

If Git is not installed, open the GitHub page, choose **Code → Download ZIP**, extract it, and open PowerShell inside the extracted folder.

### 3. Install the free dependencies

```powershell
npm install
```

### 4. Start the system

```powershell
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in the latest Chrome or Edge.

### 5. Use it

1. Press **Start monitoring**.
2. Choose **Allow** when the browser asks for camera permission.
3. Look forward naturally for five seconds while the system calibrates.
4. Keep your full face visible with even front lighting.
5. Use **Demo** instead when presenting without a webcam.
6. Press **End session** when finished. A numeric summary is stored only in this browser.

On a phone, press the gauge icon to enter **Drive Mode**: one large risk number,
one large stop button, and a screen wake lock so monitoring does not stop when
the display would normally sleep.

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| `Space` | Start or end a session |
| `V` | Toggle Drive Mode |
| `C` | Recalibrate |
| `D` | Demo mode |
| `S` | Snooze alerts for five minutes |
| `M` | Mute sound and voice |
| `F` | Fullscreen |
| `H` | Session history |
| `Esc` | Close any panel |

For macOS and Linux, the commands are the same in Terminal. A more detailed beginner guide is in [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md).

### Choose a warning language

Open **Settings → Alerts → Language** and select English, Hindi, Kannada, Marathi, Tamil, or Telugu. Press **Test voice** to hear the exact voice exposed by the current browser.

Desktop browsers do not always expose every Windows language pack through the Web Speech API. When a matching installed voice is available, the app selects it. Otherwise it leaves the voice unset and supplies the selected language code so the browser can choose its own speech engine at playback. This preserves Chrome mobile's native behavior instead of forcing an unrelated fallback voice.

## How the intelligence works

The system uses a hybrid architecture. Two pre-trained neural models find facial geometry and objects; an explainable temporal engine then decides when an observation is persistent or important enough to alert.

1. **Face geometry:** MediaPipe detects 478 normalized face and iris landmarks.
2. **Signal extraction:** EAR, MAR, gaze ratio, yaw, and pitch are calculated from stable landmark groups.
3. **Personal calibration:** robust medians establish the driver’s normal open-eye, mouth, gaze, and head-pose values.
4. **Temporal memory:** closure duration, PERCLOS, repeated yawns, off-road gaze duration, head deviation, face loss, and phone presence are tracked over time.
5. **Risk fusion:** the engine combines severity, persistence, and concurrence. A normal blink remains low risk; several sustained signals escalate quickly.
6. **Intervention:** warning and critical states trigger rate-limited local sound and optional speech.

This design is intentionally explainable: the UI identifies the primary signal driving each risk score. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [MODEL_CARD.md](MODEL_CARD.md) for the complete technical and evaluation notes.

## Commands

```bash
npm run dev       # local development at http://localhost:3000
npm run lint      # code-quality and accessibility rules
npm run test      # signal, schema, interface, and rendered-output tests
npm run test:unit # fast pure-logic tests, no build required
npm run build     # production build
npm run verify    # lint + tests + production build
npm run build:pages        # static GitHub Pages build
npm run deploy:cloudflare  # build and deploy to your Cloudflare account
```

## Free hosting

Cloudflare Workers is the recommended host for the full application. A ready-to-run GitHub Pages workflow is also included as a static fallback. Follow the noob-friendly instructions in [docs/HOSTING_GUIDE.md](docs/HOSTING_GUIDE.md).

## Install it as an app

The hosted site is an installable PWA. On Android or desktop Chrome, choose
**Install** from the browser menu; on iOS Safari, use **Share -> Add to Home
Screen**. After one successful session the models are cached, so the app keeps
working with no network at all - which is what a fully on-device system should
do.

## Project structure

```text
app/
  GuardianDashboard.tsx    live vision loop, interface, alerts, session logic
  DriveMode.tsx            full-screen glanceable view for a mounted phone
  SessionHistory.tsx       local session journal and trends
  hooks.ts                 wake lock, page visibility, shortcuts, haptics
  sessionStore.ts          numeric-only local session journal
  globals.css              minimal responsive visual system
lib/detection/
  core.mjs                 tested geometry, pose, calibration, PERCLOS, risk fusion
  features.mjs             the 13-feature window ml/train_fusion.py consumes
  learned.mjs              optional in-browser scoring of a trained model
public/models/             bundled MediaPipe and EfficientDet model weights
public/wasm/               bundled MediaPipe browser runtime
public/sw.js               offline shell and model cache
ml/                        optional subject-aware training/evaluation pipeline
docs/                      setup, architecture, safety, and presentation guides
tests/                     deterministic unit, schema, and server-render tests
```

## Privacy

- Webcam frames are read in browser memory and are not uploaded.
- No video frame, photo, face template, or biometric identity is saved.
- Session history contains only numeric counts, events, duration, and risk state.
- Session history uses browser `localStorage`, is visible under **Your sessions**,
  and can be cleared from that screen or with browser site-data controls.
- Exported reports contain no image or video data. They include a 60-second
  buffer of numeric measurements so an alert can be reviewed after the fact.
- The offline cache stores only this project's own code and model files.

### About the research recorder

Settings contains an opt-in **Record labeled windows** switch. It is off by
default and does nothing unless you turn it on.

When enabled, it writes one row per second containing the thirteen numeric
aggregates listed in `ml/train_fusion.py` (means, minima, rates, and deviations),
plus a randomly generated local tag used only to keep one person's data inside
one train/test split. It captures **no image, video frame, face template,
landmark set, or identity**, and it transmits nothing: the CSV is a file you
download yourself. Discard captured windows at any time from the same panel.

## Closing the loop with the training pipeline

The app emits exactly the feature schema the Python pipeline reads, and a test
asserts the two lists match so they cannot silently drift apart.

```bash
# 1. Enable "Record labeled windows" in Settings, tag each window, export CSV.
# 2. Train with participant-safe splits:
python ml/train_fusion.py --data your-export.csv
# 3. Optionally export the trained model for in-browser scoring:
python ml/export_browser_model.py --model ml/artifacts/driver_fusion.joblib
```

The deterministic engine remains the default and the fallback. A learned model
can only ever nudge the score, never override obvious explainable evidence such
as a two-second eye closure.

## Accuracy and honest project claims

The live system is functional, but no responsible project should call itself universally accurate without a representative test set. Lighting, glasses, camera angle, facial anatomy, skin tone, disability, hardware, and road vibration can all affect results.

For a strong college submission:

1. collect consented, non-driving recordings or use licensed research datasets;
2. keep each person entirely inside one data split to avoid identity leakage;
3. report precision, recall, macro F1, false alarms per hour, and detection latency;
4. test day/night lighting, glasses, face coverings, different cameras, and head angles;
5. document failures instead of hiding them;
6. never test unsafe behavior on a public road.

The optional `ml/` pipeline enforces subject-aware splitting and produces reproducible metrics. Synthetic data is provided only to verify the pipeline—not as evidence of real-world accuracy.

## Troubleshooting

### The camera permission was blocked

Select the camera icon in the browser address bar, allow camera access, reload the page, and start again.

### The camera is black or busy

Close Zoom, Teams, OBS, and other camera applications. Then reload the system.

### Detection is slow

Open settings and select **Eco**, disable phone detection, close other tabs, and use Chrome or Edge with hardware acceleration enabled.

### Alerts happen too often

Recalibrate in your normal sitting position, improve front lighting, keep the camera near eye level, and lower **Sensitivity** slightly.

### `npm install` fails

Confirm Node 22+ with `node --version`, delete only the project’s `node_modules` folder, and run `npm install` again. Do not delete unrelated folders.

## Good next upgrades

- benchmark across multiple public datasets with subject-independent splits
- add steering-behavior or lane-position signals from recorded simulator data
- connect a free Bluetooth vibration device through Web Bluetooth
- create a low-light infrared camera profile
- validate every multilingual alert with native speakers and accessibility testing
- evaluate alert timing in a driving simulator with ethics approval
- add signed, tamper-evident fleet reports without storing faces

See [docs/IMPROVEMENT_ROADMAP.md](docs/IMPROVEMENT_ROADMAP.md) for staged improvements from college demo to research-grade prototype.

## License and model attribution

Project code is released under the [MIT License](LICENSE). MediaPipe Tasks, the face landmarker model, and EfficientDet assets retain their original licenses and notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

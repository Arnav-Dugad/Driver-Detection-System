# Driver Drowsiness & Distraction Detection System

Driver Drowsiness & Distraction Detection System is a privacy-first college machine-learning project. It runs in the browser, uses the webcam only after permission is granted, and processes every frame on the device.

There are no API keys, paid services, subscriptions, accounts, or cloud inference. The included model files and runtime are served from this project, so camera frames never need to leave the browser.

> **Safety notice:** This system is a research prototype, not certified automotive safety equipment. It must not be relied on to prevent a crash. Never interact with a laptop or phone while driving. If you feel sleepy, pull over safely and rest.

## What is included

- 478-point face and iris landmark detection with MediaPipe
- Eye Aspect Ratio (EAR) for blinks and prolonged closure
- rolling 60-second PERCLOS fatigue measurement
- Mouth Aspect Ratio (MAR) and temporal yawn detection
- personalized gaze and head-pose baselines
- clean, mirrored facial tracking points over the live preview
- EfficientDet-Lite0 phone detection
- automatic low-light and obstructed-camera warnings
- multi-signal temporal risk fusion with concurrency escalation
- five-second personal calibration
- context-aware voice alerts for eye closure, PERCLOS, yawning, gaze, head pose, phone use, driver visibility, escalating risk, and recovery
- natural multilingual speech in English, Hindi, Kannada, Marathi, Tamil, and Telugu
- camera-hidden privacy display that keeps detection active
- synthetic demo mode for presentations without a camera
- local session journal and privacy-safe JSON reports
- clean responsive interface, light and dark modes, fullscreen camera, keyboard focus, and mobile-safe controls
- deterministic unit tests for the signal-processing and risk logic
- an optional subject-aware training pipeline under `ml/`

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
npm run test      # signal-processing and rendered-output tests
npm run build     # production build
npm run verify    # lint + tests + production build
npm run build:pages        # static GitHub Pages build
npm run deploy:cloudflare  # build and deploy to your Cloudflare account
```

## Free hosting

Cloudflare Workers is the recommended host for the full application. A ready-to-run GitHub Pages workflow is also included as a static fallback. Follow the noob-friendly instructions in [docs/HOSTING_GUIDE.md](docs/HOSTING_GUIDE.md).

## Project structure

```text
app/
  GuardianDashboard.tsx    live vision loop, interface, alerts, session logic
  globals.css              minimal responsive visual system
lib/detection/
  core.mjs                 tested geometry, calibration, PERCLOS, risk fusion
public/models/             bundled MediaPipe and EfficientDet model weights
public/wasm/               bundled MediaPipe browser runtime
ml/                        optional subject-aware training/evaluation pipeline
docs/                      setup, architecture, safety, and presentation guides
tests/                     deterministic unit and server-render tests
```

## Privacy

- Webcam frames are read in browser memory and are not uploaded.
- No video frame, photo, face template, or biometric identity is saved.
- Session history contains only numeric counts, events, duration, and risk state.
- Session history uses browser `localStorage` and can be cleared with browser site-data controls.
- Exported reports contain no image or video data.

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

- add a consent-based feature recorder and manual labeling view
- benchmark across multiple public datasets with subject-independent splits
- distill the temporal fusion model to ONNX for learned browser inference
- add steering-behavior or lane-position signals from recorded simulator data
- connect a free Bluetooth vibration device through Web Bluetooth
- package the interface as an installable PWA
- create a low-light infrared camera profile
- validate every multilingual alert with native speakers and accessibility testing
- evaluate alert timing in a driving simulator with ethics approval
- add signed, tamper-evident fleet reports without storing faces

See [docs/IMPROVEMENT_ROADMAP.md](docs/IMPROVEMENT_ROADMAP.md) for staged improvements from college demo to research-grade prototype.

## License and model attribution

Project code is released under the [MIT License](LICENSE). MediaPipe Tasks, the face landmarker model, and EfficientDet assets retain their original licenses and notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

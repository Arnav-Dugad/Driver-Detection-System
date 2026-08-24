# Beginner setup guide

This guide assumes you have never run a programming project before.

## What you need

- a Windows, macOS, or Linux computer
- a webcam
- Chrome or Edge
- an internet connection for the first dependency installation
- Node.js 22 or newer
- about 1 GB of free disk space

No GPU, API key, paid account, or Python installation is required for the live app.

## Windows — every step

1. Visit [nodejs.org](https://nodejs.org/) and download the LTS installer.
2. Run the installer. Accept the defaults.
3. Restart PowerShell if it was open during installation.
4. Download this repository from GitHub using **Code → Download ZIP**.
5. Right-click the ZIP, choose **Extract All**, and open the extracted folder.
6. Click the folder address bar, type `powershell`, and press Enter.
7. Run `npm install` and wait until the prompt returns.
8. Run `npm run dev`.
9. Leave that PowerShell window open.
10. Open Chrome or Edge and visit `http://localhost:3000`.
11. Press **Start monitoring** and allow the camera.

To stop the development server, return to PowerShell and press `Ctrl+C` once.

## macOS or Linux

Open Terminal in the project folder, then run:

```bash
npm install
npm run dev
```

Open `http://localhost:3000` in Chrome or Edge. Stop the server with `Control+C`.

## Camera placement

- Put the lens close to eye level.
- Keep the face roughly 45–75 cm from the camera.
- Keep both eyes, the nose, mouth, and jawline visible.
- Use soft front lighting. Avoid strong backlight.
- Clean the lens.
- Do not hold or adjust the computer while driving.

## Calibration

Calibration lasts five seconds. Sit in the position you will use, look forward naturally, keep your eyes open, and relax your mouth. Do not intentionally widen your eyes or force a smile. Recalibrate after changing the camera, seat, glasses, lighting, or driver.

## Multilingual voice warnings

1. Open the settings panel.
2. Keep **Voice warnings** enabled.
3. Under **Language**, choose English, Hindi, Kannada, Marathi, Tamil, or Telugu.
4. Press **Test voice**.
5. Read the status below the selector. It shows either the matching browser voice or the active fallback.

On Windows, installing a display or speech language pack does not guarantee that Chrome or Edge exposes a matching voice to websites. The app can only use the voices returned by [`speechSynthesis.getVoices()`](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis/getVoices). If the selected voice is unavailable, warnings automatically use Hindi when possible, then English, so alerts do not fail silently. Restart the entire browser after installing voices and test again. Edge administrators should also make sure Microsoft Edge's [**Configure Online Text To Speech** policy](https://learn.microsoft.com/en-us/deployedge/microsoft-edge-policies/configureonlinetexttospeech) is not disabled.

The wording changes with the detected event. Separate messages cover calibration, long eye closure, high PERCLOS, a yawn, repeated yawning, off-road gaze, head turns, phone presence, loss of driver visibility, warning, critical danger, and attention recovery. Speech is rate-limited so it remains useful instead of becoming distracting.

## Presentation mode

Press **Demo** if a classroom computer has no webcam or camera permission. The demo uses synthetic signals and cycles through focused, gaze drift, fatigue, phone, and recovery states. It does not claim to be a real prediction.

## Verify before submission

Run:

```bash
npm run verify
```

A successful run finishes without errors. Keep screenshots of the command output and add your own real evaluation results to the report.

## Common fixes

### `node` or `npm` is not recognized

Node.js is missing or PowerShell has not been restarted. Install Node.js LTS, close PowerShell, and open it again.

### The page says it cannot access the camera

Use the camera icon beside the address bar to allow permission. Windows users should also check **Settings → Privacy & security → Camera** and allow desktop apps.

### The page loads but the detector is slow

Open the system settings, choose **Eco**, and switch off phone detection. Close video calls, games, and heavy browser tabs.

### Port 3000 is already in use

Stop the other development server or use the alternate local address printed by the terminal.

### You changed code and the page looks broken

Stop the server with `Ctrl+C`, run `npm install`, then run `npm run dev` again. Your files are not deleted by these commands.

# College presentation guide

The slide deck is `docs/presentation/Driver-Drowsiness-Distraction-Detection.pptx` (16 slides,
speaker notes on every slide). The full written report is the project report
document. Both read their numbers from `ml/reports/*.json`.

To rebuild the deck after re-training (pptxgenjs is not an app dependency):

```powershell
npm install --no-save pptxgenjs@3.12.0
node docs/presentation/build_deck.cjs
```

## A 13-minute talk

| Minutes | Slides | What to say |
| --- | --- | --- |
| 0-1 | 1-2 | The problem: fatigue and distraction are temporal and visible on the face; the EU now mandates warnings. |
| 1-3 | 3-5 | What was built, the three public datasets, and why the offline experiments run the app's own code. |
| 3-5 | 6 | The aspect-ratio bug real data exposed, and why the threshold had to be re-learned. |
| 5-7 | 7-8 | EyeNet: tiny CNN, held-out drivers, int8, infrared out-of-domain test; all closed-eye signals are tied head-on. |
| 7-10 | 9-12 | Drowsiness on UTA-RLDD against the published baseline and human judges; the calibration confound we caught; errors, calibration, ablation. |
| 10-11 | 13 | Blink and yawn events against annotations. |
| 11-12 | 14 | Live Model Lab demo: start the camera, blink, close your eyes for two seconds, look down. |
| 12-13 | 15-16 | Limitations, then the three summary numbers. |

## Demo checklist

1. Open the app, then the flask icon for the **Model Lab**. Allow the camera.
2. Look ahead for five seconds while it calibrates.
3. Blink normally: EyeNet's crops flip to "closed" briefly; rule risk stays low.
4. Close your eyes for two seconds: both EyeNet and rule risk rise.
5. Wait 30 seconds for the GRU window to fill and show its class bars.
6. Without a camera, scroll the results explorer: switch the metric and the confusion-matrix model.

## Questions examiners ask

### "Where is the machine learning?"

Two networks were trained here: EyeNet, a CNN for eye state, and a GRU over
30-second signal sequences for drowsiness. They are compared with logistic
regression, random forests, gradient boosting, a temporal CNN, and the
hand-designed rule engine, all on the same held-out people.

### "What is your accuracy, and on what data?"

Give the numbers on slides 7 and 9, and say every one is for people the model
never saw: UTA-RLDD's official five folds, DMD split by driver, and MRL Eye
used only for testing.

### "How did you avoid data leakage?"

Splits are by person, never by frame. Early stopping and temperature fitting use
held-out *training* people. Confidence intervals resample people.

### "Why not just classify each frame?"

A blink and a microsleep look identical in one frame. Duration, frequency, and
co-occurring signals are what separate them, which is why the GRU reads 30
seconds and the rule engine tracks persistence.

### "Couldn't the model just learn the lighting?"

Yes, which is why brightness and contrast are excluded from every model input:
RLDD participants filmed each state at a different time of day.

### "Why is low vigilance so hard?"

It is self-rated and sits between alert and drowsy; the original paper's models
and human judges confuse it too.

### "What are the limitations?"

Labels are self-reported sleepiness and eye state, not crashes; the public DMD
subset has 13 drivers; camera-only methods fail with sunglasses, a hand over the
mouth, or night without infrared. The system is not certified safety equipment.

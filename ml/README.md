# Machine-learning experiments

The live app runs without Python. This directory holds the experiments that
train, evaluate and export the two learned models it ships, all on public data
from real people. You do not need to record anyone yourself.

| Script | What it does | Output |
| --- | --- | --- |
| `extract_signals.py` | Streams public videos, runs the app's MediaPipe model, writes per-frame signals (and 32x32 eye crops for DMD) | `ml/data/signals/`, `ml/data/crops/` |
| `train_eye_cnn.py` | Trains EyeNet on DMD eye crops, 5-fold by driver; tests on MRL Eye; exports int8 weights | `public/models/eye-state-cnn.json`, `public/results/eye_cnn.json` |
| `evaluate_signals.py` | Scores EAR, blendshape, app fusion and the CNN frame by frame against DMD labels; replays the JS engine for blink and yawn events; tunes the closed-eye threshold by leave-one-driver-out CV | `public/results/signals.json` |
| `train_drowsiness.py` | Rule engine vs logistic regression, random forest, gradient boosting, TCN and GRU on UTA-RLDD's official 5 folds; exports the calibrated GRU | `public/models/drowsiness-gru.json`, `public/results/drowsiness.json` |
| `replay_rule_engine.mjs` | Runs the app's unchanged JavaScript tracker over recorded frames | called by the two scripts above |
| `make_parity_fixture.py` | Records real MediaPipe frames for the JS/Python parity test | `tests/fixtures/signal-parity.json` |
| `train_fusion.py` | The original consented-recording pipeline (Random Forest on 13 window features) | unchanged |

Figures for the report and slides are written to `ml/reports/figures/`, and
full metric dumps to `ml/reports/*.json`.

## Data

All three datasets are public and used for non-commercial coursework with
citation. They are streamed from the Hugging Face mirror
`nyvantran/drowsy_dataset` and `oliveirabruno01/litbench-mrl-eye`; nothing is
committed to Git (`ml/data/` is ignored).

| Dataset | People | Used for | Licence |
| --- | --- | --- | --- |
| UTA-RLDD (Ghoddoosian et al., CVPRW 2019) | 60 | Alert / low-vigilant / drowsy classification | Research use |
| DMD drowsiness subset (Ortega et al., ECCV-W 2020) | 13 in the mirror | Eye-state CNN, frame-level signal evaluation, blink and yawn events | CC BY-NC-ND 4.0 |
| MRL Eye public subset (Fusek, ISVC 2018) | MRL subjects, 4,000 images | Out-of-domain test of the CNN | Research use |

No dataset frame or eye crop is reproduced in figures, the app, or Git.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r ml\requirements.txt
```

## Reproduce everything

```powershell
cd ml
python extract_signals.py dmd  --workers 6
python extract_signals.py rldd --workers 12 --segments 6 --seconds 36 --fps 15
python train_eye_cnn.py
python evaluate_signals.py
python train_drowsiness.py
cd ..
npm run test:unit     # parity tests check the exported models against PyTorch
```

Extraction downloads about 45 GB of video segments and takes roughly an hour
on a 20-thread CPU with a 25 MB/s connection; the training scripts take
15-40 minutes each on CPU. `--quick` on either training script runs a short
smoke test.

## Rules the experiments follow

- **No person in both train and test.** RLDD uses its own five folds of 12
  people; DMD uses 5-fold GroupKFold by driver.
- **Intervals resample people, not frames.** Frames from one person are
  strongly correlated, so frame-level bootstraps would be falsely narrow.
- **The rule engine is the app's own code.** It is replayed from
  `lib/detection/tracker.mjs`, not re-implemented.
- **Python ports are tested against JavaScript** on recorded MediaPipe output
  (`tests/signal-parity.test.mjs`), and both exported models are tested
  against PyTorch reference outputs (`tests/learned-models.test.mjs`).
- **Lighting is never a model input.** RLDD participants recorded each state
  at different times of day; brightness would let a model recognise the room.
- **The shipped format is what gets scored.** The CNN is evaluated after
  BatchNorm folding and int8 quantisation; the GRU after temperature scaling.

## The original recorder pipeline

`train_fusion.py` still trains a Random Forest on CSVs written by the app's
opt-in recorder (**Settings → Research → Record labeled windows**), and
`export_browser_model.py` flattens it for `lib/detection/learned.mjs`. Use it
if you later collect consented recordings of your own;
`generate_demo_data.py` produces clearly synthetic data for a smoke test only.

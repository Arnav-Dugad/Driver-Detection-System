"use client";

import { ArrowLeft, Camera, CameraOff, FlaskConical, MoonStar, ShieldCheck, Sun } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";

import { classifySignals, extractFaceSignals } from "@/lib/detection/core.mjs";
import { cropEyes } from "@/lib/detection/eyeCrop.mjs";
import { predictClosed, prepareEyeModel, type PreparedEyeModel } from "@/lib/detection/eyeNet.mjs";
import {
  createSequenceBuffer,
  predictSequence,
  prepareSequenceModel,
  sequenceFrame,
  sequenceRisk,
  type PreparedSequenceModel,
} from "@/lib/detection/sequence.mjs";
import { createTracker } from "@/lib/detection/tracker.mjs";
import { BarChart, Confusion, Reliability, RocChart, Sparkline, type BarRow } from "./charts";

const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";
const CLASS_LABELS = ["Alert", "Low vig.", "Drowsy"];

type Interval = [number, number];
type DrowsinessModel = {
  name: string;
  clip_accuracy: number;
  clip_accuracy_ci95: Interval;
  clip_macro_f1: number;
  video_accuracy: number;
  video_accuracy_ci95: Interval;
  alert_vs_drowsy_auc: number;
  ece: number;
  confusion_video: number[][];
  confusion_clip: number[][];
  reliability?: { confidence: number; accuracy: number; count: number }[];
};
type DrowsinessReport = {
  dataset: { clips: number; videos: number; people: number; clip_seconds: number; hz: number };
  published_video_accuracy: Record<string, number>;
  models: Record<string, DrowsinessModel>;
  best_model: string;
  significance: Record<string, { p_value: number; a_only_correct: number; b_only_correct: number }>;
  ablation_gradient_boosting: Record<string, DrowsinessModel>;
  permutation_importance: Record<string, number>;
  gru: { parameters: number; export_kb: number; weights_kb?: number; temperatures: number[]; protocol?: string };
  calibration_protocols?: Record<
    "enrolled_all" | "enrolled",
    {
      clips: number;
      results: Record<string, DrowsinessModel>;
      session_same_clips: Record<string, DrowsinessModel>;
      mcnemar_vs_session: Record<string, { p_value: number }>;
    }
  >;
};
type Detector = { label: string; roc_auc: number; roc_auc_ci95: Interval; best_f1: number };
type SignalReport = {
  dataset: { frames: number; labelled_frames: number; drivers: number; sessions: number; closed_share: number };
  closed_eye_detectors: Record<string, Detector>;
  roc: Record<string, { fpr: number[]; tpr: number[] }>;
  pitch_robustness: Array<Record<string, number | string>>;
  aspect_ratio_fix: Record<string, { f1: number; recall: number; false_positive_rate: number }>;
  threshold_tuning?: { recommended_cut: number; old_cut: number; cv_f1: number };
  events: Record<string, { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number; false_alarms_per_hour: number }>;
};
type EyeReport = {
  dataset: { frames: number; drivers: number };
  models: Record<string, { accuracy: number; f1_closed: number; roc_auc: number; parameters: number }>;
  external_mrl: { accuracy?: number; roc_auc?: number; f1_closed?: number; images?: number; subjects?: number };
  final_model: { parameters: number; export_kb: number; weights_kb?: number; torch_cpu_latency_ms: number };
};

const MODEL_NAMES: Record<string, string> = {
  rule_engine: "Rule engine (app)",
  logistic_regression: "Logistic regression",
  random_forest: "Random forest",
  gradient_boosting: "Gradient boosting",
  tcn: "Temporal CNN",
  gru: "GRU",
};
const EYE_NAMES: Record<string, string> = {
  pixels_pca_logreg: "Pixels + PCA + LogReg",
  eyenet_l: "EyeNet-L, float",
  eyenet_s: "EyeNet-S, float",
  eyenet_s_int8: "EyeNet-S, int8 (shipped)",
};
const pct = (value: number | undefined, digits = 1) =>
  value === undefined || Number.isNaN(value) ? "–" : `${(value * 100).toFixed(digits)}%`;

function useLabData() {
  const [data, setData] = useState<{ drowsiness?: DrowsinessReport; signals?: SignalReport; eye?: EyeReport }>({});
  const [missing, setMissing] = useState(false);
  useEffect(() => {
    let cancelled = false;
    const load = <T,>(name: string): Promise<T | null> =>
      fetch(`${BASE}/results/${name}.json`).then((response) => (response.ok ? (response.json() as Promise<T>) : null));
    Promise.all([load<DrowsinessReport>("drowsiness"), load<SignalReport>("signals"), load<EyeReport>("eye_cnn")])
      .then(([drowsiness, signals, eye]) => {
        if (cancelled) return;
        setData({ drowsiness: drowsiness ?? undefined, signals: signals ?? undefined, eye: eye ?? undefined });
        setMissing(!drowsiness || !signals || !eye);
      })
      .catch(() => !cancelled && setMissing(true));
    return () => {
      cancelled = true;
    };
  }, []);
  return { ...data, missing };
}

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = window.localStorage.getItem("driver-detection-theme");
    } catch {
      saved = null;
    }
    const preferred =
      saved === "dark" || saved === "light"
        ? saved
        : window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
    // Deferred a frame, as in the dashboard, so hydration finishes first.
    const frame = window.requestAnimationFrame(() => {
      document.documentElement.dataset.theme = preferred;
      setTheme(preferred);
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);
  const toggle = useCallback(() => {
    setTheme((current) => {
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try {
        window.localStorage.setItem("driver-detection-theme", next);
      } catch {
        // Storage can be unavailable in private windows; the toggle still works.
      }
      return next;
    });
  }, []);
  return { theme, toggle };
}

/* ------------------------------------------------------------------------ */
/* Live shadow mode                                                          */
/* ------------------------------------------------------------------------ */

type LiveState = {
  phase: "idle" | "loading" | "calibrating" | "live" | "error";
  ruleRisk: number;
  eyeLeft: number | null;
  eyeRight: number | null;
  ruleClosure: number;
  gru: Record<string, number> | null;
  gruFill: number;
  history: { rule: number[]; gru: number[]; eye: number[] };
  message?: string;
};

const initialLive: LiveState = {
  phase: "idle",
  ruleRisk: 0,
  eyeLeft: null,
  eyeRight: null,
  ruleClosure: 0,
  gru: null,
  gruFill: 0,
  history: { rule: [], gru: [], eye: [] },
};

function LiveShadow() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const leftCanvas = useRef<HTMLCanvasElement>(null);
  const rightCanvas = useRef<HTMLCanvasElement>(null);
  const scratch = useRef<HTMLCanvasElement | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameRef = useRef<number>(0);
  const eyeModel = useRef<PreparedEyeModel | null>(null);
  const seqModel = useRef<PreparedSequenceModel | null>(null);
  const [state, setState] = useState<LiveState>(initialLive);

  const stop = useCallback(() => {
    cancelAnimationFrame(frameRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setState((current) => ({ ...current, phase: "idle" }));
  }, []);

  useEffect(() => stop, [stop]);

  const paint = (canvas: HTMLCanvasElement | null, pixels: Float32Array) => {
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const image = context.createImageData(32, 32);
    for (let i = 0; i < pixels.length; i += 1) {
      const v = Math.max(0, Math.min(255, pixels[i]));
      image.data.set([v, v, v, 255], i * 4);
    }
    context.putImageData(image, 0, 0);
  };

  const start = useCallback(async () => {
    setState({ ...initialLive, phase: "loading" });
    try {
      const [eyeJson, seqJson, vision, stream] = await Promise.all([
        fetch(`${BASE}/models/eye-state-cnn.json`).then((r) => r.json()),
        fetch(`${BASE}/models/drowsiness-gru.json`).then((r) => r.json()),
        import("@mediapipe/tasks-vision"),
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false }),
      ]);
      eyeModel.current = prepareEyeModel(eyeJson);
      seqModel.current = prepareSequenceModel(seqJson);
      streamRef.current = stream;
      if (!landmarkerRef.current) {
        const fileset = await vision.FilesetResolver.forVisionTasks(`${BASE}/wasm`);
        landmarkerRef.current = await vision.FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${BASE}/models/face_landmarker.task`, delegate: "CPU" },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.55,
          minFacePresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        });
      }
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();

      const tracker = createTracker();
      const buffer = createSequenceBuffer();
      scratch.current = scratch.current ?? document.createElement("canvas");
      let lastPublish = 0;
      let lastGru = 0;
      let gru: Record<string, number> | null = null;
      const history = { rule: [] as number[], gru: [] as number[], eye: [] as number[] };

      const loop = () => {
        const now = performance.now();
        const result = landmarkerRef.current!.detectForVideo(video, now);
        const landmarks = result.faceLandmarks?.[0];
        const width = video.videoWidth;
        const height = video.videoHeight;
        const signals = landmarks
          ? extractFaceSignals(landmarks, {
              matrix: result.facialTransformationMatrixes?.[0] ?? null,
              blendshapes: result.faceBlendshapes?.[0] ?? null,
              aspect: width / height,
            })
          : null;
        const step = tracker.step({ time: now, faceFound: Boolean(signals), signals });
        let eyes: { left: number | null; right: number | null } = { left: null, right: null };
        if (signals && landmarks && scratch.current) {
          const crops = cropEyes(video, landmarks, width, height, scratch.current);
          if (crops) {
            eyes = { left: predictClosed(eyeModel.current, crops.left), right: predictClosed(eyeModel.current, crops.right) };
            if (now - lastPublish > 120) {
              paint(leftCanvas.current, crops.left);
              paint(rightCanvas.current, crops.right);
            }
          }
        }
        if (tracker.phase === "live") {
          const closure = signals ? classifySignals(signals, tracker.baseline).closure : 0;
          buffer.push(now, sequenceFrame(signals, tracker.baseline, closure));
          if (buffer.full && now - lastGru > 1000) {
            lastGru = now;
            gru = predictSequence(seqModel.current, buffer.frames());
          }
        }
        if (now - lastPublish > 120) {
          lastPublish = now;
          const eyeMean = eyes.left !== null && eyes.right !== null ? (eyes.left + eyes.right) / 2 : 0;
          history.rule = [...history.rule.slice(-299), step.risk];
          history.gru = [...history.gru.slice(-299), sequenceRisk(gru) ?? 0];
          history.eye = [...history.eye.slice(-299), eyeMean * 100];
          setState({
            phase: tracker.phase === "live" ? "live" : "calibrating",
            ruleRisk: step.risk,
            ruleClosure: step.closure ?? 0,
            eyeLeft: eyes.left,
            eyeRight: eyes.right,
            gru,
            gruFill: Math.min(1, buffer.length / 450),
            history: { ...history },
          });
        }
        frameRef.current = requestAnimationFrame(loop);
      };
      frameRef.current = requestAnimationFrame(loop);
    } catch (error) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      setState({ ...initialLive, phase: "error", message: error instanceof Error ? error.message : "Camera unavailable" });
    }
  }, []);

  const running = state.phase === "loading" || state.phase === "calibrating" || state.phase === "live";
  const gruRisk = sequenceRisk(state.gru);
  const closedLabel = (p: number | null) => (p === null ? "–" : p > 0.5 ? "closed" : "open");

  return (
    <section className="lab-card live" aria-labelledby="live-title">
      <header className="lab-card-head">
        <div>
          <p className="eyebrow">Live shadow mode</p>
          <h2 id="live-title">Three models, one camera, side by side</h2>
          <p className="lede">
            The app&apos;s rule engine, the eye-state CNN and the drowsiness GRU score the same frames in parallel. Nothing
            is recorded or sent anywhere.
          </p>
        </div>
        <button className={running ? "lab-button is-secondary" : "lab-button"} onClick={running ? stop : start}>
          {running ? <CameraOff size={17} /> : <Camera size={17} />}
          {running ? "Stop" : "Start camera"}
        </button>
      </header>

      <div className="live-grid">
        <div className="live-video">
          <video ref={videoRef} muted playsInline aria-label="Your camera, mirrored" />
          {!running && (
            <div className="live-placeholder">
              <ShieldCheck size={22} />
              <span>{state.phase === "error" ? state.message : "Camera stays off until you start it"}</span>
            </div>
          )}
          {state.phase === "calibrating" && <div className="live-badge">Calibrating · look ahead</div>}
          {state.phase === "loading" && <div className="live-badge">Loading models</div>}
        </div>

        <div className="live-panels">
          <article className="meter">
            <div className="meter-head">
              <span>Rule engine risk</span>
              <strong>{state.ruleRisk}</strong>
            </div>
            <Sparkline values={state.history.rule} label="Rule engine risk, last 36 seconds" tone="muted" />
            <p>Persistence and fusion of hand-designed signals. Explainable, no training data.</p>
          </article>

          <article className="meter">
            <div className="meter-head">
              <span>EyeNet · P(closed)</span>
              <strong>{state.eyeLeft === null ? "–" : pct(((state.eyeLeft ?? 0) + (state.eyeRight ?? 0)) / 2, 0)}</strong>
            </div>
            <div className="eye-row">
              <figure>
                <canvas ref={leftCanvas} width={32} height={32} aria-label="Left eye crop" />
                <figcaption>{closedLabel(state.eyeLeft)}</figcaption>
              </figure>
              <figure>
                <canvas ref={rightCanvas} width={32} height={32} aria-label="Right eye crop" />
                <figcaption>{closedLabel(state.eyeRight)}</figcaption>
              </figure>
              <Sparkline values={state.history.eye} label="EyeNet closed-eye probability" />
            </div>
            <p>The 32×32 crops the CNN actually sees, aligned to your eye corners.</p>
          </article>

          <article className="meter">
            <div className="meter-head">
              <span>GRU · drowsiness</span>
              <strong>{gruRisk ?? "–"}</strong>
            </div>
            {state.gru ? (
              <div className="class-bars" role="img" aria-label="GRU class probabilities">
                {(["alert", "low_vigilant", "drowsy"] as const).map((name, index) => (
                  <div key={name} className="class-bar">
                    <span>{CLASS_LABELS[index]}</span>
                    <div className="track">
                      <div className={`fill c-${index}`} style={{ width: pct(state.gru?.[name] ?? 0) }} />
                    </div>
                    <em>{pct(state.gru?.[name] ?? 0, 0)}</em>
                  </div>
                ))}
              </div>
            ) : (
              <div className="fill-progress">
                <div style={{ width: pct(state.gruFill) }} />
                <span>{state.phase === "live" ? `Filling 30 s window · ${Math.round(state.gruFill * 30)} s` : "Waits for calibration"}</span>
              </div>
            )}
            <p>Reads 30 seconds of 14 signals at 15 Hz. Trained on 60 people from UTA-RLDD.</p>
          </article>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------------ */
/* Results                                                                   */
/* ------------------------------------------------------------------------ */

function DrowsinessSection({ report }: { report: DrowsinessReport }) {
  const [metric, setMetric] = useState<"video_accuracy" | "clip_accuracy" | "clip_macro_f1" | "alert_vs_drowsy_auc">("video_accuracy");
  const [selected, setSelected] = useState(report.best_model);
  const rows: BarRow[] = Object.keys(MODEL_NAMES)
    .filter((key) => report.models[key])
    .map((key) => {
      const model = report.models[key];
      const interval = metric === "video_accuracy" ? model.video_accuracy_ci95 : metric === "clip_accuracy" ? model.clip_accuracy_ci95 : undefined;
      return {
        key,
        label: MODEL_NAMES[key],
        value: model[metric],
        low: interval?.[0],
        high: interval?.[1],
        highlight: key === report.best_model,
      };
    })
    .sort((a, b) => b.value - a.value);
  const references =
    metric === "video_accuracy"
      ? Object.entries(report.published_video_accuracy)
          .filter(([label]) => !label.startsWith("LSTM"))
          .map(([label, value]) => ({ label: label.split(" (")[0], value, tone: label.startsWith("HM") ? ("warn" as const) : ("muted" as const) }))
          .concat([{ label: "Chance", value: 1 / 3, tone: "muted" as const }])
      : metric === "alert_vs_drowsy_auc"
        ? [{ label: "Chance", value: 0.5, tone: "muted" as const }]
        : [{ label: "Chance", value: 1 / 3, tone: "muted" as const }];
  const best = report.models[report.best_model];
  const test = Object.entries(report.significance).find(([key]) => key.endsWith("vs_rule_engine"));
  const importance = Object.entries(report.permutation_importance).slice(0, 8);
  const maxImportance = Math.max(...importance.map(([, value]) => value), 0.001);
  const gru = report.models.gru;
  const raw = report.models.gru_uncalibrated;

  return (
    <section className="lab-card" aria-labelledby="rldd-title">
      <header className="lab-card-head">
        <div>
          <p className="eyebrow">Drowsiness · UTA-RLDD</p>
          <h2 id="rldd-title">
            {MODEL_NAMES[report.best_model]} reaches {pct(best.video_accuracy)} three-class video accuracy on people it never saw
          </h2>
          <p className="lede">
            {report.dataset.people} people, {report.dataset.videos} videos, {report.dataset.clips} clips of {report.dataset.clip_seconds} s.
            Official five-fold subject split; whiskers are 95% intervals from resampling people.
          </p>
        </div>
        <div className="segmented" role="tablist" aria-label="Metric">
          {(
            [
              ["video_accuracy", "Video acc."],
              ["clip_accuracy", "Clip acc."],
              ["clip_macro_f1", "Macro F1"],
              ["alert_vs_drowsy_auc", "Alert vs drowsy AUC"],
            ] as const
          ).map(([key, label]) => (
            <button key={key} role="tab" aria-selected={metric === key} onClick={() => setMetric(key)}>
              {label}
            </button>
          ))}
        </div>
      </header>
      <BarChart rows={rows} references={references} axisLabel="Held-out performance by model" />
      {test && (
        <p className="footnote">
          Exact McNemar test, {MODEL_NAMES[test[0].split("_vs_")[0]] ?? test[0]} vs rule engine on paired clips:
          {" "}p = {test[1].p_value < 0.001 ? "< 0.001" : test[1].p_value.toFixed(3)} ({test[1].a_only_correct} clips only the
          model got right, {test[1].b_only_correct} only the rule engine).
        </p>
      )}

      {report.calibration_protocols && <CalibrationProtocols protocols={report.calibration_protocols} />}

      <div className="lab-grid three">
        <div className="panel">
          <div className="panel-head">
            <h3>Where errors fall</h3>
            <select value={selected} onChange={(event) => setSelected(event.target.value)} aria-label="Model for confusion matrix">
              {Object.keys(MODEL_NAMES)
                .filter((key) => report.models[key])
                .map((key) => (
                  <option key={key} value={key}>
                    {MODEL_NAMES[key]}
                  </option>
                ))}
            </select>
          </div>
          <Confusion matrix={report.models[selected].confusion_video} labels={CLASS_LABELS} />
          <p className="caption">Per video. Low vigilance is the hard middle class, as in the original paper.</p>
        </div>
        <div className="panel">
          <h3>Calibrated confidence</h3>
          {raw && gru && (
            <Reliability
              series={[
                { key: "raw", label: `GRU raw · ECE ${raw.ece.toFixed(3)}`, points: reliabilityPoints(raw) },
                { key: "cal", label: `GRU + temperature · ECE ${gru.ece.toFixed(3)}`, points: reliabilityPoints(gru), highlight: true },
              ]}
            />
          )}
          <p className="caption">
            Temperature scaling fitted on held-out training people. ECE {raw ? raw.ece.toFixed(3) : "–"} →{" "}
            <strong>{gru ? gru.ece.toFixed(3) : "–"}</strong>.
          </p>
        </div>
        <div className="panel">
          <h3>What the boosted trees rely on</h3>
          <ol className="importance">
            {importance.map(([name, value]) => (
              <li key={name}>
                <span>{name.replaceAll("_", " ")}</span>
                <div className="track">
                  <div style={{ width: `${Math.max(2, (value / maxImportance) * 100)}%` }} />
                </div>
              </li>
            ))}
          </ol>
          <p className="caption">Permutation importance: drop in macro F1 when one feature is shuffled.</p>
        </div>
      </div>
    </section>
  );
}

const reliabilityPoints = (model: DrowsinessModel) => model.reliability ?? [];

function CalibrationProtocols({ protocols }: { protocols: NonNullable<DrowsinessReport["calibration_protocols"]> }) {
  const { enrolled_all: confounded, enrolled: controlled } = protocols;
  const keys = ["rule_engine", "logistic_regression", "gradient_boosting", "gru"].filter((key) => controlled.results[key]);
  return (
    <div className="panel protocol-compare">
      <div className="protocol-compare-text">
        <h3>A tempting result that turned out to be a confound</h3>
        <p className="caption">
          Hypothesis: calibrating on a face that is already tired hides tiredness, so one calibration taken while alert should help.
          Carrying over the whole alert baseline looked like a large win, but RLDD&apos;s states are separate recordings and the camera
          often moved, so a carried-over head-pose baseline partly recognises the recording. Carrying over only the eye and mouth
          reference removes most of the gain. The app therefore keeps per-session calibration.
        </p>
      </div>
      <table className="lab-table">
        <thead>
          <tr>
            <th scope="col">Video accuracy</th>
            <th scope="col">Per session</th>
            <th scope="col">Whole baseline carried (confounded)</th>
            <th scope="col">Eyes + mouth carried (controlled)</th>
          </tr>
        </thead>
        <tbody>
          {keys.map((key) => (
            <tr key={key}>
              <th scope="row">{MODEL_NAMES[key]}</th>
              <td>{pct(controlled.session_same_clips[key].video_accuracy)}</td>
              <td className="is-muted">{pct(confounded.results[key].video_accuracy)}</td>
              <td>{pct(controlled.results[key].video_accuracy)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SignalsSection({ signals, eye }: { signals: SignalReport; eye: EyeReport }) {
  const keys = ["score_ear", "score_blendshape", "score_fused", "score_cnn", "score_stacked"].filter((key) => signals.roc[key]);
  const curves = keys.map((key) => ({
    key,
    label: signals.closed_eye_detectors[key].label,
    fpr: signals.roc[key].fpr,
    tpr: signals.roc[key].tpr,
    auc: signals.closed_eye_detectors[key].roc_auc,
    highlight: key === "score_cnn",
  }));
  const fix = signals.aspect_ratio_fix;
  const tuning = signals.threshold_tuning;
  return (
    <section className="lab-card" aria-labelledby="dmd-title">
      <header className="lab-card-head">
        <div>
          <p className="eyebrow">Eye state · DMD and MRL Eye</p>
          <h2 id="dmd-title">
            The eye CNN matches the best signal on held-out drivers with {eye.final_model.weights_kb ?? 15} KB of weights
          </h2>
          <p className="lede">
            {signals.dataset.labelled_frames.toLocaleString("en-US")} human-labelled frames from {signals.dataset.drivers} real drivers.
            Every CNN score is out-of-fold: the network never saw that driver.
          </p>
        </div>
      </header>
      <div className="lab-grid two">
        <div className="panel">
          <h3>Closed-eye detection, ROC</h3>
          <RocChart curves={curves} title="ROC curves for closed-eye detectors on DMD" />
        </div>
        <div className="panel">
          <h3>Model size against accuracy</h3>
          <table className="lab-table">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Params</th>
                <th scope="col">Acc.</th>
                <th scope="col">AUC</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(EYE_NAMES)
                .filter(([key]) => eye.models[key])
                .map(([key, label]) => (
                  <tr key={key} className={key === "eyenet_s_int8" ? "is-accent" : ""}>
                    <th scope="row">{label}</th>
                    <td>{eye.models[key].parameters.toLocaleString("en-US")}</td>
                    <td>{pct(eye.models[key].accuracy)}</td>
                    <td>{eye.models[key].roc_auc.toFixed(3)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          <div className="stat-row">
            <div>
              <span>Out-of-domain, MRL infrared</span>
              <strong>{pct(eye.external_mrl.accuracy)}</strong>
              <em>{eye.external_mrl.images ?? "–"} images, unseen camera</em>
            </div>
            <div>
              <span>Shipped weights</span>
              <strong>{eye.final_model.weights_kb ?? Math.round(eye.final_model.parameters / 1024)} KB</strong>
              <em>int8, plain JavaScript</em>
            </div>
          </div>
          {tuning && (
            <div className="finding">
              <h4>Finding: a hidden aspect-ratio bug</h4>
              <p>
                MediaPipe normalises x and y separately, so EAR was stretched by the camera&apos;s aspect ratio. Fixing it alone
                dropped closed-eye F1 from {fix.before_fix.f1.toFixed(3)} to {fix.corrected.f1.toFixed(3)}, because the old
                threshold had silently compensated. Re-tuning the cut by leave-one-driver-out CV ({tuning.old_cut.toFixed(2)} →{" "}
                {tuning.recommended_cut.toFixed(2)}) gives <strong>{tuning.cv_f1.toFixed(3)}</strong> on unseen drivers, now on
                any camera shape.
              </p>
            </div>
          )}
        </div>
      </div>
      <div className="lab-grid two">
        <div className="panel">
          <h3>Event detection by the rule engine</h3>
          <table className="lab-table">
            <thead>
              <tr>
                <th scope="col">Event</th>
                <th scope="col">Precision</th>
                <th scope="col">Recall</th>
                <th scope="col">False alarms / h</th>
              </tr>
            </thead>
            <tbody>
              {(["blink", "yawn"] as const).map((kind) => (
                <tr key={kind}>
                  <th scope="row">{kind === "blink" ? "Blinks" : "Yawns"}</th>
                  <td>{pct(signals.events[kind].precision)}</td>
                  <td>{pct(signals.events[kind].recall)}</td>
                  <td>{signals.events[kind].false_alarms_per_hour.toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">
            The unchanged JavaScript engine replayed over every DMD session and matched against annotated intervals. Yawn recall is
            limited by yawns behind a hand, which hide the mouth.
          </p>
        </div>
        <div className="panel">
          <h3>Robustness as the head pitches down</h3>
          <table className="lab-table">
            <thead>
              <tr>
                <th scope="col">Pitch</th>
                {keys.map((key) => (
                  <th scope="col" key={key}>
                    {signals.closed_eye_detectors[key].label.split(" (")[0]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {signals.pitch_robustness.map((row) => (
                <tr key={String(row.band)}>
                  <th scope="row">{String(row.band)}</th>
                  {keys.map((key) => (
                    <td key={key}>{Number(row[key]).toFixed(3)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">ROC-AUC per band of downward head pitch relative to calibration.</p>
        </div>
      </div>
    </section>
  );
}

function Protocol() {
  const items = [
    ["Subject-independent", "Every score is for a person absent from training: RLDD's official 5 folds, DMD 5-fold by driver."],
    ["Intervals over people", "95% intervals resample participants, not frames, because one person's frames are correlated."],
    ["Confound removed", "Brightness never enters a model; RLDD states were filmed at different times of day."],
    ["Same code as the app", "The rule engine is the shipped JavaScript replayed offline; Python ports pass parity tests."],
    ["Calibrated, then shipped", "The GRU's confidence is temperature-scaled; the CNN is scored after int8 quantisation."],
    ["Honest baselines", "Published HM-LSTM and human judges are drawn on the same chart as our models."],
  ];
  return (
    <section className="lab-card protocol" aria-labelledby="protocol-title">
      <p className="eyebrow">Method</p>
      <h2 id="protocol-title">How these numbers were earned</h2>
      <div className="protocol-grid">
        {items.map(([title, body]) => (
          <div key={title}>
            <h3>{title}</h3>
            <p>{body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function ModelLab() {
  const { drowsiness, signals, eye, missing } = useLabData();
  const { theme, toggle } = useTheme();
  const headline = useMemo(() => {
    if (!drowsiness || !eye) return null;
    const best = drowsiness.models[drowsiness.best_model];
    return [
      { label: "Eye-state ROC-AUC, unseen drivers", value: eye.models.eyenet_s_int8?.roc_auc.toFixed(3) ?? "–", note: "EyeNet-S int8, DMD 5-fold" },
      { label: "Drowsiness video accuracy", value: pct(best.video_accuracy), note: `${MODEL_NAMES[drowsiness.best_model]} · published HM-LSTM 65.2%` },
      { label: "Alert vs drowsy ROC-AUC", value: best.alert_vs_drowsy_auc.toFixed(3), note: "same held-out people" },
      {
        label: "Learned weights shipped",
        value: `${Math.round((eye.final_model.weights_kb ?? eye.final_model.parameters / 1024) + (drowsiness.gru.weights_kb ?? 0))} KB`,
        note: "int8 CNN + GRU, plain JavaScript",
      },
    ];
  }, [drowsiness, eye]);

  return (
    <main className="lab-shell">
      <header className="lab-topbar">
        <a className="lab-back" href={`${BASE}/`}>
          <ArrowLeft size={17} /> Monitor
        </a>
        <span className="lab-title">
          <FlaskConical size={17} /> Model Lab
        </span>
        <button className="icon-button" onClick={toggle} aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}>
          {theme === "dark" ? <Sun size={18} /> : <MoonStar size={18} />}
        </button>
      </header>

      <section className="lab-hero">
        <p className="eyebrow">Evaluated on three public datasets</p>
        <h1>What the models learned, measured on people they never saw.</h1>
        <p>
          Two neural networks trained for this project run beside the explainable rule engine. Every figure below is loaded from the
          training artefacts, so the page cannot disagree with the experiments.
        </p>
        {headline && (
          <dl className="kpis">
            {headline.map((item) => (
              <div key={item.label}>
                <dt>{item.label}</dt>
                <dd>{item.value}</dd>
                <span>{item.note}</span>
              </div>
            ))}
          </dl>
        )}
      </section>

      <LiveShadow />
      {missing && (
        <p className="lab-missing">Experiment results are not in this build. Run the scripts in ml/ to regenerate public/results.</p>
      )}
      {drowsiness && <DrowsinessSection report={drowsiness} />}
      {signals && eye && <SignalsSection signals={signals} eye={eye} />}
      <Protocol />
      <footer className="lab-footer">
        Datasets: UTA-RLDD (Ghoddoosian et al., CVPRW 2019), DMD (Ortega et al., ECCV-W 2020, CC BY-NC-ND 4.0), MRL Eye (Fusek,
        ISVC 2018). Research prototype, not a certified safety system.
      </footer>
    </main>
  );
}

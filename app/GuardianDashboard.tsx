"use client";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  Camera,
  CameraOff,
  Check,
  ChevronRight,
  CircleHelp,
  Crosshair,
  Download,
  Eye,
  Gauge,
  Info,
  LockKeyhole,
  Maximize,
  Minimize,
  MoonStar,
  Play,
  Radio,
  RefreshCw,
  Route,
  ScanFace,
  Settings2,
  ShieldCheck,
  Siren,
  Smartphone,
  Sparkles,
  Square,
  Volume2,
  X,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  FaceLandmarker,
  NormalizedLandmark,
  ObjectDetector,
} from "@mediapipe/tasks-vision";
import {
  DEFAULT_BASELINE,
  buildCalibration,
  calculatePerclos,
  calculateRisk,
  classifySignals,
  clamp,
  extractFaceSignals,
  formatDuration,
} from "@/lib/detection/core.mjs";

type Phase = "idle" | "loading" | "calibrating" | "live" | "demo" | "paused";
type RiskState = "focused" | "caution" | "warning" | "danger";

type Baseline = {
  ear: number;
  mar: number;
  yaw: number;
  pitch: number;
  gaze: number;
};

type Telemetry = {
  ear: number;
  mar: number;
  perclos: number;
  yaw: number;
  pitch: number;
  gaze: number;
  risk: number;
  state: RiskState;
  primary: string;
  fps: number;
  faceFound: boolean;
  phoneVisible: boolean;
  eyesClosed: boolean;
  yawning: boolean;
  headAway: boolean;
  gazeAway: boolean;
};

type HistoryPoint = {
  risk: number;
  eyes: number;
  distraction: number;
};

type EventItem = {
  id: string;
  type: "attention" | "drowsiness" | "device" | "system" | "recovery";
  title: string;
  detail: string;
  time: string;
  severity: "low" | "medium" | "high";
};

type Settings = {
  sound: boolean;
  voice: boolean;
  phoneDetection: boolean;
  privacyMode: boolean;
  sensitivity: number;
  performance: "balanced" | "precision" | "eco";
};

type SessionStats = {
  blinks: number;
  yawns: number;
  distractions: number;
  phoneEvents: number;
  alerts: number;
  maxRisk: number;
};

const initialTelemetry: Telemetry = {
  ear: DEFAULT_BASELINE.ear,
  mar: DEFAULT_BASELINE.mar,
  perclos: 0,
  yaw: 0,
  pitch: DEFAULT_BASELINE.pitch,
  gaze: 0.5,
  risk: 0,
  state: "focused",
  primary: "eyes",
  fps: 0,
  faceFound: false,
  phoneVisible: false,
  eyesClosed: false,
  yawning: false,
  headAway: false,
  gazeAway: false,
};

const initialStats: SessionStats = {
  blinks: 0,
  yawns: 0,
  distractions: 0,
  phoneEvents: 0,
  alerts: 0,
  maxRisk: 0,
};

const STATE_COPY: Record<RiskState, { label: string; message: string; action: string }> = {
  focused: {
    label: "Focused",
    message: "Attention pattern is stable",
    action: "Continue monitoring",
  },
  caution: {
    label: "Drift detected",
    message: "Early fatigue or attention drift",
    action: "Recenter on the road",
  },
  warning: {
    label: "Warning",
    message: "Sustained impairment pattern",
    action: "Prepare to stop safely",
  },
  danger: {
    label: "Critical",
    message: "High-risk pattern detected",
    action: "Pull over when safe",
  },
};

const PRIMARY_COPY: Record<string, string> = {
  eyes: "prolonged eye closure",
  perclos: "elevated eye closure rate",
  yawn: "repeated yawning",
  head: "head orientation",
  gaze: "off-road gaze",
  phone: "mobile phone presence",
  missing: "driver not visible",
};

const emptyHistory = Array.from({ length: 52 }, () => ({
  risk: 4,
  eyes: 94,
  distraction: 4,
}));

function RiskOrb({ score, state }: { score: number; state: RiskState }) {
  return (
    <div className="risk-orb-shell" aria-label={`Attention risk ${score} out of 100`}>
      <div className="risk-orb-glow" />
      <div className="risk-orb-ticks" />
      <div className="risk-orb" style={{ "--risk": `${score * 3.6}deg` } as React.CSSProperties}>
        <div className="risk-orb-core">
          <span className="risk-kicker">RISK INDEX</span>
          <strong>{String(score).padStart(2, "0")}</strong>
          <span className={`risk-state state-${state}`}>{STATE_COPY[state].label}</span>
        </div>
      </div>
      <div className="orbital orbital-one" />
      <div className="orbital orbital-two" />
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  detail,
  level = "normal",
  progress,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
  level?: "normal" | "watch" | "alert";
  progress: number;
}) {
  return (
    <article className={`metric-card metric-${level}`}>
      <div className="metric-card-top">
        <span className="metric-icon">{icon}</span>
        <span className="metric-live-dot" />
      </div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
      <div className="metric-track" aria-hidden="true">
        <span style={{ width: `${clamp(progress, 0, 100)}%` }} />
      </div>
    </article>
  );
}

function FocusTimeline({ history }: { history: HistoryPoint[] }) {
  return (
    <div className="timeline-chart" aria-label="Recent attention risk timeline">
      <div className="timeline-grid" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <div className="timeline-bars">
        {history.map((point, index) => (
          <div className="timeline-column" key={`${index}-${point.risk}`}>
            <i className="bar-risk" style={{ height: `${Math.max(3, point.risk)}%` }} />
            <i className="bar-eye" style={{ height: `${Math.max(2, 100 - point.eyes)}%` }} />
            <i
              className="bar-distraction"
              style={{ height: `${Math.max(2, point.distraction)}%` }}
            />
          </div>
        ))}
      </div>
      <div className="timeline-axis">
        <span>60 sec ago</span>
        <span>now</span>
      </div>
    </div>
  );
}

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <span className="brand-mark-core" />
      <span className="brand-mark-ring brand-ring-one" />
      <span className="brand-mark-ring brand-ring-two" />
    </span>
  );
}

export default function GuardianDashboard() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [telemetry, setTelemetry] = useState<Telemetry>(initialTelemetry);
  const [history, setHistory] = useState<HistoryPoint[]>(emptyHistory);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [stats, setStats] = useState<SessionStats>(initialStats);
  const [baseline, setBaseline] = useState<Baseline>(DEFAULT_BASELINE);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [loadMessage, setLoadMessage] = useState("Preparing private vision engine");
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [sessionSeconds, setSessionSeconds] = useState(0);
  const [cameraLabel, setCameraLabel] = useState("Integrated camera");
  const [modelReady, setModelReady] = useState(false);
  const [settings, setSettings] = useState<Settings>({
    sound: true,
    voice: true,
    phoneDetection: true,
    privacyMode: false,
    sensitivity: 0.62,
    performance: "balanced",
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const objectDetectorRef = useRef<ObjectDetector | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const startedAtRef = useRef(0);
  const calibrationStartedRef = useRef(0);
  const calibrationSamplesRef = useRef<Array<Record<string, number>>>([]);
  const eyeSamplesRef = useRef<Array<{ time: number; closed: boolean }>>([]);
  const recentYawnTimesRef = useRef<number[]>([]);
  const eyeClosedAtRef = useRef(0);
  const headAwayAtRef = useRef(0);
  const gazeAwayAtRef = useRef(0);
  const faceMissingAtRef = useRef(0);
  const lastObjectDetectionRef = useRef(0);
  const phoneVisibleRef = useRef(false);
  const riskSmoothedRef = useRef(0);
  const lastAlertAtRef = useRef(0);
  const lastHistoryAtRef = useRef(0);
  const eventFlagsRef = useRef<Record<string, boolean>>({});
  const statsRef = useRef<SessionStats>(initialStats);
  const settingsRef = useRef(settings);
  const phaseRef = useRef<Phase>(phase);
  const frameCounterRef = useRef({ count: 0, started: 0, fps: 0 });

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    statsRef.current = stats;
  }, [stats]);

  const active = phase === "live" || phase === "calibrating" || phase === "demo";
  const statusCopy = STATE_COPY[telemetry.state];

  const updateStats = useCallback((patch: Partial<SessionStats>) => {
    const next = { ...statsRef.current, ...patch };
    statsRef.current = next;
    setStats(next);
  }, []);

  const pushEvent = useCallback(
    (
      type: EventItem["type"],
      title: string,
      detail: string,
      severity: EventItem["severity"] = "medium",
    ) => {
      const elapsed = startedAtRef.current
        ? (performance.now() - startedAtRef.current) / 1000
        : 0;
      const item: EventItem = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        type,
        title,
        detail,
        time: elapsed ? formatDuration(elapsed) : "NOW",
        severity,
      };
      setEvents((previous) => [item, ...previous].slice(0, 12));
    },
    [],
  );

  const unlockAudio = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    if (audioContextRef.current.state === "suspended") {
      void audioContextRef.current.resume();
    }
  }, []);

  const soundAlert = useCallback(
    (danger = false, force = false) => {
      const currentSettings = settingsRef.current;
      if ((!currentSettings.sound && !force) || typeof window === "undefined") return;
      unlockAudio();
      const context = audioContextRef.current;
      if (!context) return;
      const start = context.currentTime;
      const pulses = danger ? 3 : 2;
      for (let index = 0; index < pulses; index += 1) {
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        oscillator.type = danger ? "sawtooth" : "sine";
        oscillator.frequency.setValueAtTime(danger ? 860 : 620, start + index * 0.22);
        oscillator.frequency.exponentialRampToValueAtTime(
          danger ? 540 : 440,
          start + index * 0.22 + 0.15,
        );
        gain.gain.setValueAtTime(0.0001, start + index * 0.22);
        gain.gain.exponentialRampToValueAtTime(0.18, start + index * 0.22 + 0.015);
        gain.gain.exponentialRampToValueAtTime(0.0001, start + index * 0.22 + 0.17);
        oscillator.connect(gain).connect(context.destination);
        oscillator.start(start + index * 0.22);
        oscillator.stop(start + index * 0.22 + 0.18);
      }
      if ((currentSettings.voice || force) && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(
          danger ? "Critical alert. Pull over when safe." : "Attention warning. Please focus on the road.",
        );
        utterance.rate = 0.92;
        utterance.pitch = 0.92;
        utterance.volume = 0.9;
        window.setTimeout(() => window.speechSynthesis.speak(utterance), danger ? 720 : 520);
      }
    },
    [unlockAudio],
  );

  const prepareModels = useCallback(async () => {
    if (faceLandmarkerRef.current) return;
    setLoadMessage("Loading 478-point face geometry");
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks("/wasm");
    const faceOptions = {
      baseOptions: {
        modelAssetPath: "/models/face_landmarker.task",
        delegate: "GPU" as const,
      },
      runningMode: "VIDEO" as const,
      numFaces: 1,
      minFaceDetectionConfidence: 0.55,
      minFacePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    };
    try {
      faceLandmarkerRef.current = await vision.FaceLandmarker.createFromOptions(
        fileset,
        faceOptions,
      );
    } catch {
      setLoadMessage("Optimizing for this device");
      faceLandmarkerRef.current = await vision.FaceLandmarker.createFromOptions(fileset, {
        ...faceOptions,
        baseOptions: {
          modelAssetPath: "/models/face_landmarker.task",
          delegate: "CPU",
        },
      });
    }

    if (settingsRef.current.phoneDetection) {
      setLoadMessage("Adding mobile-device awareness");
      try {
        objectDetectorRef.current = await vision.ObjectDetector.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: "/models/efficientdet_lite0.tflite",
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          scoreThreshold: 0.42,
          maxResults: 4,
          categoryAllowlist: ["cell phone"],
        });
      } catch (phoneModelError) {
        console.warn("Phone detector unavailable; core driver monitoring remains active.", phoneModelError);
        setSettings((current) => ({ ...current, phoneDetection: false }));
      }
    }
    setModelReady(true);
  }, []);

  const resetRuntime = useCallback(() => {
    calibrationSamplesRef.current = [];
    eyeSamplesRef.current = [];
    recentYawnTimesRef.current = [];
    eyeClosedAtRef.current = 0;
    headAwayAtRef.current = 0;
    gazeAwayAtRef.current = 0;
    faceMissingAtRef.current = 0;
    lastObjectDetectionRef.current = 0;
    phoneVisibleRef.current = false;
    riskSmoothedRef.current = 0;
    lastHistoryAtRef.current = 0;
    lastAlertAtRef.current = 0;
    eventFlagsRef.current = {};
    frameCounterRef.current = { count: 0, started: performance.now(), fps: 0 };
    statsRef.current = initialStats;
    setStats(initialStats);
    setHistory(emptyHistory);
    setEvents([]);
    setTelemetry(initialTelemetry);
    setSessionSeconds(0);
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    resetRuntime();
    unlockAudio();
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser cannot access a camera. Use the latest Chrome or Edge, or open Demo mode.");
      return;
    }
    setPhase("loading");
    try {
      const [stream] = await Promise.all([
        navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 30 },
          },
          audio: false,
        }),
        prepareModels(),
      ]);
      streamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      setCameraLabel(track?.label || "Driver camera");
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      startedAtRef.current = performance.now();
      calibrationStartedRef.current = performance.now();
      setCalibrationProgress(0);
      setPhase("calibrating");
      pushEvent("system", "Private session started", "Camera frames stay on this device", "low");
    } catch (cameraError) {
      console.error(cameraError);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      const permissionDenied =
        cameraError instanceof DOMException &&
        (cameraError.name === "NotAllowedError" || cameraError.name === "SecurityError");
      setError(
        permissionDenied
          ? "Camera permission was blocked. Allow camera access in the address bar, then try again."
          : "The camera could not start. Close other camera apps, then try again or use Demo mode.",
      );
      setPhase("idle");
    }
  }, [prepareModels, pushEvent, resetRuntime, unlockAudio]);

  const startDemo = useCallback(() => {
    resetRuntime();
    unlockAudio();
    startedAtRef.current = performance.now();
    setPhase("demo");
    pushEvent("system", "Guided demo started", "Synthetic signals only — camera is off", "low");
  }, [pushEvent, resetRuntime, unlockAudio]);

  const stopSession = useCallback(() => {
    const summary = {
      endedAt: new Date().toISOString(),
      durationSeconds: sessionSeconds,
      mode: phase === "demo" ? "demo" : "camera",
      stats: statsRef.current,
      events,
      privacy: "No camera frames were stored",
    };
    try {
      const previous = JSON.parse(localStorage.getItem("aegis-session-history") || "[]");
      localStorage.setItem(
        "aegis-session-history",
        JSON.stringify([summary, ...previous].slice(0, 20)),
      );
    } catch {
      // Local reporting is optional; monitoring must never depend on storage.
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    setPhase("idle");
    setTelemetry(initialTelemetry);
    setCalibrationProgress(0);
  }, [events, phase, sessionSeconds]);

  const recalibrate = useCallback(() => {
    if (phase !== "live") return;
    calibrationSamplesRef.current = [];
    calibrationStartedRef.current = performance.now();
    setCalibrationProgress(0);
    setPhase("calibrating");
    pushEvent("system", "Calibration restarted", "Look forward with a relaxed, open gaze", "low");
  }, [phase, pushEvent]);

  const exportReport = useCallback(() => {
    const report = {
      product: "Aegis Drive",
      generatedAt: new Date().toISOString(),
      session: {
        duration: formatDuration(sessionSeconds),
        riskState: telemetry.state,
        currentRisk: telemetry.risk,
        stats,
        events,
      },
      calibration: baseline,
      note: "Assistive research prototype. No images or video are included.",
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `aegis-drive-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [baseline, events, sessionSeconds, stats, telemetry.risk, telemetry.state]);

  const drawOverlay = useCallback((landmarks: NormalizedLandmark[] | undefined) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    if (!landmarks) return;

    const xs = landmarks.map((point) => point.x * width);
    const ys = landmarks.map((point) => point.y * height);
    const left = Math.min(...xs);
    const right = Math.max(...xs);
    const top = Math.min(...ys);
    const bottom = Math.max(...ys);
    const color = telemetry.state === "danger" ? "#ff5e57" : "#c7ff62";

    context.save();
    context.shadowBlur = 12;
    context.shadowColor = color;
    context.fillStyle = "rgba(199, 255, 98, .62)";
    for (let index = 0; index < landmarks.length; index += 5) {
      const point = landmarks[index];
      context.beginPath();
      context.arc(point.x * width, point.y * height, 1.1, 0, Math.PI * 2);
      context.fill();
    }

    const eyeAndIris = [33, 133, 160, 158, 153, 144, 263, 362, 385, 387, 373, 380, 468, 473];
    context.fillStyle = color;
    for (const index of eyeAndIris) {
      const point = landmarks[index];
      if (!point) continue;
      context.beginPath();
      context.arc(point.x * width, point.y * height, 2.4, 0, Math.PI * 2);
      context.fill();
    }

    const corner = Math.max(22, (right - left) * 0.12);
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.setLineDash([8, 6]);
    const corners = [
      [left, top, 1, 1],
      [right, top, -1, 1],
      [left, bottom, 1, -1],
      [right, bottom, -1, -1],
    ];
    for (const [x, y, xDirection, yDirection] of corners) {
      context.beginPath();
      context.moveTo(x, y + corner * yDirection);
      context.lineTo(x, y);
      context.lineTo(x + corner * xDirection, y);
      context.stroke();
    }
    context.restore();
  }, [telemetry.state]);

  useEffect(() => {
    if (phase !== "calibrating" && phase !== "live") return;
    let frameId = 0;
    let lastVideoTime = -1;
    let cancelled = false;

    const processFrame = () => {
      if (cancelled) return;
      const video = videoRef.current;
      const landmarker = faceLandmarkerRef.current;
      if (!video || !landmarker || video.readyState < 2) {
        frameId = requestAnimationFrame(processFrame);
        return;
      }

      if (video.currentTime === lastVideoTime) {
        frameId = requestAnimationFrame(processFrame);
        return;
      }
      lastVideoTime = video.currentTime;
      const now = performance.now();
      let faceFound = false;

      try {
        const result = landmarker.detectForVideo(video, now);
        const landmarks = result.faceLandmarks[0];
        faceFound = Boolean(landmarks);
        drawOverlay(landmarks);

        if (settingsRef.current.phoneDetection && objectDetectorRef.current) {
          const cadence = settingsRef.current.performance === "precision" ? 450 : 760;
          if (now - lastObjectDetectionRef.current > cadence) {
            lastObjectDetectionRef.current = now;
            const detection = objectDetectorRef.current.detectForVideo(video, now);
            phoneVisibleRef.current = detection.detections.some((item) =>
              item.categories.some(
                (category) =>
                  category.categoryName.toLowerCase().includes("phone") && category.score >= 0.42,
              ),
            );
          }
        } else {
          phoneVisibleRef.current = false;
        }

        if (landmarks) {
          faceMissingAtRef.current = 0;
          const signals = extractFaceSignals(landmarks);
          if (signals) {
            if (phaseRef.current === "calibrating") {
              calibrationSamplesRef.current.push(signals);
              const progress = clamp((now - calibrationStartedRef.current) / 5200);
              setCalibrationProgress(progress);
              setTelemetry((current) => ({
                ...current,
                ...signals,
                faceFound: true,
                risk: 0,
                state: "focused",
              }));
              if (progress >= 1) {
                const learned = buildCalibration(calibrationSamplesRef.current);
                setBaseline(learned);
                setPhase("live");
                pushEvent(
                  "system",
                  "Personal baseline locked",
                  `${calibrationSamplesRef.current.length} face samples learned`,
                  "low",
                );
              }
            } else {
              const classified = classifySignals(
                signals,
                baseline,
                settingsRef.current.sensitivity,
              );
              eyeSamplesRef.current.push({ time: now, closed: classified.eyesClosed });
              eyeSamplesRef.current = eyeSamplesRef.current.filter((sample) => now - sample.time <= 60_000);
              const perclos = calculatePerclos(eyeSamplesRef.current, now);

              if (classified.eyesClosed) {
                if (!eyeClosedAtRef.current) eyeClosedAtRef.current = now;
              } else if (eyeClosedAtRef.current) {
                const closure = now - eyeClosedAtRef.current;
                if (closure > 70 && closure < 750) {
                  updateStats({ blinks: statsRef.current.blinks + 1 });
                }
                eyeClosedAtRef.current = 0;
                eventFlagsRef.current.eyes = false;
              }
              if (classified.headAway) {
                if (!headAwayAtRef.current) headAwayAtRef.current = now;
              } else {
                headAwayAtRef.current = 0;
                eventFlagsRef.current.head = false;
              }
              if (classified.gazeAway) {
                if (!gazeAwayAtRef.current) gazeAwayAtRef.current = now;
              } else {
                gazeAwayAtRef.current = 0;
                eventFlagsRef.current.gaze = false;
              }

              if (classified.yawning && !eventFlagsRef.current.yawn) {
                eventFlagsRef.current.yawn = true;
                recentYawnTimesRef.current.push(now);
                updateStats({ yawns: statsRef.current.yawns + 1 });
                pushEvent("drowsiness", "Yawn pattern detected", "Mouth geometry crossed your baseline", "medium");
              } else if (!classified.yawning) {
                eventFlagsRef.current.yawn = false;
              }
              recentYawnTimesRef.current = recentYawnTimesRef.current.filter(
                (time) => now - time <= 10 * 60_000,
              );

              const eyeClosedMs = eyeClosedAtRef.current ? now - eyeClosedAtRef.current : 0;
              const headAwayMs = headAwayAtRef.current ? now - headAwayAtRef.current : 0;
              const gazeAwayMs = gazeAwayAtRef.current ? now - gazeAwayAtRef.current : 0;
              const assessment = calculateRisk({
                eyeClosedMs,
                perclos,
                yawnActive: classified.yawning,
                recentYawns: recentYawnTimesRef.current.length,
                headAwayMs,
                gazeAwayMs,
                phoneVisible: phoneVisibleRef.current,
                sensitivity: settingsRef.current.sensitivity,
              });
              riskSmoothedRef.current =
                riskSmoothedRef.current * 0.78 + assessment.score * 0.22;
              const smoothedRisk = Math.round(riskSmoothedRef.current);
              const smoothedState: RiskState =
                smoothedRisk >= 76
                  ? "danger"
                  : smoothedRisk >= 54
                    ? "warning"
                    : smoothedRisk >= 28
                      ? "caution"
                      : "focused";

              if (eyeClosedMs > 950 && !eventFlagsRef.current.eyes) {
                eventFlagsRef.current.eyes = true;
                pushEvent(
                  "drowsiness",
                  "Microsleep signature",
                  `${(eyeClosedMs / 1000).toFixed(1)} sec continuous eye closure`,
                  "high",
                );
              }
              if (headAwayMs > 1500 && !eventFlagsRef.current.head) {
                eventFlagsRef.current.head = true;
                updateStats({ distractions: statsRef.current.distractions + 1 });
                pushEvent("attention", "Head turned away", "Sustained off-axis head pose", "medium");
              }
              if (gazeAwayMs > 1500 && !eventFlagsRef.current.gaze) {
                eventFlagsRef.current.gaze = true;
                updateStats({ distractions: statsRef.current.distractions + 1 });
                pushEvent("attention", "Off-road gaze", "Eyes remained outside the forward zone", "medium");
              }
              if (phoneVisibleRef.current && !eventFlagsRef.current.phone) {
                eventFlagsRef.current.phone = true;
                updateStats({ phoneEvents: statsRef.current.phoneEvents + 1 });
                pushEvent("device", "Phone visible", "Handheld mobile device detected in frame", "high");
              } else if (!phoneVisibleRef.current) {
                eventFlagsRef.current.phone = false;
              }

              if (smoothedRisk > statsRef.current.maxRisk) {
                updateStats({ maxRisk: smoothedRisk });
              }
              if (
                (smoothedState === "warning" || smoothedState === "danger") &&
                now - lastAlertAtRef.current > (smoothedState === "danger" ? 4200 : 8000)
              ) {
                lastAlertAtRef.current = now;
                updateStats({ alerts: statsRef.current.alerts + 1 });
                soundAlert(smoothedState === "danger");
              }

              const eyeQuality = Math.round(
                clamp(signals.ear / Math.max(0.001, baseline.ear), 0, 1.15) * 100,
              );
              const distraction = Math.round(
                clamp(Math.max(headAwayMs / 2500, gazeAwayMs / 2200, phoneVisibleRef.current ? 1 : 0)) * 100,
              );
              if (now - lastHistoryAtRef.current > 950) {
                lastHistoryAtRef.current = now;
                setHistory((previous) => [
                  ...previous.slice(-51),
                  { risk: smoothedRisk, eyes: eyeQuality, distraction },
                ]);
              }

              setTelemetry({
                ...signals,
                perclos,
                risk: smoothedRisk,
                state: smoothedState,
                primary: assessment.primary,
                fps: frameCounterRef.current.fps,
                faceFound: true,
                phoneVisible: phoneVisibleRef.current,
                ...classified,
              });
            }
          }
        }
      } catch (frameError) {
        console.warn("A vision frame was skipped.", frameError);
      }

      if (!faceFound && phaseRef.current === "live") {
        if (!faceMissingAtRef.current) faceMissingAtRef.current = now;
        const missingMs = now - faceMissingAtRef.current;
        const assessment = calculateRisk({
          faceMissingMs: missingMs,
          sensitivity: settingsRef.current.sensitivity,
        });
        riskSmoothedRef.current = riskSmoothedRef.current * 0.84 + assessment.score * 0.16;
        const risk = Math.round(riskSmoothedRef.current);
        if (missingMs > 2500 && !eventFlagsRef.current.missing) {
          eventFlagsRef.current.missing = true;
          pushEvent("attention", "Driver out of frame", "Reposition the camera for a clear face view", "medium");
        }
        setTelemetry((current) => ({
          ...current,
          faceFound: false,
          risk,
          state: risk >= 54 ? "warning" : risk >= 28 ? "caution" : "focused",
          primary: "missing",
        }));
      }

      frameCounterRef.current.count += 1;
      const fpsElapsed = now - frameCounterRef.current.started;
      if (fpsElapsed >= 1000) {
        frameCounterRef.current.fps = Math.round(
          (frameCounterRef.current.count * 1000) / fpsElapsed,
        );
        frameCounterRef.current = {
          count: 0,
          started: now,
          fps: frameCounterRef.current.fps,
        };
      }
      frameId = requestAnimationFrame(processFrame);
    };

    frameId = requestAnimationFrame(processFrame);
    return () => {
      cancelled = true;
      cancelAnimationFrame(frameId);
    };
  }, [baseline, drawOverlay, phase, pushEvent, soundAlert, updateStats]);

  useEffect(() => {
    if (phase !== "demo") return;
    const demoStarted = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      const cycle = ((now - demoStarted) / 1000) % 36;
      let risk = 8 + Math.sin(cycle * 0.8) * 3;
      let state: RiskState = "focused";
      let eyesClosed = false;
      let yawning = false;
      let headAway = false;
      let gazeAway = false;
      let phoneVisible = false;
      let primary = "eyes";

      if (cycle > 9 && cycle <= 17) {
        risk = 34 + Math.sin(cycle) * 5;
        state = "caution";
        gazeAway = true;
        primary = "gaze";
      } else if (cycle > 17 && cycle <= 26) {
        risk = 60 + (cycle - 17) * 1.2;
        state = "warning";
        eyesClosed = cycle > 21;
        yawning = cycle < 21;
        primary = eyesClosed ? "eyes" : "yawn";
      } else if (cycle > 26 && cycle <= 31) {
        risk = 86 + Math.sin(cycle * 2) * 4;
        state = "danger";
        phoneVisible = true;
        headAway = true;
        primary = "phone";
      } else if (cycle > 31) {
        risk = Math.max(10, 62 - (cycle - 31) * 11);
        state = risk > 54 ? "warning" : risk > 28 ? "caution" : "focused";
        primary = "gaze";
      }

      const point = {
        risk: Math.round(risk),
        eyes: eyesClosed ? 22 : 96,
        distraction: phoneVisible ? 100 : gazeAway || headAway ? 68 : 5,
      };
      setTelemetry({
        ear: eyesClosed ? 0.12 : 0.29,
        mar: yawning ? 0.46 : 0.11,
        perclos: eyesClosed ? 0.31 : cycle > 21 ? 0.17 : 0.04,
        yaw: headAway ? 0.17 : 0.01,
        pitch: 0.44,
        gaze: gazeAway ? 0.78 : 0.51,
        risk: point.risk,
        state,
        primary,
        fps: 30,
        faceFound: true,
        phoneVisible,
        eyesClosed,
        yawning,
        headAway,
        gazeAway,
      });
      setHistory((previous) => [...previous.slice(-51), point]);
      if (point.risk > statsRef.current.maxRisk) updateStats({ maxRisk: point.risk });

      const scene = Math.floor(cycle);
      if (scene === 10 && !eventFlagsRef.current.demoGaze) {
        eventFlagsRef.current.demoGaze = true;
        pushEvent("attention", "Off-road gaze", "Demo: attention shifted to the side", "medium");
      }
      if (scene === 18 && !eventFlagsRef.current.demoYawn) {
        eventFlagsRef.current.demoYawn = true;
        updateStats({ yawns: statsRef.current.yawns + 1 });
        pushEvent("drowsiness", "Yawn pattern detected", "Demo: fatigue signature increased", "medium");
      }
      if (scene === 27 && !eventFlagsRef.current.demoPhone) {
        eventFlagsRef.current.demoPhone = true;
        updateStats({ phoneEvents: statsRef.current.phoneEvents + 1 });
        pushEvent("device", "Phone visible", "Demo: handheld device entered the driver zone", "high");
        soundAlert(true);
      }
      if (scene === 32 && !eventFlagsRef.current.demoRecovery) {
        eventFlagsRef.current.demoRecovery = true;
        pushEvent("recovery", "Attention recovered", "Demo: gaze returned to the forward zone", "low");
      }
      if (cycle < 2) eventFlagsRef.current = {};
    }, 350);
    return () => window.clearInterval(timer);
  }, [phase, pushEvent, soundAlert, updateStats]);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      setSessionSeconds(Math.floor((performance.now() - startedAtRef.current) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [active]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      faceLandmarkerRef.current?.close();
      objectDetectorRef.current?.close();
      void audioContextRef.current?.close();
    };
  }, []);

  useEffect(() => {
    const onFullscreenChange = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void document.documentElement.requestFullscreen();
    }
  }, []);

  const eyeQuality = Math.round(clamp(telemetry.ear / Math.max(0.001, baseline.ear), 0, 1) * 100);
  const headDeviation = Math.round(
    clamp(
      Math.max(
        Math.abs(telemetry.yaw - baseline.yaw) / 0.16,
        Math.abs(telemetry.pitch - baseline.pitch) / 0.2,
      ),
    ) * 100,
  );
  const gazeQuality = Math.round(
    (1 - clamp(Math.abs(telemetry.gaze - baseline.gaze) / 0.24)) * 100,
  );

  const recommendation = useMemo(() => {
    if (phase === "idle") return "Start a private session to build your personal baseline.";
    if (phase === "loading") return "The on-device models are warming up. This happens only once per page load.";
    if (phase === "calibrating") return "Face forward, relax your expression, and keep both eyes naturally open.";
    if (!telemetry.faceFound) return "Move the camera so your full face is visible with even lighting.";
    if (telemetry.state === "danger") return "Do not fight fatigue. Pull over at the next safe location and rest.";
    if (telemetry.state === "warning") return `Risk is being driven by ${PRIMARY_COPY[telemetry.primary] || "multiple signals"}. Plan a safe stop.`;
    if (telemetry.state === "caution") return "Sit upright, look forward, and increase ventilation. Stop if symptoms continue.";
    return "You look attentive. Keep the camera stable and take a break at least every two hours.";
  }, [phase, telemetry.faceFound, telemetry.primary, telemetry.state]);

  return (
    <main className={`app-shell risk-${telemetry.state}`} data-phase={phase}>
      <div className="atmosphere" aria-hidden="true">
        <span className="aurora aurora-one" />
        <span className="aurora aurora-two" />
        <span className="grain" />
        <span className="horizon-grid" />
      </div>

      <header className="topbar">
        <a className="brand" href="#top" aria-label="Aegis Drive home">
          <BrandMark />
          <span>
            <strong>AEGIS</strong>
            <small>DRIVE / GUARDIAN OS</small>
          </span>
        </a>
        <div className="system-badges" aria-label="System status">
          <span className="system-badge">
            <LockKeyhole size={13} /> Local-only vision
          </span>
          <span className={`system-badge ${modelReady ? "is-ready" : ""}`}>
            <span className="status-dot" /> {modelReady ? "Models ready" : "Engine standby"}
          </span>
        </div>
        <div className="top-actions">
          <span className="session-clock" aria-label={`Session time ${formatDuration(sessionSeconds)}`}>
            <Radio size={13} /> {active ? formatDuration(sessionSeconds) : "OFFLINE"}
          </span>
          <button className="icon-button" onClick={() => setHelpOpen(true)} aria-label="Open quick guide">
            <CircleHelp size={18} />
          </button>
          <button className="icon-button" onClick={toggleFullscreen} aria-label="Toggle fullscreen">
            {isFullscreen ? <Minimize size={18} /> : <Maximize size={18} />}
          </button>
          <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Open settings">
            <Settings2 size={18} />
          </button>
          {active ? (
            <button className="stop-button" onClick={stopSession}>
              <Square size={14} fill="currentColor" /> End session
            </button>
          ) : (
            <button className="start-button compact" onClick={startCamera}>
              <Play size={15} fill="currentColor" /> Start monitoring
            </button>
          )}
        </div>
      </header>

      <div className="content" id="top">
        <section className="cockpit-grid" aria-label="Live driver monitoring cockpit">
          <article className="panel camera-panel">
            <div className="panel-heading camera-heading">
              <div>
                <span className="eyebrow"><ScanFace size={14} /> DRIVER VISION</span>
                <h1>See fatigue before it becomes a decision.</h1>
              </div>
              <div className="camera-meta">
                <span className={`live-pill ${active ? "active" : ""}`}>
                  <i /> {phase === "demo" ? "DEMO FEED" : active ? "MONITORING" : "STANDBY"}
                </span>
                {phase === "live" && (
                  <button className="text-button" onClick={recalibrate}>
                    <RefreshCw size={13} /> Recalibrate
                  </button>
                )}
              </div>
            </div>

            <div className={`camera-stage ${settings.privacyMode ? "privacy-on" : ""}`}>
              <video ref={videoRef} muted playsInline aria-label="Live driver camera" />
              <canvas ref={canvasRef} aria-hidden="true" />
              <div className="camera-vignette" aria-hidden="true" />
              <div className="scan-beam" aria-hidden="true" />
              <div className="hud-corner hud-top-left" aria-hidden="true" />
              <div className="hud-corner hud-top-right" aria-hidden="true" />
              <div className="hud-corner hud-bottom-left" aria-hidden="true" />
              <div className="hud-corner hud-bottom-right" aria-hidden="true" />

              {phase === "idle" && (
                <div className="stage-empty">
                  <div className="stage-emblem">
                    <Camera size={30} />
                    <span className="emblem-orbit" />
                  </div>
                  <span className="stage-kicker">PRIVATE BY DESIGN</span>
                  <h2>Your camera never becomes a recording.</h2>
                  <p>
                    478 facial landmarks are analyzed live on this device. No account,
                    subscription, upload, or API key.
                  </p>
                  <div className="stage-buttons">
                    <button className="start-button" onClick={startCamera}>
                      <Play size={17} fill="currentColor" /> Start private monitoring
                    </button>
                    <button className="secondary-button" onClick={startDemo}>
                      <Sparkles size={16} /> Explore demo
                    </button>
                  </div>
                  <small><ShieldCheck size={13} /> Camera permission is requested only after you press Start</small>
                </div>
              )}

              {phase === "loading" && (
                <div className="stage-loading" role="status">
                  <div className="neural-loader">
                    <span />
                    <span />
                    <span />
                    <span />
                    <i />
                  </div>
                  <span className="stage-kicker">ON-DEVICE INITIALIZATION</span>
                  <h2>{loadMessage}</h2>
                  <p>Nothing is being uploaded. The first start can take a few seconds.</p>
                </div>
              )}

              {phase === "calibrating" && (
                <div className="calibration-overlay" role="status">
                  <div className="calibration-reticle">
                    <ScanFace size={42} />
                    <span />
                  </div>
                  <span className="stage-kicker">PERSONALIZING / {Math.round(calibrationProgress * 100)}%</span>
                  <h2>Look forward naturally</h2>
                  <p>Relax your face, keep your eyes open, and remain still for five seconds.</p>
                  <div className="calibration-track">
                    <span style={{ width: `${calibrationProgress * 100}%` }} />
                  </div>
                </div>
              )}

              {phase === "demo" && (
                <div className="demo-driver" aria-hidden="true">
                  <div className="demo-road">
                    <span className="road-line line-one" />
                    <span className="road-line line-two" />
                  </div>
                  <div className={`demo-face ${telemetry.eyesClosed ? "eyes-shut" : ""}`}>
                    <span className="demo-eye eye-left" />
                    <span className="demo-eye eye-right" />
                    <span className="demo-nose" />
                    <span className={`demo-mouth ${telemetry.yawning ? "is-yawning" : ""}`} />
                    <span className="face-mesh-ring" />
                  </div>
                </div>
              )}

              {settings.privacyMode && active && phase !== "demo" && (
                <div className="privacy-cover">
                  <LockKeyhole size={28} />
                  <strong>Privacy display</strong>
                  <span>Detection is active. Video preview is hidden.</span>
                </div>
              )}

              {active && phase !== "calibrating" && (
                <div className="camera-risk-banner">
                  <span className={`banner-pulse state-${telemetry.state}`} />
                  <div>
                    <small>LIVE ASSESSMENT</small>
                    <strong>{statusCopy.message}</strong>
                  </div>
                  <span>{statusCopy.action}</span>
                </div>
              )}

              <div className="vision-telemetry" aria-hidden="true">
                <span>FACE {telemetry.faceFound ? "LOCK" : "SEARCH"}</span>
                <span>{telemetry.fps || "--"} FPS</span>
                <span>LANDMARKS {telemetry.faceFound ? "478" : "000"}</span>
              </div>
            </div>

            <div className="camera-footer">
              <span><Camera size={13} /> {phase === "demo" ? "Synthetic demonstration" : cameraLabel}</span>
              <span><BrainCircuit size={13} /> Temporal fusion / 60 sec window</span>
              <span><LockKeyhole size={13} /> Zero frames retained</span>
            </div>
          </article>

          <aside className="right-rail">
            <article className="panel risk-panel">
              <div className="panel-heading compact-heading">
                <div>
                  <span className="eyebrow"><Gauge size={14} /> ATTENTION STATE</span>
                  <h2>Driver condition</h2>
                </div>
                <span className="confidence-chip">
                  {telemetry.faceFound ? "HIGH SIGNAL" : active ? "SEEKING FACE" : "READY"}
                </span>
              </div>
              <RiskOrb score={telemetry.risk} state={telemetry.state} />
              <div className="risk-explanation">
                <div>
                  <span className={`risk-status-dot state-${telemetry.state}`} />
                  <strong>{statusCopy.label}</strong>
                </div>
                <p>{recommendation}</p>
              </div>
              <div className="intervention-row">
                <div>
                  <small>PRIMARY SIGNAL</small>
                  <strong>{PRIMARY_COPY[telemetry.primary] || "baseline ready"}</strong>
                </div>
                <ChevronRight size={18} />
              </div>
            </article>

            <div className="metric-grid">
              <MetricCard
                icon={<Eye size={17} />}
                label="Eye openness"
                value={`${eyeQuality}%`}
                detail={`EAR ${telemetry.ear.toFixed(3)}`}
                progress={eyeQuality}
                level={telemetry.eyesClosed ? "alert" : eyeQuality < 72 ? "watch" : "normal"}
              />
              <MetricCard
                icon={<MoonStar size={17} />}
                label="PERCLOS"
                value={`${Math.round(telemetry.perclos * 100)}%`}
                detail="60 sec closure rate"
                progress={telemetry.perclos * 100}
                level={telemetry.perclos > 0.28 ? "alert" : telemetry.perclos > 0.16 ? "watch" : "normal"}
              />
              <MetricCard
                icon={<Crosshair size={17} />}
                label="Forward focus"
                value={`${Math.min(gazeQuality, 100)}%`}
                detail={`Pose drift ${headDeviation}%`}
                progress={Math.min(gazeQuality, 100)}
                level={telemetry.headAway || telemetry.gazeAway ? "watch" : "normal"}
              />
              <MetricCard
                icon={<Smartphone size={17} />}
                label="Phone zone"
                value={telemetry.phoneVisible ? "VISIBLE" : settings.phoneDetection ? "CLEAR" : "OFF"}
                detail={settings.phoneDetection ? "EfficientDet active" : "Disabled in settings"}
                progress={telemetry.phoneVisible ? 100 : settings.phoneDetection ? 4 : 0}
                level={telemetry.phoneVisible ? "alert" : "normal"}
              />
            </div>
          </aside>
        </section>

        {error && (
          <div className="error-banner" role="alert">
            <AlertTriangle size={18} />
            <span>{error}</span>
            <button onClick={() => setError(null)} aria-label="Dismiss error"><X size={16} /></button>
          </div>
        )}

        <section className="analytics-grid">
          <article className="panel timeline-panel">
            <div className="panel-heading compact-heading">
              <div>
                <span className="eyebrow"><Activity size={14} /> TEMPORAL INTELLIGENCE</span>
                <h2>Focus continuity</h2>
              </div>
              <div className="chart-legend" aria-label="Chart legend">
                <span><i className="legend-risk" /> Risk</span>
                <span><i className="legend-eye" /> Eye closure</span>
                <span><i className="legend-distraction" /> Distraction</span>
              </div>
            </div>
            <FocusTimeline history={history} />
            <div className="session-stats">
              <div><strong>{stats.blinks}</strong><span>Blinks observed</span></div>
              <div><strong>{stats.yawns}</strong><span>Yawn patterns</span></div>
              <div><strong>{stats.distractions}</strong><span>Attention drifts</span></div>
              <div><strong>{stats.maxRisk}</strong><span>Peak risk</span></div>
            </div>
          </article>

          <article className="panel event-panel">
            <div className="panel-heading compact-heading">
              <div>
                <span className="eyebrow"><Siren size={14} /> EVENT MEMORY</span>
                <h2>Session journal</h2>
              </div>
              <button className="text-button" disabled={!events.length} onClick={exportReport}>
                <Download size={13} /> Export
              </button>
            </div>
            <div className="event-list">
              {events.length ? (
                events.slice(0, 5).map((event) => (
                  <div className={`event-item severity-${event.severity}`} key={event.id}>
                    <span className="event-icon">
                      {event.type === "device" ? (
                        <Smartphone size={15} />
                      ) : event.type === "drowsiness" ? (
                        <MoonStar size={15} />
                      ) : event.type === "system" ? (
                        <ShieldCheck size={15} />
                      ) : event.type === "recovery" ? (
                        <Check size={15} />
                      ) : (
                        <Eye size={15} />
                      )}
                    </span>
                    <div>
                      <strong>{event.title}</strong>
                      <small>{event.detail}</small>
                    </div>
                    <time>{event.time}</time>
                  </div>
                ))
              ) : (
                <div className="event-empty">
                  <BarChart3 size={24} />
                  <strong>No events yet</strong>
                  <span>Your private session journal appears here.</span>
                </div>
              )}
            </div>
          </article>
        </section>

        <section className="intelligence-strip" aria-label="System capabilities">
          <div className="intelligence-intro">
            <span className="eyebrow"><Zap size={14} /> AEGIS FUSION CORE</span>
            <h2>Six signals. One explainable decision.</h2>
            <p>
              A single blink never becomes an alarm. Aegis combines persistence,
              frequency, concurrence, and your personal baseline before escalating.
            </p>
          </div>
          <div className="capability-flow">
            {[
              ["01", "Eye geometry", "Blink + microsleep"],
              ["02", "PERCLOS", "Rolling fatigue"],
              ["03", "Head vector", "Pose deviation"],
              ["04", "Iris gaze", "Road attention"],
              ["05", "Mouth ratio", "Yawn pattern"],
              ["06", "Object zone", "Phone presence"],
            ].map(([number, title, detail]) => (
              <div className="capability-node" key={number}>
                <span>{number}</span>
                <strong>{title}</strong>
                <small>{detail}</small>
              </div>
            ))}
          </div>
        </section>

        <footer>
          <div><BrandMark /><span><strong>Aegis Drive</strong><small>Open, local, privacy-first driver assistance</small></span></div>
          <p>
            Research prototype — not a substitute for rest, responsible driving, or certified vehicle safety systems.
          </p>
          <button className="text-button" onClick={() => setHelpOpen(true)}><Info size={13} /> Safety & usage</button>
        </footer>
      </div>

      {settingsOpen && (
        <div className="drawer-backdrop">
          <aside className="settings-drawer" role="dialog" aria-modal="true" aria-label="Detection settings">
            <div className="drawer-header">
              <div><span className="eyebrow">CONTROL CENTER</span><h2>Monitoring settings</h2></div>
              <button className="icon-button" onClick={() => setSettingsOpen(false)} aria-label="Close settings"><X size={18} /></button>
            </div>

            <div className="settings-section">
              <span className="settings-label">ALERTS</span>
              <label className="toggle-row">
                <span><Volume2 size={17} /><span><strong>Alert tones</strong><small>Escalating local sound</small></span></span>
                <input type="checkbox" checked={settings.sound} onChange={(event) => setSettings((current) => ({ ...current, sound: event.target.checked }))} />
                <i />
              </label>
              <label className="toggle-row">
                <span><Siren size={17} /><span><strong>Voice guidance</strong><small>Spoken critical instructions</small></span></span>
                <input type="checkbox" checked={settings.voice} onChange={(event) => setSettings((current) => ({ ...current, voice: event.target.checked }))} />
                <i />
              </label>
              <button className="drawer-action" onClick={() => soundAlert(true, true)}><Volume2 size={15} /> Test critical alert</button>
            </div>

            <div className="settings-section">
              <span className="settings-label">VISION</span>
              <label className="toggle-row">
                <span><Smartphone size={17} /><span><strong>Phone detection</strong><small>Uses more processing power</small></span></span>
                <input type="checkbox" checked={settings.phoneDetection} onChange={(event) => setSettings((current) => ({ ...current, phoneDetection: event.target.checked }))} />
                <i />
              </label>
              <label className="toggle-row">
                <span>{settings.privacyMode ? <CameraOff size={17} /> : <Camera size={17} />}<span><strong>Privacy display</strong><small>Hide preview, keep detection active</small></span></span>
                <input type="checkbox" checked={settings.privacyMode} onChange={(event) => setSettings((current) => ({ ...current, privacyMode: event.target.checked }))} />
                <i />
              </label>
              <div className="range-control">
                <label className="range-title" htmlFor="sensitivity">Sensitivity <small>{Math.round(settings.sensitivity * 100)}%</small></label>
                <input id="sensitivity" type="range" min="0.25" max="0.9" step="0.01" value={settings.sensitivity} onChange={(event) => setSettings((current) => ({ ...current, sensitivity: Number(event.target.value) }))} />
                <span className="range-axis"><small>Fewer alerts</small><small>Earlier alerts</small></span>
              </div>
            </div>

            <div className="settings-section">
              <span className="settings-label">PERFORMANCE</span>
              <div className="segmented-control">
                {(["eco", "balanced", "precision"] as const).map((mode) => (
                  <button key={mode} className={settings.performance === mode ? "active" : ""} onClick={() => setSettings((current) => ({ ...current, performance: mode }))}>{mode}</button>
                ))}
              </div>
              <p className="settings-note"><Info size={14} /> Precision checks for phones more often. Eco is best for older laptops.</p>
            </div>

            <div className="privacy-proof">
              <LockKeyhole size={20} />
              <div><strong>No cloud. No account. No footage.</strong><span>Only numeric session summaries can be saved locally in your browser.</span></div>
            </div>
          </aside>
        </div>
      )}

      {helpOpen && (
        <div className="modal-backdrop">
          <section className="guide-modal" role="dialog" aria-modal="true" aria-label="Quick start guide">
            <div className="drawer-header">
              <div><span className="eyebrow">60-SECOND SETUP</span><h2>Get a reliable reading</h2></div>
              <button className="icon-button" onClick={() => setHelpOpen(false)} aria-label="Close guide"><X size={18} /></button>
            </div>
            <div className="guide-steps">
              <div><span>1</span><div><strong>Mount the camera</strong><p>Place your laptop or webcam near eye level, centered within roughly 45–75 cm.</p></div></div>
              <div><span>2</span><div><strong>Use even light</strong><p>Avoid a bright window behind you. Your eyes and jawline should be visible.</p></div></div>
              <div><span>3</span><div><strong>Calibrate naturally</strong><p>Look forward for five seconds without exaggerating your expression.</p></div></div>
              <div><span>4</span><div><strong>React safely</strong><p>If Aegis warns you, pull over when safe. Caffeine and loud music do not replace sleep.</p></div></div>
            </div>
            <div className="guide-warning">
              <AlertTriangle size={20} />
              <p><strong>Important:</strong> This is a college research prototype, not certified automotive safety equipment. Never interact with the screen while driving.</p>
            </div>
            <button className="start-button full" onClick={() => { setHelpOpen(false); if (!active) void startCamera(); }}>
              <Route size={17} /> {active ? "Return to monitoring" : "Start setup"}
            </button>
          </section>
        </div>
      )}
    </main>
  );
}

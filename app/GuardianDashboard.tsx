"use client";

import {
  Activity,
  AlertTriangle,
  Camera,
  CameraOff,
  Check,
  CircleHelp,
  Crosshair,
  Download,
  Eye,
  Gauge,
  Info,
  Languages,
  LockKeyhole,
  Maximize,
  Minimize,
  MoonStar,
  Play,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Siren,
  Smartphone,
  Square,
  Sun,
  Volume2,
  X,
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
  classifyCameraQuality,
  classifySignals,
  clamp,
  extractFaceSignals,
  formatDuration,
} from "@/lib/detection/core.mjs";

type Phase = "idle" | "loading" | "calibrating" | "live" | "demo" | "paused";
type RiskState = "focused" | "caution" | "warning" | "danger";
type CameraQualityState = "clear" | "low-light" | "obstructed";
type Theme = "light" | "dark";
type VoiceLanguage = "en-IN" | "hi-IN" | "kn-IN" | "mr-IN" | "ta-IN" | "te-IN";
type VoiceAlertKind =
  | "sessionStart"
  | "calibration"
  | "eyes"
  | "perclos"
  | "yawn"
  | "repeatedYawn"
  | "gaze"
  | "head"
  | "phone"
  | "missing"
  | "lowLight"
  | "obstruction"
  | "warning"
  | "danger"
  | "recovery";

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
  voiceLanguage: VoiceLanguage;
  phoneDetection: boolean;
  privacyMode: boolean;
  sensitivity: number;
  performance: "balanced" | "precision" | "eco";
};

const VOICE_LANGUAGES: Array<{ code: VoiceLanguage; label: string; nativeLabel: string }> = [
  { code: "en-IN", label: "English", nativeLabel: "English (India)" },
  { code: "hi-IN", label: "Hindi", nativeLabel: "हिन्दी" },
  { code: "kn-IN", label: "Kannada", nativeLabel: "ಕನ್ನಡ" },
  { code: "mr-IN", label: "Marathi", nativeLabel: "मराठी" },
  { code: "ta-IN", label: "Tamil", nativeLabel: "தமிழ்" },
  { code: "te-IN", label: "Telugu", nativeLabel: "తెలుగు" },
];

const VOICE_ALERTS: Record<VoiceLanguage, Record<VoiceAlertKind, string>> = {
  "en-IN": {
    sessionStart: "Monitoring has started. Please look straight ahead for calibration.",
    calibration: "Calibration complete. Driver monitoring is now active.",
    eyes: "Your eyes have been closed for too long. Open your eyes and focus on the road.",
    perclos: "Frequent eye closure detected. You may be getting drowsy. Please take a safe break.",
    yawn: "A yawn was detected. Stay alert, and consider taking a break.",
    repeatedYawn: "Repeated yawning detected. Please pull over safely and rest.",
    gaze: "Your eyes are away from the road. Please look ahead.",
    head: "Your head is turned away. Please face the road.",
    phone: "Phone detected. Put the phone away and focus on the road.",
    missing: "I cannot see the driver clearly. Please face the camera.",
    lowLight: "The camera image is too dark. Please increase the light in front of you.",
    obstruction: "The camera appears to be blocked. Please uncover or clean the lens.",
    warning: "Fatigue warning. Please prepare to stop at a safe place.",
    danger: "Critical drowsiness risk. Pull over safely and rest now.",
    recovery: "Thank you. Your attention is back on the road.",
  },
  "hi-IN": {
    sessionStart: "निगरानी शुरू हो गई है। कैलिब्रेशन के लिए कृपया सामने देखें।",
    calibration: "कैलिब्रेशन पूरा हुआ। ड्राइवर निगरानी अब चालू है।",
    eyes: "आपकी आँखें बहुत देर से बंद हैं। आँखें खोलें और सड़क पर ध्यान दें।",
    perclos: "बार-बार आँखें बंद होना पाया गया है। आप उनींदे हो सकते हैं। कृपया सुरक्षित जगह पर रुकें।",
    yawn: "जम्हाई का संकेत मिला है। सतर्क रहें और ज़रूरत हो तो विश्राम करें।",
    repeatedYawn: "बार-बार जम्हाई आ रही है। कृपया सुरक्षित जगह पर गाड़ी रोककर आराम करें।",
    gaze: "आपकी नज़र सड़क से हट गई है। कृपया सामने देखें।",
    head: "आपका सिर सड़क से दूसरी ओर है। कृपया सामने देखें।",
    phone: "फ़ोन दिखाई दे रहा है। फ़ोन दूर रखें और सड़क पर ध्यान दें।",
    missing: "ड्राइवर साफ़ दिखाई नहीं दे रहा है। कृपया कैमरे की ओर सही स्थिति में बैठें।",
    lowLight: "कैमरे के लिए रोशनी कम है। कृपया सामने की रोशनी बढ़ाएँ।",
    obstruction: "कैमरा ढका हुआ है। कृपया लेंस को साफ़ और खुला रखें।",
    warning: "थकान की चेतावनी। कृपया सुरक्षित जगह पर रुकने की तैयारी करें।",
    danger: "गंभीर उनींदापन का खतरा। कृपया तुरंत सुरक्षित जगह पर रुकें और आराम करें।",
    recovery: "धन्यवाद। आपका ध्यान फिर से सड़क पर है।",
  },
  "kn-IN": {
    sessionStart: "ಮೇಲ್ವಿಚಾರಣೆ ಆರಂಭವಾಗಿದೆ. ಕ್ಯಾಲಿಬ್ರೇಶನ್‌ಗಾಗಿ ದಯವಿಟ್ಟು ನೇರವಾಗಿ ಮುಂದೆ ನೋಡಿ.",
    calibration: "ಕ್ಯಾಲಿಬ್ರೇಶನ್ ಪೂರ್ಣಗೊಂಡಿದೆ. ಚಾಲಕ ಮೇಲ್ವಿಚಾರಣೆ ಈಗ ಸಕ್ರಿಯವಾಗಿದೆ.",
    eyes: "ನಿಮ್ಮ ಕಣ್ಣುಗಳು ಹೆಚ್ಚು ಹೊತ್ತು ಮುಚ್ಚಿವೆ. ಕಣ್ಣು ತೆರೆಯಿರಿ ಮತ್ತು ರಸ್ತೆಯತ್ತ ಗಮನ ಕೊಡಿ.",
    perclos: "ಪದೇ ಪದೇ ಕಣ್ಣು ಮುಚ್ಚುತ್ತಿರುವುದು ಕಂಡುಬಂದಿದೆ. ನಿಮಗೆ ನಿದ್ರೆ ಬರುತ್ತಿರಬಹುದು. ದಯವಿಟ್ಟು ಸುರಕ್ಷಿತ ಸ್ಥಳದಲ್ಲಿ ನಿಲ್ಲಿಸಿ.",
    yawn: "ಆಕಳಿಕೆಯ ಸೂಚನೆ ಕಂಡುಬಂದಿದೆ. ಎಚ್ಚರವಾಗಿರಿ ಮತ್ತು ಅಗತ್ಯವಿದ್ದರೆ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ.",
    repeatedYawn: "ಪದೇ ಪದೇ ಆಕಳಿಸುತ್ತಿದ್ದೀರಿ. ದಯವಿಟ್ಟು ಸುರಕ್ಷಿತವಾಗಿ ವಾಹನ ನಿಲ್ಲಿಸಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ.",
    gaze: "ನಿಮ್ಮ ದೃಷ್ಟಿ ರಸ್ತೆಯಿಂದ ಸರಿದಿದೆ. ದಯವಿಟ್ಟು ಮುಂದೆ ನೋಡಿ.",
    head: "ನಿಮ್ಮ ತಲೆ ರಸ್ತೆಯಿಂದ ಬೇರೆ ಕಡೆ ತಿರುಗಿದೆ. ದಯವಿಟ್ಟು ಮುಂದೆ ನೋಡಿ.",
    phone: "ಫೋನ್ ಕಂಡುಬಂದಿದೆ. ಫೋನ್ ದೂರ ಇಟ್ಟು ರಸ್ತೆಯತ್ತ ಗಮನ ಕೊಡಿ.",
    missing: "ಚಾಲಕ ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಕ್ಯಾಮೆರಾಕ್ಕೆ ಸರಿಯಾಗಿ ಕುಳಿತುಕೊಳ್ಳಿ.",
    lowLight: "ಕ್ಯಾಮೆರಾಗೆ ಬೆಳಕು ಕಡಿಮೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ಮುಂಭಾಗದ ಬೆಳಕನ್ನು ಹೆಚ್ಚಿಸಿ.",
    obstruction: "ಕ್ಯಾಮೆರಾ ಮುಚ್ಚಲ್ಪಟ್ಟಿದೆ. ದಯವಿಟ್ಟು ಲೆನ್ಸ್ ಅನ್ನು ಸ್ವಚ್ಛವಾಗಿ ಮತ್ತು ತೆರೆಯಾಗಿ ಇರಿಸಿ.",
    warning: "ಆಯಾಸದ ಎಚ್ಚರಿಕೆ. ಸುರಕ್ಷಿತ ಸ್ಥಳದಲ್ಲಿ ನಿಲ್ಲಿಸಲು ಸಿದ್ಧರಾಗಿ.",
    danger: "ಗಂಭೀರ ನಿದ್ರಾವಸ್ಥೆಯ ಅಪಾಯ. ತಕ್ಷಣ ಸುರಕ್ಷಿತವಾಗಿ ವಾಹನ ನಿಲ್ಲಿಸಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ.",
    recovery: "ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ಗಮನ ಮತ್ತೆ ರಸ್ತೆಯ ಮೇಲಿದೆ.",
  },
  "mr-IN": {
    sessionStart: "निरीक्षण सुरू झाले आहे. कॅलिब्रेशनसाठी कृपया सरळ समोर पाहा.",
    calibration: "कॅलिब्रेशन पूर्ण झाले. चालक निरीक्षण आता सुरू आहे.",
    eyes: "तुमचे डोळे खूप वेळ बंद आहेत. डोळे उघडा आणि रस्त्यावर लक्ष द्या.",
    perclos: "वारंवार डोळे मिटत आहेत. तुम्हाला झोप येत असू शकते. कृपया सुरक्षित ठिकाणी थांबा.",
    yawn: "जांभईचे लक्षण आढळले. सतर्क राहा आणि गरज असल्यास विश्रांती घ्या.",
    repeatedYawn: "वारंवार जांभई येत आहे. कृपया सुरक्षितपणे वाहन थांबवून विश्रांती घ्या.",
    gaze: "तुमची नजर रस्त्यावरून हटली आहे. कृपया समोर पाहा.",
    head: "तुमचे डोके रस्त्यापासून दुसरीकडे वळले आहे. कृपया समोर पाहा.",
    phone: "फोन दिसत आहे. फोन बाजूला ठेवा आणि रस्त्यावर लक्ष द्या.",
    missing: "चालक स्पष्ट दिसत नाही. कृपया कॅमेऱ्यासमोर योग्य स्थितीत बसा.",
    lowLight: "कॅमेऱ्यासाठी प्रकाश कमी आहे. कृपया समोरील प्रकाश वाढवा.",
    obstruction: "कॅमेरा झाकला आहे. कृपया लेन्स स्वच्छ आणि मोकळी ठेवा.",
    warning: "थकव्याची सूचना. सुरक्षित ठिकाणी थांबण्याची तयारी करा.",
    danger: "गंभीर झोपेचा धोका. त्वरित सुरक्षितपणे वाहन थांबवा आणि विश्रांती घ्या.",
    recovery: "धन्यवाद. तुमचे लक्ष पुन्हा रस्त्यावर आहे.",
  },
  "ta-IN": {
    sessionStart: "கண்காணிப்பு தொடங்கியது. அளவீட்டிற்காக நேராக முன்னே பாருங்கள்.",
    calibration: "அளவீடு முடிந்தது. ஓட்டுநர் கண்காணிப்பு இப்போது செயல்பாட்டில் உள்ளது.",
    eyes: "உங்கள் கண்கள் அதிக நேரம் மூடியுள்ளன. கண்களைத் திறந்து சாலையில் கவனம் செலுத்துங்கள்.",
    perclos: "அடிக்கடி கண்கள் மூடப்படுவது கண்டறியப்பட்டது. உங்களுக்கு தூக்கம் வரலாம். பாதுகாப்பான இடத்தில் நிறுத்துங்கள்.",
    yawn: "கொட்டாவி கண்டறியப்பட்டது. விழிப்புடன் இருங்கள்; தேவைப்பட்டால் ஓய்வு எடுங்கள்.",
    repeatedYawn: "மீண்டும் மீண்டும் கொட்டாவி வருகிறது. பாதுகாப்பாக வாகனத்தை நிறுத்தி ஓய்வு எடுங்கள்.",
    gaze: "உங்கள் பார்வை சாலையிலிருந்து விலகியுள்ளது. முன்னே பாருங்கள்.",
    head: "உங்கள் தலை சாலையிலிருந்து விலகித் திரும்பியுள்ளது. முன்னே பாருங்கள்.",
    phone: "தொலைபேசி கண்டறியப்பட்டது. அதை ஒதுக்கி வைத்து சாலையில் கவனம் செலுத்துங்கள்.",
    missing: "ஓட்டுநர் தெளிவாகத் தெரியவில்லை. கேமராவை நோக்கி சரியாக அமருங்கள்.",
    lowLight: "கேமராவுக்கு வெளிச்சம் குறைவாக உள்ளது. முன்புற வெளிச்சத்தை அதிகரிக்கவும்.",
    obstruction: "கேமரா மறைக்கப்பட்டுள்ளது. லென்ஸை சுத்தமாகவும் திறந்தும் வைத்திருங்கள்.",
    warning: "சோர்வு எச்சரிக்கை. பாதுகாப்பான இடத்தில் நிறுத்தத் தயாராகுங்கள்.",
    danger: "கடுமையான தூக்க அபாயம். உடனே பாதுகாப்பாக வாகனத்தை நிறுத்தி ஓய்வு எடுங்கள்.",
    recovery: "நன்றி. உங்கள் கவனம் மீண்டும் சாலையில் உள்ளது.",
  },
  "te-IN": {
    sessionStart: "పర్యవేక్షణ ప్రారంభమైంది. కాలిబ్రేషన్ కోసం దయచేసి నేరుగా ముందుకు చూడండి.",
    calibration: "కాలిబ్రేషన్ పూర్తైంది. డ్రైవర్ పర్యవేక్షణ ఇప్పుడు క్రియాశీలంగా ఉంది.",
    eyes: "మీ కళ్ళు ఎక్కువసేపు మూసుకుని ఉన్నాయి. కళ్ళు తెరిచి రోడ్డుపై దృష్టి పెట్టండి.",
    perclos: "తరచుగా కళ్ళు మూసుకోవడం గుర్తించబడింది. మీకు నిద్ర వస్తుండవచ్చు. సురక్షిత ప్రదేశంలో ఆపండి.",
    yawn: "ఆవలింత గుర్తించబడింది. అప్రమత్తంగా ఉండండి; అవసరమైతే విశ్రాంతి తీసుకోండి.",
    repeatedYawn: "పదేపదే ఆవలింత వస్తోంది. దయచేసి సురక్షితంగా వాహనం ఆపి విశ్రాంతి తీసుకోండి.",
    gaze: "మీ చూపు రోడ్డుపై నుంచి మళ్లింది. దయచేసి ముందుకు చూడండి.",
    head: "మీ తల రోడ్డుకు దూరంగా తిరిగింది. దయచేసి ముందుకు చూడండి.",
    phone: "ఫోన్ గుర్తించబడింది. ఫోన్ పక్కన పెట్టి రోడ్డుపై దృష్టి పెట్టండి.",
    missing: "డ్రైవర్ స్పష్టంగా కనిపించడం లేదు. దయచేసి కెమెరాకు సరైన స్థానంలో కూర్చోండి.",
    lowLight: "కెమెరాకు వెలుతురు తక్కువగా ఉంది. దయచేసి ముందు వెలుతురును పెంచండి.",
    obstruction: "కెమెరా కప్పబడి ఉంది. దయచేసి లెన్స్‌ను శుభ్రంగా మరియు తెరిచి ఉంచండి.",
    warning: "అలసట హెచ్చరిక. సురక్షిత ప్రదేశంలో ఆపడానికి సిద్ధం అవ్వండి.",
    danger: "తీవ్రమైన నిద్రమత్తు ప్రమాదం. వెంటనే సురక్షితంగా వాహనం ఆపి విశ్రాంతి తీసుకోండి.",
    recovery: "ధన్యవాదాలు. మీ దృష్టి మళ్లీ రోడ్డుపై ఉంది.",
  },
};

const VOICE_COOLDOWNS: Record<VoiceAlertKind, number> = {
  sessionStart: 0,
  calibration: 0,
  eyes: 18_000,
  perclos: 45_000,
  yawn: 40_000,
  repeatedYawn: 90_000,
  gaze: 16_000,
  head: 18_000,
  phone: 20_000,
  missing: 25_000,
  lowLight: 60_000,
  obstruction: 35_000,
  warning: 24_000,
  danger: 9_000,
  recovery: 15_000,
};

const ASSET_BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

function selectNaturalVoice(voices: SpeechSynthesisVoice[], language: VoiceLanguage) {
  const exactLanguage = language.toLowerCase();
  const languageRoot = exactLanguage.split("-")[0];
  return [...voices]
    .map((voice) => {
      const voiceLanguage = voice.lang.toLowerCase();
      const name = voice.name.toLowerCase();
      let score = 0;
      if (voiceLanguage === exactLanguage) score += 100;
      else if (voiceLanguage.startsWith(`${languageRoot}-`) || voiceLanguage === languageRoot) score += 72;
      if (/natural|neural|enhanced|premium/.test(name)) score += 22;
      if (/microsoft|google|siri/.test(name)) score += 10;
      if (voice.localService) score += 8;
      if (voice.default) score += 2;
      return { voice, score };
    })
    .filter(({ score }) => score >= 72)
    .sort((left, right) => right.score - left.score)[0]?.voice;
}

function measureCameraFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
) {
  const width = 32;
  const height = 18;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  context.drawImage(video, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, width, height).data;
  const luminance: number[] = [];
  let darkPixels = 0;
  let total = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    const value =
      (pixels[index] * 0.2126 + pixels[index + 1] * 0.7152 + pixels[index + 2] * 0.0722) /
      255;
    luminance.push(value);
    total += value;
    if (value < 0.16) darkPixels += 1;
  }
  const brightness = total / luminance.length;
  const variance =
    luminance.reduce((sum, value) => sum + (value - brightness) ** 2, 0) /
    luminance.length;
  return {
    brightness,
    contrast: Math.sqrt(variance),
    darkPixelRatio: darkPixels / luminance.length,
  };
}

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

const STATE_COPY: Record<RiskState, { label: string; action: string }> = {
  focused: {
    label: "Focused",
    action: "Continue monitoring",
  },
  caution: {
    label: "Drift detected",
    action: "Recenter on the road",
  },
  warning: {
    label: "Warning",
    action: "Prepare to stop safely",
  },
  danger: {
    label: "Critical",
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
  const [modelReady, setModelReady] = useState(false);
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [theme, setTheme] = useState<Theme>("light");
  const [cameraQuality, setCameraQuality] = useState<{
    state: CameraQualityState;
    brightness: number;
  }>({ state: "clear", brightness: 1 });
  const [settings, setSettings] = useState<Settings>({
    sound: true,
    voice: true,
    voiceLanguage: "en-IN",
    phoneDetection: true,
    privacyMode: false,
    sensitivity: 0.62,
    performance: "precision",
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const qualityCanvasRef = useRef<HTMLCanvasElement | null>(null);
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
  const lastQualityCheckRef = useRef(0);
  const qualityCandidateRef = useRef<{ state: CameraQualityState; count: number }>({
    state: "clear",
    count: 0,
  });
  const cameraQualityRef = useRef(cameraQuality);
  const phoneVisibleRef = useRef(false);
  const riskSmoothedRef = useRef(0);
  const lastAlertAtRef = useRef(0);
  const lastVoiceAtRef = useRef<Partial<Record<VoiceAlertKind, number>>>({});
  const lastVoiceGlobalAtRef = useRef(0);
  const voiceTimerRef = useRef<number | null>(null);
  const attentionWasImpairedRef = useRef(false);
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

  useEffect(() => {
    let savedTheme: string | null = null;
    try {
      savedTheme = window.localStorage.getItem("driver-detection-theme");
    } catch {
      // The system preference remains available when storage is restricted.
    }
    const preferredTheme: Theme =
      savedTheme === "dark" || savedTheme === "light"
        ? savedTheme
        : window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
    const frame = window.requestAnimationFrame(() => {
      document.documentElement.dataset.theme = preferredTheme;
      setTheme(preferredTheme);
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((current) => {
      const nextTheme = current === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = nextTheme;
      try {
        window.localStorage.setItem("driver-detection-theme", nextTheme);
      } catch {
        // Theme switching must still work when storage is restricted.
      }
      return nextTheme;
    });
  }, []);

  useEffect(() => {
    cameraQualityRef.current = cameraQuality;
  }, [cameraQuality]);

  useEffect(() => {
    if (!("speechSynthesis" in window)) return;
    const refreshVoices = () => setAvailableVoices([...window.speechSynthesis.getVoices()]);
    refreshVoices();
    window.speechSynthesis.addEventListener("voiceschanged", refreshVoices);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", refreshVoices);
  }, []);

  const active = phase === "live" || phase === "calibrating" || phase === "demo";
  const statusCopy = STATE_COPY[telemetry.state];
  const selectedVoice = useMemo(
    () => selectNaturalVoice(availableVoices, settings.voiceLanguage),
    [availableVoices, settings.voiceLanguage],
  );

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

  const speakAlert = useCallback(
    (kind: VoiceAlertKind, force = false) => {
      const currentSettings = settingsRef.current;
      if ((!currentSettings.voice && !force) || !("speechSynthesis" in window)) return false;

      const now = performance.now();
      const previousForKind = lastVoiceAtRef.current[kind] ?? -Infinity;
      const previousGlobal = lastVoiceGlobalAtRef.current || -Infinity;
      const urgent = kind === "danger" || kind === "eyes" || kind === "phone";
      const globalCooldown = urgent ? 0 : 2_800;
      if (
        !force &&
        (now - previousForKind < VOICE_COOLDOWNS[kind] ||
          now - previousGlobal < globalCooldown)
      ) {
        return false;
      }

      const language = currentSettings.voiceLanguage;
      const utterance = new SpeechSynthesisUtterance(VOICE_ALERTS[language][kind]);
      const naturalVoice = selectNaturalVoice(availableVoices, language);
      if (naturalVoice) utterance.voice = naturalVoice;
      utterance.lang = naturalVoice?.lang || language;
      utterance.rate = urgent ? 0.88 : language === "en-IN" ? 0.94 : 0.9;
      utterance.pitch = urgent ? 0.96 : 1;
      utterance.volume = 0.96;

      if (voiceTimerRef.current !== null) window.clearTimeout(voiceTimerRef.current);
      window.speechSynthesis.cancel();
      lastVoiceAtRef.current[kind] = now;
      lastVoiceGlobalAtRef.current = now;
      voiceTimerRef.current = window.setTimeout(() => {
        window.speechSynthesis.speak(utterance);
        voiceTimerRef.current = null;
      }, urgent ? 320 : 180);
      return true;
    },
    [availableVoices],
  );

  const soundAlert = useCallback(
    (danger = false, force = false, voiceKind?: VoiceAlertKind) => {
      const currentSettings = settingsRef.current;
      if (typeof window === "undefined") return;
      if (currentSettings.sound || force) {
        unlockAudio();
        const context = audioContextRef.current;
        if (context) {
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
        }
      }
      speakAlert(voiceKind ?? (danger ? "danger" : "warning"), force);
    },
    [speakAlert, unlockAudio],
  );

  const prepareModels = useCallback(async () => {
    if (faceLandmarkerRef.current) return;
    setLoadMessage("Loading 478-point face geometry");
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks(`${ASSET_BASE_PATH}/wasm`);
    const faceOptions = {
      baseOptions: {
        modelAssetPath: `${ASSET_BASE_PATH}/models/face_landmarker.task`,
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
          modelAssetPath: `${ASSET_BASE_PATH}/models/face_landmarker.task`,
          delegate: "CPU",
        },
      });
    }

    if (settingsRef.current.phoneDetection) {
      setLoadMessage("Adding mobile-device awareness");
      try {
        objectDetectorRef.current = await vision.ObjectDetector.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: `${ASSET_BASE_PATH}/models/efficientdet_lite0.tflite`,
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
    lastQualityCheckRef.current = 0;
    qualityCandidateRef.current = { state: "clear", count: 0 };
    cameraQualityRef.current = { state: "clear", brightness: 1 };
    phoneVisibleRef.current = false;
    riskSmoothedRef.current = 0;
    lastHistoryAtRef.current = 0;
    lastAlertAtRef.current = 0;
    lastVoiceAtRef.current = {};
    lastVoiceGlobalAtRef.current = 0;
    attentionWasImpairedRef.current = false;
    eventFlagsRef.current = {};
    frameCounterRef.current = { count: 0, started: performance.now(), fps: 0 };
    statsRef.current = initialStats;
    setStats(initialStats);
    setHistory(emptyHistory);
    setEvents([]);
    setTelemetry(initialTelemetry);
    setCameraQuality({ state: "clear", brightness: 1 });
    setSessionSeconds(0);
    const overlay = overlayCanvasRef.current?.getContext("2d");
    if (overlay && overlayCanvasRef.current) {
      overlay.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
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
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      startedAtRef.current = performance.now();
      calibrationStartedRef.current = performance.now();
      setCalibrationProgress(0);
      setPhase("calibrating");
      pushEvent("system", "Private session started", "Camera frames stay on this device", "low");
      speakAlert("sessionStart");
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
  }, [prepareModels, pushEvent, resetRuntime, speakAlert, unlockAudio]);

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
      const previous = JSON.parse(
        localStorage.getItem("driver-detection-session-history") || "[]",
      );
      localStorage.setItem(
        "driver-detection-session-history",
        JSON.stringify([summary, ...previous].slice(0, 20)),
      );
    } catch {
      // Local reporting is optional; monitoring must never depend on storage.
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const overlay = overlayCanvasRef.current?.getContext("2d");
    if (overlay && overlayCanvasRef.current) {
      overlay.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
    if (voiceTimerRef.current !== null) window.clearTimeout(voiceTimerRef.current);
    voiceTimerRef.current = null;
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    setPhase("idle");
    setTelemetry(initialTelemetry);
    cameraQualityRef.current = { state: "clear", brightness: 1 };
    setCameraQuality({ state: "clear", brightness: 1 });
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
      product: "Driver Drowsiness & Distraction Detection System",
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
    anchor.download = `driver-detection-report-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [baseline, events, sessionSeconds, stats, telemetry.risk, telemetry.state]);

  const drawTracker = useCallback(
    (landmarks: NormalizedLandmark[] | undefined) => {
      const canvas = overlayCanvasRef.current;
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

      const color = telemetry.state === "danger" ? "#ff6b64" : "#74d9ff";
      context.save();
      context.fillStyle = color;
      context.shadowBlur = 5;
      context.shadowColor = color;
      context.globalAlpha = 0.58;
      for (let index = 0; index < landmarks.length; index += 6) {
        const point = landmarks[index];
        context.beginPath();
        context.arc(point.x * width, point.y * height, 1.25, 0, Math.PI * 2);
        context.fill();
      }

      context.globalAlpha = 0.92;
      for (const index of [33, 133, 159, 145, 263, 362, 386, 374, 468, 473]) {
        const point = landmarks[index];
        if (!point) continue;
        context.beginPath();
        context.arc(point.x * width, point.y * height, 2.1, 0, Math.PI * 2);
        context.fill();
      }
      context.restore();
    },
    [telemetry.state],
  );

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
        drawTracker(landmarks);

        if (now - lastQualityCheckRef.current >= 1_200) {
          lastQualityCheckRef.current = now;
          if (!qualityCanvasRef.current) qualityCanvasRef.current = document.createElement("canvas");
          const sample = measureCameraFrame(video, qualityCanvasRef.current);
          if (sample) {
            const nextState = classifyCameraQuality({
              ...sample,
              faceFound,
            }) as CameraQualityState;
            if (qualityCandidateRef.current.state === nextState) {
              qualityCandidateRef.current.count += 1;
            } else {
              qualityCandidateRef.current = { state: nextState, count: 1 };
            }

            if (
              qualityCandidateRef.current.count >= 2 &&
              cameraQualityRef.current.state !== nextState
            ) {
              const nextQuality = { state: nextState, brightness: sample.brightness };
              cameraQualityRef.current = nextQuality;
              setCameraQuality(nextQuality);
              if (nextState === "clear") {
                eventFlagsRef.current.lowLight = false;
                eventFlagsRef.current.obstruction = false;
              } else if (
                phaseRef.current === "calibrating" ||
                phaseRef.current === "live"
              ) {
                const flag = nextState === "low-light" ? "lowLight" : "obstruction";
                if (!eventFlagsRef.current[flag]) {
                  eventFlagsRef.current[flag] = true;
                  pushEvent(
                    "system",
                    nextState === "low-light" ? "Low light" : "Camera obstructed",
                    nextState === "low-light"
                      ? "Increase the light in front of the driver"
                      : "Uncover or clean the camera lens",
                    "medium",
                  );
                  soundAlert(false, false, flag);
                }
              }
            }
          }
        }

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
          eventFlagsRef.current.missing = false;
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
                speakAlert("calibration");
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
              if (perclos > 0.28 && !eventFlagsRef.current.perclos) {
                eventFlagsRef.current.perclos = true;
                pushEvent(
                  "drowsiness",
                  "Frequent eye closure",
                  `${Math.round(perclos * 100)}% closure rate in the rolling window`,
                  "high",
                );
                speakAlert("perclos");
              } else if (perclos < 0.18) {
                eventFlagsRef.current.perclos = false;
              }

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
                recentYawnTimesRef.current = recentYawnTimesRef.current.filter(
                  (time) => now - time <= 10 * 60_000,
                );
                updateStats({ yawns: statsRef.current.yawns + 1 });
                pushEvent("drowsiness", "Yawn pattern detected", "Mouth geometry crossed your baseline", "medium");
                if (
                  recentYawnTimesRef.current.length >= 3 &&
                  !eventFlagsRef.current.repeatedYawn
                ) {
                  eventFlagsRef.current.repeatedYawn = true;
                  pushEvent(
                    "drowsiness",
                    "Repeated yawning",
                    `${recentYawnTimesRef.current.length} yawn patterns within ten minutes`,
                    "high",
                  );
                  speakAlert("repeatedYawn");
                } else {
                  speakAlert("yawn");
                }
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
                speakAlert("eyes");
              }
              if (headAwayMs > 1500 && !eventFlagsRef.current.head) {
                eventFlagsRef.current.head = true;
                updateStats({ distractions: statsRef.current.distractions + 1 });
                pushEvent("attention", "Head turned away", "Sustained off-axis head pose", "medium");
                speakAlert("head");
              }
              if (gazeAwayMs > 1500 && !eventFlagsRef.current.gaze) {
                eventFlagsRef.current.gaze = true;
                updateStats({ distractions: statsRef.current.distractions + 1 });
                pushEvent("attention", "Off-road gaze", "Eyes remained outside the forward zone", "medium");
                speakAlert("gaze");
              }
              if (phoneVisibleRef.current && !eventFlagsRef.current.phone) {
                eventFlagsRef.current.phone = true;
                updateStats({ phoneEvents: statsRef.current.phoneEvents + 1 });
                pushEvent("device", "Phone visible", "Handheld mobile device detected in frame", "high");
                lastAlertAtRef.current = now;
                soundAlert(true, false, "phone");
              } else if (!phoneVisibleRef.current) {
                eventFlagsRef.current.phone = false;
              }

              if (smoothedState === "warning" || smoothedState === "danger") {
                attentionWasImpairedRef.current = true;
              } else if (smoothedState === "focused" && attentionWasImpairedRef.current) {
                attentionWasImpairedRef.current = false;
                pushEvent("recovery", "Attention recovered", "Driver signals returned to the safe zone", "low");
                speakAlert("recovery");
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
                soundAlert(
                  smoothedState === "danger",
                  false,
                  smoothedState === "danger" ? "danger" : "warning",
                );
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
        if (
          missingMs > 2500 &&
          cameraQualityRef.current.state === "clear" &&
          !eventFlagsRef.current.missing
        ) {
          eventFlagsRef.current.missing = true;
          pushEvent("attention", "Driver out of frame", "Reposition the camera for a clear face view", "medium");
          speakAlert("missing");
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
  }, [baseline, drawTracker, phase, pushEvent, soundAlert, speakAlert, updateStats]);

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
        speakAlert("gaze");
      }
      if (scene === 18 && !eventFlagsRef.current.demoYawn) {
        eventFlagsRef.current.demoYawn = true;
        updateStats({ yawns: statsRef.current.yawns + 1 });
        pushEvent("drowsiness", "Yawn pattern detected", "Demo: fatigue signature increased", "medium");
        speakAlert("yawn");
      }
      if (scene === 27 && !eventFlagsRef.current.demoPhone) {
        eventFlagsRef.current.demoPhone = true;
        updateStats({ phoneEvents: statsRef.current.phoneEvents + 1 });
        pushEvent("device", "Phone visible", "Demo: handheld device entered the driver zone", "high");
        soundAlert(true, false, "phone");
      }
      if (scene === 32 && !eventFlagsRef.current.demoRecovery) {
        eventFlagsRef.current.demoRecovery = true;
        pushEvent("recovery", "Attention recovered", "Demo: gaze returned to the forward zone", "low");
        speakAlert("recovery");
      }
      if (cycle < 2) eventFlagsRef.current = {};
    }, 350);
    return () => window.clearInterval(timer);
  }, [phase, pushEvent, soundAlert, speakAlert, updateStats]);

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
      if (voiceTimerRef.current !== null) window.clearTimeout(voiceTimerRef.current);
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
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
    if (phase === "idle") return "Start monitoring to calibrate.";
    if (phase === "loading") return "Loading on-device models.";
    if (cameraQuality.state === "obstructed") return "Uncover or clean the camera lens.";
    if (cameraQuality.state === "low-light") return "Increase the light in front of you.";
    if (phase === "calibrating") return "Face forward with both eyes open.";
    if (!telemetry.faceFound) return "Center your face in even light.";
    if (telemetry.state === "danger") return "Pull over safely and rest now.";
    if (telemetry.state === "warning") return "Prepare to stop safely.";
    if (telemetry.state === "caution") return "Look forward. Stop if this continues.";
    return "Attention is stable.";
  }, [cameraQuality.state, phase, telemetry.faceFound, telemetry.state]);

  return (
    <main className={`app-shell risk-${telemetry.state}`} data-phase={phase}>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Driver Drowsiness and Distraction Detection System home">
          <BrandMark />
          <span>
            <strong>Driver Drowsiness &amp; Distraction</strong>
            <small>Detection System</small>
          </span>
        </a>
        <span className={`system-status ${modelReady ? "is-ready" : ""}`}>
          <span className="status-dot" /> {active ? "Monitoring" : modelReady ? "Ready" : "Standby"}
        </span>
        <div className="top-actions">
          <button className="icon-button" onClick={() => setHelpOpen(true)} aria-label="Open quick guide">
            <CircleHelp size={18} />
          </button>
          <button
            className="icon-button"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            aria-pressed={theme === "dark"}
          >
            {theme === "dark" ? <Sun size={18} /> : <MoonStar size={18} />}
          </button>
          <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Open settings">
            <Settings2 size={18} />
          </button>
          {active ? (
            <button className="stop-button" onClick={stopSession} aria-label="End session">
              <Square size={14} fill="currentColor" /> <span>End session</span>
            </button>
          ) : (
            <button className="start-button compact" onClick={startCamera} aria-label="Start monitoring">
              <Play size={15} fill="currentColor" /> <span>Start monitoring</span>
            </button>
          )}
        </div>
      </header>

      <div className="content" id="top">
        <section className="cockpit-grid" aria-label="Live driver monitoring">
          <article className="panel camera-panel">
            <div className="panel-heading camera-heading">
              <div>
                <h1>Driver monitoring</h1>
                <span className="session-clock">{active ? formatDuration(sessionSeconds) : "On-device processing"}</span>
              </div>
              <div className="camera-meta">
                {active && phase !== "demo" && cameraQuality.state !== "clear" && (
                  <span className={`quality-pill quality-${cameraQuality.state}`}>
                    {cameraQuality.state === "low-light" ? <Sun size={13} /> : <CameraOff size={13} />}
                    <span>{cameraQuality.state === "low-light" ? "Low light" : "Camera blocked"}</span>
                  </span>
                )}
                <span className={`live-pill ${active ? "active" : ""}`}>
                  <i /> {phase === "demo" ? "Demo" : active ? "Live" : "Ready"}
                </span>
                {phase === "live" && (
                  <button className="icon-button small" onClick={recalibrate} aria-label="Recalibrate">
                    <RefreshCw size={15} />
                  </button>
                )}
                <button className="icon-button small" onClick={toggleFullscreen} aria-label="Toggle fullscreen">
                  {isFullscreen ? <Minimize size={16} /> : <Maximize size={16} />}
                </button>
              </div>
            </div>

            <div className={`camera-stage ${settings.privacyMode ? "privacy-on" : ""}`}>
              <video ref={videoRef} muted playsInline aria-label="Live driver camera" />
              <canvas ref={overlayCanvasRef} className="face-tracker" aria-hidden="true" />

              {phase === "idle" && (
                <div className="stage-empty">
                  <div className="stage-emblem"><Camera size={26} /></div>
                  <h2>Ready to monitor</h2>
                  <p>Private. On-device. No video saved.</p>
                  <div className="stage-buttons">
                    <button className="start-button" onClick={startCamera}>
                      <Play size={17} fill="currentColor" /> Start monitoring
                    </button>
                    <button className="secondary-button" onClick={startDemo}>
                      Demo
                    </button>
                  </div>
                </div>
              )}

              {phase === "loading" && (
                <div className="stage-loading" role="status">
                  <div className="stage-emblem"><Camera size={26} /></div>
                  <h2>{loadMessage}</h2>
                  <p>Initializing on this device…</p>
                </div>
              )}

              {phase === "calibrating" && (
                <div className="calibration-overlay" role="status">
                  <h2>Look straight ahead</h2>
                  <p>{Math.round(calibrationProgress * 100)}% calibrated</p>
                  <div className="calibration-track">
                    <span style={{ width: `${calibrationProgress * 100}%` }} />
                  </div>
                </div>
              )}

              {phase === "demo" && (
                <div className="demo-driver" aria-hidden="true">
                  <div className={`demo-face ${telemetry.eyesClosed ? "eyes-shut" : ""}`}>
                    <span className="demo-eye eye-left" />
                    <span className="demo-eye eye-right" />
                    <span className="demo-nose" />
                    <span className={`demo-mouth ${telemetry.yawning ? "is-yawning" : ""}`} />
                  </div>
                </div>
              )}

              {settings.privacyMode && active && phase !== "demo" && (
                <div className="privacy-cover">
                  <LockKeyhole size={28} />
                  <strong>Privacy display</strong>
                  <span>Detection remains active</span>
                </div>
              )}

              {active && phase !== "calibrating" && (
                <div className="camera-risk-banner">
                  <span className={`risk-status-dot state-${telemetry.state}`} />
                  <strong>{statusCopy.label}</strong>
                  <span>{statusCopy.action}</span>
                </div>
              )}
            </div>
          </article>

          <aside className="right-rail">
            <article className="panel risk-panel">
              <div className="panel-heading compact-heading">
                <h2><Gauge size={17} /> Risk</h2>
                <span className="confidence-chip">{telemetry.faceFound ? "Face detected" : active ? "Finding face" : "Ready"}</span>
              </div>
              <div className="risk-score" aria-label={`Attention risk ${telemetry.risk} out of 100`}>
                <strong>{telemetry.risk}</strong><span>/ 100</span>
              </div>
              <div className="risk-explanation">
                <div>
                  <span className={`risk-status-dot state-${telemetry.state}`} />
                  <strong>{statusCopy.label}</strong>
                </div>
                <p>{recommendation}</p>
                {active && <small>{PRIMARY_COPY[telemetry.primary] || "Baseline stable"}</small>}
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
              <h2><Activity size={17} /> Last minute</h2>
              <div className="chart-legend" aria-label="Chart legend">
                <span><i className="legend-risk" /> Risk</span>
                <span><i className="legend-eye" /> Eye closure</span>
              </div>
            </div>
            <FocusTimeline history={history} />
            <div className="session-stats">
              <div><strong>{stats.blinks}</strong><span>Blinks</span></div>
              <div><strong>{stats.yawns}</strong><span>Yawns</span></div>
              <div><strong>{stats.distractions}</strong><span>Distractions</span></div>
              <div><strong>{stats.maxRisk}</strong><span>Peak risk</span></div>
            </div>
          </article>

          <article className="panel event-panel">
            <div className="panel-heading compact-heading">
              <h2><Siren size={17} /> Events</h2>
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
                  <strong>No events</strong>
                </div>
              )}
            </div>
          </article>
        </section>

        <footer>
          <p>Research prototype. Never use it as a substitute for rest or safe driving.</p>
        </footer>
      </div>

      {settingsOpen && (
        <div className="drawer-backdrop">
          <aside className="settings-drawer" role="dialog" aria-modal="true" aria-label="Detection settings">
            <div className="drawer-header">
              <h2>Settings</h2>
              <button className="icon-button" onClick={() => setSettingsOpen(false)} aria-label="Close settings"><X size={18} /></button>
            </div>

            <div className="settings-section">
              <span className="settings-label">ALERTS</span>
              <label className="toggle-row">
                <span><Volume2 size={17} /><span><strong>Alert tones</strong></span></span>
                <input type="checkbox" checked={settings.sound} onChange={(event) => setSettings((current) => ({ ...current, sound: event.target.checked }))} />
                <i />
              </label>
              <label className="toggle-row">
                <span><Siren size={17} /><span><strong>Voice warnings</strong></span></span>
                <input type="checkbox" checked={settings.voice} onChange={(event) => setSettings((current) => ({ ...current, voice: event.target.checked }))} />
                <i />
              </label>
              <div className="language-control">
                <label htmlFor="voice-language"><Languages size={17} /><span><strong>Language</strong></span></label>
                <select
                  id="voice-language"
                  value={settings.voiceLanguage}
                  onChange={(event) => setSettings((current) => ({ ...current, voiceLanguage: event.target.value as VoiceLanguage }))}
                >
                  {VOICE_LANGUAGES.map((language) => (
                    <option key={language.code} value={language.code}>{language.label} — {language.nativeLabel}</option>
                  ))}
                </select>
                <div className={`voice-status ${selectedVoice ? "voice-found" : "voice-browser"}`}>
                  <span className="status-dot" />
                  <span>
                    <strong>{selectedVoice?.name || "Browser-selected voice"}</strong>
                    <small>{selectedVoice ? "Available in this browser" : "Selected by language at playback"}</small>
                  </span>
                </div>
              </div>
              <button className="drawer-action" onClick={() => speakAlert("warning", true)}><Volume2 size={15} /> Test voice</button>
            </div>

            <div className="settings-section">
              <span className="settings-label">VISION</span>
              <label className="toggle-row">
                <span><Smartphone size={17} /><span><strong>Phone detection</strong></span></span>
                <input type="checkbox" checked={settings.phoneDetection} onChange={(event) => setSettings((current) => ({ ...current, phoneDetection: event.target.checked }))} />
                <i />
              </label>
              <label className="toggle-row">
                <span>{settings.privacyMode ? <CameraOff size={17} /> : <Camera size={17} />}<span><strong>Hide video</strong></span></span>
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
              <p className="settings-note"><Info size={14} /> Precision is enabled by default.</p>
            </div>

            <div className="privacy-proof">
              <LockKeyhole size={20} />
              <div><strong>Local processing</strong><span>No footage is saved.</span></div>
            </div>
          </aside>
        </div>
      )}

      {helpOpen && (
        <div className="modal-backdrop">
          <section className="guide-modal" role="dialog" aria-modal="true" aria-label="Quick start guide">
            <div className="drawer-header">
              <h2>Quick start</h2>
              <button className="icon-button" onClick={() => setHelpOpen(false)} aria-label="Close guide"><X size={18} /></button>
            </div>
            <div className="guide-steps">
              <div><span>1</span><div><strong>Camera at eye level</strong><p>Keep your face centered and well lit.</p></div></div>
              <div><span>2</span><div><strong>Look ahead</strong><p>Calibration takes five seconds.</p></div></div>
              <div><span>3</span><div><strong>Stop safely</strong><p>Never interact with this screen while driving.</p></div></div>
            </div>
            <div className="guide-warning">
              <AlertTriangle size={20} />
              <p>Research prototype, not certified automotive safety equipment.</p>
            </div>
            <button className="start-button full" onClick={() => { setHelpOpen(false); if (!active) void startCamera(); }}>
              {active ? "Done" : "Start monitoring"}
            </button>
          </section>
        </div>
      )}
    </main>
  );
}

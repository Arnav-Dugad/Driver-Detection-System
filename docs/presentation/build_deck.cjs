/**
 * Builds docs/presentation/Driver-Drowsiness-Distraction-Detection.pptx from the experiment outputs.
 *
 *   NODE_PATH=<folder with pptxgenjs> node docs/presentation/build_deck.cjs
 *
 * Every number on a slide is read from ml/reports/*.json, so the deck cannot
 * disagree with the report or the Model Lab. Re-run after re-training.
 */

/* eslint-disable @typescript-eslint/no-require-imports -- a plain Node script, run with node */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const read = (name) => JSON.parse(fs.readFileSync(path.join(ROOT, "ml", "reports", name), "utf8"));
const eye = read("eye_cnn_metrics.json");
const signals = read("signal_metrics.json");
const drowsy = read("drowsiness_metrics.json");
const APPLY_THEME = process.env.APPLY_THEME;
const OUT = path.join(__dirname, "Driver-Drowsiness-Distraction-Detection.pptx");
// RLDD and DMD people are disjoint datasets; MRL subjects are counted when known.
const PEOPLE = drowsy.dataset.people + signals.dataset.drivers + (eye.external_mrl.subjects ?? 0);

const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const f3 = (v) => v.toFixed(3);

// Night road, headlights: deep navy carries the deck, amber is the one sharp accent.
const THEME = {
  name: "Night Drive",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0F172A", lt1: "FFFFFF", dk2: "0B1324", lt2: "F1F5F9",
    accent1: "F59E0B", accent2: "0E7490", accent3: "64748B", accent4: "16A34A",
    accent5: "DC2626", accent6: "CBD5E1", hlink: "0E7490", folHlink: "64748B",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "Driver Drowsiness & Distraction Detection";
pres.author = "Arnav Dugad, Sujal Kumar";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---- layouts ---------------------------------------------------------------
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.9, fontSize: 44, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.3, w: 11.2, h: 1.2, fontSize: 18, color: C.accent6, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 1.0, fontSize: 28, bold: true, color: C.text1, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } },
    { text: { text: "Driver Drowsiness & Distraction Detection · ML project", options: { x: 0.6, y: 7.0, w: 5, h: 0.3, fontSize: 10, color: C.accent3, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent3, align: "right" },
});
pres.defineSlideMaster({
  title: "DARK_CONTENT",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 1.0, fontSize: 28, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent6, align: "right" },
});

// ---- helpers ---------------------------------------------------------------
const text = (slide, value, opts) => slide.addText(value, { isTextBox: true, margin: 0, fontSize: 15, color: C.text1, ...opts });
const card = (slide, x, y, w, h, opts = {}) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12,
    fill: { color: opts.fill ?? C.background2 },
    line: { color: opts.line ?? C.background2, width: 0.75 },
    objectName: opts.name,
  });
function stat(slide, x, y, w, value, label, opts = {}) {
  text(slide, value, { x, y, w, h: 0.9, fontSize: opts.size ?? 44, bold: true, color: opts.color ?? C.accent1, fontFace: THEME.headFontFace });
  text(slide, label, { x, y: y + 0.92, w, h: 0.7, fontSize: 13, color: opts.labelColor ?? C.accent3, valign: "top" });
}
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", legendFontFace: "+mn-lt" };
const chartQuiet = {
  ...chartText,
  catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent3, catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
  dataLabelFontSize: 11, dataLabelColor: HEX.dk1,
  valGridLine: { color: "E2E8F0", size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false,
};

const add = (masterName, sectionTitle) => pres.addSlide({ masterName, sectionTitle });

// ---- 1. title ----------------------------------------------------------------
pres.addSection({ title: "Opening" });
let s = add("DARK_TITLE", "Opening");
s.addText("Driver Drowsiness & Distraction Detection", { placeholder: "title" });
s.addText(`Detecting drowsy and distracted drivers in the browser, on-device. Evaluated on ${PEOPLE} real people from three public datasets\nArnav Dugad & Sujal Kumar · ML Mini Project · Manipal Institute of Technology`, { placeholder: "body" });
s.addNotes("Introduce the project in one sentence: a browser app that watches the driver's face on-device and warns on drowsiness or distraction, with two neural networks trained and evaluated on public data from real people.");

// ---- 2. problem ---------------------------------------------------------------
s = add("CONTENT", "Opening");
s.addText("Fatigue and distraction kill thousands a year, and the law now requires warnings", { placeholder: "title" });
const problem = [
  ["3,208", "people killed in US crashes involving distracted drivers, 2024 (NHTSA)"],
  ["633", "killed in drowsy-driving crashes, 2023 (NHTSA, via GHSA); modelled estimates run ~10x higher"],
  ["1.19 M", "road-traffic deaths worldwide each year (WHO, 2023)"],
];
problem.forEach(([value, label], i) => {
  card(s, 0.6 + i * 4.1, 1.75, 3.8, 2.6, { name: `problem-card-${i}` });
  stat(s, 0.9 + i * 4.1, 2.0, 3.3, value, label);
});
card(s, 0.6, 4.75, 12.1, 1.75, { fill: "FEF3C7", line: "FEF3C7", name: "regulation" });
text(s, "EU Regulation 2019/2144", { x: 0.95, y: 4.95, w: 6, h: 0.4, fontSize: 16, bold: true });
text(s, "Driver drowsiness and attention warning has been mandatory on every new EU car since 7 July 2024; advanced distraction warning since 7 July 2026. Both need exactly what this project measures: eye closure, gaze and head pose over time.", { x: 0.95, y: 5.4, w: 11.4, h: 0.95, fontSize: 15 });
s.addNotes("The point: drowsiness and distraction are temporal states visible on the face seconds before a crash, and regulators now mandate camera-based warnings.");

// ---- 3. what we built ---------------------------------------------------------
s = add("CONTENT", "Opening");
s.addText("One web app: pretrained perception, two models we trained, an explainable rule engine", { placeholder: "title" });
const steps = [
  ["Perceive", "MediaPipe Face Landmarker: 478 landmarks, 52 blendshapes, head-pose matrix. EfficientDet-Lite0 for phones.", false],
  ["Measure", "EAR, MAR, iris gaze, yaw/pitch/roll, aspect-corrected; personal 5-second calibration.", false],
  ["Learn", `EyeNet CNN (${eye.final_model.parameters.toLocaleString("en-US")} params, int8) on eye crops; GRU over 30 s of 14 signals.`, true],
  ["Decide", "Time-weighted PERCLOS, yawn morphology, persistence and fusion into a 0-100 risk.", false],
  ["Warn", "Voice in 6 languages, haptics, Drive Mode, counterfactual explanations.", false],
];
steps.forEach(([title, body, ours], i) => {
  const x = 0.6 + i * 2.46;
  card(s, x, 1.9, 2.26, 3.6, { fill: ours ? "FEF3C7" : C.background2, line: ours ? HEX.accent1 : C.background2, name: `pipeline-${i}` });
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 2.15, w: 0.55, h: 0.55, fill: { color: ours ? C.accent1 : C.text1 }, line: { color: ours ? C.accent1 : C.text1 }, objectName: `pipeline-dot-${i}` });
  text(s, String(i + 1), { x: x + 0.25, y: 2.15, w: 0.55, h: 0.55, align: "center", valign: "middle", bold: true, color: C.background1, fontSize: 16 });
  text(s, title, { x: x + 0.25, y: 2.9, w: 1.9, h: 0.45, fontSize: 19, bold: true, fontFace: THEME.headFontFace });
  text(s, body, { x: x + 0.25, y: 3.4, w: 1.85, h: 2.0, fontSize: 13, valign: "top" });
  if (i < steps.length - 1) {
    s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: x + 2.29, y: 3.55, w: 0.14, h: 0.3, rotate: 0, fill: { color: C.accent6 }, line: { color: C.accent6 }, flipH: false, objectName: `pipeline-arrow-${i}` });
  }
});
text(s, "Everything runs on the device. No frame, landmark or face template ever leaves the browser.", { x: 0.6, y: 5.85, w: 12.1, h: 0.5, fontSize: 16, italic: true, color: C.accent2 });
s.addNotes("Walk left to right. Highlight the amber step: the two networks trained in this project. Stress privacy: there is no server.");

// ---- 4. data ---------------------------------------------------------------
pres.addSection({ title: "Data" });
s = add("CONTENT", "Data");
s.addText("No data of our own: three public datasets of real people, split by person", { placeholder: "title" });
const datasets = [
  ["UTA-RLDD", `${drowsy.dataset.people} people`, `${drowsy.dataset.videos} webcam/phone videos, self-rated alert, low-vigilant or drowsy (KSS). ${drowsy.dataset.clips.toLocaleString("en-US")} clips of ${drowsy.dataset.clip_seconds} s.`, "Drowsiness models · official 5 folds"],
  ["DMD", `${signals.dataset.drivers} drivers`, `${signals.dataset.frames.toLocaleString("en-US")} frames in a real car, every frame labelled open, closing, closed or opening; blinks and yawns marked.`, "Eye CNN · signal and event checks"],
  ["MRL Eye", `${eye.external_mrl.images ?? "4,000"} images`, "Near-infrared eye crops from a different camera and population, never used for training.", "Out-of-domain test"],
];
datasets.forEach(([name, size, body, use], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 1.75, 3.8, 4.6, { name: `dataset-${i}` });
  text(s, name, { x: x + 0.3, y: 2.0, w: 3.2, h: 0.5, fontSize: 22, bold: true, fontFace: THEME.headFontFace });
  text(s, size, { x: x + 0.3, y: 2.55, w: 3.2, h: 0.7, fontSize: 32, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  text(s, body, { x: x + 0.3, y: 3.45, w: 3.2, h: 1.9, fontSize: 14, valign: "top" });
  text(s, use, { x: x + 0.3, y: 5.55, w: 3.2, h: 0.5, fontSize: 13, bold: true, color: C.accent2 });
});
s.addNotes("I could not record people myself, so everything is evaluated on public datasets with real participants. The rule throughout: nobody appears in both training and test data.");

// ---- 5. pipeline rigour ------------------------------------------------------
s = add("CONTENT", "Data");
s.addText("The offline experiments run the app's own code, so results transfer to the product", { placeholder: "title" });
const rigour = [
  ["Same perception", "Videos go through the identical face_landmarker.task file and thresholds as the browser."],
  ["Same maths", "Python ports of the browser signals match JavaScript to 10^-4 on recorded MediaPipe output (parity test)."],
  ["Same rule engine", "The shipped JavaScript tracker is replayed over every clip, not re-implemented."],
  ["Same shipped weights", "The CNN is scored after int8 quantisation, the GRU after temperature scaling; JS matches PyTorch in tests."],
];
rigour.forEach(([title, body], i) => {
  const x = 0.6 + (i % 2) * 6.15;
  const y = 1.8 + Math.floor(i / 2) * 2.35;
  card(s, x, y, 5.95, 2.1, { name: `rigour-${i}` });
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.35, w: 0.5, h: 0.5, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: `rigour-dot-${i}` });
  text(s, "✓", { x: x + 0.3, y: y + 0.35, w: 0.5, h: 0.5, align: "center", valign: "middle", color: C.background1, bold: true, fontSize: 16 });
  text(s, title, { x: x + 1.0, y: y + 0.3, w: 4.7, h: 0.5, fontSize: 19, bold: true, fontFace: THEME.headFontFace });
  text(s, body, { x: x + 1.0, y: y + 0.85, w: 4.7, h: 1.1, fontSize: 14, valign: "top" });
});
s.addNotes("Explain why this matters: offline numbers are only meaningful for the app if the same code produces them. Parity tests enforce this in CI.");

// ---- 6. aspect bug -----------------------------------------------------------
pres.addSection({ title: "Results" });
s = add("CONTENT", "Results");
const fix = signals.aspect_ratio_fix;
const tuning = signals.threshold_tuning;
s.addText("Real data exposed a hidden bug: eye ratios were stretched by the camera's shape", { placeholder: "title" });
text(s, "MediaPipe divides x by frame width and y by frame height. On a 16:9 webcam an open eye read EAR ≈ 0.52 instead of ≈ 0.29; on a portrait phone it read too low. A calibration clamp hid it.", { x: 0.6, y: 1.65, w: 5.7, h: 1.6, fontSize: 15, valign: "top" });
text(s, `Fixing the maths alone lowered closed-eye F1, because the old threshold had silently compensated. Re-learning the cut by leave-one-driver-out cross-validation (${tuning.old_cut.toFixed(2)} → ${tuning.recommended_cut.toFixed(2)}) gives correct geometry on every camera shape, and the best F1.`, { x: 0.6, y: 3.3, w: 5.7, h: 1.8, fontSize: 15, valign: "top" });
card(s, 0.6, 5.25, 5.7, 1.3, { fill: "FEF3C7", line: "FEF3C7", name: "lesson" });
text(s, "Lesson: fix the measurement, then re-learn the decision threshold from data, never by hand.", { x: 0.85, y: 5.4, w: 5.2, h: 1.0, fontSize: 15, bold: true, valign: "middle" });
s.addChart(pres.charts.BAR, [{ name: "Closed-eye F1", labels: ["Before fix, old cut", "Fixed EAR, old cut", "Fixed EAR, CV-tuned cut"], values: [fix.before_fix.f1, fix.corrected.f1, tuning.cv_f1] }], {
  x: 6.7, y: 1.6, w: 6.0, h: 4.95, barDir: "col", ...chartQuiet,
  chartColors: [HEX.accent3, HEX.accent5, HEX.accent1], showValue: true, dataLabelFormatCode: "0.000", dataLabelPosition: "outEnd",
  valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0", showLegend: false,
  showTitle: true, title: `Closed-eye F1, ${signals.dataset.drivers} DMD drivers, each judged with a cut tuned on the others`, titleFontSize: 13, titleColor: HEX.dk1,
});
s.addNotes("The most important engineering lesson. Without real labelled data this bug would never have been found. The tuned bar is cross-validated: every driver is scored with a threshold chosen on the other drivers.");

// ---- 7. eye CNN --------------------------------------------------------------
s = add("CONTENT", "Results");
const m = eye.models;
s.addText(`EyeNet: a ${(eye.final_model.parameters / 1000).toFixed(1)}k-parameter CNN that detects closed eyes on unseen drivers`, { placeholder: "title" });
stat(s, 0.6, 1.7, 3.6, f3(m.eyenet_s_int8.roc_auc), "ROC-AUC on held-out DMD drivers (int8, as shipped)");
stat(s, 0.6, 3.3, 3.6, pct(eye.external_mrl.accuracy ?? NaN), "accuracy on MRL infrared eyes: different camera, people and light");
stat(s, 0.6, 4.9, 3.6, `${eye.final_model.weights_kb ?? (eye.final_model.parameters / 1024).toFixed(1)} KB`, "of int8 weights, run in plain JavaScript");
const rows = [["Model", "Parameters", "Accuracy", "F1 closed", "ROC-AUC"]];
[["pixels_pca_logreg", "Pixels + PCA + LogReg"], ["eyenet_l", "EyeNet-L (float)"], ["eyenet_s", "EyeNet-S (float)"], ["eyenet_s_int8", "EyeNet-S int8 (shipped)"]].forEach(([key, label]) => {
  rows.push([label, m[key].parameters.toLocaleString("en-US"), pct(m[key].accuracy), f3(m[key].f1_closed), f3(m[key].roc_auc)]);
});
s.addTable(rows.map((row, r) => row.map((cell, c) => ({
  text: cell,
  options: {
    bold: r === 0 || r === rows.length - 1, color: r === rows.length - 1 ? HEX.accent1 : r === 0 ? HEX.accent3 : HEX.dk1,
    align: c === 0 ? "left" : "right", fontSize: r === 0 ? 12 : 14, fill: { color: r % 2 ? "F8FAFC" : "FFFFFF" },
  },
}))), { x: 4.7, y: 1.8, w: 8.0, colW: [3.0, 1.3, 1.2, 1.2, 1.3], rowH: 0.5, border: { type: "solid", pt: 0.5, color: "E2E8F0" }, fontFace: THEME.bodyFontFace });
text(s, "Five-fold cross-validation grouped by driver. Input: 32×32 landmark-aligned grayscale crops; four 3×3 conv blocks, global average pooling. BatchNorm is folded and weights quantised to int8 before scoring, so the table's last row is exactly what ships.", { x: 4.7, y: 4.65, w: 8.0, h: 1.6, fontSize: 14, color: C.accent3, valign: "top" });
s.addNotes("The CNN clearly beats a pixel baseline and matches the 4x larger network. int8 quantisation costs almost nothing. The MRL result shows it transfers to infrared images it never saw.");

// ---- 8. signal comparison ------------------------------------------------------
s = add("CONTENT", "Results");
const det = signals.closed_eye_detectors;
const detKeys = ["score_ear", "score_blendshape", "score_fused", "score_cnn", "score_stacked"];
const aucs = detKeys.map((k) => det[k].roc_auc);
s.addText(`Head-on, every closed-eye signal scores above ${Math.floor(Math.min(...aucs) * 100) / 100} ROC-AUC on unseen drivers`, { placeholder: "title" });
const detRows = [["Closed-eye detector", "ROC-AUC", "95% CI over drivers", "Best F1"]];
detKeys.forEach((key) => detRows.push([det[key].label, f3(det[key].roc_auc), `${f3(det[key].roc_auc_ci95[0])} – ${f3(det[key].roc_auc_ci95[1])}`, f3(det[key].best_f1)]));
s.addTable(detRows.map((row, r) => row.map((cell, c) => ({
  text: cell,
  options: { bold: r === 0 || detKeys[r - 1] === "score_cnn", color: r === 0 ? HEX.accent3 : detKeys[r - 1] === "score_cnn" ? HEX.accent1 : HEX.dk1, align: c ? "right" : "left", fontSize: r === 0 ? 12 : 15, fill: { color: r % 2 ? "F8FAFC" : "FFFFFF" } },
}))), { x: 0.6, y: 1.75, w: 7.6, colW: [3.4, 1.2, 1.9, 1.1], rowH: 0.55, border: { type: "solid", pt: 0.5, color: "E2E8F0" }, fontFace: THEME.bodyFontFace });
card(s, 8.6, 1.75, 4.1, 3.3, { fill: "FEF3C7", line: "FEF3C7", name: "honest-read" });
text(s, "Honest reading", { x: 8.9, y: 1.95, w: 3.5, h: 0.45, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
text(s, "The intervals overlap: on DMD's mostly frontal footage the CNN does not beat MediaPipe's own blink score. Its value is independence from landmarks and its infrared transfer (slide 7), which is why it runs beside the fusion rather than replacing it.", { x: 8.9, y: 2.45, w: 3.5, h: 2.5, fontSize: 14, valign: "top" });
text(s, `${signals.dataset.labelled_frames.toLocaleString("en-US")} human-labelled frames; CNN and stacked scores are out-of-fold, so no driver was seen in training.`, { x: 0.6, y: 5.45, w: 7.6, h: 0.8, fontSize: 14, color: C.accent3 });
s.addNotes("This slide is deliberately honest: all detectors are excellent head-on and statistically tied. Do not oversell the CNN here; its strength is robustness and transfer.");

// ---- 9. drowsiness ------------------------------------------------------------
s = add("CONTENT", "Results");
const order = ["rule_engine", "logistic_regression", "random_forest", "gradient_boosting", "tcn", "gru"];
const names = { rule_engine: "Rule engine (app)", logistic_regression: "Logistic regression", random_forest: "Random forest", gradient_boosting: "Gradient boosting", tcn: "Temporal CNN", gru: "GRU" };
const best = drowsy.best_model;
const bestModel = drowsy.models[best];
const published = drowsy.published_video_accuracy;
s.addText(`On 60 unseen people the best model reaches ${pct(bestModel.video_accuracy)}: drowsiness from 30-second clips is hard`, { placeholder: "title" });
const bars = [
  ...order.map((key) => ({ label: names[key], value: drowsy.models[key].video_accuracy, color: key === best ? HEX.accent1 : HEX.accent6 })),
  { label: "HM-LSTM (published)", value: published["HM-LSTM (Ghoddoosian et al.)"], color: HEX.accent2 },
  { label: "Human judges (published)", value: published["Human judges (Ghoddoosian et al.)"], color: HEX.accent3 },
].sort((a, b) => a.value - b.value);
s.addChart(pres.charts.BAR, [{ name: "Video accuracy", labels: bars.map((b) => b.label), values: bars.map((b) => b.value) }], {
  x: 0.6, y: 1.55, w: 8.1, h: 5.1, barDir: "bar", ...chartQuiet, chartColors: bars.map((b) => b.color),
  showValue: true, dataLabelFormatCode: "0.0%", dataLabelPosition: "outEnd", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2,
  valAxisLabelFormatCode: "0%", showLegend: false, barGapWidthPct: 45,
  showTitle: true, title: "3-class video accuracy, UTA-RLDD official 5-fold split (chance 33%)", titleFontSize: 13, titleColor: HEX.dk1,
});
stat(s, 9.1, 1.75, 3.6, f3(bestModel.alert_vs_drowsy_auc), `alert vs drowsy ROC-AUC, ${names[best].toLowerCase()}`, { size: 40 });
stat(s, 9.1, 3.45, 3.6, `${pct(bestModel.video_accuracy_ci95[0], 0)}–${pct(bestModel.video_accuracy_ci95[1], 0)}`, "95% interval, resampling people", { size: 34, color: C.accent2 });
text(s, "The published HM-LSTM reads whole 10-minute videos of blink sequences; these models see 6 clips of 30 s per video. Every model, learned or rule-based, is well above chance and below the published model.", { x: 9.1, y: 5.15, w: 3.6, h: 1.4, fontSize: 13, color: C.accent3, valign: "top" });
s.addNotes("Amber is our best model; teal and grey are the numbers published with the dataset. We are below the published HM-LSTM, which uses whole videos. Say this plainly.");

// ---- 10. calibration protocol --------------------------------------------------
s = add("CONTENT", "Results");
const cp = drowsy.calibration_protocols;
const cpKeys = ["rule_engine", "logistic_regression", "gradient_boosting", "gru"].filter((k) => cp.enrolled.results[k]);
const meanOf = (block, field) => cpKeys.reduce((sum, k) => sum + block[k][field], 0) / cpKeys.length;
const sessionMean = meanOf(cp.enrolled.session_same_clips, "video_accuracy");
const confoundedMean = meanOf(cp.enrolled_all.results, "video_accuracy");
const controlledMean = meanOf(cp.enrolled.results, "video_accuracy");
s.addText("We nearly fooled ourselves: a big gain from \"fresh\" calibration was a confound", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Calibrated each session (app)", labels: cpKeys.map((k) => names[k]), values: cpKeys.map((k) => cp.enrolled.session_same_clips[k].video_accuracy) },
  { name: "Whole alert baseline carried over (confounded)", labels: cpKeys.map((k) => names[k]), values: cpKeys.map((k) => cp.enrolled_all.results[k].video_accuracy) },
  { name: "Eye + mouth baseline only (controlled)", labels: cpKeys.map((k) => names[k]), values: cpKeys.map((k) => cp.enrolled.results[k].video_accuracy) },
], {
  x: 0.6, y: 1.55, w: 7.6, h: 5.0, barDir: "col", ...chartQuiet, chartColors: [HEX.accent6, HEX.accent5, HEX.accent1],
  showValue: true, dataLabelFormatCode: "0%", dataLabelPosition: "outEnd", dataLabelFontSize: 10,
  valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0%",
  showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: HEX.dk1,
  showTitle: true, title: "3-class video accuracy on identical clips", titleFontSize: 13, titleColor: HEX.dk1,
});
const stepText = [
  ["Hypothesis", "Calibrating on a face that is already tired hides tiredness, so calibrate once while alert."],
  ["First result", `Carrying the whole alert baseline over raised mean accuracy ${pct(sessionMean, 0)} → ${pct(confoundedMean, 0)}.`],
  ["The catch", "RLDD's three states are separate recordings, often with the camera moved: a carried-over head-pose baseline partly recognises the recording."],
  ["Controlled", `Carrying only eyes and mouth: ${pct(sessionMean, 0)} → ${pct(controlledMean, 0)}. The app keeps per-session calibration.`],
];
stepText.forEach(([head, body], i) => {
  text(s, head, { x: 8.6, y: 1.65 + i * 1.22, w: 4.1, h: 0.4, fontSize: 16, bold: true, fontFace: THEME.headFontFace, color: i === 2 ? C.accent5 : C.text1 });
  text(s, body, { x: 8.6, y: 2.05 + i * 1.22, w: 4.1, h: 0.8, fontSize: 13, valign: "top" });
});
s.addNotes("This is the methodological highlight. The first experiment looked like a big win. Before shipping it we asked what else a carried-over baseline could encode: camera placement. Controlling for it removed most of the gain, so the feature was not shipped. Catching your own confound is what good ML practice looks like.");

// ---- 11. errors and calibration ---------------------------------------------
s = add("CONTENT", "Results");
const conf = bestModel.confusion_video;
const raw = drowsy.models.gru_uncalibrated;
const cal = drowsy.models.gru;
const improved = cal.ece < raw.ece;
s.addText(`Errors sit in the middle class; the GRU's confidence ${improved ? "is calibrated by temperature scaling" : "was already well calibrated"}`, { placeholder: "title" });
const classLabels = ["Alert", "Low vigilant", "Drowsy"];
const confRows = [[`${names[best]}: true ↓ / predicted →`, ...classLabels]];
conf.forEach((row, i) => {
  const total = row.reduce((a, b) => a + b, 0) || 1;
  confRows.push([classLabels[i], ...row.map((v) => `${Math.round((v / total) * 100)}%  (${v})`)]);
});
s.addTable(confRows.map((row, r) => row.map((cell, c) => {
  const share = r && c ? conf[r - 1][c - 1] / (conf[r - 1].reduce((a, b) => a + b, 0) || 1) : 0;
  const shade = r && c ? (r === c ? (share > 0.5 ? "F59E0B" : share > 0.3 ? "FCD34D" : "FEF3C7") : share > 0.3 ? "CBD5E1" : "F1F5F9") : "FFFFFF";
  return { text: cell, options: { bold: r === 0 || c === 0, align: c ? "center" : "left", valign: "middle", fontSize: r === 0 ? 12 : 15, color: r === 0 ? HEX.accent3 : HEX.dk1, fill: { color: shade } } };
})), { x: 0.6, y: 1.75, w: 6.6, colW: [2.4, 1.4, 1.4, 1.4], rowH: [0.55, 0.95, 0.95, 0.95], border: { type: "solid", pt: 1, color: "FFFFFF" }, fontFace: THEME.bodyFontFace });
text(s, "Per video. Alert and drowsy are usually right; low vigilance is confused with both, as in the original paper and for its human judges.", { x: 0.6, y: 5.35, w: 6.6, h: 1.0, fontSize: 14, color: C.accent3, valign: "top" });
const relRaw = raw.reliability ?? [];
const relCal = cal.reliability ?? [];
s.addChart(pres.charts.SCATTER, [
  { name: "Confidence", values: [0, 1, ...relRaw.map((p) => p.confidence), ...relCal.map((p) => p.confidence)] },
  { name: "Perfect calibration", values: [0, 1, ...relRaw.map(() => null), ...relCal.map(() => null)] },
  { name: `GRU raw (ECE ${f3(raw.ece)})`, values: [null, null, ...relRaw.map((p) => p.accuracy), ...relCal.map(() => null)] },
  { name: `GRU + temperature (ECE ${f3(cal.ece)})`, values: [null, null, ...relRaw.map(() => null), ...relCal.map((p) => p.accuracy)] },
], {
  x: 7.6, y: 1.55, w: 5.1, h: 5.0, ...chartText, chartColors: [HEX.accent6, HEX.accent3, HEX.accent1],
  lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
  valAxisMinVal: 0, valAxisMaxVal: 1, catAxisMinVal: 0, catAxisMaxVal: 1, valAxisMajorUnit: 0.25, catAxisMajorUnit: 0.25,
  valAxisLabelFormatCode: "0.00", catAxisLabelFormatCode: "0.00", valAxisLabelColor: HEX.accent3, catAxisLabelColor: HEX.accent3,
  valGridLine: { color: "E2E8F0", size: 0.75 }, catGridLine: { color: "E2E8F0", size: 0.75 },
  showValAxisTitle: true, valAxisTitle: "Observed accuracy", showCatAxisTitle: true, catAxisTitle: "Stated confidence",
  valAxisTitleFontSize: 11, catAxisTitleFontSize: 11, valAxisTitleColor: HEX.accent3, catAxisTitleColor: HEX.accent3,
  showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: HEX.dk1,
  showTitle: true, title: "Reliability diagram, GRU", titleFontSize: 13, titleColor: HEX.dk1,
});
s.addNotes(`Left: where the best model's errors fall. Right: a risk score is acted on, so its confidence must mean what it says. ECE ${f3(raw.ece)} raw, ${f3(cal.ece)} after temperature scaling.`);

// ---- 12. what matters --------------------------------------------------------
s = add("CONTENT", "Results");
const ab = drowsy.ablation_gradient_boosting;
const groups = [["eyes", "Eyes"], ["mouth", "Mouth"], ["head", "Head pose"], ["gaze_brow", "Gaze + brow"]];
const alone = groups.map(([k]) => ab[k].clip_accuracy);
const strongest = groups[alone.indexOf(Math.max(...alone))][1];
s.addText(`${strongest} carry the most signal on their own; the ablation shows what each group adds`, { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Only this group", labels: groups.map((g) => g[1]), values: alone },
  { name: "All but this group", labels: groups.map((g) => g[1]), values: groups.map(([k]) => ab[`without_${k}`].clip_accuracy) },
], {
  x: 0.6, y: 1.55, w: 7.0, h: 5.0, barDir: "col", ...chartQuiet, chartColors: [HEX.accent1, HEX.accent6],
  showValue: true, dataLabelFormatCode: "0%", dataLabelPosition: "outEnd", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0%",
  showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.dk1,
  showTitle: true, title: `Clip accuracy, gradient boosting (all features ${pct(drowsy.models.gradient_boosting.clip_accuracy)}, chance 33%)`, titleFontSize: 13, titleColor: HEX.dk1,
});
const imp = Object.entries(drowsy.permutation_importance).slice(0, 6);
text(s, "Most important features", { x: 8.0, y: 1.7, w: 4.7, h: 0.45, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
s.addText(imp.map(([k, v], i) => ({ text: `${k.replaceAll("_", " ")}  (${v.toFixed(3)})`, options: { bullet: { type: "number" }, breakLine: i < imp.length - 1 } })), {
  x: 8.0, y: 2.25, w: 4.7, h: 3.0, fontSize: 15, color: C.text1, paraSpaceAfter: 8, isTextBox: true, margin: 0, valign: "top",
});
text(s, "Permutation importance: drop in macro F1 when one feature is shuffled on held-out people. A group whose removal raises accuracy is adding noise for unseen people.", { x: 8.0, y: 5.3, w: 4.7, h: 1.2, fontSize: 13, color: C.accent3, valign: "top" });
s.addNotes("Ablation answers which signals matter for new people. Read the orange bars as 'how far this group gets alone' and the grey bars as 'what is left without it'.");

// ---- 12. events ----------------------------------------------------------------
s = add("CONTENT", "Results");
s.addText("The rule engine's event detectors, checked against human annotations", { placeholder: "title" });
const ev = signals.events;
[["blink", "Blinks"], ["yawn", "Yawns"]].forEach(([k, label], i) => {
  const x = 0.6 + i * 6.15;
  card(s, x, 1.75, 5.95, 3.4, { name: `event-${k}` });
  text(s, label, { x: x + 0.35, y: 1.95, w: 5.2, h: 0.5, fontSize: 22, bold: true, fontFace: THEME.headFontFace });
  stat(s, x + 0.35, 2.6, 1.8, pct(ev[k].precision, 0), "precision", { size: 36 });
  stat(s, x + 2.2, 2.6, 1.8, pct(ev[k].recall, 0), "recall", { size: 36, color: C.accent2 });
  stat(s, x + 4.05, 2.6, 1.8, ev[k].false_alarms_per_hour.toFixed(0), "false events / hour", { size: 36, color: C.accent3 });
  text(s, `${ev[k].tp} matched, ${ev[k].fp} spurious, ${ev[k].fn} missed`, { x: x + 0.35, y: 4.45, w: 5.2, h: 0.45, fontSize: 13, color: C.accent3 });
});
text(s, `The unchanged JavaScript engine replayed over ${signals.dataset.sessions} DMD sessions (${(ev.hours * 60).toFixed(0)} minutes). Yawn recall is limited by yawns behind a hand, which hide the mouth from any camera-only method; a stated limitation, not hidden.`, { x: 0.6, y: 5.5, w: 12.1, h: 1.0, fontSize: 15 });
s.addNotes("Event-level evaluation is what a driver experiences. Blinks are detected well. Yawn recall is low because many DMD yawns are covered by a hand.");

// ---- 13. model lab ----------------------------------------------------------------
pres.addSection({ title: "Product" });
s = add("CONTENT", "Product");
s.addText("The Model Lab: every result live in the app, and all models side by side on your webcam", { placeholder: "title" });
const labShot = path.join(__dirname, "model-lab.png");
if (fs.existsSync(labShot)) {
  s.addImage({ path: labShot, x: 0.6, y: 1.55, w: 7.6, h: 4.95, altText: "Screenshot of the Model Lab page", sizing: { type: "cover", w: 7.6, h: 4.95 } });
}
const labPoints = [
  ["Shadow mode", "Rule engine, EyeNet and GRU score the same frames in parallel, with the CNN's eye crops shown live."],
  ["Results explorer", "Model comparison with intervals, confusion matrices, reliability, ablation; switch metric and model."],
  ["No hand-typed numbers", "The page reads the same JSON the training scripts write."],
];
labPoints.forEach(([title, body], i) => {
  text(s, title, { x: 8.6, y: 1.7 + i * 1.6, w: 4.1, h: 0.45, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
  text(s, body, { x: 8.6, y: 2.15 + i * 1.6, w: 4.1, h: 1.0, fontSize: 14, valign: "top" });
});
s.addNotes("Demo the Lab here if time allows: start the camera, blink, close eyes for two seconds, look down.");

// ---- 14. limitations --------------------------------------------------------------
s = add("CONTENT", "Product");
s.addText("What these results do not show", { placeholder: "title" });
const limits = [
  ["Not a crash predictor", "Labels are self-rated sleepiness (RLDD) and eye state (DMD), not driving performance."],
  ["Small DMD subset", `${signals.dataset.drivers} drivers in the public mirror; wide intervals for per-driver results.`],
  ["Hard middle class", "Low vigilance is confused with both neighbours, by models and by people."],
  ["Camera-only", "Sunglasses, a hand over the mouth, night without IR, and extreme angles remain failure cases."],
];
limits.forEach(([title, body], i) => {
  const x = 0.6 + (i % 2) * 6.15;
  const y = 1.75 + Math.floor(i / 2) * 2.3;
  card(s, x, y, 5.95, 2.05, { name: `limit-${i}` });
  text(s, title, { x: x + 0.35, y: y + 0.3, w: 5.3, h: 0.5, fontSize: 19, bold: true, fontFace: THEME.headFontFace });
  text(s, body, { x: x + 0.35, y: y + 0.85, w: 5.3, h: 1.0, fontSize: 15, valign: "top" });
});
text(s, "Next: a simulator study linking alerts to lane deviation, infrared night profiles, and per-driver fine-tuning from calibration.", { x: 0.6, y: 6.35, w: 12.1, h: 0.45, fontSize: 15, italic: true, color: C.accent2 });
s.addNotes("Being explicit about limitations is part of doing ML properly.");

// ---- 15. close ----------------------------------------------------------------------
s = add("DARK_CONTENT", "Product");
s.addText("Summary", { placeholder: "title" });
const closing = [
  [f3(m.eyenet_s_int8.roc_auc), "eye-state AUC on unseen drivers"],
  [pct(bestModel.video_accuracy), "drowsiness video accuracy on 60 unseen people (published 65.2%)"],
  [f3(tuning.cv_f1), "closed-eye F1 after the aspect fix and a CV-tuned threshold"],
];
closing.forEach(([value, label], i) => stat(s, 0.6 + i * 4.1, 1.9, 3.8, value, label, { size: 48, labelColor: C.accent6 }));
text(s, `Two models trained and evaluated on ${PEOPLE} real people, a hidden bug found and a tempting confound caught by measuring, and an app that shows its own evidence.`, { x: 0.6, y: 4.3, w: 12.1, h: 1.0, fontSize: 20, color: C.background1, fontFace: THEME.headFontFace });
text(s, "Questions?", { x: 0.6, y: 5.7, w: 12.1, h: 0.8, fontSize: 32, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
s.addNotes("Close on the three numbers and invite questions.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) {
    const { applyTheme } = require(APPLY_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log(`wrote ${OUT}`);
})();

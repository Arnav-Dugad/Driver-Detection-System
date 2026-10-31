/**
 * Live input channels and a plain-JavaScript GRU for the temporal drowsiness
 * model trained by ml/train_drowsiness.py on UTA-RLDD.
 *
 * SEQUENCE_CHANNELS and sequenceFrame() mirror ml/sequence_features.py. The
 * model sees 30 seconds at 15 Hz; createSequenceBuffer() resamples the app's
 * variable camera rate onto that fixed grid by sample-and-hold, which is what
 * ffmpeg's fps filter did to the training videos.
 */

export const SEQUENCE_CHANNELS = [
  "closure",
  "ear_rel",
  "blink",
  "squint",
  "wide",
  "mar_rel",
  "jaw_open",
  "yaw_rel",
  "pitch_rel",
  "roll_rel",
  "gaze_rel",
  "gaze_bs",
  "brow_inner_up",
  "face_found",
];
export const SEQUENCE_HZ = 15;

/** One channel vector from a frame's signals, its fused closure, and the baseline. */
export function sequenceFrame(signals, baseline, closure) {
  const vector = new Float32Array(SEQUENCE_CHANNELS.length);
  if (!signals) return vector;
  const shapes = signals.blendshapes ?? {};
  vector[0] = closure ?? 0;
  vector[1] = signals.ear / Math.max(0.001, baseline.ear);
  vector[2] = shapes.blink ?? 0;
  vector[3] = shapes.squint ?? 0;
  vector[4] = shapes.wide ?? 0;
  vector[5] = signals.mar - baseline.mar;
  vector[6] = shapes.jawOpen ?? 0;
  vector[7] = signals.yaw - (baseline.yaw ?? 0);
  vector[8] = signals.pitch - (baseline.pitch ?? 0);
  vector[9] = (signals.roll ?? 0) - (baseline.roll ?? 0);
  vector[10] = signals.gaze - (baseline.gaze ?? 0.5);
  vector[11] = (shapes.gaze ?? 0.5) - 0.5;
  vector[12] = shapes.browInnerUp ?? 0;
  vector[13] = 1;
  return vector;
}

/** Fixed-rate rolling buffer of channel vectors. */
export function createSequenceBuffer({ seconds = 30, hz = SEQUENCE_HZ } = {}) {
  const capacity = Math.round(seconds * hz);
  const step = 1000 / hz;
  let frames = [];
  let nextTime = null;
  let latest = new Float32Array(SEQUENCE_CHANNELS.length);
  return {
    push(time, vector) {
      latest = vector;
      if (nextTime === null) nextTime = time;
      // Emit every grid point that has elapsed, holding the latest vector.
      while (time >= nextTime) {
        frames.push(latest);
        nextTime += step;
      }
      if (frames.length > capacity) frames = frames.slice(frames.length - capacity);
    },
    get full() {
      return frames.length >= capacity;
    },
    get length() {
      return frames.length;
    },
    frames: () => frames,
    reset() {
      frames = [];
      nextTime = null;
    },
  };
}

export function isSequenceModel(model) {
  return Boolean(
    model &&
      model.format === "drowsiness-gru" &&
      Array.isArray(model.channels) &&
      model.channels.join() === SEQUENCE_CHANNELS.join() &&
      model.gru?.weight_ih,
  );
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** Expand the JSON payload into typed arrays once. */
export function prepareSequenceModel(model) {
  if (!isSequenceModel(model)) return null;
  const f32 = (values) => Float32Array.from(values.flat(Infinity));
  return {
    classes: model.classes,
    hidden: model.gru.hidden,
    inputs: model.channels.length,
    mean: f32(model.normalize.mean),
    std: f32(model.normalize.std),
    wih: f32(model.gru.weight_ih),
    whh: f32(model.gru.weight_hh),
    bih: f32(model.gru.bias_ih),
    bhh: f32(model.gru.bias_hh),
    headW: f32(model.head.weight),
    headB: f32(model.head.bias),
    temperature: model.temperature ?? 1,
  };
}

/**
 * Class probabilities for a sequence of channel vectors.
 * GRU gates follow PyTorch: rows of weight_ih are [reset; update; new].
 * The head reads the mean and the max of the hidden states over time.
 */
export function predictSequence(prepared, frames) {
  if (!prepared || !frames?.length) return null;
  const { hidden: H, inputs: C, mean, std, wih, whh, bih, bhh } = prepared;
  let h = new Float32Array(H);
  const sum = new Float32Array(H);
  const max = new Float32Array(H).fill(-Infinity);
  const x = new Float32Array(C);
  const gi = new Float32Array(3 * H);
  const gh = new Float32Array(3 * H);
  for (const frame of frames) {
    for (let c = 0; c < C; c += 1) x[c] = (frame[c] - mean[c]) / std[c];
    for (let row = 0; row < 3 * H; row += 1) {
      let a = bih[row];
      const offset = row * C;
      for (let c = 0; c < C; c += 1) a += wih[offset + c] * x[c];
      gi[row] = a;
      let b = bhh[row];
      const hOffset = row * H;
      for (let k = 0; k < H; k += 1) b += whh[hOffset + k] * h[k];
      gh[row] = b;
    }
    const next = new Float32Array(H);
    for (let k = 0; k < H; k += 1) {
      const r = sigmoid(gi[k] + gh[k]);
      const z = sigmoid(gi[H + k] + gh[H + k]);
      const n = Math.tanh(gi[2 * H + k] + r * gh[2 * H + k]);
      next[k] = (1 - z) * n + z * h[k];
      sum[k] += next[k];
      if (next[k] > max[k]) max[k] = next[k];
    }
    h = next;
  }
  const pooled = new Float32Array(2 * H);
  for (let k = 0; k < H; k += 1) {
    pooled[k] = sum[k] / frames.length;
    pooled[H + k] = max[k];
  }
  const K = prepared.classes.length;
  const logits = new Float64Array(K);
  for (let o = 0; o < K; o += 1) {
    let value = prepared.headB[o];
    for (let i = 0; i < 2 * H; i += 1) value += prepared.headW[o * 2 * H + i] * pooled[i];
    logits[o] = value / prepared.temperature;
  }
  const peak = Math.max(...logits);
  const exps = Array.from(logits, (value) => Math.exp(value - peak));
  const total = exps.reduce((a, b) => a + b, 0);
  return Object.fromEntries(prepared.classes.map((name, index) => [name, exps[index] / total]));
}

/**
 * Map class probabilities onto the app's 0..100 risk scale. Low vigilance is
 * weighted halfway, matching how RLDD places it between alert and drowsy.
 */
export function sequenceRisk(probabilities) {
  if (!probabilities) return null;
  const drowsy = probabilities.drowsy ?? 0;
  const low = probabilities.low_vigilant ?? 0;
  return Math.round(100 * Math.min(1, drowsy + 0.5 * low));
}

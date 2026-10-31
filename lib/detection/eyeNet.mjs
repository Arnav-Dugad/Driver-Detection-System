/**
 * Plain-JavaScript forward pass for the eye-state CNN trained by
 * ml/train_eye_cnn.py and exported to public/models/eye-state-cnn.json.
 *
 * The network is four 3x3 convolutions (BatchNorm already folded in by the
 * exporter) with ReLU and 2x2 max pooling, global average pooling, and a
 * two-way linear head. Weights ship as symmetric int8 with one scale per
 * tensor and are expanded to Float32Array once at load. A 32x32 crop costs
 * roughly 0.3 M multiply-adds, so no WebGL, WASM, or ONNX runtime is needed.
 *
 * tests/eye-net.test.mjs checks this implementation against PyTorch outputs
 * recorded by the exporter.
 */

export const EYE_INPUT_SIZE = 32;

/** Reject anything that is not the payload shape the exporter writes. */
export function isEyeModel(model) {
  return Boolean(
    model &&
      model.format === "eye-state-cnn" &&
      Array.isArray(model.layers) &&
      model.layers.length > 0 &&
      model.head?.weight?.q,
  );
}

const dequantize = ({ q, scale }) => Float32Array.from(q, (value) => value * scale);

/** Expand int8 weights once; the result is what `predictClosed` consumes. */
export function prepareEyeModel(model) {
  if (!isEyeModel(model)) return null;
  return {
    layers: model.layers.map((layer) => ({
      weight: dequantize(layer.weight),
      bias: Float32Array.from(layer.bias),
      out: layer.weight.shape[0],
      in: layer.weight.shape[1],
      pool: Boolean(layer.pool),
    })),
    head: {
      weight: dequantize(model.head.weight),
      bias: Float32Array.from(model.head.bias),
      out: model.head.weight.shape[0],
      in: model.head.weight.shape[1],
    },
  };
}

/** Per-crop standardization, identical to standardize() in train_eye_cnn.py. */
export function standardizeCrop(pixels) {
  const n = pixels.length;
  let sum = 0;
  for (let i = 0; i < n; i += 1) sum += pixels[i];
  const mean = sum / n;
  let variance = 0;
  for (let i = 0; i < n; i += 1) variance += (pixels[i] - mean) ** 2;
  const scale = 1 / (Math.sqrt(variance / n) + 8);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) out[i] = (pixels[i] - mean) * scale;
  return out;
}

function conv3x3Relu(input, size, layer) {
  const { weight, bias, out: outChannels, in: inChannels } = layer;
  const plane = size * size;
  const output = new Float32Array(outChannels * plane);
  for (let o = 0; o < outChannels; o += 1) {
    const target = o * plane;
    output.fill(bias[o], target, target + plane);
    for (let c = 0; c < inChannels; c += 1) {
      const source = c * plane;
      const kernel = (o * inChannels + c) * 9;
      for (let ky = 0; ky < 3; ky += 1) {
        for (let kx = 0; kx < 3; kx += 1) {
          const w = weight[kernel + ky * 3 + kx];
          if (w === 0) continue;
          const dy = ky - 1;
          const dx = kx - 1;
          const yStart = Math.max(0, -dy);
          const yEnd = Math.min(size, size - dy);
          const xStart = Math.max(0, -dx);
          const xEnd = Math.min(size, size - dx);
          for (let y = yStart; y < yEnd; y += 1) {
            const row = target + y * size;
            const sourceRow = source + (y + dy) * size + dx;
            for (let x = xStart; x < xEnd; x += 1) {
              output[row + x] += w * input[sourceRow + x];
            }
          }
        }
      }
    }
    for (let i = target; i < target + plane; i += 1) {
      if (output[i] < 0) output[i] = 0;
    }
  }
  return output;
}

function maxPool2(input, channels, size) {
  const half = size >> 1;
  const output = new Float32Array(channels * half * half);
  for (let c = 0; c < channels; c += 1) {
    for (let y = 0; y < half; y += 1) {
      for (let x = 0; x < half; x += 1) {
        const base = c * size * size + 2 * y * size + 2 * x;
        output[c * half * half + y * half + x] = Math.max(
          input[base],
          input[base + 1],
          input[base + size],
          input[base + size + 1],
        );
      }
    }
  }
  return output;
}

/**
 * Probability that the eye in a 32x32 grayscale crop (0..255, row-major) is
 * closed. Returns null when no usable model is loaded.
 */
export function predictClosed(prepared, pixels) {
  if (!prepared || !pixels || pixels.length !== EYE_INPUT_SIZE * EYE_INPUT_SIZE) return null;
  let activation = standardizeCrop(pixels);
  let size = EYE_INPUT_SIZE;
  for (const layer of prepared.layers) {
    activation = conv3x3Relu(activation, size, layer);
    if (layer.pool) {
      activation = maxPool2(activation, layer.out, size);
      size >>= 1;
    }
  }
  const channels = prepared.layers[prepared.layers.length - 1].out;
  const plane = size * size;
  const pooled = new Float32Array(channels);
  for (let c = 0; c < channels; c += 1) {
    let sum = 0;
    for (let i = 0; i < plane; i += 1) sum += activation[c * plane + i];
    pooled[c] = sum / plane;
  }
  const { weight, bias, out, in: width } = prepared.head;
  const logits = new Float32Array(out);
  for (let o = 0; o < out; o += 1) {
    let value = bias[o];
    for (let i = 0; i < width; i += 1) value += weight[o * width + i] * pooled[i];
    logits[o] = value;
  }
  // Two-way softmax reduces to a logistic of the logit difference.
  return 1 / (1 + Math.exp(logits[0] - logits[1]));
}

import { evaluateActivation, roundNum } from './cnnMath';
import type { ActivationFnId, Matrix2D, PoolingType } from './presets';

/**
 * Downsamples an RGBA buffer of size (width x height) into an 8x8 grayscale
 * intensity matrix normalized to [0.00, 1.00].
 */
export function sampleRgbaToMatrix8x8(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  invert = false
): Matrix2D {
  const size = 8;
  const matrix: Matrix2D = [];

  for (let r = 0; r < size; r++) {
    const row: number[] = [];
    const yStart = Math.floor((r * height) / size);
    const yEnd = Math.max(yStart + 1, Math.floor(((r + 1) * height) / size));

    for (let c = 0; c < size; c++) {
      const xStart = Math.floor((c * width) / size);
      const xEnd = Math.max(xStart + 1, Math.floor(((c + 1) * width) / size));

      let sumLum = 0;
      let count = 0;

      for (let y = yStart; y < yEnd; y++) {
        for (let x = xStart; x < xEnd; x++) {
          const idx = (y * width + x) * 4;
          const rVal = rgba[idx] ?? 0;
          const gVal = rgba[idx + 1] ?? 0;
          const bVal = rgba[idx + 2] ?? 0;
          const lum = (0.299 * rVal + 0.587 * gVal + 0.114 * bVal) / 255;
          sumLum += lum;
          count++;
        }
      }

      const avg = count > 0 ? sumLum / count : 0;
      const normalized = invert ? 1 - avg : avg;
      row.push(roundNum(Math.max(0, Math.min(1, normalized)), 2));
    }
    matrix.push(row);
  }

  return matrix;
}

/**
 * Runs a real 2D convolution + activation (+ optional 2x2 pooling) directly
 * over a working-resolution RGBA buffer and produces RGBA visual output buffers
 * for live photographic inspection.
 */
export function processImageConvolution(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  kernel: Matrix2D,
  bias: number,
  activationFn: ActivationFnId,
  leakySlope: number,
  poolMode: PoolingType,
  invert: boolean
): {
  convRgba: Uint8ClampedArray;
  actRgba: Uint8ClampedArray;
  poolRgba: Uint8ClampedArray;
  poolWidth: number;
  poolHeight: number;
} {
  // Convert RGBA -> 2D luminance [0..1]
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    const lum =
      (0.299 * rgba[idx] + 0.587 * rgba[idx + 1] + 0.114 * rgba[idx + 2]) /
      255;
    gray[i] = invert ? 1 - lum : lum;
  }

  const convOut = new Float32Array(width * height);
  const actOut = new Float32Array(width * height);

  let maxAbsConv = 0.001;
  let maxAbsAct = 0.001;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let kr = -1; kr <= 1; kr++) {
        for (let kc = -1; kc <= 1; kc++) {
          const sy = Math.min(height - 1, Math.max(0, y + kr));
          const sx = Math.min(width - 1, Math.max(0, x + kc));
          sum += gray[sy * width + sx] * kernel[kr + 1][kc + 1];
        }
      }
      const z = sum + bias;
      const a = evaluateActivation(z, activationFn, leakySlope);
      const pos = y * width + x;
      convOut[pos] = z;
      actOut[pos] = a;
      if (Math.abs(z) > maxAbsConv) maxAbsConv = Math.abs(z);
      if (Math.abs(a) > maxAbsAct) maxAbsAct = Math.abs(a);
    }
  }

  const convRgba = new Uint8ClampedArray(width * height * 4);
  const actRgba = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    // Conv visualization: positive = azure (#0284C7), negative = coral (#DC2626), 0 = slate-50 (#F8FAFC)
    const cv = convOut[i] / maxAbsConv;
    if (cv >= 0) {
      convRgba[idx] = Math.round(248 - cv * (248 - 2));
      convRgba[idx + 1] = Math.round(250 - cv * (250 - 132));
      convRgba[idx + 2] = Math.round(252 - cv * (252 - 199));
    } else {
      const mag = Math.min(1, -cv);
      convRgba[idx] = Math.round(248 - mag * (248 - 220));
      convRgba[idx + 1] = Math.round(250 - mag * (250 - 38));
      convRgba[idx + 2] = Math.round(252 - mag * (252 - 38));
    }
    convRgba[idx + 3] = 255;

    // Activated visualization
    const av = actOut[i] / maxAbsAct;
    if (av >= 0) {
      actRgba[idx] = Math.round(248 - av * (248 - 2));
      actRgba[idx + 1] = Math.round(250 - av * (250 - 132));
      actRgba[idx + 2] = Math.round(252 - av * (252 - 199));
    } else {
      const mag = Math.min(1, -av);
      actRgba[idx] = Math.round(248 - mag * (248 - 220));
      actRgba[idx + 1] = Math.round(250 - mag * (250 - 38));
      actRgba[idx + 2] = Math.round(252 - mag * (252 - 38));
    }
    actRgba[idx + 3] = 255;
  }

  // 2x2 Pooling pass
  const poolWidth = Math.max(1, Math.floor(width / 2));
  const poolHeight = Math.max(1, Math.floor(height / 2));
  const poolRgba = new Uint8ClampedArray(poolWidth * poolHeight * 4);

  for (let py = 0; py < poolHeight; py++) {
    for (let px = 0; px < poolWidth; px++) {
      const y0 = py * 2;
      const x0 = px * 2;
      const vals = [
        actOut[y0 * width + x0],
        actOut[y0 * width + Math.min(width - 1, x0 + 1)],
        actOut[Math.min(height - 1, y0 + 1) * width + x0],
        actOut[Math.min(height - 1, y0 + 1) * width + Math.min(width - 1, x0 + 1)],
      ];
      const pooledVal =
        poolMode === 'avg'
          ? (vals[0] + vals[1] + vals[2] + vals[3]) / 4
          : Math.max(...vals);

      const pv = pooledVal / maxAbsAct;
      const outIdx = (py * poolWidth + px) * 4;
      if (pv >= 0) {
        poolRgba[outIdx] = Math.round(248 - pv * (248 - 5));
        poolRgba[outIdx + 1] = Math.round(250 - pv * (250 - 150));
        poolRgba[outIdx + 2] = Math.round(252 - pv * (252 - 105));
      } else {
        const mag = Math.min(1, -pv);
        poolRgba[outIdx] = Math.round(248 - mag * (248 - 220));
        poolRgba[outIdx + 1] = Math.round(250 - mag * (250 - 38));
        poolRgba[outIdx + 2] = Math.round(252 - mag * (252 - 38));
      }
      poolRgba[outIdx + 3] = 255;
    }
  }

  return {
    convRgba,
    actRgba,
    poolRgba,
    poolWidth,
    poolHeight,
  };
}

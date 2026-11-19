import { evaluateActivation, roundNum, softmax } from './cnnMath';
import type { ActivationFnId, Matrix2D, PoolingType } from './presets';
import {
  generateProceduralSample24,
  STUDIO_SIZE,
  type ProceduralSampleId,
} from './studioSamples';

export const STUDIO_CONV_SIZE = STUDIO_SIZE - 2; // 22
export const STUDIO_POOL_SIZE = STUDIO_CONV_SIZE / 2; // 11

export interface StudioFilterSpec {
  id: string;
  name: string;
  shortName: string;
  description: string;
  rgb: string;
  hex: string;
  kernel: Matrix2D;
  div: number;
}

export const DEFAULT_STUDIO_FILTERS: StudioFilterSpec[] = [
  {
    id: 'vertical',
    name: 'Filtro 1: Bordes Verticales',
    shortName: 'Bordes Verticales',
    description: 'Detecta líneas y transiciones de arriba a abajo',
    rgb: '56, 189, 248',
    hex: '#38bdf8',
    kernel: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ],
    div: 4,
  },
  {
    id: 'horizontal',
    name: 'Filtro 2: Bordes Horizontales',
    shortName: 'Bordes Horizontales',
    description: 'Detecta líneas y horizontes de izquierda a derecha',
    rgb: '251, 191, 36',
    hex: '#fbbf24',
    kernel: [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ],
    div: 4,
  },
  {
    id: 'contours',
    name: 'Filtro 3: Detección de Contornos',
    shortName: 'Contornos (Laplaciano)',
    description: 'Resalta cambios bruscos de luz en cualquier ángulo',
    rgb: '251, 113, 133',
    hex: '#fb7185',
    kernel: [
      [-1, -1, -1],
      [-1, 8, -1],
      [-1, -1, -1],
    ],
    div: 4,
  },
];

export interface StudioPrediction {
  id: ProceduralSampleId;
  label: string;
  color: string;
  probability: number;
}

export interface StudioCNNResult {
  rawMaps: Matrix2D[]; // 3 x 22x22 pre-activation
  convMaps: Matrix2D[]; // 3 x 22x22 post-activation
  poolMaps: Matrix2D[]; // 3 x 11x11
  flattenVector: Array<{ val: number; rgb: string }>;
  predictions: StudioPrediction[];
}

export function computeStudioCNN(
  inputGrid: Matrix2D,
  filters: StudioFilterSpec[],
  poolMode: PoolingType,
  activationFn: ActivationFnId,
  leakySlope: number,
  bias: number
): StudioCNNResult {
  const rawMaps: Matrix2D[] = [];
  const convMaps: Matrix2D[] = [];
  const poolMaps: Matrix2D[] = [];

  for (let c = 0; c < filters.length; c++) {
    const f = filters[c];
    const rMap: Matrix2D = [];
    const cMap: Matrix2D = [];

    for (let y = 0; y < STUDIO_CONV_SIZE; y++) {
      const rRow: number[] = [];
      const cRow: number[] = [];
      for (let x = 0; x < STUDIO_CONV_SIZE; x++) {
        let sum = 0;
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            sum += inputGrid[y + ky][x + kx] * f.kernel[ky][kx];
          }
        }
        const scaledRaw = roundNum(sum / f.div + bias, 3);
        const actVal = evaluateActivation(scaledRaw, activationFn, leakySlope);
        rRow.push(scaledRaw);
        cRow.push(actVal);
      }
      rMap.push(rRow);
      cMap.push(cRow);
    }

    rawMaps.push(rMap);
    convMaps.push(cMap);

    const pMap: Matrix2D = [];
    for (let py = 0; py < STUDIO_POOL_SIZE; py++) {
      const pRow: number[] = [];
      for (let px = 0; px < STUDIO_POOL_SIZE; px++) {
        const vals = [
          cMap[py * 2][px * 2],
          cMap[py * 2][px * 2 + 1],
          cMap[py * 2 + 1][px * 2],
          cMap[py * 2 + 1][px * 2 + 1],
        ];
        const pooled =
          poolMode === 'max'
            ? Math.max(...vals)
            : (vals[0] + vals[1] + vals[2] + vals[3]) / 4;
        pRow.push(roundNum(pooled, 3));
      }
      pMap.push(pRow);
    }
    poolMaps.push(pMap);
  }

  const flattenVector: Array<{ val: number; rgb: string }> = [];
  for (let c = 0; c < poolMaps.length; c++) {
    const rgb = filters[c].rgb;
    for (let y = 0; y < STUDIO_POOL_SIZE; y++) {
      for (let x = 0; x < STUDIO_POOL_SIZE; x++) {
        flattenVector.push({ val: poolMaps[c][y][x], rgb });
      }
    }
  }

  const predictions = classifyStudioPoolMaps(poolMaps);

  return {
    rawMaps,
    convMaps,
    poolMaps,
    flattenVector,
    predictions,
  };
}

function flattenPools(poolMaps: Matrix2D[]): number[] {
  const vec: number[] = [];
  for (const m of poolMaps) {
    for (const row of m) {
      for (const v of row) {
        vec.push(v);
      }
    }
  }
  return vec;
}

function cosineSim(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function computeCanonicalPrototype(id: ProceduralSampleId): number[] {
  const grid = generateProceduralSample24(id);
  const pools: Matrix2D[] = [];
  for (const f of DEFAULT_STUDIO_FILTERS) {
    const cMap: Matrix2D = [];
    for (let y = 0; y < STUDIO_CONV_SIZE; y++) {
      const row: number[] = [];
      for (let x = 0; x < STUDIO_CONV_SIZE; x++) {
        let sum = 0;
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            sum += grid[y + ky][x + kx] * f.kernel[ky][kx];
          }
        }
        row.push(Math.max(0, sum / f.div));
      }
      cMap.push(row);
    }
    const pMap: Matrix2D = [];
    for (let py = 0; py < STUDIO_POOL_SIZE; py++) {
      const row: number[] = [];
      for (let px = 0; px < STUDIO_POOL_SIZE; px++) {
        row.push(
          Math.max(
            cMap[py * 2][px * 2],
            cMap[py * 2][px * 2 + 1],
            cMap[py * 2 + 1][px * 2],
            cMap[py * 2 + 1][px * 2 + 1]
          )
        );
      }
      pMap.push(row);
    }
    pools.push(pMap);
  }
  return flattenPools(pools);
}

const STUDIO_PROTOTYPES: Record<ProceduralSampleId, number[]> = {
  cat: computeCanonicalPrototype('cat'),
  car: computeCanonicalPrototype('car'),
  digit8: computeCanonicalPrototype('digit8'),
  house: computeCanonicalPrototype('house'),
};

const STUDIO_CLASSES: Array<{
  id: ProceduralSampleId;
  label: string;
  color: string;
}> = [
  { id: 'cat', label: 'Gato / Animal', color: '#38bdf8' },
  { id: 'car', label: 'Vehículo', color: '#fbbf24' },
  { id: 'digit8', label: 'Dígito Manuscrito', color: '#34d399' },
  { id: 'house', label: 'Casa / Arquitectura', color: '#fb7185' },
];

export function classifyStudioPoolMaps(poolMaps: Matrix2D[]): StudioPrediction[] {
  const vec = flattenPools(poolMaps);
  const temperature = 7.5;
  const logits = STUDIO_CLASSES.map(
    (cls) => cosineSim(vec, STUDIO_PROTOTYPES[cls.id]) * temperature
  );
  const probs = softmax(logits);
  return STUDIO_CLASSES.map((cls, idx) => ({
    ...cls,
    probability: probs[idx],
  })).sort((a, b) => b.probability - a.probability);
}

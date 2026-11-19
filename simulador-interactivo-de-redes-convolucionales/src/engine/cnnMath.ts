import {
  INPUT_PRESETS,
  KERNEL_PRESETS,
  type ActivationFnId,
  type InputPresetId,
  type Matrix2D,
  type PoolingType,
} from './presets';

export interface ConvolutionStepDetail {
  index: number;
  outRow: number;
  outCol: number;
  startRow: number;
  startCol: number;
  patch: Matrix2D;
  products: Matrix2D;
  sum: number;
  bias: number;
  output: number;
  reluOutput: number;
}

export interface ConvolutionResult {
  paddedInput: Matrix2D;
  featureMap: Matrix2D;
  steps: ConvolutionStepDetail[];
  outRows: number;
  outCols: number;
}

export interface ActivationResult {
  activated: Matrix2D;
  zeroedCount: number;
  positiveCount: number;
  negativeCount: number;
  minVal: number;
  maxVal: number;
}

export interface PoolStepDetail {
  index: number;
  outRow: number;
  outCol: number;
  startRow: number;
  startCol: number;
  patch: Matrix2D;
  maxVal: number;
  avgVal: number;
  selectedVal: number;
  maxLocalRow: number;
  maxLocalCol: number;
}

export interface PoolResult {
  pooled: Matrix2D;
  steps: PoolStepDetail[];
  outRows: number;
  outCols: number;
}

export interface PipelineBranch {
  id: string;
  name: string;
  kernel: Matrix2D;
  conv: Matrix2D;
  relu: Matrix2D;
  pooled: Matrix2D;
}

export interface PipelineClassification {
  id: InputPresetId;
  label: string;
  score: number;
  probability: number;
}

export interface ArchitectureLayerSpec {
  id: string;
  name: string;
  operation: string;
  height: number;
  width: number;
  channels: number;
  params: number;
  notes: string;
}

export function roundNum(val: number, decimals = 2): number {
  const factor = 10 ** decimals;
  return Math.round(val * factor) / factor;
}

export function evaluateActivation(
  z: number,
  fn: ActivationFnId,
  leakySlope = 0.1
): number {
  switch (fn) {
    case 'relu':
      return roundNum(Math.max(0, z), 2);
    case 'leaky_relu':
      return roundNum(z > 0 ? z : z * leakySlope, 2);
    case 'sigmoid':
      return roundNum(1 / (1 + Math.exp(-z)), 2);
    case 'tanh':
      return roundNum(Math.tanh(z), 2);
    case 'linear':
      return roundNum(z, 2);
  }
}

export function padMatrix(matrix: Matrix2D, padding: number): Matrix2D {
  if (padding <= 0) {
    return matrix.map((r) => [...r]);
  }
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const paddedRows = rows + padding * 2;
  const paddedCols = cols + padding * 2;
  const result: Matrix2D = [];

  for (let r = 0; r < paddedRows; r++) {
    const row: number[] = [];
    for (let c = 0; c < paddedCols; c++) {
      const srcR = r - padding;
      const srcC = c - padding;
      if (srcR >= 0 && srcR < rows && srcC >= 0 && srcC < cols) {
        row.push(matrix[srcR][srcC]);
      } else {
        row.push(0);
      }
    }
    result.push(row);
  }
  return result;
}

export function computeOutputDim(
  inputSize: number,
  kernelSize: number,
  padding: number,
  stride: number
): number {
  if (stride <= 0 || kernelSize <= 0) return 0;
  const raw = Math.floor((inputSize + 2 * padding - kernelSize) / stride) + 1;
  return Math.max(0, raw);
}

export function convolve2D(
  input: Matrix2D,
  kernel: Matrix2D,
  padding: number,
  stride: number,
  bias = 0,
  activationFn: ActivationFnId = 'relu',
  leakySlope = 0.1
): ConvolutionResult {
  const paddedInput = padMatrix(input, padding);
  const inRows = paddedInput.length;
  const inCols = paddedInput[0]?.length ?? 0;
  const kRows = kernel.length;
  const kCols = kernel[0]?.length ?? 0;

  const outRows = Math.max(0, Math.floor((inRows - kRows) / stride) + 1);
  const outCols = Math.max(0, Math.floor((inCols - kCols) / stride) + 1);

  const featureMap: Matrix2D = [];
  const steps: ConvolutionStepDetail[] = [];

  for (let outR = 0; outR < outRows; outR++) {
    const rowVals: number[] = [];
    for (let outC = 0; outC < outCols; outC++) {
      const startRow = outR * stride;
      const startCol = outC * stride;
      const patch: Matrix2D = [];
      const products: Matrix2D = [];
      let sum = 0;

      for (let kr = 0; kr < kRows; kr++) {
        const pRow: number[] = [];
        const prodRow: number[] = [];
        for (let kc = 0; kc < kCols; kc++) {
          const val = paddedInput[startRow + kr][startCol + kc];
          const w = kernel[kr][kc];
          const prod = roundNum(val * w, 3);
          pRow.push(val);
          prodRow.push(prod);
          sum += val * w;
        }
        patch.push(pRow);
        products.push(prodRow);
      }

      const roundedSum = roundNum(sum, 2);
      const output = roundNum(sum + bias, 2);
      const reluOutput = evaluateActivation(output, activationFn, leakySlope);
      rowVals.push(output);

      steps.push({
        index: steps.length,
        outRow: outR,
        outCol: outC,
        startRow,
        startCol,
        patch,
        products,
        sum: roundedSum,
        bias,
        output,
        reluOutput,
      });
    }
    featureMap.push(rowVals);
  }

  return {
    paddedInput,
    featureMap,
    steps,
    outRows,
    outCols,
  };
}

export function applyActivation(
  matrix: Matrix2D,
  fn: ActivationFnId = 'relu',
  leakySlope = 0.1
): ActivationResult {
  let zeroedCount = 0;
  let positiveCount = 0;
  let negativeCount = 0;
  let minVal = Infinity;
  let maxVal = -Infinity;

  const activated: Matrix2D = matrix.map((row) =>
    row.map((val) => {
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
      const out = evaluateActivation(val, fn, leakySlope);
      if (out === 0) {
        zeroedCount++;
      } else if (out > 0) {
        positiveCount++;
      } else {
        negativeCount++;
      }
      return out;
    })
  );

  if (minVal === Infinity) minVal = 0;
  if (maxVal === -Infinity) maxVal = 0;

  return {
    activated,
    zeroedCount,
    positiveCount,
    negativeCount,
    minVal: roundNum(minVal, 2),
    maxVal: roundNum(maxVal, 2),
  };
}

export function applyReLU(matrix: Matrix2D): ActivationResult {
  return applyActivation(matrix, 'relu');
}

export function pool2D(
  matrix: Matrix2D,
  poolSize = 2,
  stride = 2,
  mode: PoolingType = 'max'
): PoolResult {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;

  const outRows = Math.max(0, Math.floor((rows - poolSize) / stride) + 1);
  const outCols = Math.max(0, Math.floor((cols - poolSize) / stride) + 1);

  const pooled: Matrix2D = [];
  const steps: PoolStepDetail[] = [];

  for (let outR = 0; outR < outRows; outR++) {
    const outRowVals: number[] = [];
    for (let outC = 0; outC < outCols; outC++) {
      const startRow = outR * stride;
      const startCol = outC * stride;
      const patch: Matrix2D = [];
      let maxVal = -Infinity;
      let maxLocalRow = 0;
      let maxLocalCol = 0;
      let sum = 0;
      let count = 0;

      for (let pr = 0; pr < poolSize; pr++) {
        const pRow: number[] = [];
        for (let pc = 0; pc < poolSize; pc++) {
          const val = matrix[startRow + pr][startCol + pc];
          pRow.push(val);
          sum += val;
          count++;
          if (val > maxVal) {
            maxVal = val;
            maxLocalRow = pr;
            maxLocalCol = pc;
          }
        }
        patch.push(pRow);
      }

      const roundedMax = roundNum(maxVal, 2);
      const roundedAvg = roundNum(count > 0 ? sum / count : 0, 2);
      const selectedVal = mode === 'avg' ? roundedAvg : roundedMax;

      outRowVals.push(selectedVal);
      steps.push({
        index: steps.length,
        outRow: outR,
        outCol: outC,
        startRow,
        startCol,
        patch,
        maxVal: roundedMax,
        avgVal: roundedAvg,
        selectedVal,
        maxLocalRow,
        maxLocalCol,
      });
    }
    pooled.push(outRowVals);
  }

  return {
    pooled,
    steps,
    outRows,
    outCols,
  };
}

export function maxPool2D(
  matrix: Matrix2D,
  poolSize = 2,
  stride = 2
): PoolResult {
  return pool2D(matrix, poolSize, stride, 'max');
}

export function softmax(logits: number[]): number[] {
  if (logits.length === 0) return [];
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((e) => (sumExps > 0 ? e / sumExps : 0));
}

function flattenMatrices(matrices: Matrix2D[]): number[] {
  const flat: number[] = [];
  for (const m of matrices) {
    for (const row of m) {
      for (const val of row) {
        flat.push(val);
      }
    }
  }
  return flat;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

const PIPELINE_KERNELS = [
  { id: 'vertical', name: 'Filtro Vertical', matrix: KERNEL_PRESETS.vertical.matrix },
  { id: 'horizontal', name: 'Filtro Horizontal', matrix: KERNEL_PRESETS.horizontal.matrix },
  { id: 'diagonal', name: 'Filtro Diagonal', matrix: KERNEL_PRESETS.diagonal.matrix },
  { id: 'laplacian', name: 'Filtro Contornos', matrix: KERNEL_PRESETS.laplacian.matrix },
];

function extractFeatureVector(
  input: Matrix2D,
  activationFn: ActivationFnId = 'relu',
  leakySlope = 0.1,
  poolMode: PoolingType = 'max'
): {
  branches: PipelineBranch[];
  vector: number[];
} {
  const branches: PipelineBranch[] = PIPELINE_KERNELS.map((k) => {
    const conv = convolve2D(input, k.matrix, 0, 1).featureMap;
    const relu = applyActivation(conv, activationFn, leakySlope).activated;
    const pooled = pool2D(relu, 2, 2, poolMode).pooled;
    return {
      id: k.id,
      name: k.name,
      kernel: k.matrix,
      conv,
      relu,
      pooled,
    };
  });
  const vector = flattenMatrices(branches.map((b) => b.pooled));
  return { branches, vector };
}

const PROTOTYPE_VECTORS: Record<InputPresetId, number[]> = {
  cross: extractFeatureVector(INPUT_PRESETS.cross.matrix).vector,
  verticalBar: extractFeatureVector(INPUT_PRESETS.verticalBar.matrix).vector,
  horizontalBar: extractFeatureVector(INPUT_PRESETS.horizontalBar.matrix).vector,
  diagonal: extractFeatureVector(INPUT_PRESETS.diagonal.matrix).vector,
  square: extractFeatureVector(INPUT_PRESETS.square.matrix).vector,
};

export function classifyShapePipeline(
  input: Matrix2D,
  activationFn: ActivationFnId = 'relu',
  leakySlope = 0.1,
  poolMode: PoolingType = 'max'
): {
  branches: PipelineBranch[];
  predictions: PipelineClassification[];
} {
  const { branches, vector } = extractFeatureVector(
    input,
    activationFn,
    leakySlope,
    poolMode
  );
  const presetIds: InputPresetId[] = [
    'cross',
    'verticalBar',
    'horizontalBar',
    'diagonal',
    'square',
  ];

  // Temperature-scaled cosine similarity between extracted pooled CNN features and learned shape prototypes
  const temperature = 6.5;
  const rawScores = presetIds.map((id) => {
    const sim = cosineSimilarity(vector, PROTOTYPE_VECTORS[id]);
    return roundNum(sim * temperature, 3);
  });

  const probs = softmax(rawScores);

  const predictions: PipelineClassification[] = presetIds
    .map((id, idx) => ({
      id,
      label: INPUT_PRESETS[id].name,
      score: rawScores[idx],
      probability: probs[idx],
    }))
    .sort((a, b) => b.probability - a.probability);

  return { branches, predictions };
}

export interface ArchitectureParams {
  inputSize: number;
  inputChannels: number;
  numFilters1: number;
  kernel1: number;
  padding1: number;
  stride1: number;
  pool1: number;
  numFilters2: number;
  kernel2: number;
  padding2: number;
  stride2: number;
  pool2: number;
  numClasses: number;
}

export function computeArchitectureSummary(params: ArchitectureParams): {
  layers: ArchitectureLayerSpec[];
  totalParams: number;
  valid: boolean;
} {
  const c1 = computeOutputDim(
    params.inputSize,
    params.kernel1,
    params.padding1,
    params.stride1
  );
  const p1 = params.pool1 > 1 ? Math.floor(c1 / params.pool1) : c1;
  const c2 = computeOutputDim(
    p1,
    params.kernel2,
    params.padding2,
    params.stride2
  );
  const p2 = params.pool2 > 1 ? Math.floor(c2 / params.pool2) : c2;

  const valid = c1 > 0 && p1 > 0 && c2 > 0 && p2 > 0;

  const conv1Params =
    params.kernel1 * params.kernel1 * params.inputChannels * params.numFilters1 +
    params.numFilters1;
  const conv2Params =
    params.kernel2 * params.kernel2 * params.numFilters1 * params.numFilters2 +
    params.numFilters2;
  const flatSize = Math.max(0, p2) * Math.max(0, p2) * params.numFilters2;
  const fcParams = flatSize * params.numClasses + params.numClasses;

  const layers: ArchitectureLayerSpec[] = [
    {
      id: 'input',
      name: 'Imagen de Entrada',
      operation: 'Tensor RGB / Escala de Grises',
      height: params.inputSize,
      width: params.inputSize,
      channels: params.inputChannels,
      params: 0,
      notes: 'Matriz inicial de intensidades de píxeles.',
    },
    {
      id: 'conv1',
      name: 'Convolución 1 + Activación',
      operation: `${params.numFilters1} filtros ${params.kernel1}×${params.kernel1} (S=${params.stride1}, P=${params.padding1})`,
      height: c1,
      width: c1,
      channels: params.numFilters1,
      params: conv1Params,
      notes: 'Detecta bordes, esquinas y gradientes simples.',
    },
    {
      id: 'pool1',
      name: 'Pooling 1',
      operation: `Ventana ${params.pool1}×${params.pool1} (Stride ${params.pool1})`,
      height: p1,
      width: p1,
      channels: params.numFilters1,
      params: 0,
      notes: 'Reduce resolución espacial y aporta invariancia.',
    },
    {
      id: 'conv2',
      name: 'Convolución 2 + Activación',
      operation: `${params.numFilters2} filtros ${params.kernel2}×${params.kernel2} (S=${params.stride2}, P=${params.padding2})`,
      height: c2,
      width: c2,
      channels: params.numFilters2,
      params: conv2Params,
      notes: 'Combina bordes previos en texturas y partes de objetos.',
    },
    {
      id: 'pool2',
      name: 'Pooling 2',
      operation: `Ventana ${params.pool2}×${params.pool2} (Stride ${params.pool2})`,
      height: p2,
      width: p2,
      channels: params.numFilters2,
      params: 0,
      notes: 'Comprime la representación antes de aplanar.',
    },
    {
      id: 'fc',
      name: 'Flatten + Densa (Softmax)',
      operation: `Vector ${flatSize} → ${params.numClasses} clases`,
      height: 1,
      width: 1,
      channels: params.numClasses,
      params: fcParams,
      notes: 'Calcula la probabilidad final de cada categoría.',
    },
  ];

  return {
    layers,
    totalParams: conv1Params + conv2Params + fcParams,
    valid,
  };
}

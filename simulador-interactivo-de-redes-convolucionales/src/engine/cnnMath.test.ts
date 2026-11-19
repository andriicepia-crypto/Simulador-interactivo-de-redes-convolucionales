import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyReLU,
  classifyShapePipeline,
  computeArchitectureSummary,
  computeOutputDim,
  convolve2D,
  maxPool2D,
  padMatrix,
  softmax,
} from './cnnMath';
import { INPUT_PRESETS, KERNEL_PRESETS } from './presets';

test('computeOutputDim obeys standard CNN dimension formula', () => {
  assert.equal(computeOutputDim(8, 3, 0, 1), 6);
  assert.equal(computeOutputDim(8, 3, 1, 1), 8);
  assert.equal(computeOutputDim(8, 3, 0, 2), 3);
  assert.equal(computeOutputDim(8, 3, 1, 2), 4);
});

test('padMatrix adds zero-valued border around input matrix', () => {
  const m = [
    [1, 2],
    [3, 4],
  ];
  const padded = padMatrix(m, 1);
  assert.equal(padded.length, 4);
  assert.equal(padded[0].length, 4);
  assert.deepEqual(padded[0], [0, 0, 0, 0]);
  assert.deepEqual(padded[1], [0, 1, 2, 0]);
  assert.deepEqual(padded[2], [0, 3, 4, 0]);
  assert.deepEqual(padded[3], [0, 0, 0, 0]);
});

test('convolve2D computes exact dot product and feature map dimensions', () => {
  const input = INPUT_PRESETS.verticalBar.matrix;
  const kernel = KERNEL_PRESETS.vertical.matrix;

  const noPad = convolve2D(input, kernel, 0, 1);
  assert.equal(noPad.outRows, 6);
  assert.equal(noPad.outCols, 6);
  assert.equal(noPad.steps.length, 36);

  // Left edge of vertical bar (cols 2..4 are 1s):
  // At outCol=0 (cols 0,1,2 -> [0,0,1]), vertical Sobel [1,0,-1; 2,0,-2; 1,0,-1] gives -4
  assert.equal(noPad.featureMap[0][0], -4);
  // At outCol=3 (cols 3,4,5 -> [1,1,0]), vertical Sobel gives +4
  assert.equal(noPad.featureMap[0][3], 4);

  const withPad = convolve2D(input, kernel, 1, 1);
  assert.equal(withPad.outRows, 8);
  assert.equal(withPad.outCols, 8);
});

test('applyReLU clamps negative activations to zero and preserves positives', () => {
  const mat = [
    [-4, 0, 3.5],
    [-0.1, 2, -8],
  ];
  const res = applyReLU(mat);
  assert.deepEqual(res.activated, [
    [0, 0, 3.5],
    [0, 2, 0],
  ]);
  assert.equal(res.zeroedCount, 4);
  assert.equal(res.positiveCount, 2);
});

test('maxPool2D reduces spatial dimensions and tracks winning coordinates', () => {
  const mat = [
    [1, 3, 2, 4],
    [5, 2, 0, 1],
    [0, 1, 9, 2],
    [2, 2, 1, 3],
  ];
  const res = maxPool2D(mat, 2, 2);
  assert.deepEqual(res.pooled, [
    [5, 4],
    [2, 9],
  ]);
  assert.equal(res.steps[0].maxVal, 5);
  assert.equal(res.steps[0].maxLocalRow, 1);
  assert.equal(res.steps[0].maxLocalCol, 0);
  assert.equal(res.steps[2].maxVal, 2);
  assert.equal(res.steps[3].maxVal, 9);
});

test('softmax probabilities sum to 1 and preserve ordering', () => {
  const probs = softmax([2, 1, 0]);
  const sum = probs.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-6);
  assert.ok(probs[0] > probs[1] && probs[1] > probs[2]);
});

test('classifyShapePipeline accurately identifies each canonical input shape', () => {
  const presetKeys = Object.keys(INPUT_PRESETS) as Array<keyof typeof INPUT_PRESETS>;
  for (const key of presetKeys) {
    const res = classifyShapePipeline(INPUT_PRESETS[key].matrix);
    assert.equal(res.branches.length, 4);
    assert.equal(res.predictions[0].id, key);
    assert.ok(res.predictions[0].probability > 0.45);
  }
});

test('computeArchitectureSummary calculates tensor dimensions and parameter counts', () => {
  const summary = computeArchitectureSummary({
    inputSize: 28,
    inputChannels: 1,
    numFilters1: 8,
    kernel1: 3,
    padding1: 1,
    stride1: 1,
    pool1: 2,
    numFilters2: 16,
    kernel2: 3,
    padding2: 0,
    stride2: 1,
    pool2: 2,
    numClasses: 10,
  });
  assert.equal(summary.valid, true);
  assert.equal(summary.layers[1].height, 28); // conv1 28x28x8
  assert.equal(summary.layers[2].height, 14); // pool1 14x14x8
  assert.equal(summary.layers[3].height, 12); // conv2 12x12x16
  assert.equal(summary.layers[4].height, 6);  // pool2 6x6x16
  assert.ok(summary.totalParams > 0);
});

import {
  applyActivation,
  evaluateActivation,
  pool2D,
} from './cnnMath';
import { sampleRgbaToMatrix8x8 } from './imageProcessor';

test('evaluateActivation supports ReLU, Leaky ReLU, Sigmoid, Tanh, and Linear', () => {
  assert.equal(evaluateActivation(-2, 'relu'), 0);
  assert.equal(evaluateActivation(-2, 'leaky_relu', 0.1), -0.2);
  assert.equal(evaluateActivation(0, 'sigmoid'), 0.5);
  assert.equal(evaluateActivation(0, 'tanh'), 0);
  assert.equal(evaluateActivation(-3.5, 'linear'), -3.5);
});

test('applyActivation counts positive, zeroed, and negative outputs accurately', () => {
  const res = applyActivation(
    [
      [-2, 0],
      [1, 3],
    ],
    'leaky_relu',
    0.2
  );
  assert.deepEqual(res.activated, [
    [-0.4, 0],
    [1, 3],
  ]);
  assert.equal(res.negativeCount, 1);
  assert.equal(res.zeroedCount, 1);
  assert.equal(res.positiveCount, 2);
});

test('pool2D supports average pooling in addition to max pooling', () => {
  const mat = [
    [2, 4, 0, 2],
    [2, 0, 2, 4],
    [1, 1, 3, 3],
    [1, 1, 3, 3],
  ];
  const avgRes = pool2D(mat, 2, 2, 'avg');
  assert.deepEqual(avgRes.pooled, [
    [2, 2],
    [1, 3],
  ]);
});

test('sampleRgbaToMatrix8x8 downsamples RGBA buffer into normalized 8x8 luminance matrix', () => {
  const w = 16;
  const h = 16;
  const rgba = new Uint8ClampedArray(w * h * 4);
  // Left half white (255), right half black (0)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const val = x < 8 ? 255 : 0;
      const idx = (y * w + x) * 4;
      rgba[idx] = val;
      rgba[idx + 1] = val;
      rgba[idx + 2] = val;
      rgba[idx + 3] = 255;
    }
  }
  const normal = sampleRgbaToMatrix8x8(rgba, w, h, false);
  assert.equal(normal.length, 8);
  assert.equal(normal[0][0], 1);
  assert.equal(normal[0][7], 0);

  const inverted = sampleRgbaToMatrix8x8(rgba, w, h, true);
  assert.equal(inverted[0][0], 0);
  assert.equal(inverted[0][7], 1);
});


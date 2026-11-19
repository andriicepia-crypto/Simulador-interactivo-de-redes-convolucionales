import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeStudioCNN,
  DEFAULT_STUDIO_FILTERS,
  STUDIO_CONV_SIZE,
  STUDIO_POOL_SIZE,
} from './studioEngine';
import {
  generateProceduralSample24,
  sampleRgbaToMatrix,
  STUDIO_SIZE,
  type ProceduralSampleId,
} from './studioSamples';

test('procedural 24x24 samples generate normalized grids with expected dimensions', () => {
  const ids: ProceduralSampleId[] = ['cat', 'car', 'digit8', 'house'];
  for (const id of ids) {
    const grid = generateProceduralSample24(id);
    assert.equal(grid.length, STUDIO_SIZE);
    assert.equal(grid[0].length, STUDIO_SIZE);
    const maxVal = Math.max(...grid.flat());
    assert.ok(maxVal > 0.7 && maxVal <= 1);
  }
});

test('computeStudioCNN computes 3 parallel 22x22 conv maps, 11x11 pool maps, 363 flatten values, and accurate classification', () => {
  const ids: ProceduralSampleId[] = ['cat', 'car', 'digit8', 'house'];
  for (const id of ids) {
    const grid = generateProceduralSample24(id);
    const res = computeStudioCNN(
      grid,
      DEFAULT_STUDIO_FILTERS,
      'max',
      'relu',
      0.1,
      0
    );
    assert.equal(res.convMaps.length, 3);
    assert.equal(res.convMaps[0].length, STUDIO_CONV_SIZE);
    assert.equal(res.poolMaps.length, 3);
    assert.equal(res.poolMaps[0].length, STUDIO_POOL_SIZE);
    assert.equal(res.flattenVector.length, 363);
    assert.equal(res.predictions[0].id, id);
    assert.ok(res.predictions[0].probability > 0.5);
  }
});

test('sampleRgbaToMatrix downsamples any buffer to 24x24 with optional inversion', () => {
  const w = 48;
  const h = 48;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = 255;
    rgba[i * 4 + 1] = 255;
    rgba[i * 4 + 2] = 255;
    rgba[i * 4 + 3] = 255;
  }
  const white = sampleRgbaToMatrix(rgba, w, h, 24, false);
  assert.equal(white.length, 24);
  assert.equal(white[0][0], 1);

  const inverted = sampleRgbaToMatrix(rgba, w, h, 24, true);
  assert.equal(inverted[0][0], 0);
});

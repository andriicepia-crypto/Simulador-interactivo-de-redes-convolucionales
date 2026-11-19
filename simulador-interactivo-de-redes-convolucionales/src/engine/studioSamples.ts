import type { Matrix2D } from './presets';

export const STUDIO_SIZE = 24;

export type ProceduralSampleId = 'cat' | 'car' | 'digit8' | 'house';

function createBlank24(): Matrix2D {
  return Array.from({ length: STUDIO_SIZE }, () => Array(STUDIO_SIZE).fill(0));
}

function setPixel(grid: Matrix2D, x: number, y: number, val: number) {
  const xi = Math.round(x);
  const yi = Math.round(y);
  if (xi >= 0 && xi < STUDIO_SIZE && yi >= 0 && yi < STUDIO_SIZE) {
    grid[yi][xi] = Math.max(0, Math.min(1, Math.round(val * 100) / 100));
  }
}

function fillRect(
  grid: Matrix2D,
  x0: number,
  y0: number,
  w: number,
  h: number,
  val: number
) {
  for (let y = y0; y < y0 + h; y++) {
    for (let x = x0; x < x0 + w; x++) {
      setPixel(grid, x, y, val);
    }
  }
}

function fillEllipse(
  grid: Matrix2D,
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  valFn: (nx: number, ny: number, dist: number) => number
) {
  for (let y = 0; y < STUDIO_SIZE; y++) {
    for (let x = 0; x < STUDIO_SIZE; x++) {
      const nx = (x - cx) / rx;
      const ny = (y - cy) / ry;
      const dist = Math.sqrt(nx * nx + ny * ny);
      if (dist <= 1) {
        setPixel(grid, x, y, valFn(nx, ny, dist));
      }
    }
  }
}

function strokeCircle(
  grid: Matrix2D,
  cx: number,
  cy: number,
  r: number,
  thickness: number,
  val: number
) {
  for (let y = 0; y < STUDIO_SIZE; y++) {
    for (let x = 0; x < STUDIO_SIZE; x++) {
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      if (Math.abs(dist - r) <= thickness / 2) {
        const edge = 1 - Math.abs(dist - r) / (thickness / 2 + 0.3);
        setPixel(grid, x, y, Math.max(grid[y][x], val * Math.min(1, edge + 0.35)));
      }
    }
  }
}

function fillTriangle(
  grid: Matrix2D,
  p1: [number, number],
  p2: [number, number],
  p3: [number, number],
  val: number
) {
  const sign = (
    a: [number, number],
    b: [number, number],
    c: [number, number]
  ) => (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1]);

  for (let y = 0; y < STUDIO_SIZE; y++) {
    for (let x = 0; x < STUDIO_SIZE; x++) {
      const pt: [number, number] = [x, y];
      const d1 = sign(pt, p1, p2);
      const d2 = sign(pt, p2, p3);
      const d3 = sign(pt, p3, p1);
      const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
      const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
      if (!(hasNeg && hasPos)) {
        setPixel(grid, x, y, val);
      }
    }
  }
}

export function generateProceduralSample24(id: ProceduralSampleId): Matrix2D {
  const grid = createBlank24();

  if (id === 'cat') {
    // Subtle studio vignette
    fillEllipse(grid, 12, 13, 13, 13, (_nx, _ny, dist) =>
      Math.max(0.04, 0.22 * (1 - dist))
    );
    // Pointed ears
    fillTriangle(grid, [4, 10], [6, 2], [10, 8], 0.82);
    fillTriangle(grid, [20, 10], [18, 2], [14, 8], 0.82);
    // Shaded head
    fillEllipse(grid, 12, 14, 7.6, 6.6, (nx, ny, dist) => {
      const light = 0.96 - dist * 0.42 - ny * 0.08 + nx * 0.04;
      return Math.max(0.35, light);
    });
    // Eyes + specular highlight
    fillRect(grid, 8, 12, 2, 2, 0.06);
    fillRect(grid, 14, 12, 2, 2, 0.06);
    setPixel(grid, 8, 12, 0.98);
    setPixel(grid, 14, 12, 0.98);
    // Nose & whiskers
    fillRect(grid, 11, 15, 2, 1, 0.14);
    for (let x = 2; x <= 6; x++) setPixel(grid, x, 15, 0.65);
    for (let x = 17; x <= 21; x++) setPixel(grid, x, 15, 0.65);
  } else if (id === 'car') {
    // Sky + road horizon
    fillRect(grid, 0, 0, 24, 17, 0.14);
    fillRect(grid, 0, 17, 24, 7, 0.06);
    fillRect(grid, 0, 17, 24, 1, 0.42);
    // Cabin + metallic chassis gradient
    for (let y = 8; y < 12; y++) {
      for (let x = 6; x < 17; x++) {
        setPixel(grid, x, y, 0.92 - (y - 8) * 0.06);
      }
    }
    for (let y = 12; y < 17; y++) {
      for (let x = 3; x < 21; x++) {
        setPixel(grid, x, y, 0.86 - (y - 12) * 0.1);
      }
    }
    // Dark windows
    fillRect(grid, 7, 9, 4, 3, 0.1);
    fillRect(grid, 12, 9, 4, 3, 0.1);
    // Wheels + rims
    fillEllipse(grid, 7, 17, 2.6, 2.6, () => 0.05);
    fillEllipse(grid, 17, 17, 2.6, 2.6, () => 0.05);
    fillRect(grid, 6, 16, 2, 2, 0.88);
    fillRect(grid, 16, 16, 2, 2, 0.88);
  } else if (id === 'digit8') {
    strokeCircle(grid, 12, 7.5, 4.2, 2.5, 1.0);
    strokeCircle(grid, 12, 16.2, 4.8, 2.5, 1.0);
  } else if (id === 'house') {
    fillRect(grid, 0, 0, 24, 24, 0.08);
    // Roof
    fillTriangle(grid, [3, 11], [12, 3], [21, 11], 0.92);
    // Facade wall
    fillRect(grid, 5, 11, 14, 10, 0.62);
    // Door + windows
    fillRect(grid, 10, 15, 4, 6, 0.1);
    fillRect(grid, 7, 13, 2, 2, 0.98);
    fillRect(grid, 15, 13, 2, 2, 0.98);
  }

  return grid;
}

export function sampleRgbaToMatrix(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  targetSize: number,
  invert = false
): Matrix2D {
  const matrix: Matrix2D = [];
  for (let r = 0; r < targetSize; r++) {
    const row: number[] = [];
    const yStart = Math.floor((r * height) / targetSize);
    const yEnd = Math.max(
      yStart + 1,
      Math.floor(((r + 1) * height) / targetSize)
    );
    for (let c = 0; c < targetSize; c++) {
      const xStart = Math.floor((c * width) / targetSize);
      const xEnd = Math.max(
        xStart + 1,
        Math.floor(((c + 1) * width) / targetSize)
      );
      let sumLum = 0;
      let count = 0;
      for (let y = yStart; y < yEnd; y++) {
        for (let x = xStart; x < xEnd; x++) {
          const idx = (y * width + x) * 4;
          const lum =
            (0.299 * (rgba[idx] ?? 0) +
              0.587 * (rgba[idx + 1] ?? 0) +
              0.114 * (rgba[idx + 2] ?? 0)) /
            255;
          sumLum += lum;
          count++;
        }
      }
      const avg = count > 0 ? sumLum / count : 0;
      const val = invert ? 1 - avg : avg;
      row.push(Math.round(Math.max(0, Math.min(1, val)) * 100) / 100);
    }
    matrix.push(row);
  }
  return matrix;
}

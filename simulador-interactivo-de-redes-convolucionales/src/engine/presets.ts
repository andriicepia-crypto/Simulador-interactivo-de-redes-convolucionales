export type Matrix2D = number[][];

export type InputPresetId =
  | 'cross'
  | 'verticalBar'
  | 'horizontalBar'
  | 'diagonal'
  | 'square';

export type SampleImageId = 'facade' | 'leaf' | 'digit';

export type KernelPresetId =
  | 'vertical'
  | 'horizontal'
  | 'laplacian'
  | 'diagonal'
  | 'sharpen'
  | 'blur'
  | 'custom';

export type ActivationFnId =
  | 'relu'
  | 'leaky_relu'
  | 'sigmoid'
  | 'tanh'
  | 'linear';

export type PoolingType = 'max' | 'avg';

export type StageBackdropId = 'technical_grid' | 'studio_wave' | 'blueprint';

export interface InputPreset {
  id: InputPresetId;
  name: string;
  shortLabel: string;
  description: string;
  matrix: Matrix2D;
}

export interface SampleImagePreset {
  id: SampleImageId;
  name: string;
  shortLabel: string;
  description: string;
  recommendedKernel: Exclude<KernelPresetId, 'custom'>;
  src: string;
  invertByDefault: boolean;
}

export interface KernelPreset {
  id: Exclude<KernelPresetId, 'custom'>;
  name: string;
  shortLabel: string;
  description: string;
  matrix: Matrix2D;
}

export interface ActivationMeta {
  id: ActivationFnId;
  name: string;
  shortLabel: string;
  formula: string;
  description: string;
}

export const HERO_BACKDROP_SRC =
  '/src/assets/images/cnn_hero_backdrop_1795082275074.jpg';

export const SAMPLE_IMAGES: Record<SampleImageId, SampleImagePreset> = {
  facade: {
    id: 'facade',
    name: 'Fachada Arquitectónica',
    shortLabel: 'Fachada',
    description:
      'Columnas y losas de concreto ideales para observar detectores de bordes verticales y horizontales.',
    recommendedKernel: 'vertical',
    src: '/src/assets/images/sample_arch_facade_1795082275277.jpg',
    invertByDefault: false,
  },
  leaf: {
    id: 'leaf',
    name: 'Hoja Botánica',
    shortLabel: 'Hoja',
    description:
      'Nervaduras diagonales y silueta orgánica para probar filtros laplacianos y diagonales.',
    recommendedKernel: 'laplacian',
    src: '/src/assets/images/sample_botanical_leaf_1795082275849.jpg',
    invertByDefault: true,
  },
  digit: {
    id: 'digit',
    name: 'Trazo Tinta (7)',
    shortLabel: 'Dígito 7',
    description:
      'Trazo de tinta negra sobre papel claro similar a conjuntos clásicos de reconocimiento de dígitos.',
    recommendedKernel: 'diagonal',
    src: '/src/assets/images/sample_ink_digit_1795082275503.jpg',
    invertByDefault: true,
  },
};

export const ACTIVATION_FUNCTIONS: Record<ActivationFnId, ActivationMeta> = {
  relu: {
    id: 'relu',
    name: 'ReLU (Rectified Linear Unit)',
    shortLabel: 'ReLU',
    formula: 'f(z) = máx(0, z)',
    description:
      'Anula activaciones negativas y deja pasar los valores positivos sin saturación.',
  },
  leaky_relu: {
    id: 'leaky_relu',
    name: 'Leaky ReLU',
    shortLabel: 'Leaky ReLU',
    formula: 'f(z) = z si z > 0, α·z si z ≤ 0',
    description:
      'Mantiene una pequeña pendiente positiva en valores negativos para evitar neuronas muertas.',
  },
  sigmoid: {
    id: 'sigmoid',
    name: 'Sigmoide Logística',
    shortLabel: 'Sigmoide',
    formula: 'f(z) = 1 / (1 + e⁻ᶻ)',
    description:
      'Comprime cualquier salida lineal en un rango suave entre 0 y 1.',
  },
  tanh: {
    id: 'tanh',
    name: 'Tangente Hiperbólica (Tanh)',
    shortLabel: 'Tanh',
    formula: 'f(z) = tanh(z)',
    description:
      'Centrada en cero, proyecta las activaciones en el intervalo [-1, +1].',
  },
  linear: {
    id: 'linear',
    name: 'Identidad (Sin activación)',
    shortLabel: 'Lineal',
    formula: 'f(z) = z',
    description:
      'Transmite la suma convolucional sin transformación no lineal para comparación directa.',
  },
};

export const INPUT_PRESETS: Record<InputPresetId, InputPreset> = {
  cross: {
    id: 'cross',
    name: 'Cruz (+)',
    shortLabel: 'Cruz',
    description: 'Combina bordes verticales y horizontales interceptados en el centro.',
    matrix: [
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
    ],
  },
  verticalBar: {
    id: 'verticalBar',
    name: 'Barra Vertical',
    shortLabel: 'Vertical',
    description: 'Franja vertical intensa con transiciones marcadas a izquierda y derecha.',
    matrix: [
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0],
    ],
  },
  horizontalBar: {
    id: 'horizontalBar',
    name: 'Barra Horizontal',
    shortLabel: 'Horizontal',
    description: 'Banda horizontal continua que activa detectores horizontales.',
    matrix: [
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
    ],
  },
  diagonal: {
    id: 'diagonal',
    name: 'Diagonal (/)',
    shortLabel: 'Diagonal',
    description: 'Trazo inclinado a 45 grados ideal para probar filtros direccionales.',
    matrix: [
      [0, 0, 0, 0, 0, 0, 1, 1],
      [0, 0, 0, 0, 0, 1, 1, 0],
      [0, 0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0],
      [1, 1, 0, 0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0],
    ],
  },
  square: {
    id: 'square',
    name: 'Cuadrado (□)',
    shortLabel: 'Cuadrado',
    description: 'Contorno cerrado con cuatro esquinas y bordes rectos bien definidos.',
    matrix: [
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 0, 0, 0, 0, 1, 0],
      [0, 1, 0, 0, 0, 0, 1, 0],
      [0, 1, 0, 0, 0, 0, 1, 0],
      [0, 1, 0, 0, 0, 0, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
    ],
  },
};

export const KERNEL_PRESETS: Record<Exclude<KernelPresetId, 'custom'>, KernelPreset> = {
  vertical: {
    id: 'vertical',
    name: 'Detector de Bordes Verticales',
    shortLabel: 'Vertical',
    description: 'Asigna pesos positivos a la izquierda y negativos a la derecha para resaltar cambios verticales.',
    matrix: [
      [1, 0, -1],
      [2, 0, -2],
      [1, 0, -1],
    ],
  },
  horizontal: {
    id: 'horizontal',
    name: 'Detector de Bordes Horizontales',
    shortLabel: 'Horizontal',
    description: 'Detecta transiciones de arriba hacia abajo destacando líneas y contornos horizontales.',
    matrix: [
      [1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1],
    ],
  },
  laplacian: {
    id: 'laplacian',
    name: 'Contornos Totales (Laplaciano)',
    shortLabel: 'Contornos',
    description: 'Compara cada píxel central contra sus 8 vecinos para extraer bordes en cualquier dirección.',
    matrix: [
      [-1, -1, -1],
      [-1, 8, -1],
      [-1, -1, -1],
    ],
  },
  diagonal: {
    id: 'diagonal',
    name: 'Detector Diagonal (/)',
    shortLabel: 'Diagonal',
    description: 'Responde con intensidad máxima cuando los píxeles siguen una pendiente ascendente.',
    matrix: [
      [-1, -1, 2],
      [-1, 2, -1],
      [2, -1, -1],
    ],
  },
  sharpen: {
    id: 'sharpen',
    name: 'Realce de Nitidez',
    shortLabel: 'Nitidez',
    description: 'Amplifica el píxel central y resta los vecinos cardinales para acentuar detalles.',
    matrix: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
  blur: {
    id: 'blur',
    name: 'Suavizado Promedio',
    shortLabel: 'Suavizado',
    description: 'Promedia la vecindad 3×3 reduciendo ruido y transiciones bruscas.',
    matrix: [
      [0.11, 0.11, 0.11],
      [0.11, 0.11, 0.11],
      [0.11, 0.11, 0.11],
    ],
  },
};

export function cloneMatrix(matrix: Matrix2D): Matrix2D {
  return matrix.map((row) => [...row]);
}

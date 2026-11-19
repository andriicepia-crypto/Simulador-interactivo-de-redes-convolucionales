import React, { useRef, useState } from 'react';
import {
  ACTIVATION_FUNCTIONS,
  INPUT_PRESETS,
  KERNEL_PRESETS,
  SAMPLE_IMAGES,
  type ActivationFnId,
  type InputPresetId,
  type KernelPresetId,
  type Matrix2D,
  type PoolingType,
  type SampleImageId,
  type StageBackdropId,
} from '../engine/presets';
import {
  ArrowLeftRight,
  Eraser,
  Image as ImageIcon,
  Pencil,
  Trash2,
  Upload,
} from 'lucide-react';

export type LearningStage = 'input' | 'conv' | 'relu' | 'pool' | 'pipeline';

interface ControlDeckProps {
  stage: LearningStage;
  selectedSource: InputPresetId | SampleImageId | 'uploaded' | 'custom';
  activePhotoLabel: string | null;
  invertPhoto: boolean;
  selectedKernelPreset: KernelPresetId;
  kernel: Matrix2D;
  padding: number;
  stride: number;
  bias: number;
  activationFn: ActivationFnId;
  leakySlope: number;
  poolMode: PoolingType;
  backdrop: StageBackdropId;
  brushMode: 'draw' | 'erase';
  outSize: number;
  uploadError: string | null;
  onSelectInputPreset: (id: InputPresetId) => void;
  onSelectSampleImage: (id: SampleImageId) => void;
  onUploadImageFile: (file: File) => void;
  onToggleInvertPhoto: () => void;
  onClearInput: () => void;
  onSetBrushMode: (mode: 'draw' | 'erase') => void;
  onSelectKernelPreset: (id: Exclude<KernelPresetId, 'custom'>) => void;
  onUpdateKernelCell: (row: number, col: number, delta: number) => void;
  onSetPadding: (pad: number) => void;
  onSetStride: (stride: number) => void;
  onSetBias: (bias: number) => void;
  onSetActivationFn: (fn: ActivationFnId) => void;
  onSetLeakySlope: (slope: number) => void;
  onSetPoolMode: (mode: PoolingType) => void;
  onSetBackdrop: (backdrop: StageBackdropId) => void;
}

const STAGE_INSIGHTS: Record<
  LearningStage,
  { title: string; summary: string; detail: string }
> = {
  input: {
    title: 'De fotografía real a matriz numérica',
    summary:
      'Una CNN convierte cada imagen o dibujo en una cuadrícula numérica de intensidades entre 0.0 y 1.0.',
    detail:
      'Elige una fotografía de ejemplo, sube tu propia imagen o pinta píxeles manualmente para ver cómo se cuantifica la luz.',
  },
  conv: {
    title: 'Extracción de características con Kernels',
    summary:
      'Un filtro 3×3 se desliza por la imagen calculando productos escalares más un sesgo (bias) para detectar bordes y texturas.',
    detail:
      'Los pesos positivos (azul) buscan píxeles brillantes; los negativos (rosa) penalizan la presencia de píxeles en esa posición.',
  },
  relu: {
    title: 'Funciones de activación personalizables',
    summary:
      'Tras la convolución, la función de activación (ReLU, Leaky ReLU, Sigmoide o Tanh) introduce no linealidad en cada neurona.',
    detail:
      'Cambia la función activa para comparar cómo ReLU anula los negativos frente a curvas suaves como Sigmoide o Tanh.',
  },
  pool: {
    title: 'Compresión espacial (Max o Average)',
    summary:
      'Resume bloques de 2×2 píxeles conservando el valor máximo (Max) o el promedio (Average) para reducir resolución.',
    detail:
      'Disminuye el coste computacional en capas profundas y aporta tolerancia ante pequeños desplazamientos del objeto.',
  },
  pipeline: {
    title: 'Clasificación de extremo a extremo',
    summary:
      'Cuatro ramas convolucionales paralelas extraen rasgos complementarios antes de calcular probabilidades Softmax.',
    detail:
      'Prueba cambiando la función de activación o el método de pooling para ver cómo influyen en la predicción final.',
  },
};

export const ControlDeck: React.FC<ControlDeckProps> = ({
  stage,
  selectedSource,
  activePhotoLabel,
  invertPhoto,
  selectedKernelPreset,
  kernel,
  padding,
  stride,
  bias,
  activationFn,
  leakySlope,
  poolMode,
  backdrop,
  brushMode,
  outSize,
  uploadError,
  onSelectInputPreset,
  onSelectSampleImage,
  onUploadImageFile,
  onToggleInvertPhoto,
  onClearInput,
  onSetBrushMode,
  onSelectKernelPreset,
  onUpdateKernelCell,
  onSetPadding,
  onSetStride,
  onSetBias,
  onSetActivationFn,
  onSetLeakySlope,
  onSetPoolMode,
  onSetBackdrop,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [brokenThumbs, setBrokenThumbs] = useState<Record<string, boolean>>({});

  const insight = STAGE_INSIGHTS[stage];
  const inputPresetList = Object.values(INPUT_PRESETS);
  const sampleImageList = Object.values(SAMPLE_IMAGES);
  const kernelPresetList = Object.values(KERNEL_PRESETS);
  const activationList = Object.values(ACTIVATION_FUNCTIONS);

  return (
    <aside
      aria-label="Panel de control y conceptos"
      className="flex flex-col gap-4"
    >
      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 sm:p-5">
        <p className="text-xs font-medium text-sky-400 mb-1">
          Concepto de la etapa
        </p>
        <h2 className="font-display text-lg font-semibold text-slate-50 leading-snug">
          {insight.title}
        </h2>
        <p className="text-sm text-slate-300 mt-2 leading-relaxed">
          {insight.summary}
        </p>
        <p className="text-xs text-slate-400 mt-2 leading-relaxed">
          {insight.detail}
        </p>
      </section>

      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 sm:p-5">
        <div className="flex items-center justify-between gap-2 mb-3">
          <h3 className="text-sm font-semibold text-slate-100">
            Imágenes y patrones de entrada
          </h3>
          <span className="text-xs font-mono tabular-nums text-sky-400 truncate max-w-[120px]">
            {activePhotoLabel
              ? activePhotoLabel
              : selectedSource === 'custom'
                ? 'Dibujo libre'
                : selectedSource in INPUT_PRESETS
                  ? INPUT_PRESETS[selectedSource as InputPresetId].shortLabel
                  : 'Imagen'}
          </span>
        </div>

        <p className="text-[11px] font-medium text-slate-400 mb-1.5">
          Ejemplos visuales reales
        </p>
        <div className="grid grid-cols-3 gap-2 mb-3">
          {sampleImageList.map((img) => {
            const active = selectedSource === img.id;
            return (
              <button
                key={img.id}
                type="button"
                aria-label={`Ejemplo visual: ${img.name}`}
                aria-pressed={active}
                onClick={() => onSelectSampleImage(img.id)}
                className={`flex flex-col items-center p-1.5 rounded-lg border text-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                  active
                    ? 'bg-sky-500/15 border-sky-400 text-sky-200'
                    : 'bg-[#080d1a] border-slate-800 text-slate-300 hover:border-slate-600'
                }`}
              >
                <div className="w-full aspect-square rounded overflow-hidden bg-slate-900 flex items-center justify-center mb-1">
                  {!brokenThumbs[img.id] ? (
                    <img
                      src={img.src}
                      alt={img.name}
                      referrerPolicy="no-referrer"
                      onError={() =>
                        setBrokenThumbs((prev) => ({ ...prev, [img.id]: true }))
                      }
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <ImageIcon className="w-4 h-4 text-slate-500" aria-hidden="true" />
                  )}
                </div>
                <span className="text-[11px] font-medium whitespace-nowrap truncate w-full">
                  {img.shortLabel}
                </span>
              </button>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-2 mb-3 pb-3 border-b border-slate-800">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            aria-label="Cargar imagen desde tu dispositivo"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                onUploadImageFile(file);
                e.target.value = '';
              }
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold text-slate-950 bg-emerald-400 hover:bg-emerald-300 rounded-lg whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300"
          >
            <Upload className="w-3.5 h-3.5" aria-hidden="true" />
            Subir imagen
          </button>

          {activePhotoLabel && (
            <button
              type="button"
              onClick={onToggleInvertPhoto}
              aria-pressed={invertPhoto}
              className={`inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                invertPhoto
                  ? 'bg-amber-400/20 text-amber-300 border-amber-400'
                  : 'bg-[#080d1a] text-slate-300 border-slate-700 hover:border-slate-500'
              }`}
            >
              <ArrowLeftRight className="w-3.5 h-3.5" aria-hidden="true" />
              Invertir
            </button>
          )}
        </div>

        {uploadError && (
          <p className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg px-2.5 py-1.5 mb-3">
            {uploadError}
          </p>
        )}

        <p className="text-[11px] font-medium text-slate-400 mb-1.5">
          Patrones geométricos (8×8)
        </p>
        <div className="grid grid-cols-3 gap-1.5 mb-3">
          {inputPresetList.map((preset) => {
            const active = selectedSource === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                aria-label={`Entrada: ${preset.name}`}
                aria-pressed={active}
                onClick={() => onSelectInputPreset(preset.id)}
                className={`px-2 py-1.5 text-xs font-medium rounded-lg border whitespace-nowrap truncate focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                  active
                    ? 'bg-sky-500/20 text-sky-300 border-sky-400'
                    : 'bg-[#080d1a] text-slate-300 border-slate-800 hover:border-slate-600'
                }`}
              >
                {preset.shortLabel}
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => onSetBrushMode('draw')}
            aria-pressed={brushMode === 'draw'}
            className={`flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
              brushMode === 'draw'
                ? 'bg-sky-500/20 text-sky-300 border-sky-400'
                : 'bg-[#080d1a] text-slate-300 border-slate-800 hover:border-slate-600'
            }`}
          >
            <Pencil className="w-3.5 h-3.5" aria-hidden="true" />
            Pintar (1)
          </button>
          <button
            type="button"
            onClick={() => onSetBrushMode('erase')}
            aria-pressed={brushMode === 'erase'}
            className={`flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
              brushMode === 'erase'
                ? 'bg-sky-500/20 text-sky-300 border-sky-400'
                : 'bg-[#080d1a] text-slate-300 border-slate-800 hover:border-slate-600'
            }`}
          >
            <Eraser className="w-3.5 h-3.5" aria-hidden="true" />
            Borrar (0)
          </button>
          <button
            type="button"
            onClick={onClearInput}
            aria-label="Limpiar cuadrícula de entrada"
            className="inline-flex items-center justify-center px-2.5 py-2 text-xs font-medium text-slate-300 hover:text-rose-300 hover:bg-rose-500/10 bg-[#080d1a] border border-slate-800 hover:border-rose-500/40 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
          </button>
        </div>
      </section>

      {stage !== 'pipeline' && (
        <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 sm:p-5">
          <div className="flex items-center justify-between gap-2 mb-3">
            <h3 className="text-sm font-semibold text-slate-100">
              Filtro / Kernel (3×3)
            </h3>
            <span className="text-xs font-mono text-sky-400">
              {selectedKernelPreset === 'custom'
                ? 'Personalizado'
                : KERNEL_PRESETS[selectedKernelPreset].shortLabel}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-1.5 mb-3">
            {kernelPresetList.map((kp) => {
              const active = selectedKernelPreset === kp.id;
              return (
                <button
                  key={kp.id}
                  type="button"
                  aria-label={`Kernel: ${kp.name}`}
                  aria-pressed={active}
                  onClick={() => onSelectKernelPreset(kp.id)}
                  className={`px-2 py-1.5 text-xs font-medium rounded-lg border whitespace-nowrap truncate focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                    active
                      ? 'bg-sky-500/20 text-sky-300 border-sky-400'
                      : 'bg-[#080d1a] text-slate-300 border-slate-800 hover:border-slate-600'
                  }`}
                >
                  {kp.shortLabel}
                </button>
              );
            })}
          </div>

          <div className="pt-3 border-t border-slate-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-slate-300">
                Editar pesos del Kernel
              </span>
              <span className="text-[11px] text-slate-400">
                Usa − / + en cada peso
              </span>
            </div>

            <div className="grid grid-cols-3 gap-1.5">
              {kernel.map((row, r) =>
                row.map((weight, c) => (
                  <div
                    key={`ke-${r}-${c}`}
                    className={`flex items-center justify-between px-1.5 py-1 rounded-lg border ${
                      weight > 0
                        ? 'bg-sky-500/10 border-sky-500/35'
                        : weight < 0
                          ? 'bg-rose-500/10 border-rose-500/35'
                          : 'bg-[#080d1a] border-slate-800'
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => onUpdateKernelCell(r, c, -1)}
                      aria-label={`Disminuir peso fila ${r + 1} columna ${c + 1}`}
                      className="w-6 h-6 flex items-center justify-center text-xs font-semibold text-slate-300 hover:bg-slate-800 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
                    >
                      −
                    </button>
                    <span className="text-xs font-mono font-semibold tabular-nums text-slate-100">
                      {Number.isInteger(weight) ? weight : weight.toFixed(1)}
                    </span>
                    <button
                      type="button"
                      onClick={() => onUpdateKernelCell(r, c, 1)}
                      aria-label={`Aumentar peso fila ${r + 1} columna ${c + 1}`}
                      className="w-6 h-6 flex items-center justify-center text-xs font-semibold text-slate-300 hover:bg-slate-800 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
                    >
                      +
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      )}

      <section className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 sm:p-5">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Funciones personalizables y entorno
        </h3>

        <div className="space-y-3.5">
          <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="font-medium text-slate-300">
                Función de activación
              </span>
              <span className="font-mono text-sky-400 font-medium">
                {ACTIVATION_FUNCTIONS[activationFn].shortLabel}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-1.5">
              {activationList.map((act) => {
                const active = activationFn === act.id;
                return (
                  <button
                    key={act.id}
                    type="button"
                    aria-label={`Activación: ${act.name}`}
                    aria-pressed={active}
                    onClick={() => onSetActivationFn(act.id)}
                    className={`px-2 py-1.5 text-xs font-medium rounded-lg border whitespace-nowrap truncate focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                      active
                        ? 'bg-sky-500/20 text-sky-300 border-sky-400'
                        : 'bg-[#080d1a] text-slate-300 border-slate-800 hover:border-slate-600'
                    }`}
                  >
                    {act.shortLabel}
                  </button>
                );
              })}
            </div>

            {activationFn === 'leaky_relu' && (
              <div className="mt-2.5">
                <div className="flex justify-between text-xs mb-1">
                  <label htmlFor="leaky-alpha" className="text-slate-400">
                    Pendiente negativa (α)
                  </label>
                  <span className="font-mono tabular-nums font-semibold text-sky-300">
                    α = {leakySlope.toFixed(2)}
                  </span>
                </div>
                <input
                  id="leaky-alpha"
                  type="range"
                  min={0.05}
                  max={0.4}
                  step={0.05}
                  value={leakySlope}
                  onChange={(e) => onSetLeakySlope(Number(e.target.value))}
                  className="w-full accent-sky-400"
                />
              </div>
            )}
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <label htmlFor="conv-bias" className="font-medium text-slate-300">
                Sesgo (Bias b)
              </label>
              <span className="font-mono tabular-nums font-semibold text-sky-300">
                {bias > 0 ? `+${bias.toFixed(1)}` : bias.toFixed(1)}
              </span>
            </div>
            <input
              id="conv-bias"
              type="range"
              min={-3}
              max={3}
              step={0.5}
              value={bias}
              onChange={(e) => onSetBias(Number(e.target.value))}
              className="w-full accent-sky-400"
            />
          </div>

          <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="font-medium text-slate-300">Método de Pooling</span>
              <span className="font-mono text-emerald-400">
                {poolMode === 'max' ? 'Máximo' : 'Promedio'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-1.5 p-1 bg-[#080d1a] border border-slate-800 rounded-lg">
              <button
                type="button"
                onClick={() => onSetPoolMode('max')}
                aria-pressed={poolMode === 'max'}
                className={`py-1.5 text-xs font-medium rounded-md whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                  poolMode === 'max'
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Max Pooling
              </button>
              <button
                type="button"
                onClick={() => onSetPoolMode('avg')}
                aria-pressed={poolMode === 'avg'}
                className={`py-1.5 text-xs font-medium rounded-md whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                  poolMode === 'avg'
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Average Pooling
              </button>
            </div>
          </div>

          {stage !== 'pipeline' && (
            <div className="grid grid-cols-2 gap-2.5 pt-1">
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-medium text-slate-300">Padding</span>
                  <span className="font-mono tabular-nums text-slate-400">
                    P={padding}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-1 p-1 bg-[#080d1a] border border-slate-800 rounded-lg">
                  <button
                    type="button"
                    onClick={() => onSetPadding(0)}
                    aria-pressed={padding === 0}
                    className={`py-1 text-xs font-medium rounded-md whitespace-nowrap ${
                      padding === 0
                        ? 'bg-slate-800 text-slate-50'
                        : 'text-slate-400'
                    }`}
                  >
                    0
                  </button>
                  <button
                    type="button"
                    onClick={() => onSetPadding(1)}
                    aria-pressed={padding === 1}
                    className={`py-1 text-xs font-medium rounded-md whitespace-nowrap ${
                      padding === 1
                        ? 'bg-slate-800 text-slate-50'
                        : 'text-slate-400'
                    }`}
                  >
                    1
                  </button>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-medium text-slate-300">Stride</span>
                  <span className="font-mono tabular-nums text-slate-400">
                    S={stride}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-1 p-1 bg-[#080d1a] border border-slate-800 rounded-lg">
                  <button
                    type="button"
                    onClick={() => onSetStride(1)}
                    aria-pressed={stride === 1}
                    className={`py-1 text-xs font-medium rounded-md whitespace-nowrap ${
                      stride === 1
                        ? 'bg-slate-800 text-slate-50'
                        : 'text-slate-400'
                    }`}
                  >
                    1
                  </button>
                  <button
                    type="button"
                    onClick={() => onSetStride(2)}
                    aria-pressed={stride === 2}
                    className={`py-1 text-xs font-medium rounded-md whitespace-nowrap ${
                      stride === 2
                        ? 'bg-slate-800 text-slate-50'
                        : 'text-slate-400'
                    }`}
                  >
                    2
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="pt-2 border-t border-slate-800">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="font-medium text-slate-300">
                Fondo del escenario
              </span>
            </div>
            <div className="grid grid-cols-3 gap-1.5">
              {[
                { id: 'technical_grid' as const, label: 'Rejilla' },
                { id: 'studio_wave' as const, label: 'Óptico' },
                { id: 'blueprint' as const, label: 'Blueprint' },
              ].map((bg) => (
                <button
                  key={bg.id}
                  type="button"
                  aria-pressed={backdrop === bg.id}
                  onClick={() => onSetBackdrop(bg.id)}
                  className={`py-1.5 px-2 text-xs font-medium rounded-lg border whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                    backdrop === bg.id
                      ? 'bg-slate-700 text-slate-50 border-slate-500'
                      : 'bg-[#080d1a] text-slate-400 border-slate-800 hover:text-slate-200'
                  }`}
                >
                  {bg.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {stage !== 'pipeline' && (
          <div className="mt-3.5 pt-3 border-t border-slate-800 text-xs font-mono tabular-nums text-slate-400 flex items-center justify-between">
            <span>Salida: ⌊(8 + 2·{padding} − 3)/{stride}⌋ + 1</span>
            <span className="font-semibold text-sky-400">
              {outSize}×{outSize}
            </span>
          </div>
        )}
      </section>
    </aside>
  );
};

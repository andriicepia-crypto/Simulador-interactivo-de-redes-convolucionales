import React from 'react';
import {
  ArrowRight,
  Pause,
  Play,
  RotateCcw,
  SkipBack,
  SkipForward,
} from 'lucide-react';
import {
  applyActivation,
  classifyShapePipeline,
  convolve2D,
  pool2D,
} from '../engine/cnnMath';
import {
  ACTIVATION_FUNCTIONS,
  type ActivationFnId,
  type Matrix2D,
  type PoolingType,
  type StageBackdropId,
} from '../engine/presets';
import { ActivationCurveChart } from './ActivationCurveChart';
import type { LearningStage } from './ControlDeck';
import { DotProductBreakdown } from './DotProductBreakdown';
import { MatrixGrid } from './MatrixGrid';
import { PhotoConvolutionPreview } from './PhotoConvolutionPreview';

interface StageSimulatorProps {
  stage: LearningStage;
  onChangeStage: (stage: LearningStage) => void;
  inputMatrix: Matrix2D;
  kernel: Matrix2D;
  padding: number;
  stride: number;
  bias: number;
  activationFn: ActivationFnId;
  leakySlope: number;
  poolMode: PoolingType;
  backdrop: StageBackdropId;
  brushMode: 'draw' | 'erase';
  convStepIndex: number;
  poolStepIndex: number;
  isPlaying: boolean;
  speedMs: number;
  activePhotoSrc: string | null;
  activePhotoLabel: string | null;
  photoRgbaBuffer: Uint8ClampedArray | null;
  photoBufferWidth: number;
  photoBufferHeight: number;
  invertPhoto: boolean;
  onToggleInputCell: (row: number, col: number, value?: number) => void;
  onSetConvStep: (idx: number) => void;
  onSetPoolStep: (idx: number) => void;
  onTogglePlay: () => void;
  onStepPrev: () => void;
  onStepNext: () => void;
  onResetPlayback: () => void;
  onSetSpeedMs: (ms: number) => void;
}

export const StageSimulator: React.FC<StageSimulatorProps> = ({
  stage,
  onChangeStage,
  inputMatrix,
  kernel,
  padding,
  stride,
  bias,
  activationFn,
  leakySlope,
  poolMode,
  backdrop,
  brushMode,
  convStepIndex,
  poolStepIndex,
  isPlaying,
  speedMs,
  activePhotoSrc,
  activePhotoLabel,
  photoRgbaBuffer,
  photoBufferWidth,
  photoBufferHeight,
  invertPhoto,
  onToggleInputCell,
  onSetConvStep,
  onSetPoolStep,
  onTogglePlay,
  onStepPrev,
  onStepNext,
  onResetPlayback,
  onSetSpeedMs,
}) => {
  const actMeta = ACTIVATION_FUNCTIONS[activationFn];
  const convRes = convolve2D(
    inputMatrix,
    kernel,
    padding,
    stride,
    bias,
    activationFn,
    leakySlope
  );
  const actRes = applyActivation(convRes.featureMap, activationFn, leakySlope);
  const poolRes = pool2D(actRes.activated, 2, 2, poolMode);

  const safeConvIdx = Math.min(
    convStepIndex,
    Math.max(0, convRes.steps.length - 1)
  );
  const activeConvStep = convRes.steps[safeConvIdx];

  const safePoolIdx = Math.min(
    poolStepIndex,
    Math.max(0, poolRes.steps.length - 1)
  );
  const activePoolStep = poolRes.steps[safePoolIdx];

  const pipelineRes =
    stage === 'pipeline'
      ? classifyShapePipeline(inputMatrix, activationFn, leakySlope, poolMode)
      : null;

  const showPlayback = stage === 'conv' || stage === 'pool';
  const totalSteps =
    stage === 'pool' ? poolRes.steps.length : convRes.steps.length;
  const currentStepDisplay = stage === 'pool' ? safePoolIdx + 1 : safeConvIdx + 1;

  const activePixels = inputMatrix.flat().filter((v) => v > 0.5).length;

  const stages: Array<{ id: LearningStage; label: string }> = [
    { id: 'input', label: '1. Entrada' },
    { id: 'conv', label: '2. Convolución' },
    { id: 'relu', label: `3. ${actMeta.shortLabel}` },
    { id: 'pool', label: poolMode === 'max' ? '4. Max Pooling' : '4. Avg Pooling' },
    { id: 'pipeline', label: '5. Pipeline 8×8' },
  ];

  const backdropClass = `stage-backdrop-${backdrop}`;

  return (
    <div className="flex flex-col gap-4">
      {/* Stage Progression Tabs */}
      <div
        role="tablist"
        aria-label="Etapas del laboratorio 8 por 8"
        className="flex items-center gap-1 p-1.5 bg-[#0f172a] border border-slate-800 rounded-xl overflow-x-auto"
      >
        {stages.map((st) => {
          const active = stage === st.id;
          return (
            <button
              key={st.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => onChangeStage(st.id)}
              className={`flex-1 min-w-fit px-3.5 py-2 text-xs sm:text-sm font-medium rounded-lg whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                active
                  ? 'bg-sky-500/20 text-sky-300 border border-sky-400/70'
                  : 'text-slate-400 hover:text-slate-100 border border-transparent'
              }`}
            >
              {st.label}
            </button>
          );
        })}
      </div>

      {/* Main Simulation Canvas Card with Selectable Scientific Backdrop */}
      <div
        className={`${backdropClass} border border-slate-800 rounded-2xl p-4 sm:p-5`}
        data-backdrop={backdrop}
      >
        {showPlayback && (
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3.5 mb-4 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onTogglePlay}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-slate-950 bg-sky-400 hover:bg-sky-300 rounded-lg whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-3.5 h-3.5" aria-hidden="true" />
                    Pausar
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" aria-hidden="true" />
                    Reproducir
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={onStepPrev}
                aria-label="Paso anterior"
                className="inline-flex items-center justify-center w-9 h-9 text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                <SkipBack className="w-4 h-4" aria-hidden="true" />
              </button>

              <button
                type="button"
                onClick={onStepNext}
                aria-label="Paso siguiente"
                className="inline-flex items-center justify-center w-9 h-9 text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                <SkipForward className="w-4 h-4" aria-hidden="true" />
              </button>

              <button
                type="button"
                onClick={onResetPlayback}
                aria-label="Reiniciar animación"
                className="inline-flex items-center justify-center w-9 h-9 text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                <RotateCcw className="w-4 h-4" aria-hidden="true" />
              </button>
            </div>

            <div className="flex items-center gap-4">
              <span
                className="text-xs font-mono tabular-nums text-slate-300"
                data-testid="step-counter"
              >
                Paso {currentStepDisplay} / {totalSteps}
              </span>

              <div className="flex items-center gap-1 p-0.5 bg-slate-900 border border-slate-800 rounded-lg">
                {[
                  { label: '0.5x', ms: 600 },
                  { label: '1x', ms: 300 },
                  { label: '2x', ms: 140 },
                ].map((spd) => (
                  <button
                    key={spd.label}
                    type="button"
                    onClick={() => onSetSpeedMs(spd.ms)}
                    className={`px-2.5 py-1 text-xs font-mono rounded-md whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                      speedMs === spd.ms
                        ? 'bg-sky-500/20 text-sky-300 font-semibold'
                        : 'text-slate-400 hover:text-slate-100'
                    }`}
                  >
                    {spd.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* STAGE 1: INPUT PIXELS */}
        {stage === 'input' && (
          <div className="flex flex-col items-center gap-6">
            <div className="flex flex-col lg:flex-row items-center justify-center gap-8 w-full">
              <MatrixGrid
                title="Matriz de Entrada (Editable)"
                subtitle="8 × 8 intensidades (0.0 – 1.0)"
                matrix={inputMatrix}
                mode="input"
                editable={true}
                brushMode={brushMode}
                onCellToggle={onToggleInputCell}
                testId="input-grid"
              />

              <ArrowRight
                className="w-6 h-6 text-slate-500 rotate-90 lg:rotate-0 shrink-0"
                aria-hidden="true"
              />

              <div className="flex flex-col items-center bg-[#080d1a] border border-slate-800 rounded-xl p-5 min-w-[220px]">
                <h3 className="text-sm font-semibold text-slate-100 mb-1">
                  Vista visual cuantificada
                </h3>
                <p className="text-xs text-slate-400 mb-4">
                  Escala de grises 8 × 8
                </p>

                <div
                  className="grid gap-0 border border-slate-700 rounded overflow-hidden"
                  style={{
                    gridTemplateColumns: 'repeat(8, 22px)',
                    gridTemplateRows: 'repeat(8, 22px)',
                  }}
                  aria-hidden="true"
                >
                  {inputMatrix.map((row, r) =>
                    row.map((v, c) => {
                      const shade = Math.round(
                        Math.max(0, Math.min(1, v)) * 245 + 8
                      );
                      return (
                        <div
                          key={`pv-${r}-${c}`}
                          style={{
                            backgroundColor: `rgb(${shade}, ${shade}, ${shade})`,
                          }}
                        />
                      );
                    })
                  )}
                </div>

                <div className="w-full mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs font-mono tabular-nums text-slate-400">
                  <span>Celdas &gt; 0.5:</span>
                  <span className="font-semibold text-slate-100">
                    {activePixels} / 64
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* STAGE 2: CONVOLUTION */}
        {stage === 'conv' && activeConvStep && (
          <div className="flex flex-col gap-5">
            <div className="flex flex-col xl:flex-row items-center justify-center gap-5">
              <MatrixGrid
                title={
                  padding > 0
                    ? `Entrada con Padding (${convRes.paddedInput.length}×${convRes.paddedInput.length})`
                    : 'Entrada (8×8)'
                }
                subtitle={`Ventana en (${activeConvStep.startRow + 1}, ${activeConvStep.startCol + 1})`}
                matrix={convRes.paddedInput}
                mode="input"
                padding={padding}
                editable={true}
                brushMode={brushMode}
                onCellToggle={onToggleInputCell}
                highlightWindow={{
                  startRow: activeConvStep.startRow,
                  startCol: activeConvStep.startCol,
                  size: 3,
                  variant: 'conv',
                }}
                testId="conv-input-grid"
              />

              <span
                className="text-xl font-semibold text-slate-500"
                aria-hidden="true"
              >
                ∗
              </span>

              <MatrixGrid
                title="Kernel"
                subtitle="3 × 3"
                matrix={kernel}
                mode="kernel"
                compact={true}
                testId="conv-kernel-grid"
              />

              <ArrowRight
                className="w-5 h-5 text-slate-500 rotate-90 xl:rotate-0 shrink-0"
                aria-hidden="true"
              />

              <MatrixGrid
                title="Mapa de Características"
                subtitle={`${convRes.outRows} × ${convRes.outCols} · Clic para inspeccionar`}
                matrix={convRes.featureMap}
                mode="feature"
                selectedCell={{
                  row: activeConvStep.outRow,
                  col: activeConvStep.outCol,
                }}
                onCellSelect={(r, c) => {
                  const idx = convRes.steps.findIndex(
                    (s) => s.outRow === r && s.outCol === c
                  );
                  if (idx !== -1) onSetConvStep(idx);
                }}
                testId="conv-feature-grid"
              />
            </div>

            <DotProductBreakdown
              step={activeConvStep}
              kernel={kernel}
              activationLabel={actMeta.shortLabel}
            />
          </div>
        )}

        {/* STAGE 3: CUSTOMIZABLE ACTIVATION */}
        {stage === 'relu' && (
          <div className="flex flex-col gap-5">
            <div className="flex flex-col xl:flex-row items-center justify-center gap-6">
              <MatrixGrid
                title="Salida Lineal (Antes)"
                subtitle={`Mín: ${actRes.minVal} · Máx: ${actRes.maxVal}`}
                matrix={convRes.featureMap}
                mode="feature"
                selectedCell={
                  activeConvStep
                    ? { row: activeConvStep.outRow, col: activeConvStep.outCol }
                    : null
                }
                onCellSelect={(r, c) => {
                  const idx = convRes.steps.findIndex(
                    (s) => s.outRow === r && s.outCol === c
                  );
                  if (idx !== -1) onSetConvStep(idx);
                }}
                testId="relu-before-grid"
              />

              <div className="w-full max-w-[260px] shrink-0">
                <ActivationCurveChart
                  activationFn={activationFn}
                  leakySlope={leakySlope}
                  currentInput={activeConvStep?.output ?? 0}
                  currentOutput={activeConvStep?.reluOutput ?? 0}
                />
              </div>

              <MatrixGrid
                title={`Mapa Activado (${actMeta.shortLabel})`}
                subtitle={`Cero: ${actRes.zeroedCount} · Positivos: ${actRes.positiveCount}`}
                matrix={actRes.activated}
                mode="relu"
                referenceMatrix={
                  activationFn === 'relu' ? convRes.featureMap : undefined
                }
                selectedCell={
                  activeConvStep
                    ? { row: activeConvStep.outRow, col: activeConvStep.outCol }
                    : null
                }
                onCellSelect={(r, c) => {
                  const idx = convRes.steps.findIndex(
                    (s) => s.outRow === r && s.outCol === c
                  );
                  if (idx !== -1) onSetConvStep(idx);
                }}
                testId="relu-after-grid"
              />
            </div>

            <div className="bg-[#080d1a] border border-slate-800 rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-300">
              <span>
                Haz clic en cualquier celda o cambia entre ReLU, Leaky ReLU, Sigmoide y Tanh en el panel derecho.
              </span>
              <span className="font-mono tabular-nums text-sky-400">
                {actMeta.formula}
              </span>
            </div>
          </div>
        )}

        {/* STAGE 4: POOLING (MAX OR AVERAGE) */}
        {stage === 'pool' && activePoolStep && (
          <div className="flex flex-col gap-5">
            <div className="flex flex-col lg:flex-row items-center justify-center gap-6">
              <MatrixGrid
                title={`Mapa ${actMeta.shortLabel} de Entrada`}
                subtitle={`${actRes.activated.length} × ${actRes.activated[0]?.length ?? 0}`}
                matrix={actRes.activated}
                mode="relu"
                referenceMatrix={
                  activationFn === 'relu' ? convRes.featureMap : undefined
                }
                highlightWindow={{
                  startRow: activePoolStep.startRow,
                  startCol: activePoolStep.startCol,
                  size: 2,
                  variant: 'pool',
                  winnerLocalRow:
                    poolMode === 'max' ? activePoolStep.maxLocalRow : undefined,
                  winnerLocalCol:
                    poolMode === 'max' ? activePoolStep.maxLocalCol : undefined,
                }}
                testId="pool-input-grid"
              />

              <div className="flex flex-col items-center gap-2">
                <ArrowRight
                  className="w-6 h-6 text-emerald-400 rotate-90 lg:rotate-0"
                  aria-hidden="true"
                />
                <div className="px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/40 rounded-lg text-center">
                  <div className="text-[11px] font-medium text-emerald-300">
                    Ventana 2×2
                  </div>
                  <div className="text-xs font-mono font-semibold tabular-nums text-emerald-200">
                    {poolMode === 'max' ? 'MÁX' : 'PROM'} = {activePoolStep.selectedVal}
                  </div>
                </div>
              </div>

              <MatrixGrid
                title={
                  poolMode === 'max'
                    ? 'Mapa Comprimido (Max Pool)'
                    : 'Mapa Comprimido (Avg Pool)'
                }
                subtitle={`${poolRes.outRows} × ${poolRes.outCols} · −75% resolución`}
                matrix={poolRes.pooled}
                mode="pool"
                selectedCell={{
                  row: activePoolStep.outRow,
                  col: activePoolStep.outCol,
                }}
                onCellSelect={(r, c) => {
                  const idx = poolRes.steps.findIndex(
                    (s) => s.outRow === r && s.outCol === c
                  );
                  if (idx !== -1) onSetPoolStep(idx);
                }}
                testId="pool-output-grid"
              />
            </div>

            <div className="bg-[#080d1a] border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3">
              <div className="text-xs text-slate-300">
                Bloque 2×2 analizado:{' '}
                <span className="font-mono tabular-nums font-semibold text-slate-100">
                  [{activePoolStep.patch.flat().join(', ')}]
                </span>
              </div>
              <div className="text-xs font-mono tabular-nums text-emerald-400 font-semibold">
                {poolMode === 'max'
                  ? `Valor máximo conservado: ${activePoolStep.selectedVal}`
                  : `Promedio calculado: ${activePoolStep.selectedVal}`}
              </div>
            </div>
          </div>
        )}

        {/* STAGE 5: FULL PIPELINE 8x8 */}
        {stage === 'pipeline' && pipelineRes && (
          <div className="flex flex-col gap-6" data-testid="pipeline-view">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
              <div className="lg:col-span-5 flex flex-col items-center">
                <MatrixGrid
                  title="Entrada (Editable)"
                  subtitle="8 × 8 píxeles"
                  matrix={inputMatrix}
                  mode="input"
                  editable={true}
                  brushMode={brushMode}
                  compact={true}
                  onCellToggle={onToggleInputCell}
                />
              </div>

              <div className="lg:col-span-7 bg-[#080d1a] border border-slate-800 rounded-xl p-4">
                <div className="flex items-center justify-between gap-2 mb-3">
                  <h3 className="text-sm font-semibold text-slate-100 whitespace-nowrap">
                    Clasificación Softmax (8×8)
                  </h3>
                  <span className="text-xs font-mono tabular-nums text-slate-400 whitespace-nowrap">
                    {actMeta.shortLabel} · {poolMode === 'max' ? 'MaxPool' : 'AvgPool'}
                  </span>
                </div>

                <div className="space-y-2.5" data-testid="softmax-predictions">
                  {pipelineRes.predictions.map((pred, idx) => {
                    const pct = Math.round(pred.probability * 100);
                    const isTop = idx === 0;
                    return (
                      <div key={pred.id} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span
                            className={
                              isTop
                                ? 'font-semibold text-slate-100'
                                : 'text-slate-400'
                            }
                          >
                            {isTop ? `★ ${pred.label}` : pred.label}
                          </span>
                          <span
                            className={`font-mono tabular-nums ${
                              isTop
                                ? 'font-semibold text-sky-400'
                                : 'text-slate-400'
                            }`}
                          >
                            {pct}%
                          </span>
                        </div>
                        <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              isTop ? 'bg-sky-400' : 'bg-slate-600'
                            }`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-800 flex flex-col gap-3">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-sm font-semibold text-slate-100">
                  4 Ramas Paralelas (Conv 3×3 → {actMeta.shortLabel} →{' '}
                  {poolMode === 'max' ? 'MaxPool' : 'AvgPool'} 2×2)
                </h3>
                <span className="text-xs font-mono tabular-nums text-slate-400 whitespace-nowrap">
                  4 × 3×3 = 36 valores
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {pipelineRes.branches.map((branch) => (
                  <div
                    key={branch.id}
                    className="p-3.5 bg-[#080d1a] border border-slate-800 rounded-xl flex items-center justify-between gap-3"
                  >
                    <div>
                      <div className="text-xs font-semibold text-slate-100 whitespace-nowrap">
                        {branch.name.replace('Filtro ', '')}
                      </div>
                      <div className="text-[11px] font-mono tabular-nums text-slate-400 mt-0.5 whitespace-nowrap">
                        6×6 → 3×3
                      </div>
                    </div>

                    <div
                      className="grid grid-cols-3 gap-0.5 p-1 bg-slate-950 border border-slate-800 rounded shrink-0"
                      aria-label={`Mapa comprimido de ${branch.name}`}
                    >
                      {branch.pooled.flat().map((val, idx) => {
                        const intensity = Math.min(1, Math.abs(val) / 6);
                        return (
                          <div
                            key={idx}
                            style={{
                              backgroundColor:
                                val > 0
                                  ? `rgba(56, 189, 248, ${(0.22 + intensity * 0.78).toFixed(2)})`
                                  : val < 0
                                    ? `rgba(251, 113, 133, ${(0.22 + intensity * 0.78).toFixed(2)})`
                                    : '#0f172a',
                            }}
                            className="w-3.5 h-3.5 rounded-[2px]"
                            title={String(val)}
                          />
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activePhotoSrc && activePhotoLabel && (
          <div className="mt-5 pt-5 border-t border-slate-800">
            <PhotoConvolutionPreview
              imageSrc={activePhotoSrc}
              imageLabel={activePhotoLabel}
              rgbaBuffer={photoRgbaBuffer}
              bufferWidth={photoBufferWidth}
              bufferHeight={photoBufferHeight}
              kernel={kernel}
              bias={bias}
              activationFn={activationFn}
              leakySlope={leakySlope}
              poolMode={poolMode}
              invert={invertPhoto}
            />
          </div>
        )}
      </div>
    </div>
  );
};

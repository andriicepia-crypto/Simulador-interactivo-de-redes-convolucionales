import React, { useState } from 'react';
import { Check, Code2, Copy, Microscope, RotateCcw } from 'lucide-react';
import { ACTIVATION_FUNCTIONS, type ActivationFnId, type Matrix2D, type PoolingType } from '../engine/presets';
import type { StudioFilterSpec } from '../engine/studioEngine';
import { ActivationCurveChart } from './ActivationCurveChart';

interface StudioXRayDockProps {
  inputGrid: Matrix2D;
  activeFilter: StudioFilterSpec;
  activeChannelIndex: number;
  isFilterModified: boolean;
  convMap: Matrix2D;
  rawMap: Matrix2D;
  poolMap: Matrix2D;
  focusRow: number;
  focusCol: number;
  poolMode: PoolingType;
  activationFn: ActivationFnId;
  leakySlope: number;
  bias: number;
  onUpdateFilterWeight: (row: number, col: number, delta: number) => void;
  onResetActiveFilter: () => void;
}

export const StudioXRayDock: React.FC<StudioXRayDockProps> = ({
  inputGrid,
  activeFilter,
  isFilterModified,
  convMap,
  rawMap,
  poolMap,
  focusRow,
  focusCol,
  poolMode,
  activationFn,
  leakySlope,
  bias,
  onUpdateFilterWeight,
  onResetActiveFilter,
}) => {
  const [copied, setCopied] = useState(false);

  const actMeta = ACTIVATION_FUNCTIONS[activationFn];
  const pRow = Math.floor(focusRow / 2);
  const pCol = Math.floor(focusCol / 2);

  const patch3x3: Matrix2D = [];
  for (let ky = 0; ky < 3; ky++) {
    const r: number[] = [];
    for (let kx = 0; kx < 3; kx++) {
      r.push(inputGrid[focusRow + ky]?.[focusCol + kx] ?? 0);
    }
    patch3x3.push(r);
  }

  const rawVal = rawMap[focusRow]?.[focusCol] ?? 0;
  const actVal = convMap[focusRow]?.[focusCol] ?? 0;

  const poolPatch: number[] = [
    convMap[pRow * 2]?.[pCol * 2] ?? 0,
    convMap[pRow * 2]?.[pCol * 2 + 1] ?? 0,
    convMap[pRow * 2 + 1]?.[pCol * 2] ?? 0,
    convMap[pRow * 2 + 1]?.[pCol * 2 + 1] ?? 0,
  ];
  const pooledVal = poolMap[pRow]?.[pCol] ?? 0;

  const kerasActivation =
    activationFn === 'leaky_relu'
      ? `LeakyReLU(alpha=${leakySlope.toFixed(2)})`
      : `'${activationFn}'`;

  const kerasPoolLayer =
    poolMode === 'max'
      ? 'MaxPooling2D(pool_size=(2, 2))'
      : 'AveragePooling2D(pool_size=(2, 2))';

  const kerasSnippet =
    activationFn === 'leaky_relu'
      ? `model = Sequential([
  # Capa 2: 3 filtros 3x3 paralelos + Leaky ReLU (b=${bias.toFixed(2)})
  Conv2D(filters=3, kernel_size=(3, 3),
         use_bias=${bias !== 0 ? 'True' : 'False'}, input_shape=(24, 24, 1)),
  ${kerasActivation},

  # Capa 3: Reduce dimensiones de 22x22 a 11x11
  ${kerasPoolLayer},

  # Capa 4: Aplana 3 mapas (11*11*3 = 363) y clasifica
  Flatten(),
  Dense(units=4, activation='softmax')
])`
      : `model = Sequential([
  # Capa 2: 3 filtros 3x3 paralelos + activación ${actMeta.shortLabel}
  Conv2D(filters=3, kernel_size=(3, 3), activation=${kerasActivation},
         use_bias=${bias !== 0 ? 'True' : 'False'}, input_shape=(24, 24, 1)),

  # Capa 3: Reduce dimensiones de 22x22 a 11x11
  ${kerasPoolLayer},

  # Capa 4: Aplana 3 mapas (11*11*3 = 363) y clasifica
  Flatten(),
  Dense(units=4, activation='softmax')
])`;

  const handleCopyCode = async () => {
    try {
      await navigator.clipboard.writeText(kerasSnippet);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <section
      className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 sm:p-5 grid grid-cols-1 xl:grid-cols-12 gap-5"
      data-testid="studio-xray-dock"
    >
      {/* Left: Rayos X Matemáticos en Vivo */}
      <div className="xl:col-span-7 bg-[#080d1a] border border-slate-800/90 rounded-xl p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3.5">
          <div className="flex items-center gap-2 min-w-0">
            <Microscope className="w-4 h-4 text-amber-400 shrink-0" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-amber-300 truncate">
              Rayos X Matemáticos en Vivo —{' '}
              <span style={{ color: activeFilter.hex }}>{activeFilter.name}</span>
            </h3>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-[11px] font-mono tabular-nums text-slate-400">
              Conv ({focusRow}, {focusCol}) · Pool ({pRow}, {pCol})
            </span>
            {isFilterModified && (
              <button
                type="button"
                onClick={onResetActiveFilter}
                className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-amber-300 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 rounded-md whitespace-nowrap"
              >
                <RotateCcw className="w-3 h-3" aria-hidden="true" />
                Restaurar filtro
              </button>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Patch 3x3 */}
          <div className="flex flex-col items-center">
            <span className="text-[11px] text-slate-400 mb-1.5">
              Parche Entrada (3×3)
            </span>
            <div className="grid grid-cols-3 gap-1">
              {patch3x3.map((row, r) =>
                row.map((pVal, c) => {
                  const bgAlpha = Math.min(0.88, Math.max(0.06, pVal * 0.85));
                  return (
                    <div
                      key={`xp-${r}-${c}`}
                      style={{
                        backgroundColor: `rgba(248, 250, 252, ${bgAlpha.toFixed(2)})`,
                      }}
                      className={`w-10 h-8 rounded border border-slate-700 flex items-center justify-center font-mono text-[11px] font-semibold tabular-nums ${
                        pVal > 0.52 ? 'text-slate-950' : 'text-slate-100'
                      }`}
                    >
                      {pVal.toFixed(2)}
                    </div>
                  );
                })
              )}
            </div>
          </div>

          <span className="text-lg font-bold text-slate-500" aria-hidden="true">
            ×
          </span>

          {/* Editable Kernel 3x3 */}
          <div className="flex flex-col items-center">
            <span className="text-[11px] text-slate-400 mb-1.5">
              Pesos Kernel (Editables)
            </span>
            <div className="grid grid-cols-3 gap-1">
              {activeFilter.kernel.map((row, r) =>
                row.map((kVal, c) => (
                  <div
                    key={`xk-${r}-${c}`}
                    className={`w-[54px] h-8 rounded border flex items-center justify-between px-0.5 font-mono text-[11px] font-semibold tabular-nums ${
                      kVal > 0
                        ? 'bg-emerald-500/10 border-emerald-500/35 text-emerald-300'
                        : kVal < 0
                          ? 'bg-rose-500/10 border-rose-500/35 text-rose-300'
                          : 'bg-slate-900 border-slate-700 text-slate-400'
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => onUpdateFilterWeight(r, c, -1)}
                      aria-label={`Disminuir peso kernel fila ${r + 1} columna ${c + 1}`}
                      className="w-3.5 h-6 flex items-center justify-center rounded hover:bg-slate-700/70 text-slate-300"
                    >
                      −
                    </button>
                    <span>{kVal > 0 ? `+${kVal}` : kVal}</span>
                    <button
                      type="button"
                      onClick={() => onUpdateFilterWeight(r, c, 1)}
                      aria-label={`Aumentar peso kernel fila ${r + 1} columna ${c + 1}`}
                      className="w-3.5 h-6 flex items-center justify-center rounded hover:bg-slate-700/70 text-slate-300"
                    >
                      +
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          <span className="text-lg font-bold text-slate-500" aria-hidden="true">
            =
          </span>

          {/* Raw + Activation Result */}
          <div className="px-3.5 py-2.5 rounded-xl bg-slate-900 border border-sky-500/50 text-center font-mono tabular-nums min-w-[116px]">
            <div className="text-[11px] text-slate-400">
              z → {actMeta.shortLabel}
            </div>
            <div className="text-lg font-bold text-sky-400 mt-0.5">
              {actVal.toFixed(2)}
            </div>
            <div className="text-[11px] text-slate-400">
              Bruto: {rawVal.toFixed(2)}
            </div>
          </div>

          <span className="text-lg font-bold text-slate-500" aria-hidden="true">
            →
          </span>

          {/* 2x2 Pooling Grid */}
          <div className="flex flex-col items-center">
            <span className="text-[11px] text-slate-400 mb-1.5">
              Bloque Pooling (2×2)
            </span>
            <div className="grid grid-cols-2 gap-1">
              {poolPatch.map((v, idx) => {
                const isWinner =
                  poolMode === 'max' &&
                  Math.abs(v - pooledVal) < 0.002 &&
                  v !== 0;
                return (
                  <div
                    key={`xp2-${idx}`}
                    className={`w-11 h-8 rounded border flex items-center justify-center font-mono text-[11px] font-semibold tabular-nums ${
                      isWinner
                        ? 'bg-emerald-500/15 border-emerald-400 text-emerald-300'
                        : 'bg-slate-900 border-slate-700 text-slate-200'
                    }`}
                  >
                    {v.toFixed(2)}
                  </div>
                );
              })}
            </div>
          </div>

          <span className="text-lg font-bold text-slate-500" aria-hidden="true">
            =
          </span>

          {/* Final Pool Result */}
          <div className="px-3.5 py-2.5 rounded-xl bg-slate-900 border border-emerald-500/50 text-center font-mono tabular-nums min-w-[96px]">
            <div className="text-[11px] text-slate-400">
              {poolMode === 'max' ? 'Max Pool' : 'Avg Pool'}
            </div>
            <div className="text-lg font-bold text-emerald-400 mt-0.5">
              {pooledVal.toFixed(2)}
            </div>
          </div>
        </div>

        {/* Inline Activation Function Curve Strip */}
        <div className="mt-4 pt-3.5 border-t border-slate-800/80 grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
          <div className="md:col-span-5">
            <ActivationCurveChart
              activationFn={activationFn}
              leakySlope={leakySlope}
              currentInput={rawVal}
              currentOutput={actVal}
              compact={true}
            />
          </div>
          <div className="md:col-span-7 text-xs text-slate-300 space-y-1.5 leading-relaxed">
            <p className="font-semibold text-slate-100">
              {actMeta.name}{' '}
              <span className="font-mono text-sky-400 ml-1">
                ({actMeta.formula})
              </span>
            </p>
            <p className="text-slate-400">{actMeta.description}</p>
            <p className="text-[11px] text-slate-400 font-mono tabular-nums">
              Divisor de escala: ÷{activeFilter.div} · Sesgo aplicado: b ={' '}
              {bias >= 0 ? `+${bias.toFixed(2)}` : bias.toFixed(2)}
            </p>
          </div>
        </div>
      </div>

      {/* Right: Dynamic Python / Keras Equivalent */}
      <div className="xl:col-span-5 bg-[#080d1a] border border-slate-800/90 rounded-xl p-4 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2">
              <Code2 className="w-4 h-4 text-violet-400 shrink-0" aria-hidden="true" />
              <h3 className="text-sm font-semibold text-violet-300">
                Definición Equivalente en Python (Keras)
              </h3>
            </div>
            <button
              type="button"
              onClick={handleCopyCode}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md whitespace-nowrap"
            >
              {copied ? (
                <>
                  <Check className="w-3 h-3 text-emerald-400" aria-hidden="true" />
                  Copiado
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" aria-hidden="true" />
                  Copiar
                </>
              )}
            </button>
          </div>

          <pre className="bg-[#030712] border border-sky-500/20 rounded-xl p-3.5 text-xs font-mono text-sky-200 overflow-x-auto leading-relaxed">
            <code>{kerasSnippet}</code>
          </pre>
        </div>

        <p className="text-xs text-slate-400 mt-3 leading-relaxed">
          Al cambiar la función de activación, el tipo de pooling o el sesgo en la barra superior, esta arquitectura Keras se actualiza al instante reflejando el grafo exacto que estás simulando.
        </p>
      </div>
    </section>
  );
};

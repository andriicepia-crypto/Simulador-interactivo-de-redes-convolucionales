import React, { useState } from 'react';
import {
  computeArchitectureSummary,
  type ArchitectureParams,
} from '../engine/cnnMath';
import { ArrowRight, AlertTriangle } from 'lucide-react';

const INITIAL_PARAMS: ArchitectureParams = {
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
};

export const ArchitectureExplorer: React.FC = () => {
  const [params, setParams] = useState<ArchitectureParams>(INITIAL_PARAMS);
  const summary = computeArchitectureSummary(params);

  const updateParam = <K extends keyof ArchitectureParams>(
    key: K,
    val: ArchitectureParams[K]
  ) => {
    setParams((prev) => ({ ...prev, [key]: val }));
  };

  return (
    <div className="flex flex-col gap-6" data-testid="architecture-explorer">
      <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <h1 className="font-display text-2xl font-bold text-slate-50">
              Calculadora de Arquitectura y Tensores
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Ajusta los hiperparámetros de dos bloques convolucionales para observar cómo cambian las dimensiones espaciales y la cantidad de parámetros entrenables.
            </p>
          </div>

          <button
            type="button"
            onClick={() => setParams(INITIAL_PARAMS)}
            className="px-3.5 py-2 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            Restaurar valores LeNet/MNIST
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pb-6 border-b border-slate-800">
          {/* Input */}
          <div className="space-y-4">
            <h2 className="text-sm font-semibold text-slate-100">
              1. Tensor de entrada
            </h2>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <label htmlFor="arch-input-size" className="text-slate-300 font-medium">
                  Resolución espacial
                </label>
                <span className="font-mono tabular-nums text-sky-400 font-semibold">
                  {params.inputSize} × {params.inputSize} px
                </span>
              </div>
              <input
                id="arch-input-size"
                type="range"
                min={16}
                max={64}
                step={4}
                value={params.inputSize}
                onChange={(e) => updateParam('inputSize', Number(e.target.value))}
                className="w-full accent-sky-400"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300 font-medium">Canales de color</span>
                <span className="font-mono tabular-nums text-sky-400 font-semibold">
                  {params.inputChannels === 1 ? '1 (Grises)' : '3 (RGB)'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-1.5 p-1 bg-[#080d1a] border border-slate-800 rounded-lg">
                {[1, 3].map((ch) => (
                  <button
                    key={ch}
                    type="button"
                    onClick={() => updateParam('inputChannels', ch)}
                    className={`py-1.5 text-xs font-medium rounded-md ${
                      params.inputChannels === ch
                        ? 'bg-sky-500/20 text-sky-300'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {ch === 1 ? '1 Canal' : '3 Canales (RGB)'}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Conv Block 1 */}
          <div className="space-y-4">
            <h2 className="text-sm font-semibold text-slate-100">
              2. Bloque Convolucional 1
            </h2>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <label htmlFor="arch-filters-1" className="text-slate-300 font-medium">
                  Número de filtros
                </label>
                <span className="font-mono tabular-nums text-sky-400 font-semibold">
                  {params.numFilters1} filtros
                </span>
              </div>
              <input
                id="arch-filters-1"
                type="range"
                min={4}
                max={32}
                step={4}
                value={params.numFilters1}
                onChange={(e) => updateParam('numFilters1', Number(e.target.value))}
                className="w-full accent-sky-400"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <label htmlFor="arch-kernel-1" className="text-slate-300 font-medium">
                    Kernel
                  </label>
                  <span className="font-mono tabular-nums text-sky-400 font-semibold">
                    {params.kernel1}×{params.kernel1}
                  </span>
                </div>
                <input
                  id="arch-kernel-1"
                  type="range"
                  min={3}
                  max={5}
                  step={2}
                  value={params.kernel1}
                  onChange={(e) => updateParam('kernel1', Number(e.target.value))}
                  className="w-full accent-sky-400"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <label htmlFor="arch-pad-1" className="text-slate-300 font-medium">
                    Padding
                  </label>
                  <span className="font-mono tabular-nums text-sky-400 font-semibold">
                    P = {params.padding1}
                  </span>
                </div>
                <input
                  id="arch-pad-1"
                  type="range"
                  min={0}
                  max={2}
                  step={1}
                  value={params.padding1}
                  onChange={(e) => updateParam('padding1', Number(e.target.value))}
                  className="w-full accent-sky-400"
                />
              </div>
            </div>
          </div>

          {/* Conv Block 2 & Output */}
          <div className="space-y-4">
            <h2 className="text-sm font-semibold text-slate-100">
              3. Bloque Convolucional 2 y Salida
            </h2>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <label htmlFor="arch-filters-2" className="text-slate-300 font-medium">
                  Filtros Conv 2
                </label>
                <span className="font-mono tabular-nums text-sky-400 font-semibold">
                  {params.numFilters2} filtros
                </span>
              </div>
              <input
                id="arch-filters-2"
                type="range"
                min={8}
                max={64}
                step={8}
                value={params.numFilters2}
                onChange={(e) => updateParam('numFilters2', Number(e.target.value))}
                className="w-full accent-sky-400"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <label htmlFor="arch-classes" className="text-slate-300 font-medium">
                  Clases de salida (Softmax)
                </label>
                <span className="font-mono tabular-nums text-sky-400 font-semibold">
                  {params.numClasses} clases
                </span>
              </div>
              <input
                id="arch-classes"
                type="range"
                min={2}
                max={20}
                step={1}
                value={params.numClasses}
                onChange={(e) => updateParam('numClasses', Number(e.target.value))}
                className="w-full accent-sky-400"
              />
            </div>
          </div>
        </div>

        {!summary.valid && (
          <div className="mt-5 p-3.5 bg-rose-500/10 border border-rose-500/40 rounded-xl flex items-center gap-2.5 text-xs text-rose-300">
            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" aria-hidden="true" />
            <span>
              Las dimensiones espaciales colapsaron a 0 en una capa profunda. Aumenta el tamaño de entrada o añade Padding.
            </span>
          </div>
        )}

        <div className="mt-6">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
            <h2 className="text-sm font-semibold text-slate-100">
              Flujo de dimensiones tensoriales (Alto × Ancho × Canales)
            </h2>
            <div className="text-xs font-mono tabular-nums text-slate-300">
              Parámetros entrenables totales:{' '}
              <span className="font-semibold text-sky-400">
                {summary.totalParams.toLocaleString('es-ES')}
              </span>
            </div>
          </div>

          <div className="flex flex-wrap lg:flex-nowrap items-stretch gap-2.5 overflow-x-auto pb-2">
            {summary.layers.map((layer, index) => {
              const boxScale = Math.max(
                28,
                Math.min(72, layer.height * 1.6)
              );
              return (
                <React.Fragment key={layer.id}>
                  <div className="flex-1 min-w-[165px] bg-[#080d1a] border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
                    <div>
                      <div className="text-xs font-semibold text-slate-100">
                        {layer.name}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {layer.operation}
                      </div>
                    </div>

                    <div className="my-4 flex items-center justify-center h-20">
                      <div
                        className="rounded-md border border-sky-400/50 bg-sky-500/15 flex items-center justify-center font-mono tabular-nums text-xs font-semibold text-sky-300"
                        style={{
                          width: `${boxScale}px`,
                          height: `${boxScale}px`,
                        }}
                      >
                        {layer.height}×{layer.width}
                      </div>
                    </div>

                    <div className="pt-2.5 border-t border-slate-800 flex items-center justify-between text-[11px] font-mono tabular-nums">
                      <span className="text-slate-300">
                        {layer.height}×{layer.width}×{layer.channels}
                      </span>
                      <span className="text-slate-400">
                        {layer.params > 0
                          ? `${layer.params.toLocaleString('es-ES')} p.`
                          : '0 p.'}
                      </span>
                    </div>
                  </div>
                  {index < summary.layers.length - 1 && (
                    <div className="hidden lg:flex items-center justify-center shrink-0">
                      <ArrowRight
                        className="w-4 h-4 text-slate-500"
                        aria-hidden="true"
                      />
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

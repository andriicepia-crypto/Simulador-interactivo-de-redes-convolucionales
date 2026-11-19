import React from 'react';
import type { ConvolutionStepDetail } from '../engine/cnnMath';
import type { Matrix2D } from '../engine/presets';

interface DotProductBreakdownProps {
  step: ConvolutionStepDetail;
  kernel: Matrix2D;
  activationLabel?: string;
}

export const DotProductBreakdown: React.FC<DotProductBreakdownProps> = ({
  step,
  kernel,
  activationLabel = 'ReLU',
}) => {
  return (
    <div
      className="bg-[#080d1a] border border-slate-800 rounded-xl p-4"
      data-testid="dot-product-breakdown"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <h3 className="text-sm font-semibold text-slate-100">
          Cálculo en celda ({step.outRow + 1}, {step.outCol + 1})
        </h3>
        <span className="text-xs font-mono tabular-nums text-slate-400">
          Producto escalar elemento a elemento (3×3)
        </span>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3 sm:gap-4">
        <div className="flex flex-col items-center">
          <span className="text-[11px] font-medium text-slate-400 mb-1">
            Parche de entrada
          </span>
          <div className="grid grid-cols-3 gap-1 p-1.5 bg-sky-500/10 border border-sky-500/30 rounded-lg">
            {step.patch.map((row, r) =>
              row.map((val, c) => (
                <div
                  key={`p-${r}-${c}`}
                  className="w-8 h-7 rounded bg-slate-900 border border-slate-700 flex items-center justify-center text-xs font-mono tabular-nums text-slate-100"
                >
                  {Number.isInteger(val) ? val : val.toFixed(1)}
                </div>
              ))
            )}
          </div>
        </div>

        <span className="text-sm font-semibold text-slate-500" aria-hidden="true">
          ×
        </span>

        <div className="flex flex-col items-center">
          <span className="text-[11px] font-medium text-slate-400 mb-1">
            Kernel (Pesos)
          </span>
          <div className="grid grid-cols-3 gap-1 p-1.5 bg-slate-900 border border-slate-800 rounded-lg">
            {kernel.map((row, r) =>
              row.map((w, c) => (
                <div
                  key={`k-${r}-${c}`}
                  className={`w-8 h-7 rounded border flex items-center justify-center text-xs font-mono tabular-nums ${
                    w > 0
                      ? 'bg-sky-500/15 border-sky-500/40 text-sky-300'
                      : w < 0
                        ? 'bg-rose-500/15 border-rose-500/40 text-rose-300'
                        : 'bg-slate-950 border-slate-800 text-slate-500'
                  }`}
                >
                  {w}
                </div>
              ))
            )}
          </div>
        </div>

        <span className="text-sm font-semibold text-slate-500" aria-hidden="true">
          =
        </span>

        <div className="flex flex-col items-center">
          <span className="text-[11px] font-medium text-slate-400 mb-1">
            Productos
          </span>
          <div className="grid grid-cols-3 gap-1 p-1.5 bg-slate-900 border border-slate-800 rounded-lg">
            {step.products.map((row, r) =>
              row.map((prod, c) => (
                <div
                  key={`pr-${r}-${c}`}
                  className={`w-9 h-7 rounded border flex items-center justify-center text-xs font-mono tabular-nums ${
                    prod > 0
                      ? 'bg-sky-500/15 border-sky-500/40 text-sky-300 font-semibold'
                      : prod < 0
                        ? 'bg-rose-500/15 border-rose-500/40 text-rose-300 font-semibold'
                        : 'bg-slate-950 border-slate-800 text-slate-500'
                  }`}
                >
                  {Number.isInteger(prod) ? prod : prod.toFixed(1)}
                </div>
              ))
            )}
          </div>
        </div>

        <span className="text-sm font-semibold text-slate-500" aria-hidden="true">
          →
        </span>

        <div className="flex flex-col items-center justify-center px-3.5 py-2.5 bg-slate-900 border border-sky-500/40 text-white rounded-lg min-w-[116px]">
          <span className="text-[11px] text-slate-400">
            {step.bias !== 0 ? `Suma + b (${step.bias})` : 'Suma total'}
          </span>
          <span className="text-lg font-mono font-semibold tabular-nums text-slate-50">
            {step.output > 0 ? `+${step.output}` : step.output}
          </span>
          <span className="text-[11px] font-mono tabular-nums text-sky-400 mt-0.5">
            {activationLabel}: {step.reluOutput}
          </span>
        </div>
      </div>
    </div>
  );
};

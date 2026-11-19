import React from 'react';
import { evaluateActivation } from '../engine/cnnMath';
import { ACTIVATION_FUNCTIONS, type ActivationFnId } from '../engine/presets';

interface ActivationCurveChartProps {
  activationFn: ActivationFnId;
  leakySlope: number;
  currentInput: number;
  currentOutput: number;
  compact?: boolean;
}

export const ActivationCurveChart: React.FC<ActivationCurveChartProps> = ({
  activationFn,
  leakySlope,
  currentInput,
  currentOutput,
  compact = false,
}) => {
  const meta = ACTIVATION_FUNCTIONS[activationFn];
  const isBoundedUnit = activationFn === 'sigmoid' || activationFn === 'tanh';

  const minX = -6;
  const maxX = 6;
  const minY =
    activationFn === 'tanh' ? -1.2 : activationFn === 'sigmoid' ? -0.2 : -3;
  const maxY = isBoundedUnit ? 1.2 : 6;

  const mapX = (x: number) => {
    const clamped = Math.max(minX, Math.min(maxX, x));
    return 24 + ((clamped - minX) / (maxX - minX)) * 192;
  };

  const mapY = (y: number) => {
    const clamped = Math.max(minY, Math.min(maxY, y));
    return 112 - ((clamped - minY) / (maxY - minY)) * 90;
  };

  const zeroY = mapY(0);
  const zeroX = mapX(0);

  const samples = 48;
  const points: string[] = [];
  for (let i = 0; i <= samples; i++) {
    const z = minX + (i / samples) * (maxX - minX);
    const fz = evaluateActivation(z, activationFn, leakySlope);
    points.push(`${mapX(z).toFixed(1)},${mapY(fz).toFixed(1)}`);
  }

  const ptX = mapX(currentInput);
  const ptY = mapY(currentOutput);
  const isZeroOrNeg = currentOutput <= 0;

  return (
    <div
      className={`bg-[#080d1a] border border-slate-800 rounded-xl flex flex-col items-center ${
        compact ? 'p-3' : 'p-4'
      }`}
      data-testid="activation-curve-chart"
    >
      <div className="w-full flex items-center justify-between gap-2 mb-1">
        <span className="text-xs font-semibold text-slate-100 whitespace-nowrap">
          {meta.shortLabel}
        </span>
        <span
          className={`text-[11px] font-mono tabular-nums font-semibold whitespace-nowrap ${
            isZeroOrNeg ? 'text-rose-400' : 'text-sky-400'
          }`}
        >
          {currentOutput > 0
            ? 'ACTIVO'
            : currentOutput < 0
              ? 'NEGATIVO'
              : 'CERO'}
        </span>
      </div>
      <div className="w-full text-[11px] font-mono text-slate-400 mb-2 truncate">
        {meta.formula}
      </div>

      <svg
        viewBox="0 0 240 136"
        className={`w-full h-auto ${compact ? 'max-w-[200px]' : 'max-w-[260px]'}`}
        role="img"
        aria-label={`Gráfico de ${meta.name} mostrando entrada z=${currentInput} y salida ${currentOutput}`}
      >
        <line
          x1="24"
          y1={zeroY}
          x2="216"
          y2={zeroY}
          stroke="#334155"
          strokeWidth="1"
        />
        <line
          x1={zeroX}
          y1="16"
          x2={zeroX}
          y2="118"
          stroke="#334155"
          strokeWidth="1"
        />

        <text
          x="220"
          y={zeroY + 3}
          fontSize="10"
          fill="#94a3b8"
          fontFamily="IBM Plex Mono, monospace"
        >
          z
        </text>
        <text
          x={zeroX + 5}
          y="20"
          fontSize="10"
          fill="#94a3b8"
          fontFamily="IBM Plex Mono, monospace"
        >
          f(z)
        </text>
        <text
          x="24"
          y="130"
          fontSize="9"
          fill="#64748b"
          fontFamily="IBM Plex Mono, monospace"
        >
          -6
        </text>
        <text
          x={zeroX - 3}
          y="130"
          fontSize="9"
          fill="#64748b"
          fontFamily="IBM Plex Mono, monospace"
        >
          0
        </text>
        <text
          x="208"
          y="130"
          fontSize="9"
          fill="#64748b"
          fontFamily="IBM Plex Mono, monospace"
        >
          +6
        </text>

        <polyline
          fill="none"
          stroke="#38bdf8"
          strokeWidth="2.75"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points.join(' ')}
        />

        <line
          x1={ptX}
          y1={zeroY}
          x2={ptX}
          y2={ptY}
          stroke={isZeroOrNeg ? '#fb7185' : '#38bdf8'}
          strokeWidth="1"
          strokeDasharray="3 2"
        />

        <circle
          cx={ptX}
          cy={ptY}
          r="5"
          fill={isZeroOrNeg ? '#fb7185' : '#38bdf8'}
          stroke="#0f172a"
          strokeWidth="2"
        />
      </svg>

      <div className="w-full mt-2 pt-2 border-t border-slate-800 flex items-center justify-between text-[11px] font-mono tabular-nums text-slate-400">
        <span>z = {currentInput.toFixed(2)}</span>
        <span>→</span>
        <span
          className={
            isZeroOrNeg
              ? 'text-rose-400 font-semibold'
              : 'text-sky-400 font-semibold'
          }
        >
          f(z) = {currentOutput.toFixed(2)}
        </span>
      </div>
    </div>
  );
};

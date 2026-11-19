import React, { useState } from 'react';
import type { Matrix2D } from '../engine/presets';

export type GridMode = 'input' | 'feature' | 'relu' | 'pool' | 'kernel';

interface HighlightWindow {
  startRow: number;
  startCol: number;
  size: number;
  variant: 'conv' | 'pool';
  winnerLocalRow?: number;
  winnerLocalCol?: number;
}

interface MatrixGridProps {
  title: string;
  subtitle?: string;
  matrix: Matrix2D;
  mode: GridMode;
  padding?: number;
  highlightWindow?: HighlightWindow | null;
  selectedCell?: { row: number; col: number } | null;
  referenceMatrix?: Matrix2D;
  editable?: boolean;
  brushMode?: 'draw' | 'erase';
  compact?: boolean;
  onCellToggle?: (row: number, col: number, value?: number) => void;
  onCellSelect?: (row: number, col: number) => void;
  testId?: string;
}

function formatCellVal(val: number, mode: GridMode): string {
  if (mode === 'input') {
    return Number.isInteger(val) ? String(val) : val.toFixed(1);
  }
  if (Number.isInteger(val)) return String(val);
  return val.toFixed(1);
}

export const MatrixGrid: React.FC<MatrixGridProps> = ({
  title,
  subtitle,
  matrix,
  mode,
  padding = 0,
  highlightWindow = null,
  selectedCell = null,
  referenceMatrix,
  editable = false,
  brushMode = 'draw',
  compact = false,
  onCellToggle,
  onCellSelect,
  testId,
}) => {
  const [isDragging, setIsDragging] = useState(false);

  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;

  let maxAbs = 1;
  for (const r of matrix) {
    for (const v of r) {
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
  }

  const handleInputCellAction = (r: number, c: number, isClick: boolean) => {
    if (!editable || !onCellToggle) return;
    const origR = r - padding;
    const origC = c - padding;
    const origRows = rows - padding * 2;
    const origCols = cols - padding * 2;
    if (origR < 0 || origR >= origRows || origC < 0 || origC >= origCols) {
      return;
    }
    if (isClick) {
      const current = matrix[r][c];
      onCellToggle(origR, origC, current > 0.5 ? 0 : 1);
    } else {
      onCellToggle(origR, origC, brushMode === 'draw' ? 1 : 0);
    }
  };

  const cellSizeClass = compact
    ? 'w-7 h-7 text-[11px]'
    : rows >= 10
      ? 'w-8 h-8 sm:w-9 sm:h-9 text-xs'
      : rows >= 8
        ? 'w-9 h-9 sm:w-10 sm:h-10 text-xs sm:text-sm'
        : 'w-10 h-10 sm:w-11 sm:h-11 text-xs sm:text-sm';

  return (
    <div
      className="flex flex-col items-center"
      onMouseLeave={() => setIsDragging(false)}
      onMouseUp={() => setIsDragging(false)}
      data-testid={testId}
    >
      <div className="mb-2 text-center">
        <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
        {subtitle && (
          <p className="text-xs text-slate-400 font-mono tabular-nums mt-0.5">
            {subtitle}
          </p>
        )}
      </div>

      <div
        className="inline-grid gap-1 p-2 bg-[#080d1a] border border-slate-800 rounded-xl select-none"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
        role="grid"
        aria-label={title}
      >
        {matrix.map((row, rIdx) =>
          row.map((val, cIdx) => {
            const isPaddingCell =
              padding > 0 &&
              (rIdx < padding ||
                rIdx >= rows - padding ||
                cIdx < padding ||
                cIdx >= cols - padding);

            const inWindow =
              highlightWindow !== null &&
              rIdx >= highlightWindow.startRow &&
              rIdx < highlightWindow.startRow + highlightWindow.size &&
              cIdx >= highlightWindow.startCol &&
              cIdx < highlightWindow.startCol + highlightWindow.size;

            const isPoolWinner =
              inWindow &&
              highlightWindow?.variant === 'pool' &&
              highlightWindow.winnerLocalRow === rIdx - highlightWindow.startRow &&
              highlightWindow.winnerLocalCol === cIdx - highlightWindow.startCol;

            const isSelected =
              selectedCell !== null &&
              selectedCell.row === rIdx &&
              selectedCell.col === cIdx;

            const wasClampedByRelu =
              mode === 'relu' &&
              referenceMatrix &&
              referenceMatrix[rIdx]?.[cIdx] < 0 &&
              val === 0;

            let bgStyle: React.CSSProperties = {};
            let textClass = 'text-slate-200';
            let borderClass = 'border-slate-800';

            if (isPaddingCell) {
              bgStyle = { backgroundColor: '#0b1222' };
              textClass = 'text-slate-500';
              borderClass = 'border-dashed border-slate-700';
            } else if (mode === 'input') {
              const intensity = Math.max(0, Math.min(1, val));
              const shade = Math.round(intensity * 235 + 12);
              bgStyle = { backgroundColor: `rgb(${shade}, ${shade}, ${shade})` };
              textClass =
                intensity >= 0.55
                  ? 'text-slate-950 font-semibold'
                  : 'text-slate-200 font-medium';
              borderClass =
                intensity >= 0.55 ? 'border-slate-300' : 'border-slate-800';
            } else if (
              mode === 'kernel' ||
              mode === 'feature' ||
              mode === 'relu' ||
              mode === 'pool'
            ) {
              const norm = Math.min(1, Math.abs(val) / maxAbs);
              if (val > 0.001) {
                const alpha = 0.16 + norm * 0.72;
                bgStyle = {
                  backgroundColor: `rgba(56, 189, 248, ${alpha.toFixed(2)})`,
                };
                textClass =
                  norm > 0.55
                    ? 'text-slate-950 font-bold'
                    : 'text-sky-200 font-medium';
                borderClass = 'border-sky-500/40';
              } else if (val < -0.001) {
                const alpha = 0.16 + norm * 0.72;
                bgStyle = {
                  backgroundColor: `rgba(251, 113, 133, ${alpha.toFixed(2)})`,
                };
                textClass =
                  norm > 0.55
                    ? 'text-slate-950 font-bold'
                    : 'text-rose-200 font-medium';
                borderClass = 'border-rose-500/40';
              } else {
                bgStyle = {
                  backgroundColor: wasClampedByRelu
                    ? 'rgba(251, 113, 133, 0.08)'
                    : '#0f172a',
                };
                textClass = wasClampedByRelu
                  ? 'text-rose-400 font-medium'
                  : 'text-slate-500';
                borderClass = wasClampedByRelu
                  ? 'border-rose-500/40'
                  : 'border-slate-800';
              }
            }

            let ringClass = '';
            if (isPoolWinner) {
              ringClass =
                'ring-2 ring-emerald-400 ring-offset-1 ring-offset-[#080d1a] z-10';
            } else if (inWindow) {
              ringClass =
                highlightWindow?.variant === 'pool'
                  ? 'ring-2 ring-emerald-400 z-10'
                  : 'ring-2 ring-amber-400 z-10';
            } else if (isSelected) {
              ringClass =
                mode === 'pool'
                  ? 'ring-2 ring-emerald-400 ring-offset-1 ring-offset-[#080d1a] z-10'
                  : 'ring-2 ring-sky-400 ring-offset-1 ring-offset-[#080d1a] z-10';
            }

            const isInteractive =
              (editable && !isPaddingCell) || Boolean(onCellSelect);

            const ariaLabel = isPaddingCell
              ? `Celda de padding (${rIdx}, ${cIdx}): valor 0`
              : `${title} fila ${rIdx + 1} columna ${cIdx + 1}: valor ${formatCellVal(val, mode)}${
                  wasClampedByRelu ? ' (anulado por ReLU)' : ''
                }${isPoolWinner ? ' (máximo elegido)' : ''}`;

            return (
              <button
                key={`${rIdx}-${cIdx}`}
                type="button"
                disabled={!isInteractive}
                onMouseDown={() => {
                  if (editable && !isPaddingCell) {
                    setIsDragging(true);
                    handleInputCellAction(rIdx, cIdx, true);
                  } else if (onCellSelect) {
                    onCellSelect(rIdx, cIdx);
                  }
                }}
                onMouseEnter={() => {
                  if (isDragging && editable && !isPaddingCell) {
                    handleInputCellAction(rIdx, cIdx, false);
                  }
                }}
                onClick={(e) => {
                  if (e.detail === 0) {
                    if (editable && !isPaddingCell) {
                      handleInputCellAction(rIdx, cIdx, true);
                    } else if (onCellSelect) {
                      onCellSelect(rIdx, cIdx);
                    }
                  }
                }}
                style={bgStyle}
                aria-label={ariaLabel}
                className={`relative ${cellSizeClass} border rounded flex items-center justify-center font-mono tabular-nums focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${borderClass} ${textClass} ${ringClass} ${
                  isInteractive
                    ? 'cursor-pointer hover:scale-[1.04]'
                    : 'cursor-default'
                }`}
              >
                <span>{formatCellVal(val, mode)}</span>
                {wasClampedByRelu && (
                  <span
                    aria-hidden="true"
                    className="absolute top-0.5 right-0.5 w-1.5 h-1.5 rounded-full bg-rose-400"
                  />
                )}
                {isPoolWinner && (
                  <span
                    aria-hidden="true"
                    className="absolute -top-1 -right-1 px-1 text-[8px] leading-tight font-sans font-bold bg-emerald-400 text-slate-950 rounded"
                  >
                    MÁX
                  </span>
                )}
              </button>
            );
          })
        )}
      </div>
    </div>
  );
};

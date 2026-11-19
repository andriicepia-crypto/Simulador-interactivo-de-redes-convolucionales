import React, { useEffect, useRef, useState } from 'react';
import { processImageConvolution } from '../engine/imageProcessor';
import {
  ACTIVATION_FUNCTIONS,
  type ActivationFnId,
  type Matrix2D,
  type PoolingType,
} from '../engine/presets';
import { Image as ImageIcon } from 'lucide-react';

interface PhotoConvolutionPreviewProps {
  imageSrc: string;
  imageLabel: string;
  rgbaBuffer: Uint8ClampedArray | null;
  bufferWidth: number;
  bufferHeight: number;
  kernel: Matrix2D;
  bias: number;
  activationFn: ActivationFnId;
  leakySlope: number;
  poolMode: PoolingType;
  invert: boolean;
}

export const PhotoConvolutionPreview: React.FC<PhotoConvolutionPreviewProps> = ({
  imageSrc,
  imageLabel,
  rgbaBuffer,
  bufferWidth,
  bufferHeight,
  kernel,
  bias,
  activationFn,
  leakySlope,
  poolMode,
  invert,
}) => {
  const convCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const actCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const poolCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [imgFailed, setImgFailed] = useState(false);

  useEffect(() => {
    setImgFailed(false);
  }, [imageSrc]);

  useEffect(() => {
    if (!rgbaBuffer || bufferWidth <= 0 || bufferHeight <= 0) return;

    const result = processImageConvolution(
      rgbaBuffer,
      bufferWidth,
      bufferHeight,
      kernel,
      bias,
      activationFn,
      leakySlope,
      poolMode,
      invert
    );

    const drawRgba = (
      canvas: HTMLCanvasElement | null,
      data: Uint8ClampedArray,
      w: number,
      h: number
    ) => {
      if (!canvas) return;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const imageData = ctx.createImageData(w, h);
      imageData.data.set(data);
      ctx.putImageData(imageData, 0, 0);
    };

    drawRgba(convCanvasRef.current, result.convRgba, bufferWidth, bufferHeight);
    drawRgba(actCanvasRef.current, result.actRgba, bufferWidth, bufferHeight);
    drawRgba(
      poolCanvasRef.current,
      result.poolRgba,
      result.poolWidth,
      result.poolHeight
    );
  }, [
    rgbaBuffer,
    bufferWidth,
    bufferHeight,
    kernel,
    bias,
    activationFn,
    leakySlope,
    poolMode,
    invert,
  ]);

  const actLabel = ACTIVATION_FUNCTIONS[activationFn].shortLabel;

  return (
    <section
      className="bg-[#080d1a] border border-slate-800 rounded-xl p-4"
      data-testid="photo-convolution-preview"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">
            Visor Fotográfico en Vivo · {imageLabel}
          </h3>
          <p className="text-xs text-slate-400">
            Observa cómo el kernel actual y la función {actLabel} filtran la fotografía real ({bufferWidth}×{bufferHeight} px).
          </p>
        </div>
        <span className="text-xs font-mono tabular-nums text-sky-400">
          Azul (+) · Rojo (−)
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="flex flex-col items-center bg-[#0f172a] border border-slate-800 rounded-lg p-2.5">
          <div className="w-24 h-24 rounded border border-slate-700 overflow-hidden bg-slate-900 flex items-center justify-center">
            {!imgFailed ? (
              <img
                src={imageSrc}
                alt={`Imagen original: ${imageLabel}`}
                referrerPolicy="no-referrer"
                onError={() => setImgFailed(true)}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="flex flex-col items-center justify-center text-slate-500 p-2 text-center">
                <ImageIcon className="w-5 h-5 mb-1" aria-hidden="true" />
                <span className="text-[10px]">{imageLabel}</span>
              </div>
            )}
          </div>
          <span className="text-[11px] font-medium text-slate-300 mt-2 whitespace-nowrap">
            1. Original
          </span>
        </div>

        <div className="flex flex-col items-center bg-[#0f172a] border border-slate-800 rounded-lg p-2.5">
          <canvas
            ref={convCanvasRef}
            aria-label="Mapa convolucionado de la imagen"
            className="w-24 h-24 rounded border border-slate-700 bg-slate-950 [image-rendering:pixelated]"
          />
          <span className="text-[11px] font-medium text-slate-300 mt-2 whitespace-nowrap">
            2. Convolución 3×3
          </span>
        </div>

        <div className="flex flex-col items-center bg-[#0f172a] border border-slate-800 rounded-lg p-2.5">
          <canvas
            ref={actCanvasRef}
            aria-label={`Mapa tras función ${actLabel}`}
            className="w-24 h-24 rounded border border-slate-700 bg-slate-950 [image-rendering:pixelated]"
          />
          <span className="text-[11px] font-medium text-slate-300 mt-2 whitespace-nowrap">
            3. {actLabel}
          </span>
        </div>

        <div className="flex flex-col items-center bg-[#0f172a] border border-slate-800 rounded-lg p-2.5">
          <canvas
            ref={poolCanvasRef}
            aria-label="Mapa comprimido tras Pooling 2×2"
            className="w-24 h-24 rounded border border-slate-700 bg-slate-950 [image-rendering:pixelated]"
          />
          <span className="text-[11px] font-medium text-slate-300 mt-2 whitespace-nowrap">
            4. {poolMode === 'max' ? 'Max Pool 2×2' : 'Avg Pool 2×2'}
          </span>
        </div>
      </div>
    </section>
  );
};

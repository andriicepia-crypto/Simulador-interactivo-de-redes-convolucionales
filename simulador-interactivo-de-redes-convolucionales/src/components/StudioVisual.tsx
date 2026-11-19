import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ArrowLeftRight,
  Eraser,
  Pause,
  Pencil,
  Play,
  SkipForward,
  Trash2,
  Upload,
} from 'lucide-react';
import {
  ACTIVATION_FUNCTIONS,
  HERO_BACKDROP_SRC,
  SAMPLE_IMAGES,
  type ActivationFnId,
  type Matrix2D,
  type PoolingType,
  type SampleImageId,
  type StageBackdropId,
} from '../engine/presets';
import {
  computeStudioCNN,
  DEFAULT_STUDIO_FILTERS,
  STUDIO_CONV_SIZE,
  STUDIO_POOL_SIZE,
  type StudioFilterSpec,
} from '../engine/studioEngine';
import {
  generateProceduralSample24,
  sampleRgbaToMatrix,
  STUDIO_SIZE,
  type ProceduralSampleId,
} from '../engine/studioSamples';
import { StudioXRayDock } from './StudioXRayDock';

type StudioSourceId =
  | ProceduralSampleId
  | SampleImageId
  | 'uploaded'
  | 'custom';

interface StudioVisualProps {
  resetSignal: number;
}

const PROCEDURAL_SAMPLES: Array<{ id: ProceduralSampleId; label: string }> = [
  { id: 'cat', label: 'Gato' },
  { id: 'car', label: 'Coche' },
  { id: 'digit8', label: 'Dígito 8' },
  { id: 'house', label: 'Casa' },
];

function cloneFilters(filters: StudioFilterSpec[]): StudioFilterSpec[] {
  return filters.map((f) => ({
    ...f,
    kernel: f.kernel.map((row) => [...row]),
  }));
}

function drawMatrixCanvas(
  canvas: HTMLCanvasElement | null,
  matrix: Matrix2D,
  rgbColor: string,
  highlightRect: {
    x: number;
    y: number;
    w: number;
    h: number;
    color: string;
  } | null,
  showNums: boolean,
  isConvSigned = false
) {
  if (!canvas || matrix.length === 0) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const rows = matrix.length;
  const cols = matrix[0].length;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const rawVal = matrix[y][x];
      const clamped = Math.min(1, Math.max(0, Math.abs(rawVal)));

      if (isConvSigned && rawVal < -0.001) {
        ctx.fillStyle = `rgba(251, 113, 133, ${(clamped * 0.85).toFixed(3)})`;
      } else {
        ctx.fillStyle = `rgba(${rgbColor}, ${Math.min(1, Math.max(0, rawVal)).toFixed(3)})`;
      }
      ctx.fillRect(x * cellW, y * cellH, cellW, cellH);

      if (cellW >= 8) {
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.14)';
        ctx.lineWidth = 0.6;
        ctx.strokeRect(x * cellW, y * cellH, cellW, cellH);
      }

      if (showNums && cellW >= 10) {
        ctx.fillStyle = rawVal > 0.55 ? '#020617' : '#e2e8f0';
        ctx.font = '600 6.5px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const formatted = rawVal.toFixed(1).replace(/^0\./, '.');
        ctx.fillText(
          formatted,
          x * cellW + cellW / 2,
          y * cellH + cellH / 2
        );
      }
    }
  }

  if (highlightRect) {
    ctx.strokeStyle = highlightRect.color;
    ctx.lineWidth = 2;
    ctx.strokeRect(
      highlightRect.x * cellW + 0.5,
      highlightRect.y * cellH + 0.5,
      highlightRect.w * cellW - 1,
      highlightRect.h * cellH - 1
    );
  }
}

export const StudioVisual: React.FC<StudioVisualProps> = ({ resetSignal }) => {
  const [selectedSource, setSelectedSource] = useState<StudioSourceId>('cat');
  const [inputGrid, setInputGrid] = useState<Matrix2D>(() =>
    generateProceduralSample24('cat')
  );
  const [activePhotoSrc, setActivePhotoSrc] = useState<string | null>(null);
  const [activePhotoLabel, setActivePhotoLabel] = useState<string | null>(null);
  const [invertPhoto, setInvertPhoto] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [filters, setFilters] = useState<StudioFilterSpec[]>(() =>
    cloneFilters(DEFAULT_STUDIO_FILTERS)
  );
  const [activeChannel, setActiveChannel] = useState<number>(0);
  const [poolMode, setPoolMode] = useState<PoolingType>('max');
  const [activationFn, setActivationFn] = useState<ActivationFnId>('relu');
  const [leakySlope, setLeakySlope] = useState<number>(0.1);
  const [bias, setBias] = useState<number>(0);
  const [backdrop, setBackdrop] = useState<StageBackdropId>('studio_wave');

  const [brushMode, setBrushMode] = useState<'draw' | 'erase'>('draw');
  const [showNumbers, setShowNumbers] = useState<boolean>(false);
  const [focusRow, setFocusRow] = useState<number>(11);
  const [focusCol, setFocusCol] = useState<number>(6);

  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speedValue, setSpeedValue] = useState<number>(280); // higher = faster (delay = 500 - speedValue)
  const [isDrawing, setIsDrawing] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pipelineWrapperRef = useRef<HTMLDivElement | null>(null);
  const connectorSvgRef = useRef<SVGSVGElement | null>(null);

  const canvasInputRef = useRef<HTMLCanvasElement | null>(null);
  const canvasConvRefs = useRef<Array<HTMLCanvasElement | null>>([
    null,
    null,
    null,
  ]);
  const canvasPoolRefs = useRef<Array<HTMLCanvasElement | null>>([
    null,
    null,
    null,
  ]);
  const canvasFlattenRef = useRef<HTMLCanvasElement | null>(null);

  const [connectorPaths, setConnectorPaths] = useState<{
    convPoly: string;
    poolPoly: string;
    flatPath: string;
    visible: boolean;
  }>({ convPoly: '', poolPoly: '', flatPath: '', visible: false });

  // Reset when TopBar Reiniciar is triggered
  useEffect(() => {
    if (resetSignal === 0) return;
    setIsPlaying(false);
    setSelectedSource('cat');
    setActivePhotoSrc(null);
    setActivePhotoLabel(null);
    setInvertPhoto(false);
    setUploadError(null);
    setInputGrid(generateProceduralSample24('cat'));
    setFilters(cloneFilters(DEFAULT_STUDIO_FILTERS));
    setActiveChannel(0);
    setPoolMode('max');
    setActivationFn('relu');
    setLeakySlope(0.1);
    setBias(0);
    setBackdrop('studio_wave');
    setBrushMode('draw');
    setShowNumbers(false);
    setFocusRow(11);
    setFocusCol(6);
  }, [resetSignal]);

  // Sample real or uploaded photo into 24x24 grid whenever activePhotoSrc or invertPhoto changes
  useEffect(() => {
    if (!activePhotoSrc) return;
    let cancelled = false;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (cancelled) return;
      const off = document.createElement('canvas');
      off.width = 96;
      off.height = 96;
      const ctx = off.getContext('2d');
      if (!ctx) return;

      const minSide = Math.min(img.width, img.height);
      const sx = Math.floor((img.width - minSide) / 2);
      const sy = Math.floor((img.height - minSide) / 2);
      ctx.drawImage(img, sx, sy, minSide, minSide, 0, 0, 96, 96);
      const data = ctx.getImageData(0, 0, 96, 96).data;
      setInputGrid(sampleRgbaToMatrix(data, 96, 96, STUDIO_SIZE, invertPhoto));
      setUploadError(null);
    };
    img.onerror = () => {
      if (cancelled) return;
      setUploadError('No se pudo procesar la imagen seleccionada.');
    };
    img.src = activePhotoSrc;

    return () => {
      cancelled = true;
    };
  }, [activePhotoSrc, invertPhoto]);

  const cnnResult = useMemo(
    () =>
      computeStudioCNN(
        inputGrid,
        filters,
        poolMode,
        activationFn,
        leakySlope,
        bias
      ),
    [inputGrid, filters, poolMode, activationFn, leakySlope, bias]
  );

  // Scanner playback loop
  useEffect(() => {
    if (!isPlaying) return;
    const delay = Math.max(60, 500 - speedValue);
    const id = window.setInterval(() => {
      setFocusCol((prevCol) => {
        const nextCol = prevCol + 2;
        if (nextCol >= STUDIO_CONV_SIZE) {
          setFocusRow((prevRow) =>
            prevRow + 2 >= STUDIO_CONV_SIZE ? 0 : prevRow + 2
          );
          return 0;
        }
        return nextCol;
      });
    }, delay);
    return () => window.clearInterval(id);
  }, [isPlaying, speedValue]);

  // Global mouse/touch release for painting
  useEffect(() => {
    const stop = () => setIsDrawing(false);
    window.addEventListener('mouseup', stop);
    window.addEventListener('touchend', stop);
    return () => {
      window.removeEventListener('mouseup', stop);
      window.removeEventListener('touchend', stop);
    };
  }, []);

  const pRow = Math.floor(focusRow / 2);
  const pCol = Math.floor(focusCol / 2);

  // Render all 2D canvases
  useEffect(() => {
    drawMatrixCanvas(
      canvasInputRef.current,
      inputGrid,
      '255, 255, 255',
      { x: focusCol, y: focusRow, w: 3, h: 3, color: '#fbbf24' },
      showNumbers,
      false
    );

    const isSigned =
      activationFn === 'leaky_relu' ||
      activationFn === 'tanh' ||
      activationFn === 'linear';

    for (let c = 0; c < filters.length; c++) {
      drawMatrixCanvas(
        canvasConvRefs.current[c],
        cnnResult.convMaps[c],
        filters[c].rgb,
        c === activeChannel
          ? { x: pCol * 2, y: pRow * 2, w: 2, h: 2, color: '#34d399' }
          : null,
        false,
        isSigned
      );
      drawMatrixCanvas(
        canvasPoolRefs.current[c],
        cnnResult.poolMaps[c],
        '52, 211, 153',
        c === activeChannel
          ? { x: pCol, y: pRow, w: 1, h: 1, color: '#fbbf24' }
          : null,
        false,
        isSigned
      );
    }

    const fCanvas = canvasFlattenRef.current;
    if (fCanvas) {
      const fCtx = fCanvas.getContext('2d');
      if (fCtx) {
        fCtx.clearRect(0, 0, fCanvas.width, fCanvas.height);
        const barW = fCanvas.width / cnnResult.flattenVector.length;
        cnnResult.flattenVector.forEach((item, i) => {
          const alpha = Math.min(1, Math.max(0.08, Math.abs(item.val) * 1.4));
          fCtx.fillStyle =
            item.val < 0
              ? `rgba(251, 113, 133, ${alpha.toFixed(2)})`
              : `rgba(${item.rgb}, ${alpha.toFixed(2)})`;
          fCtx.fillRect(i * barW, 0, Math.ceil(barW), fCanvas.height);
        });
      }
    }
  }, [
    inputGrid,
    cnnResult,
    filters,
    activeChannel,
    focusRow,
    focusCol,
    pRow,
    pCol,
    showNumbers,
    activationFn,
  ]);

  // Compute SVG receptive-field connector polygons & Bezier
  const updateConnectors = useCallback(() => {
    const wrapperEl = pipelineWrapperRef.current;
    const inEl = canvasInputRef.current;
    const convEl = canvasConvRefs.current[activeChannel];
    const poolEl = canvasPoolRefs.current[activeChannel];
    const flatEl = canvasFlattenRef.current;

    if (!wrapperEl || !inEl || !convEl || !poolEl || !flatEl) return;

    const wrapper = wrapperEl.getBoundingClientRect();
    if (wrapper.width < 1140) {
      setConnectorPaths((prev) =>
        prev.visible ? { ...prev, visible: false } : prev
      );
      return;
    }

    const inRect = inEl.getBoundingClientRect();
    const convRect = convEl.getBoundingClientRect();
    const poolRect = poolEl.getBoundingClientRect();
    const flatRect = flatEl.getBoundingClientRect();

    const cellIn = inRect.width / STUDIO_SIZE;
    const x1 = inRect.left - wrapper.left + (focusCol + 3) * cellIn;
    const y1Top = inRect.top - wrapper.top + focusRow * cellIn;
    const y1Bot = y1Top + 3 * cellIn;

    const cellConv = convRect.width / STUDIO_CONV_SIZE;
    const x2 = convRect.left - wrapper.left + focusCol * cellConv;
    const y2 = convRect.top - wrapper.top + (focusRow + 0.5) * cellConv;

    const x2Right = convRect.left - wrapper.left + (pCol * 2 + 2) * cellConv;
    const y2PoolTop = convRect.top - wrapper.top + pRow * 2 * cellConv;
    const y2PoolBot = y2PoolTop + 2 * cellConv;

    const cellPool = poolRect.width / STUDIO_POOL_SIZE;
    const x3 = poolRect.left - wrapper.left + pCol * cellPool;
    const y3 = poolRect.top - wrapper.top + (pRow + 0.5) * cellPool;

    const x3Right = poolRect.left - wrapper.left + poolRect.width;
    const y3Mid = poolRect.top - wrapper.top + poolRect.height / 2;
    const x4 = flatRect.left - wrapper.left;
    const y4 = flatRect.top - wrapper.top + flatRect.height / 2;

    setConnectorPaths({
      convPoly: `${x1.toFixed(1)},${y1Top.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)} ${x1.toFixed(1)},${y1Bot.toFixed(1)}`,
      poolPoly: `${x2Right.toFixed(1)},${y2PoolTop.toFixed(1)} ${x3.toFixed(1)},${y3.toFixed(1)} ${x2Right.toFixed(1)},${y2PoolBot.toFixed(1)}`,
      flatPath: `M ${x3Right.toFixed(1)} ${y3Mid.toFixed(1)} C ${(x3Right + 32).toFixed(1)} ${y3Mid.toFixed(1)}, ${(x4 - 32).toFixed(1)} ${y4.toFixed(1)}, ${x4.toFixed(1)} ${y4.toFixed(1)}`,
      visible: true,
    });
  }, [activeChannel, focusRow, focusCol, pRow, pCol]);

  useLayoutEffect(() => {
    updateConnectors();
  }, [updateConnectors, cnnResult]);

  useEffect(() => {
    const el = pipelineWrapperRef.current;
    if (!el) return;
    const obs = new ResizeObserver(() => {
      updateConnectors();
    });
    obs.observe(el);
    window.addEventListener('resize', updateConnectors);
    return () => {
      obs.disconnect();
      window.removeEventListener('resize', updateConnectors);
    };
  }, [updateConnectors]);

  const handleLoadProcedural = (id: ProceduralSampleId) => {
    setIsPlaying(false);
    setSelectedSource(id);
    setActivePhotoSrc(null);
    setActivePhotoLabel(null);
    setInvertPhoto(false);
    setUploadError(null);
    setInputGrid(generateProceduralSample24(id));
  };

  const handleLoadRealSample = (id: SampleImageId) => {
    setIsPlaying(false);
    const sample = SAMPLE_IMAGES[id];
    setSelectedSource(id);
    setActivePhotoSrc(sample.src);
    setActivePhotoLabel(sample.shortLabel);
    setInvertPhoto(sample.invertByDefault);
    setUploadError(null);
  };

  const handleImageUpload = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setUploadError('Selecciona un archivo de imagen válido (PNG, JPG, WebP).');
      return;
    }
    setIsPlaying(false);
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setSelectedSource('uploaded');
        setActivePhotoSrc(reader.result);
        setActivePhotoLabel(file.name);
        setInvertPhoto(false);
        setUploadError(null);
      }
    };
    reader.onerror = () => {
      setUploadError('Error al leer la imagen seleccionada.');
    };
    reader.readAsDataURL(file);
  };

  const handleInvertCurrentGrid = () => {
    if (activePhotoSrc) {
      setInvertPhoto((v) => !v);
    } else {
      setInputGrid((prev) =>
        prev.map((row) =>
          row.map((v) => Math.round(Math.max(0, Math.min(1, 1 - v)) * 100) / 100)
        )
      );
      setInvertPhoto((v) => !v);
    }
  };

  const handleClearGrid = () => {
    setIsPlaying(false);
    setSelectedSource('custom');
    setActivePhotoSrc(null);
    setActivePhotoLabel(null);
    setInvertPhoto(false);
    setInputGrid(
      Array.from({ length: STUDIO_SIZE }, () => Array(STUDIO_SIZE).fill(0))
    );
  };

  const paintAtClientCoords = (
    clientX: number,
    clientY: number,
    shouldPaint: boolean
  ) => {
    const canvas = canvasInputRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((clientX - rect.left) / rect.width) * STUDIO_SIZE);
    const y = Math.floor(((clientY - rect.top) / rect.height) * STUDIO_SIZE);
    if (x < 0 || x >= STUDIO_SIZE || y < 0 || y >= STUDIO_SIZE) return;

    setFocusRow(Math.min(STUDIO_CONV_SIZE - 1, Math.max(0, y - 1)));
    setFocusCol(Math.min(STUDIO_CONV_SIZE - 1, Math.max(0, x - 1)));

    if (!shouldPaint) return;

    setSelectedSource('custom');
    setInputGrid((prev) => {
      const next = prev.map((r) => [...r]);
      if (brushMode === 'draw') {
        next[y][x] = 1.0;
        if (x + 1 < STUDIO_SIZE) next[y][x + 1] = Math.max(next[y][x + 1], 0.55);
        if (y + 1 < STUDIO_SIZE) next[y + 1][x] = Math.max(next[y + 1][x], 0.55);
      } else {
        next[y][x] = 0;
        if (x + 1 < STUDIO_SIZE) next[y][x + 1] = 0;
        if (y + 1 < STUDIO_SIZE) next[y + 1][x] = 0;
      }
      return next;
    });
  };

  const handleConvPointerInspect = (
    channelIdx: number,
    clientX: number,
    clientY: number
  ) => {
    if (isPlaying) return;
    const canvas = canvasConvRefs.current[channelIdx];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(
      ((clientX - rect.left) / rect.width) * STUDIO_CONV_SIZE
    );
    const y = Math.floor(
      ((clientY - rect.top) / rect.height) * STUDIO_CONV_SIZE
    );
    if (x >= 0 && x < STUDIO_CONV_SIZE && y >= 0 && y < STUDIO_CONV_SIZE) {
      setActiveChannel(channelIdx);
      setFocusRow(y);
      setFocusCol(x);
    }
  };

  const handleStepScanner = () => {
    setIsPlaying(false);
    setFocusCol((prevCol) => {
      const nextCol = prevCol + 2;
      if (nextCol >= STUDIO_CONV_SIZE) {
        setFocusRow((prevRow) =>
          prevRow + 2 >= STUDIO_CONV_SIZE ? 0 : prevRow + 2
        );
        return 0;
      }
      return nextCol;
    });
  };

  const handleUpdateFilterWeight = (r: number, c: number, delta: number) => {
    setFilters((prev) =>
      prev.map((f, idx) => {
        if (idx !== activeChannel) return f;
        return {
          ...f,
          kernel: f.kernel.map((row, ri) =>
            row.map((val, ci) =>
              ri === r && ci === c ? Math.max(-8, Math.min(8, val + delta)) : val
            )
          ),
        };
      })
    );
  };

  const handleResetActiveFilter = () => {
    setFilters((prev) =>
      prev.map((f, idx) =>
        idx === activeChannel
          ? {
              ...DEFAULT_STUDIO_FILTERS[idx],
              kernel: DEFAULT_STUDIO_FILTERS[idx].kernel.map((r) => [...r]),
            }
          : f
      )
    );
  };

  const activeFilter = filters[activeChannel];
  const defaultFilter = DEFAULT_STUDIO_FILTERS[activeChannel];
  const isFilterModified = activeFilter.kernel.some((row, r) =>
    row.some((val, c) => val !== defaultFilter.kernel[r][c])
  );

  const actMeta = ACTIVATION_FUNCTIONS[activationFn];
  const activationList = Object.values(ACTIVATION_FUNCTIONS);
  const sampleImageList = Object.values(SAMPLE_IMAGES);
  const isCustomOrPhoto =
    selectedSource === 'facade' ||
    selectedSource === 'leaf' ||
    selectedSource === 'digit' ||
    selectedSource === 'uploaded';

  return (
    <div className="space-y-4" data-testid="studio-visual">
      {/* Studio Observatory Hero Card */}
      <section className="relative overflow-hidden rounded-2xl border border-slate-800 bg-[#0f172a]">
        <img
          src={HERO_BACKDROP_SRC}
          alt=""
          aria-hidden="true"
          referrerPolicy="no-referrer"
          className="absolute inset-0 w-full h-full object-cover object-center opacity-20 pointer-events-none"
        />
        <div
          aria-hidden="true"
          className="absolute inset-0 bg-gradient-to-r from-[#070b14]/95 via-[#0f172a]/85 to-[#0f172a]/70 pointer-events-none"
        />
        <div className="relative px-5 py-4 sm:px-6 sm:py-5 flex flex-wrap items-center justify-between gap-4">
          <div className="max-w-2xl">
            <h1 className="font-display text-xl sm:text-2xl font-bold tracking-tight text-slate-50 text-balance">
              Estudio Visual Interactivo de Redes Neuronales Convolucionales
            </h1>
            <p className="text-xs sm:text-sm text-slate-300 mt-1 leading-relaxed">
              Carga fotografías reales, dibuja en el lienzo 24×24 y observa 3 filtros paralelos, activaciones personalizables y pooling en vivo.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setIsPlaying((p) => !p)}
              className={`inline-flex items-center gap-1.5 px-4 py-2 text-xs sm:text-sm font-semibold rounded-lg whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                isPlaying
                  ? 'bg-amber-400 text-slate-950 hover:bg-amber-300'
                  : 'bg-sky-400 text-slate-950 hover:bg-sky-300'
              }`}
            >
              {isPlaying ? (
                <>
                  <Pause className="w-4 h-4" aria-hidden="true" />
                  Pausar Escáner
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" aria-hidden="true" />
                  Iniciar Escáner
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleStepScanner}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs sm:text-sm font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
            >
              <SkipForward className="w-4 h-4" aria-hidden="true" />
              Paso
            </button>
          </div>
        </div>
      </section>

      {/* Studio Control Toolbar */}
      <section
        aria-label="Barra de controles del estudio"
        className="bg-[#0f172a] border border-slate-800 rounded-2xl p-3.5 sm:px-5 space-y-3"
      >
        {/* Row 1: Samples, Real Photos, Upload & Canvas Tools */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Procedural 24x24 Samples */}
          <div className="flex flex-wrap items-center gap-1.5 pr-3 border-r border-slate-800">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mr-1">
              Muestras:
            </span>
            {PROCEDURAL_SAMPLES.map((s) => {
              const active = selectedSource === s.id;
              return (
                <button
                  key={s.id}
                  type="button"
                  aria-pressed={active}
                  onClick={() => handleLoadProcedural(s.id)}
                  className={`px-2.5 py-1.5 text-xs font-medium rounded-lg border whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                    active
                      ? 'bg-sky-500/20 text-sky-300 border-sky-400'
                      : 'bg-slate-900 text-slate-200 border-slate-700 hover:border-slate-500'
                  }`}
                >
                  {s.label}
                </button>
              );
            })}
          </div>

          {/* Real Photographic Samples + Upload */}
          <div className="flex flex-wrap items-center gap-1.5 pr-3 border-r border-slate-800">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mr-1">
              Fotos reales:
            </span>
            {sampleImageList.map((img) => {
              const active = selectedSource === img.id;
              return (
                <button
                  key={img.id}
                  type="button"
                  aria-label={`Cargar fotografía: ${img.name}`}
                  aria-pressed={active}
                  onClick={() => handleLoadRealSample(img.id)}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-lg border whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                    active
                      ? 'bg-sky-500/20 text-sky-300 border-sky-400'
                      : 'bg-slate-900 text-slate-200 border-slate-700 hover:border-slate-500'
                  }`}
                >
                  <img
                    src={img.src}
                    alt=""
                    aria-hidden="true"
                    referrerPolicy="no-referrer"
                    className="w-4 h-4 rounded-xs object-cover"
                  />
                  {img.shortLabel}
                </button>
              );
            })}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              aria-label="Subir foto desde tu dispositivo"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) {
                  handleImageUpload(f);
                  e.target.value = '';
                }
              }}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-950 bg-emerald-400 hover:bg-emerald-300 rounded-lg whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-300"
            >
              <Upload className="w-3.5 h-3.5" aria-hidden="true" />
              Subir Foto
            </button>

            <button
              type="button"
              onClick={handleInvertCurrentGrid}
              aria-pressed={invertPhoto}
              className={`inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-lg border whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                invertPhoto
                  ? 'bg-amber-400/20 text-amber-300 border-amber-400'
                  : 'bg-slate-900 text-slate-300 border-slate-700 hover:border-slate-500'
              }`}
            >
              <ArrowLeftRight className="w-3.5 h-3.5" aria-hidden="true" />
              Invertir
            </button>
          </div>

          {/* Brush & Canvas Controls */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mr-1">
              Lienzo:
            </span>
            <button
              type="button"
              onClick={() => setBrushMode('draw')}
              aria-pressed={brushMode === 'draw'}
              className={`inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-lg border whitespace-nowrap ${
                brushMode === 'draw'
                  ? 'bg-sky-500/20 text-sky-300 border-sky-400'
                  : 'bg-slate-900 text-slate-300 border-slate-700 hover:border-slate-500'
              }`}
            >
              <Pencil className="w-3.5 h-3.5" aria-hidden="true" />
              Pintar
            </button>
            <button
              type="button"
              onClick={() => setBrushMode('erase')}
              aria-pressed={brushMode === 'erase'}
              className={`inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-lg border whitespace-nowrap ${
                brushMode === 'erase'
                  ? 'bg-sky-500/20 text-sky-300 border-sky-400'
                  : 'bg-slate-900 text-slate-300 border-slate-700 hover:border-slate-500'
              }`}
            >
              <Eraser className="w-3.5 h-3.5" aria-hidden="true" />
              Borrar
            </button>
            <button
              type="button"
              onClick={handleClearGrid}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-200 bg-slate-900 hover:bg-rose-500/15 hover:text-rose-300 border border-slate-700 hover:border-rose-500/50 rounded-lg whitespace-nowrap"
            >
              <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
              Limpiar
            </button>

            <label className="inline-flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer ml-1 select-none whitespace-nowrap">
              <input
                type="checkbox"
                checked={showNumbers}
                onChange={(e) => setShowNumbers(e.target.checked)}
                className="accent-sky-400 w-3.5 h-3.5 rounded"
              />
              Ver valores
            </label>
          </div>
        </div>

        {/* Row 2: Customizable Functions, Pooling, Bias, Speed & Backdrop */}
        <div className="pt-2.5 border-t border-slate-800/80 flex flex-wrap items-center gap-2.5">
          {/* Activation Functions */}
          <div className="flex flex-wrap items-center gap-1 pr-2.5 border-r border-slate-800">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mr-1">
              Activación:
            </span>
            {activationList.map((act) => {
              const active = activationFn === act.id;
              return (
                <button
                  key={act.id}
                  type="button"
                  aria-label={`Función de activación: ${act.name}`}
                  aria-pressed={active}
                  onClick={() => setActivationFn(act.id)}
                  className={`px-2 py-1 text-xs font-medium rounded-lg border whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                    active
                      ? 'bg-sky-500/20 text-sky-300 border-sky-400'
                      : 'bg-slate-900 text-slate-300 border-slate-700 hover:border-slate-500'
                  }`}
                >
                  {act.shortLabel}
                </button>
              );
            })}

            {activationFn === 'leaky_relu' && (
              <label className="inline-flex items-center gap-1 text-[11px] font-mono text-sky-300 ml-1">
                <span>α={leakySlope.toFixed(2)}</span>
                <input
                  type="range"
                  min={0.05}
                  max={0.4}
                  step={0.05}
                  value={leakySlope}
                  onChange={(e) => setLeakySlope(Number(e.target.value))}
                  aria-label="Pendiente negativa Leaky ReLU"
                  className="w-14 accent-sky-400"
                />
              </label>
            )}
          </div>

          {/* Pooling Type */}
          <div className="flex items-center gap-1 pr-2.5 border-r border-slate-800">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mr-1">
              Pooling:
            </span>
            <button
              type="button"
              onClick={() => setPoolMode('max')}
              aria-pressed={poolMode === 'max'}
              className={`px-2 py-1 text-xs font-medium rounded-lg border whitespace-nowrap ${
                poolMode === 'max'
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-400'
                  : 'bg-slate-900 text-slate-300 border-slate-700 hover:border-slate-500'
              }`}
            >
              Max Pooling
            </button>
            <button
              type="button"
              onClick={() => setPoolMode('avg')}
              aria-pressed={poolMode === 'avg'}
              className={`px-2 py-1 text-xs font-medium rounded-lg border whitespace-nowrap ${
                poolMode === 'avg'
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-400'
                  : 'bg-slate-900 text-slate-300 border-slate-700 hover:border-slate-500'
              }`}
            >
              Average Pooling
            </button>
          </div>

          {/* Bias & Speed Sliders */}
          <div className="flex flex-wrap items-center gap-3 pr-2.5 border-r border-slate-800">
            <label className="inline-flex items-center gap-1.5 text-xs text-slate-300">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 whitespace-nowrap">
                Sesgo b:
              </span>
              <input
                type="range"
                min={-1.5}
                max={1.5}
                step={0.25}
                value={bias}
                onChange={(e) => setBias(Number(e.target.value))}
                aria-label="Sesgo convolucional"
                className="w-16 accent-sky-400"
              />
              <span className="font-mono tabular-nums text-sky-300 w-10">
                {bias >= 0 ? `+${bias.toFixed(2)}` : bias.toFixed(2)}
              </span>
            </label>

            <label className="inline-flex items-center gap-1.5 text-xs text-slate-300">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 whitespace-nowrap">
                Velocidad:
              </span>
              <input
                type="range"
                min={60}
                max={420}
                step={20}
                value={speedValue}
                onChange={(e) => setSpeedValue(Number(e.target.value))}
                aria-label="Velocidad del escáner"
                className="w-16 accent-sky-400"
              />
            </label>
          </div>

          {/* Stage Backdrop */}
          <div className="flex items-center gap-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mr-1">
              Fondo:
            </span>
            {[
              { id: 'studio_wave' as const, label: 'Óptico' },
              { id: 'technical_grid' as const, label: 'Rejilla' },
              { id: 'blueprint' as const, label: 'Blueprint' },
            ].map((bg) => (
              <button
                key={bg.id}
                type="button"
                aria-pressed={backdrop === bg.id}
                onClick={() => setBackdrop(bg.id)}
                className={`px-2 py-1 text-xs font-medium rounded-lg border whitespace-nowrap ${
                  backdrop === bg.id
                    ? 'bg-slate-700 text-slate-50 border-slate-500'
                    : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
                }`}
              >
                {bg.label}
              </button>
            ))}
          </div>
        </div>

        {uploadError && (
          <p className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg px-3 py-1.5">
            {uploadError}
          </p>
        )}
      </section>

      {/* 4-COLUMN VISUAL CNN PIPELINE WITH DYNAMIC SVG RECEPTIVE-FIELD CONNECTORS */}
      <section
        ref={pipelineWrapperRef}
        data-backdrop={backdrop}
        className={`relative stage-backdrop-${backdrop} border border-slate-800 rounded-2xl p-4 sm:p-5 overflow-hidden`}
        data-testid="studio-pipeline-stage"
      >
        {/* Dynamic Receptive Field Cones (overlay above cards) */}
        <svg
          ref={connectorSvgRef}
          aria-hidden="true"
          className="absolute inset-0 w-full h-full pointer-events-none z-30"
        >
          {connectorPaths.visible && (
            <>
              <polygon
                points={connectorPaths.convPoly}
                fill="rgba(251, 191, 36, 0.14)"
                stroke="#fbbf24"
                strokeWidth="1.25"
                strokeDasharray="3,2"
              />
              <polygon
                points={connectorPaths.poolPoly}
                fill="rgba(52, 211, 153, 0.14)"
                stroke="#34d399"
                strokeWidth="1.25"
                strokeDasharray="3,2"
              />
              <path
                d={connectorPaths.flatPath}
                fill="none"
                stroke="rgba(167, 139, 250, 0.7)"
                strokeWidth="2"
                strokeDasharray="4,3"
              />
            </>
          )}
        </svg>

        <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-[272px_1fr_1fr_272px] gap-4 sm:gap-5 items-stretch">
          {/* COLUMN 1: CAPA 1 ENTRADA 24x24x1 */}
          <article className="bg-[#0f172a]/95 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
            <div className="w-full flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider mb-1">
              <span className="text-slate-200">Capa 1 · Entrada</span>
              <span className="font-mono tabular-nums text-slate-400">
                24 × 24 × 1
              </span>
            </div>
            <h2 className="font-display text-base font-bold text-slate-50 text-center">
              Imagen en Escala de Grises
            </h2>
            <p className="text-xs text-slate-400 text-center mt-1 mb-3 leading-snug">
              Cada píxel oscila entre <strong className="text-slate-200">0.0</strong> (negro) y <strong className="text-slate-200">1.0</strong> (blanco). Dibuja encima o sube una foto.
            </p>

            <canvas
              ref={canvasInputRef}
              width={240}
              height={240}
              aria-label="Lienzo interactivo de entrada 24 por 24 píxeles"
              onMouseDown={(e) => {
                setIsDrawing(true);
                paintAtClientCoords(e.clientX, e.clientY, true);
              }}
              onMouseMove={(e) => {
                paintAtClientCoords(e.clientX, e.clientY, isDrawing);
              }}
              onTouchStart={(e) => {
                e.preventDefault();
                setIsDrawing(true);
                const t = e.touches[0];
                if (t) paintAtClientCoords(t.clientX, t.clientY, true);
              }}
              onTouchMove={(e) => {
                e.preventDefault();
                const t = e.touches[0];
                if (t) paintAtClientCoords(t.clientX, t.clientY, true);
              }}
              className="w-[240px] h-[240px] rounded-lg bg-[#020617] border-2 border-slate-700 cursor-crosshair touch-none shadow-inner"
            />

            <div className="w-full mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between gap-2">
              <span className="text-[11px] font-mono tabular-nums text-amber-300">
                Ventana 3×3: ({focusRow}, {focusCol})
              </span>
              {activePhotoSrc && activePhotoLabel && (
                <div className="flex items-center gap-1.5">
                  <img
                    src={activePhotoSrc}
                    alt={activePhotoLabel}
                    referrerPolicy="no-referrer"
                    className="w-6 h-6 rounded border border-slate-700 object-cover"
                  />
                  <span className="text-[10px] text-slate-400 truncate max-w-[80px]">
                    {activePhotoLabel}
                  </span>
                </div>
              )}
            </div>
          </article>

          {/* COLUMN 2: CAPA 2 CONV2D + ACTIVACION (3 CHANNELS) */}
          <article className="bg-[#0f172a]/95 border border-slate-800 rounded-xl p-4 flex flex-col">
            <div className="w-full flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider mb-1">
              <span className="text-sky-400">
                Capa 2 · Conv2D + {actMeta.shortLabel}
              </span>
              <span className="font-mono tabular-nums text-sky-400">
                22 × 22 × 3
              </span>
            </div>
            <h2 className="font-display text-base font-bold text-slate-50 text-center">
              Mapas de Características
            </h2>
            <p className="text-xs text-slate-400 text-center mt-1 mb-3 leading-snug">
              3 filtros paralelos escanean la imagen. Selecciona o pasa el cursor sobre un mapa para inspeccionar.
            </p>

            <div className="flex flex-col gap-2.5 flex-1 justify-center">
              {filters.map((f, idx) => {
                const isSelected = activeChannel === idx;
                return (
                  <div
                    key={f.id}
                    role="button"
                    tabIndex={0}
                    aria-pressed={isSelected}
                    onClick={() => setActiveChannel(idx)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setActiveChannel(idx);
                      }
                    }}
                    className={`rounded-xl p-2.5 border flex items-center justify-between gap-2.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                      isSelected
                        ? 'bg-amber-400/10 border-amber-400/80'
                        : 'bg-[#080d1a]/80 border-slate-800 hover:border-sky-500/60'
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div
                        className="text-xs font-bold flex items-center gap-1.5 truncate"
                        style={{ color: f.hex }}
                      >
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ backgroundColor: f.hex }}
                          aria-hidden="true"
                        />
                        <span className="truncate">{f.name}</span>
                      </div>
                      <div className="text-[11px] text-slate-400 truncate mt-0.5">
                        {f.description}
                      </div>

                      {/* Mini 3x3 Kernel Matrix */}
                      <div className="inline-grid grid-cols-3 gap-0.5 mt-1.5">
                        {f.kernel.flat().map((kv, kIdx) => (
                          <span
                            key={`mk-${idx}-${kIdx}`}
                            className="w-5 text-center text-[10px] font-mono tabular-nums bg-slate-800/90 text-slate-200 rounded-[2px] py-0.5"
                          >
                            {kv > 0 ? `+${kv}` : kv}
                          </span>
                        ))}
                      </div>
                    </div>

                    <canvas
                      ref={(el) => {
                        canvasConvRefs.current[idx] = el;
                      }}
                      width={88}
                      height={88}
                      aria-label={`Mapa de características ${f.shortName}`}
                      onMouseMove={(e) =>
                        handleConvPointerInspect(idx, e.clientX, e.clientY)
                      }
                      onClick={(e) => {
                        e.stopPropagation();
                        handleConvPointerInspect(idx, e.clientX, e.clientY);
                      }}
                      className="w-[88px] h-[88px] rounded-lg bg-[#020617] border border-slate-700 shrink-0 cursor-crosshair"
                    />
                  </div>
                );
              })}
            </div>
          </article>

          {/* COLUMN 3: CAPA 3 POOLING 2x2 (11x11x3) */}
          <article className="bg-[#0f172a]/95 border border-slate-800 rounded-xl p-4 flex flex-col">
            <div className="w-full flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider mb-1">
              <span className="text-emerald-400">
                Capa 3 · {poolMode === 'max' ? 'Max Pool' : 'Avg Pool'} (2×2)
              </span>
              <span className="font-mono tabular-nums text-emerald-400">
                11 × 11 × 3
              </span>
            </div>
            <h2 className="font-display text-base font-bold text-slate-50 text-center">
              Submuestreo Espacial
            </h2>
            <p className="text-xs text-slate-400 text-center mt-1 mb-3 leading-snug">
              Comprime cada mapa a la mitad (<strong className="text-slate-200">−75% de datos</strong>), preservando rasgos clave.
            </p>

            <div className="flex flex-col gap-2.5 flex-1 justify-center">
              {filters.map((f, idx) => {
                const isSelected = activeChannel === idx;
                return (
                  <div
                    key={`pool-${f.id}`}
                    role="button"
                    tabIndex={0}
                    aria-pressed={isSelected}
                    onClick={() => setActiveChannel(idx)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setActiveChannel(idx);
                      }
                    }}
                    className={`rounded-xl p-2.5 border flex items-center justify-between gap-2.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                      isSelected
                        ? 'bg-emerald-500/10 border-emerald-400/80'
                        : 'bg-[#080d1a]/80 border-slate-800 hover:border-emerald-500/50'
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="text-xs font-bold text-emerald-300 flex items-center gap-1.5">
                        <span
                          className="w-2 h-2 rounded-full bg-emerald-400 shrink-0"
                          aria-hidden="true"
                        />
                        <span>Mapa Reducido {idx + 1}</span>
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5 truncate">
                        Resumen: {f.shortName} (11×11)
                      </div>
                      <div className="text-[10px] font-mono tabular-nums text-slate-500 mt-1">
                        {poolMode === 'max' ? 'máx(2×2)' : 'prom(2×2)'} · Paso 2
                      </div>
                    </div>

                    <canvas
                      ref={(el) => {
                        canvasPoolRefs.current[idx] = el;
                      }}
                      width={88}
                      height={88}
                      aria-label={`Mapa comprimido ${idx + 1}: ${f.shortName}`}
                      className="w-[88px] h-[88px] rounded-lg bg-[#020617] border border-slate-700 shrink-0"
                    />
                  </div>
                );
              })}
            </div>
          </article>

          {/* COLUMN 4: CAPA 4 FLATTEN + SOFTMAX */}
          <article className="bg-[#0f172a]/95 border border-slate-800 rounded-xl p-4 flex flex-col">
            <div className="w-full flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider mb-1">
              <span className="text-violet-400">Capa 4 · Flatten + Dense</span>
              <span className="font-mono tabular-nums text-violet-400">
                363 → 4
              </span>
            </div>
            <h2 className="font-display text-base font-bold text-slate-50 text-center">
              Decisión Final (Softmax)
            </h2>
            <p className="text-xs text-slate-400 text-center mt-1 mb-3 leading-snug">
              Los 3 mapas 11×11 se estiran en un vector de <strong className="text-slate-200">363 neuronas</strong> antes de clasificar.
            </p>

            <div className="w-full flex items-center justify-between text-[11px] font-mono text-slate-400 mb-1">
              <span>Vector Flatten (363)</span>
              <span>Activación</span>
            </div>
            <canvas
              ref={canvasFlattenRef}
              width={236}
              height={18}
              aria-label="Tira espectral del vector Flatten de 363 neuronas"
              className="w-full h-[18px] rounded border border-slate-700 bg-[#020617] mb-4"
            />

            <div className="space-y-2.5 flex-1" data-testid="studio-predictions">
              {cnnResult.predictions.map((pred, idx) => {
                const pct = Math.round(pred.probability * 100);
                const isWinner = idx === 0;
                return (
                  <div key={pred.id}>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span
                        className={
                          isWinner
                            ? 'font-semibold text-slate-50'
                            : 'text-slate-300'
                        }
                      >
                        {isWinner ? `★ ${pred.label}` : pred.label}
                      </span>
                      <span
                        className="font-mono tabular-nums font-bold"
                        style={{ color: pred.color }}
                      >
                        {pct}%
                      </span>
                    </div>
                    <div className="w-full h-2.5 bg-[#020617] rounded-full overflow-hidden border border-slate-800/80">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${pct}%`,
                          backgroundColor: pred.color,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {isCustomOrPhoto && (
              <p className="text-[11px] text-slate-400 mt-3 pt-2.5 border-t border-slate-800/80 leading-snug">
                Similitud estructural de la fotografía frente a los 4 prototipos visuales base.
              </p>
            )}
          </article>
        </div>
      </section>

      {/* BOTTOM DOCK: RAYOS X MATEMÁTICOS + CÓDIGO KERAS */}
      <StudioXRayDock
        inputGrid={inputGrid}
        activeFilter={activeFilter}
        activeChannelIndex={activeChannel}
        isFilterModified={isFilterModified}
        convMap={cnnResult.convMaps[activeChannel]}
        rawMap={cnnResult.rawMaps[activeChannel]}
        poolMap={cnnResult.poolMaps[activeChannel]}
        focusRow={focusRow}
        focusCol={focusCol}
        poolMode={poolMode}
        activationFn={activationFn}
        leakySlope={leakySlope}
        bias={bias}
        onUpdateFilterWeight={handleUpdateFilterWeight}
        onResetActiveFilter={handleResetActiveFilter}
      />
    </div>
  );
};

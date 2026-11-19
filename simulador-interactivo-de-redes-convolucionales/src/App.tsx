import { useEffect, useState } from 'react';
import { ArchitectureExplorer } from './components/ArchitectureExplorer';
import { ConceptsReference } from './components/ConceptsReference';
import { ControlDeck, type LearningStage } from './components/ControlDeck';
import { StageSimulator } from './components/StageSimulator';
import { StudioVisual } from './components/StudioVisual';
import { TopBar, type AppView } from './components/TopBar';
import { computeOutputDim } from './engine/cnnMath';
import { sampleRgbaToMatrix8x8 } from './engine/imageProcessor';
import {
  cloneMatrix,
  HERO_BACKDROP_SRC,
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
} from './engine/presets';

const WORK_BUFFER_SIZE = 80;

export default function App() {
  const [activeView, setActiveView] = useState<AppView>('studio');
  const [studioResetSignal, setStudioResetSignal] = useState<number>(0);

  // Laboratorio 8x8 state
  const [stage, setStage] = useState<LearningStage>('conv');
  const [selectedSource, setSelectedSource] = useState<
    InputPresetId | SampleImageId | 'uploaded' | 'custom'
  >('cross');
  const [inputMatrix, setInputMatrix] = useState<Matrix2D>(() =>
    cloneMatrix(INPUT_PRESETS.cross.matrix)
  );

  const [activePhotoSrc, setActivePhotoSrc] = useState<string | null>(null);
  const [activePhotoLabel, setActivePhotoLabel] = useState<string | null>(null);
  const [invertPhoto, setInvertPhoto] = useState<boolean>(false);
  const [photoRgbaBuffer, setPhotoRgbaBuffer] = useState<Uint8ClampedArray | null>(
    null
  );
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [selectedKernelPreset, setSelectedKernelPreset] =
    useState<KernelPresetId>('vertical');
  const [kernel, setKernel] = useState<Matrix2D>(() =>
    cloneMatrix(KERNEL_PRESETS.vertical.matrix)
  );

  const [padding, setPadding] = useState<number>(0);
  const [stride, setStride] = useState<number>(1);
  const [bias, setBias] = useState<number>(0);
  const [activationFn, setActivationFn] = useState<ActivationFnId>('relu');
  const [leakySlope, setLeakySlope] = useState<number>(0.1);
  const [poolMode, setPoolMode] = useState<PoolingType>('max');
  const [backdrop, setBackdrop] = useState<StageBackdropId>('technical_grid');
  const [brushMode, setBrushMode] = useState<'draw' | 'erase'>('draw');

  const [convStepIndex, setConvStepIndex] = useState<number>(0);
  const [poolStepIndex, setPoolStepIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speedMs, setSpeedMs] = useState<number>(300);

  const outSize = computeOutputDim(8, 3, padding, stride);
  const totalConvSteps = outSize * outSize;
  const poolDim = Math.floor(outSize / 2);
  const totalPoolSteps = Math.max(1, poolDim * poolDim);

  useEffect(() => {
    if (!activePhotoSrc) return;
    let cancelled = false;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (cancelled) return;
      const canvas = document.createElement('canvas');
      canvas.width = WORK_BUFFER_SIZE;
      canvas.height = WORK_BUFFER_SIZE;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const minSide = Math.min(img.width, img.height);
      const sx = Math.floor((img.width - minSide) / 2);
      const sy = Math.floor((img.height - minSide) / 2);
      ctx.drawImage(
        img,
        sx,
        sy,
        minSide,
        minSide,
        0,
        0,
        WORK_BUFFER_SIZE,
        WORK_BUFFER_SIZE
      );
      const imageData = ctx.getImageData(
        0,
        0,
        WORK_BUFFER_SIZE,
        WORK_BUFFER_SIZE
      );
      const bufferCopy = new Uint8ClampedArray(imageData.data);
      setPhotoRgbaBuffer(bufferCopy);
      setInputMatrix(
        sampleRgbaToMatrix8x8(
          bufferCopy,
          WORK_BUFFER_SIZE,
          WORK_BUFFER_SIZE,
          invertPhoto
        )
      );
      setUploadError(null);
    };
    img.onerror = () => {
      if (cancelled) return;
      setUploadError('No se pudo cargar la imagen seleccionada.');
    };
    img.src = activePhotoSrc;

    return () => {
      cancelled = true;
    };
  }, [activePhotoSrc, invertPhoto]);

  useEffect(() => {
    if (!isPlaying || activeView !== 'lab') return;
    if (stage !== 'conv' && stage !== 'pool') return;

    const timer = window.setInterval(() => {
      if (stage === 'conv') {
        setConvStepIndex((prev) =>
          totalConvSteps > 0 ? (prev + 1) % totalConvSteps : 0
        );
      } else if (stage === 'pool') {
        setPoolStepIndex((prev) =>
          totalPoolSteps > 0 ? (prev + 1) % totalPoolSteps : 0
        );
      }
    }, speedMs);

    return () => window.clearInterval(timer);
  }, [isPlaying, activeView, stage, speedMs, totalConvSteps, totalPoolSteps]);

  const handleSelectView = (view: AppView) => {
    setIsPlaying(false);
    setActiveView(view);
  };

  const handleChangeStage = (nextStage: LearningStage) => {
    setIsPlaying(false);
    setStage(nextStage);
  };

  const handleResetDemo = () => {
    setIsPlaying(false);
    if (activeView === 'studio') {
      setStudioResetSignal((n) => n + 1);
      return;
    }
    setActiveView('studio');
    setStudioResetSignal((n) => n + 1);
    setStage('conv');
    setSelectedSource('cross');
    setActivePhotoSrc(null);
    setActivePhotoLabel(null);
    setPhotoRgbaBuffer(null);
    setInvertPhoto(false);
    setUploadError(null);
    setInputMatrix(cloneMatrix(INPUT_PRESETS.cross.matrix));
    setSelectedKernelPreset('vertical');
    setKernel(cloneMatrix(KERNEL_PRESETS.vertical.matrix));
    setPadding(0);
    setStride(1);
    setBias(0);
    setActivationFn('relu');
    setLeakySlope(0.1);
    setPoolMode('max');
    setBackdrop('technical_grid');
    setBrushMode('draw');
    setConvStepIndex(0);
    setPoolStepIndex(0);
  };

  const handleToggleInputCell = (row: number, col: number, value?: number) => {
    setSelectedSource('custom');
    setInputMatrix((prev) =>
      prev.map((r, rIdx) =>
        rIdx === row
          ? r.map((v, cIdx) =>
              cIdx === col ? (value !== undefined ? value : v > 0.5 ? 0 : 1) : v
            )
          : r
      )
    );
  };

  const handleSelectInputPreset = (id: InputPresetId) => {
    setSelectedSource(id);
    setActivePhotoSrc(null);
    setActivePhotoLabel(null);
    setPhotoRgbaBuffer(null);
    setUploadError(null);
    setInputMatrix(cloneMatrix(INPUT_PRESETS[id].matrix));
    setConvStepIndex(0);
    setPoolStepIndex(0);
  };

  const handleSelectSampleImage = (id: SampleImageId) => {
    const sample = SAMPLE_IMAGES[id];
    setSelectedSource(id);
    setActivePhotoSrc(sample.src);
    setActivePhotoLabel(sample.shortLabel);
    setInvertPhoto(sample.invertByDefault);
    setUploadError(null);
    setSelectedKernelPreset(sample.recommendedKernel);
    setKernel(cloneMatrix(KERNEL_PRESETS[sample.recommendedKernel].matrix));
    setConvStepIndex(0);
    setPoolStepIndex(0);
  };

  const handleUploadImageFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setUploadError('Selecciona un archivo de imagen válido (PNG, JPG, WebP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setSelectedSource('uploaded');
        setActivePhotoSrc(reader.result);
        setActivePhotoLabel(file.name);
        setInvertPhoto(false);
        setUploadError(null);
        setConvStepIndex(0);
        setPoolStepIndex(0);
      }
    };
    reader.onerror = () => {
      setUploadError('Error al leer el archivo de imagen.');
    };
    reader.readAsDataURL(file);
  };

  const handleClearInput = () => {
    setSelectedSource('custom');
    setActivePhotoSrc(null);
    setActivePhotoLabel(null);
    setPhotoRgbaBuffer(null);
    setInputMatrix(Array.from({ length: 8 }, () => Array(8).fill(0)));
  };

  const handleSelectKernelPreset = (id: Exclude<KernelPresetId, 'custom'>) => {
    setSelectedKernelPreset(id);
    setKernel(cloneMatrix(KERNEL_PRESETS[id].matrix));
  };

  const handleUpdateKernelCell = (row: number, col: number, delta: number) => {
    setSelectedKernelPreset('custom');
    setKernel((prev) =>
      prev.map((r, rIdx) =>
        rIdx === row
          ? r.map((w, cIdx) => {
              if (cIdx !== col) return w;
              return Math.max(-8, Math.min(8, Math.round((w + delta) * 10) / 10));
            })
          : r
      )
    );
  };

  const handleSetPadding = (pad: number) => {
    setIsPlaying(false);
    setPadding(pad);
    setConvStepIndex(0);
    setPoolStepIndex(0);
  };

  const handleSetStride = (s: number) => {
    setIsPlaying(false);
    setStride(s);
    setConvStepIndex(0);
    setPoolStepIndex(0);
  };

  const handleStepPrev = () => {
    setIsPlaying(false);
    if (stage === 'pool') {
      setPoolStepIndex((prev) => (prev - 1 + totalPoolSteps) % totalPoolSteps);
    } else {
      setConvStepIndex((prev) => (prev - 1 + totalConvSteps) % totalConvSteps);
    }
  };

  const handleStepNext = () => {
    setIsPlaying(false);
    if (stage === 'pool') {
      setPoolStepIndex((prev) => (prev + 1) % totalPoolSteps);
    } else {
      setConvStepIndex((prev) => (prev + 1) % totalConvSteps);
    }
  };

  const handleResetPlayback = () => {
    setIsPlaying(false);
    setConvStepIndex(0);
    setPoolStepIndex(0);
  };

  const handleJumpToStage = (targetStage: LearningStage) => {
    setIsPlaying(false);
    setStage(targetStage);
    setActiveView('lab');
  };

  return (
    <div className="min-h-screen flex flex-col text-slate-50">
      <TopBar
        activeView={activeView}
        onSelectView={handleSelectView}
        onResetDemo={handleResetDemo}
      />

      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-5">
        {activeView === 'studio' && (
          <StudioVisual resetSignal={studioResetSignal} />
        )}

        {activeView === 'lab' && (
          <div className="space-y-4">
            <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-[#0f172a]">
              <img
                src={HERO_BACKDROP_SRC}
                alt=""
                aria-hidden="true"
                referrerPolicy="no-referrer"
                className="absolute inset-0 w-full h-full object-cover object-center opacity-20 pointer-events-none"
              />
              <div
                aria-hidden="true"
                className="absolute inset-0 bg-gradient-to-r from-[#070b14]/95 via-[#0f172a]/85 to-[#0f172a]/65 pointer-events-none"
              />
              <div className="relative px-5 py-4 sm:px-6 sm:py-5 max-w-3xl">
                <h1 className="font-display text-xl sm:text-2xl font-bold tracking-tight text-slate-50 text-balance">
                  Laboratorio Matemático Paso a Paso (8×8)
                </h1>
                <p className="text-xs sm:text-sm text-slate-300 mt-1">
                  Inspecciona celda por celda cada producto escalar, ajusta Padding, Stride y funciones de activación o prueba el visor fotográfico de alta resolución.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              <div className="lg:col-span-8">
                <StageSimulator
                  stage={stage}
                  onChangeStage={handleChangeStage}
                  inputMatrix={inputMatrix}
                  kernel={kernel}
                  padding={padding}
                  stride={stride}
                  bias={bias}
                  activationFn={activationFn}
                  leakySlope={leakySlope}
                  poolMode={poolMode}
                  backdrop={backdrop}
                  brushMode={brushMode}
                  convStepIndex={convStepIndex}
                  poolStepIndex={poolStepIndex}
                  isPlaying={isPlaying}
                  speedMs={speedMs}
                  activePhotoSrc={activePhotoSrc}
                  activePhotoLabel={activePhotoLabel}
                  photoRgbaBuffer={photoRgbaBuffer}
                  photoBufferWidth={WORK_BUFFER_SIZE}
                  photoBufferHeight={WORK_BUFFER_SIZE}
                  invertPhoto={invertPhoto}
                  onToggleInputCell={handleToggleInputCell}
                  onSetConvStep={(idx) => {
                    setIsPlaying(false);
                    setConvStepIndex(idx);
                  }}
                  onSetPoolStep={(idx) => {
                    setIsPlaying(false);
                    setPoolStepIndex(idx);
                  }}
                  onTogglePlay={() => setIsPlaying((p) => !p)}
                  onStepPrev={handleStepPrev}
                  onStepNext={handleStepNext}
                  onResetPlayback={handleResetPlayback}
                  onSetSpeedMs={setSpeedMs}
                />
              </div>

              <div className="lg:col-span-4">
                <ControlDeck
                  stage={stage}
                  selectedSource={selectedSource}
                  activePhotoLabel={activePhotoLabel}
                  invertPhoto={invertPhoto}
                  selectedKernelPreset={selectedKernelPreset}
                  kernel={kernel}
                  padding={padding}
                  stride={stride}
                  bias={bias}
                  activationFn={activationFn}
                  leakySlope={leakySlope}
                  poolMode={poolMode}
                  backdrop={backdrop}
                  brushMode={brushMode}
                  outSize={outSize}
                  uploadError={uploadError}
                  onSelectInputPreset={handleSelectInputPreset}
                  onSelectSampleImage={handleSelectSampleImage}
                  onUploadImageFile={handleUploadImageFile}
                  onToggleInvertPhoto={() => setInvertPhoto((v) => !v)}
                  onClearInput={handleClearInput}
                  onSetBrushMode={setBrushMode}
                  onSelectKernelPreset={handleSelectKernelPreset}
                  onUpdateKernelCell={handleUpdateKernelCell}
                  onSetPadding={handleSetPadding}
                  onSetStride={handleSetStride}
                  onSetBias={setBias}
                  onSetActivationFn={setActivationFn}
                  onSetLeakySlope={setLeakySlope}
                  onSetPoolMode={setPoolMode}
                  onSetBackdrop={setBackdrop}
                />
              </div>
            </div>
          </div>
        )}

        {activeView === 'architecture' && <ArchitectureExplorer />}

        {activeView === 'concepts' && (
          <ConceptsReference onJumpToStage={handleJumpToStage} />
        )}
      </main>
    </div>
  );
}

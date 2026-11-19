import React from 'react';
import type { LearningStage } from './ControlDeck';

interface ConceptsReferenceProps {
  onJumpToStage: (stage: LearningStage) => void;
}

const CONCEPTS: Array<{
  title: string;
  formula: string;
  summary: string;
  whyItMatters: string;
  targetStage: LearningStage;
  actionLabel: string;
}> = [
  {
    title: 'Kernel (Filtro Convolucional)',
    formula: 'z = Σ (Entrada · Peso)',
    summary:
      'Es una pequeña matriz de pesos (por ejemplo, 3×3) que se desliza sobre toda la imagen multiplicando y sumando valores locales.',
    whyItMatters:
      'Permite detectar el mismo patrón (un borde vertical, una esquina o una curva) sin importar en qué parte de la imagen aparezca.',
    targetStage: 'conv',
    actionLabel: 'Probar en Convolución 8×8',
  },
  {
    title: 'Padding (Relleno de Ceros)',
    formula: 'N_pad = N + 2·P',
    summary:
      'Añade un marco de ceros alrededor de la matriz de entrada antes de aplicar el filtro convolucional.',
    whyItMatters:
      'Evita que los bordes de la imagen pierdan información rápidamente y permite mantener exactamente el mismo tamaño espacial (Same Padding).',
    targetStage: 'conv',
    actionLabel: 'Activar Padding = 1',
  },
  {
    title: 'Stride (Paso de Deslizamiento)',
    formula: '⌊(N + 2P − K) / S⌋ + 1',
    summary:
      'Define cuántos píxeles salta la ventana del filtro en cada movimiento horizontal y vertical.',
    whyItMatters:
      'Un stride de 2 reduce el mapa de características a la mitad en cada eje, acelerando el cómputo a cambio de menor resolución.',
    targetStage: 'conv',
    actionLabel: 'Ajustar Stride',
  },
  {
    title: 'Activación ReLU y No Linealidad',
    formula: 'f(z) = máx(0, z)',
    summary:
      'Reemplaza cualquier valor negativo por cero y deja intactos los valores positivos que superan el umbral.',
    whyItMatters:
      'Introduce no linealidad en la red para que pueda aprender formas complejas y desactiva neuronas irrelevantes.',
    targetStage: 'relu',
    actionLabel: 'Explorar curvas de activación',
  },
  {
    title: 'Max y Average Pooling (2×2)',
    formula: 'y = máx(bloque 2×2)',
    summary:
      'Divide el mapa activado en regiones de 2×2 y selecciona únicamente la activación más alta o su promedio.',
    whyItMatters:
      'Reduce el número de parámetros en un 75% y hace que la red sea tolerante a pequeñas traslaciones.',
    targetStage: 'pool',
    actionLabel: 'Ver Pooling 2×2',
  },
  {
    title: 'Flatten y Softmax',
    formula: 'P(i) = e^(z_i) / Σ e^(z_j)',
    summary:
      'Convierte los mapas 2D finales en un vector lineal y transforma los puntajes en probabilidades que suman 100%.',
    whyItMatters:
      'Conecta las características visuales extraídas por las capas convolucionales con la decisión final de clasificación.',
    targetStage: 'pipeline',
    actionLabel: 'Ver Pipeline 8×8',
  },
];

export const ConceptsReference: React.FC<ConceptsReferenceProps> = ({
  onJumpToStage,
}) => {
  return (
    <div className="space-y-6" data-testid="concepts-reference">
      <div className="max-w-2xl">
        <h1 className="font-display text-2xl font-bold text-slate-50">
          Fundamentos de una Red Neuronal Convolucional
        </h1>
        <p className="text-sm text-slate-400 mt-1.5 leading-relaxed">
          Cada bloque de una CNN transforma datos visuales brutos en representaciones cada vez más abstractas. Selecciona cualquier concepto para experimentarlo directamente en el simulador.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {CONCEPTS.map((item, idx) => (
          <article
            key={item.title}
            className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between gap-2 text-xs text-slate-400 mb-2">
                <span className="font-medium">0{idx + 1}. Pilar CNN</span>
                <span className="font-mono tabular-nums text-sky-400">
                  {item.formula}
                </span>
              </div>

              <h2 className="font-display text-lg font-semibold text-slate-50">
                {item.title}
              </h2>

              <p className="text-sm text-slate-300 mt-2 leading-relaxed">
                {item.summary}
              </p>

              <p className="text-xs text-slate-400 mt-3 pt-3 border-t border-slate-800 leading-relaxed">
                {item.whyItMatters}
              </p>
            </div>

            <div className="mt-4 pt-2">
              <button
                type="button"
                onClick={() => onJumpToStage(item.targetStage)}
                className="w-full px-3.5 py-2 text-xs font-semibold text-sky-300 bg-sky-500/10 hover:bg-sky-500/20 border border-sky-500/30 rounded-lg whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                {item.actionLabel}
              </button>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
};

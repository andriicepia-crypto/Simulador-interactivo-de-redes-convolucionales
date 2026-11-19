import React from 'react';

export type AppView = 'studio' | 'lab' | 'architecture' | 'concepts';

interface TopBarProps {
  activeView: AppView;
  onSelectView: (view: AppView) => void;
  onResetDemo: () => void;
}

const NAV_ITEMS: Array<{ id: AppView; label: string }> = [
  { id: 'studio', label: 'Estudio Visual' },
  { id: 'lab', label: 'Laboratorio 8×8' },
  { id: 'architecture', label: 'Arquitectura' },
  { id: 'concepts', label: 'Conceptos' },
];

export const TopBar: React.FC<TopBarProps> = ({
  activeView,
  onSelectView,
  onResetDemo,
}) => {
  return (
    <header className="md:sticky md:top-0 z-30 bg-[#0b1220]/90 backdrop-blur-md border-b border-slate-800/90">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 h-14 sm:h-16 flex items-center justify-between gap-4 sm:gap-8">
        {/* Zone 1: Single-line wordmark */}
        <button
          type="button"
          onClick={() => onSelectView('studio')}
          className="font-display text-xl font-bold tracking-tight text-slate-50 whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 rounded"
        >
          ConvLab
        </button>

        {/* Zone 2: 4 single-line text nav links (desktop) */}
        <nav
          aria-label="Navegación principal"
          className="hidden md:flex items-center gap-6"
        >
          {NAV_ITEMS.map((item) => {
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectView(item.id)}
                aria-current={isActive ? 'page' : undefined}
                className={`relative py-1.5 text-sm font-medium whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 rounded ${
                  isActive
                    ? 'text-sky-400'
                    : 'text-slate-400 hover:text-slate-100'
                }`}
              >
                {item.label}
                {isActive && (
                  <span
                    aria-hidden="true"
                    className="absolute left-0 right-0 -bottom-[1px] h-0.5 bg-sky-400 rounded-full"
                  />
                )}
              </button>
            );
          })}
        </nav>

        {/* Zone 3: 1 primary action */}
        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={onResetDemo}
            className="px-4 py-2 text-xs sm:text-sm font-semibold text-slate-950 bg-sky-400 hover:bg-sky-300 rounded-lg whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0b1220] focus-visible:ring-sky-400"
          >
            Reiniciar
          </button>
        </div>
      </div>

      {/* Mobile navigation strip (non-clipping) */}
      <nav
        aria-label="Navegación móvil"
        className="md:hidden border-t border-slate-800/80 px-2 py-1.5 flex items-center justify-between gap-1 bg-[#080d18]"
      >
        {NAV_ITEMS.map((item) => {
          const isActive = activeView === item.id;
          return (
            <button
              key={`mob-${item.id}`}
              type="button"
              onClick={() => onSelectView(item.id)}
              aria-current={isActive ? 'page' : undefined}
              className={`flex-1 py-2 px-1 text-[11px] font-medium text-center whitespace-nowrap rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                isActive
                  ? 'bg-slate-800 text-sky-400 font-semibold'
                  : 'text-slate-400 hover:text-slate-100'
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
};

import { useMeshStore } from '../store/useMeshStore';
import type { PropAnnouncement } from '../types/mesh';

function capabilityLabel(cap: string): string {
  const parts = cap.split('.');
  return parts[parts.length - 1] || cap;
}

export default function PropTile({ source, prop }: { source: string; prop: PropAnnouncement }) {
  const fire = useMeshStore((s) => s.fire);
  const activeEffects = useMeshStore((s) => s.activeEffects);
  const online = Boolean((prop as unknown as Record<string, number>).last_seen)
    ? Date.now() - ((prop as unknown as Record<string, number>).last_seen || 0) < 10000
    : true;

  return (
    <div className={`tile ${!online ? 'opacity-60' : ''}`}>
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-bold text-sm">{prop.name || prop.id || source}</h3>
          <p className="text-xs text-slate-400">{source} {!online && '• offline'}</p>
        </div>
        <div className={`w-3 h-3 rounded-full ${online ? 'bg-emerald-500' : 'bg-slate-600'}`} />
      </div>
      <div className="flex flex-wrap gap-1">
        {(prop.capabilities || []).map((cap) => {
          const key = `${source}::${cap}`;
          const active = activeEffects[key];
          return (
            <button
              key={cap}
              onClick={() => fire(cap, {})}
              disabled={!online}
              className={`text-xs px-2 py-1 rounded border ${
                active
                  ? 'bg-amber-500 border-amber-400 text-black'
                  : 'bg-slate-800 border-slate-600 hover:bg-slate-700'
              }`}
            >
              {capabilityLabel(cap)}
            </button>
          );
        })}
      </div>
    </div>
  );
}

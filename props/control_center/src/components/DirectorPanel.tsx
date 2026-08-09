import { useMeshStore } from '../store/useMeshStore';

export default function DirectorPanel() {
  const { world, family_mode, toggleFamilyMode, fireCannon, nextPumpkinSong, testPortraitLine } = useMeshStore();

  return (
    <div className="tile">
      <div className="flex justify-between items-center">
        <h3 className="font-bold text-sm">Director</h3>
        <span className={`text-xs font-bold px-2 py-0.5 rounded ${world.audience_present ? 'bg-amber-500 text-black' : 'bg-slate-700 text-slate-300'}`}>
          {world.audience_present ? 'AUDIENCE PRESENT' : 'NO AUDIENCE'}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <button
          onClick={toggleFamilyMode}
          className={`btn text-sm py-2 ${family_mode ? 'btn-scene-active' : 'btn-primary'}`}
        >
          {family_mode ? 'Family Mode ON' : 'Family Mode'}
        </button>

        <button
          onClick={fireCannon}
          disabled={family_mode}
          className={`btn text-sm py-2 ${family_mode ? 'bg-slate-600 text-slate-400' : 'btn-danger'}`}
        >
          🔥 Fire Cannon
        </button>

        <button
          onClick={nextPumpkinSong}
          className="btn text-sm py-2 btn-scene"
        >
          🎃 Next Song
        </button>

        <button
          onClick={testPortraitLine}
          className="btn text-sm py-2 btn-scene"
        >
          🎭 Portrait Test
        </button>
      </div>

      {world.linger_alert && (
        <div className="bg-red-900/60 border border-red-500 text-red-100 text-sm px-3 py-2 rounded animate-pulse">
          ⚠️ Sideyard visitor lingering &gt;3 minutes
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
        {world.zones.map((z) => (
          <div
            key={z.zone}
            className={`rounded p-2 border ${z.occupied ? 'bg-emerald-900/40 border-emerald-500' : 'bg-slate-800 border-slate-600'}`}
          >
            <div className="font-semibold capitalize">{z.zone.replace('_', ' ')}</div>
            <div className="text-slate-400">{z.occupied ? `${z.count} person${z.count > 1 ? 's' : ''}` : 'empty'}</div>
            {z.occupied && z.linger_s > 0 && (
              <div className="text-amber-400">{Math.round(z.linger_s)}s</div>
            )}
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        {world.pumpkins_singing && (
          <span className="px-2 py-1 rounded bg-purple-600 text-white">🎵 Pumpkins singing</span>
        )}
        {world.portrait_speaking && (
          <span className="px-2 py-1 rounded bg-sky-600 text-white">🗣️ Portrait speaking</span>
        )}
        {world.cannon_cooldown && (
          <span className="px-2 py-1 rounded bg-orange-700 text-white">💨 Cannon cooling</span>
        )}
        {world.fog_cooldown && (
          <span className="px-2 py-1 rounded bg-slate-600 text-white">🌫️ Fog cooling</span>
        )}
        {world.thunder_cooldown && (
          <span className="px-2 py-1 rounded bg-slate-600 text-white">⚡ Thunder cooling</span>
        )}
      </div>
    </div>
  );
}

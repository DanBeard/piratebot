import { useMeshStore } from '../store/useMeshStore';

const SCENES = ['idle', 'show', 'quiet', 'estop', 'resume'] as const;

export default function SceneSelector() {
  const { scene, estop, setSceneCommand } = useMeshStore();

  return (
    <div className="flex flex-wrap gap-2 items-center">
      <span className="text-sm uppercase tracking-wider text-slate-400 mr-1">Scene:</span>
      {SCENES.map((s) => {
        const active = s === scene || (s === 'estop' && estop);
        return (
          <button
            key={s}
            onClick={() => setSceneCommand(s)}
            className={`btn ${s === 'estop' ? 'btn-danger' : active ? 'btn-scene-active' : 'btn-scene'}`}
          >
            {s}
          </button>
        );
      })}
      {estop && (
        <span className="ml-2 animate-pulse text-red-500 font-bold">ESTOP ACTIVE</span>
      )}
    </div>
  );
}

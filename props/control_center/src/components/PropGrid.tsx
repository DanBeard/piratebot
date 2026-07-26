import { useMeshStore } from '../store/useMeshStore';
import PropTile from './PropTile';

export default function PropGrid() {
  const manifest = useMeshStore((s) => s.manifest);
  const entries = Object.entries(manifest);

  if (entries.length === 0) {
    return (
      <div className="text-slate-500 text-sm italic p-4">
        No props announced yet.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
      {entries.map(([source, prop]) => (
        <PropTile key={source} source={source} prop={prop} />
      ))}
    </div>
  );
}

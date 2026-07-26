import { useMeshStore } from '../store/useMeshStore';

export default function ConnectionStatus() {
  const { connected, brokerUrl } = useMeshStore();

  return (
    <div className="flex items-center gap-2 text-sm">
      <div className={`w-3 h-3 rounded-full ${connected ? 'bg-emerald-500' : 'bg-red-500 animate-pulse'}`} />
      <span className={connected ? 'text-emerald-400' : 'text-red-400'}>
        {connected ? 'Connected' : 'Reconnecting...'}
      </span>
      <span className="text-xs text-slate-500 truncate max-w-[200px]">{brokerUrl}</span>
    </div>
  );
}

import { useState } from 'react';
import { useMeshStore } from '../store/useMeshStore';

export default function EventLog() {
  const logs = useMeshStore((s) => s.logs);
  const clearLogs = useMeshStore((s) => s.clearLogs);
  const [filter, setFilter] = useState('');

  const filtered = filter
    ? logs.filter((l) => l.topic.includes(filter) || l.source.includes(filter))
    : logs;

  return (
    <div className="tile h-96 overflow-hidden">
      <div className="flex justify-between items-center">
        <h3 className="font-bold text-sm">Event Log</h3>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="filter topic/source"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="text-xs bg-slate-800 border border-slate-600 rounded px-2 py-1"
          />
          <button
            onClick={clearLogs}
            className="text-xs px-2 py-1 rounded bg-slate-800 hover:bg-slate-700"
          >
            Clear
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto font-mono text-xs space-y-1 pr-1">
        {filtered.slice(0, 100).map((entry) => (
          <div key={entry.id} className="border-b border-slate-800 pb-1">
            <span className="text-slate-500">{entry.time}</span>{' '}
            <span className="text-sky-400">{entry.topic}</span>{' '}
            <span className="text-amber-400">[{entry.source}]</span>{' '}
            <span className="text-slate-300">{entry.payload}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

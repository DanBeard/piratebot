import ConnectionStatus from './components/ConnectionStatus';
import EventLog from './components/EventLog';
import PropGrid from './components/PropGrid';
import SceneSelector from './components/SceneSelector';

function App() {
  return (
    <div className="min-h-screen flex flex-col p-3 gap-3">
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
        <div>
          <h1 className="text-xl font-bold text-sky-400">PirateBot Control Center</h1>
          <p className="text-xs text-slate-400">Prop mesh command, monitoring, and debug</p>
        </div>
        <ConnectionStatus />
      </header>

      <section className="flex flex-col gap-3">
        <SceneSelector />
        <PropGrid />
      </section>

      <section className="flex-1 min-h-0">
        <EventLog />
      </section>
    </div>
  );
}

export default App;

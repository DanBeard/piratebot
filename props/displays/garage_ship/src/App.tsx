import { useEffect, useRef, useState } from 'react';
import { onMeshMessage, useMeshStore } from './store/useMeshStore';
import type { MeshMessage } from './types/mesh';

function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [firing, setFiring] = useState(false);
  const [flash, setFlash] = useState(false);
  const connected = useMeshStore((s) => s.connected);
  const send = useMeshStore((s) => s.send);

  const fireCannon = () => {
    if (firing) return;
    setFiring(true);
    setFlash(true);
    send({
      topic: 'world.garage.start',
      source: 'garage_ship',
      payload: { triggered: 'local' },
    });

    setTimeout(() => setFlash(false), 120);

    const ball = document.createElement('div');
    ball.className = 'cannonball';
    if (containerRef.current) containerRef.current.appendChild(ball);

    const ship = containerRef.current?.querySelector('.ship') as HTMLElement | null;
    const startX = ship ? ship.offsetLeft + ship.offsetWidth - 40 : 100;
    const startY = ship ? ship.offsetTop + 80 : 300;
    const endX = window.innerWidth + 100;
    const endY = startY - 200 - Math.random() * 200;

    ball.animate(
      [
        { transform: `translate(${startX}px, ${startY}px)`, opacity: 1 },
        { transform: `translate(${endX}px, ${endY}px) scale(0.6)`, opacity: 1, offset: 0.85 },
        { transform: `translate(${endX + 100}px, ${endY + 100}px) scale(0.2)`, opacity: 0 },
      ],
      { duration: 1400, easing: 'cubic-bezier(0.2, 0.8, 0.6, 1)' }
    );

    const smoke = document.createElement('div');
    smoke.className = 'smoke';
    if (containerRef.current) containerRef.current.appendChild(smoke);
    smoke.animate(
      [
        { transform: `translate(${startX - 80}px, ${startY - 100}px) scale(0.3)`, opacity: 0.8 },
        { transform: `translate(${startX + 160}px, ${startY - 260}px) scale(2.5)`, opacity: 0 },
      ],
      { duration: 2000, easing: 'ease-out' }
    );

    setTimeout(() => {
      ball.remove();
      smoke.remove();
      setFiring(false);
    }, 2200);
  };

  useEffect(() => {
    onMeshMessage((msg: MeshMessage) => {
      if (msg.topic === 'effects.cannon.fire' || msg.topic === 'world.garage.start') {
        fireCannon();
      }
      if (msg.topic === 'effects.thunder.clap') {
        setFlash(true);
        setTimeout(() => setFlash(false), 250);
      }
    });
  }, []);

  return (
    <div ref={containerRef} className="ship-scene" onClick={fireCannon}>
      <div className={`flash ${flash ? 'opacity-100' : 'opacity-0'}`} />
      <div className="absolute top-4 right-4 text-slate-400 text-xs">
        {connected ? '● online' : '● reconnecting'}
      </div>
      <div className="absolute top-4 left-4 text-slate-300 text-sm">
        Tap/click anywhere or send effects.cannon.fire to fire
      </div>

      <div className="ship">
        <div className="mast" />
        <div className="sail" />
      </div>
    </div>
  );
}

export default App;

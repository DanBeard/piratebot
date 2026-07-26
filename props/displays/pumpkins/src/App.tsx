import { useEffect, useRef, useState } from 'react';
import { onMeshMessage, useMeshStore } from './store/useMeshStore';
import type { MeshMessage } from './types/mesh';

const SONGS = [
  { id: 'eliza_lee', title: 'Eliza Lee', duration: 120 },
  { id: 'old_maui', title: 'Old Maui', duration: 135 },
  { id: 'randy', title: 'Randy Dandy Oh', duration: 110 },
  { id: 'fire_marengo', title: 'Fire Marengo', duration: 140 },
  { id: 'rosibella', title: 'Rosibella', duration: 125 },
  { id: 'joli_rouge', title: 'Joli Rouge', duration: 115 },
];

function App() {
  const connected = useMeshStore((s) => s.connected);
  const singing = useMeshStore((s) => s.singing);
  const currentSong = useMeshStore((s) => s.currentSong);
  const setSinging = useMeshStore((s) => s.setSinging);
  const send = useMeshStore((s) => s.send);
  const [timeLeft, setTimeLeft] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const playSong = (songId: string | null) => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (!songId) {
      setSinging(false, null);
      send({ topic: 'scene.pumpkin.idle', source: 'pumpkins', payload: {} });
      return;
    }
    const song = SONGS.find((s) => s.id === songId) || SONGS[0];
    setSinging(true, song.title);
    setTimeLeft(song.duration);
    send({ topic: 'scene.pumpkin.sing', source: 'pumpkins', payload: { song: songId, title: song.title } });

    timerRef.current = setInterval(() => {
      setTimeLeft((t) => {
        if (t <= 1) {
          if (timerRef.current) clearInterval(timerRef.current);
          setSinging(false, null);
          send({ topic: 'scene.pumpkin.idle', source: 'pumpkins', payload: {} });
          return 0;
        }
        return t - 1;
      });
    }, 1000);
  };

  useEffect(() => {
    onMeshMessage((msg: MeshMessage) => {
      if (msg.topic === 'scene.pumpkin.sing') {
        const requested = msg.payload.song as string;
        playSong(requested);
      }
      if (msg.topic === 'scene.pumpkin.idle' || msg.topic === 'scene.stop') {
        playSong(null);
      }
    });
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  return (
    <div className="pumpkin-stage">
      <div className={`pumpkin ${singing ? 'singing' : ''}`}>
        <div className="face">
          <div className="eye-left" />
          <div className="eye-right" />
          <div className="nose" />
          <div className="mouth" />
        </div>
      </div>

      <div className="absolute bottom-8 left-0 right-0 text-center">
        <p className="text-amber-400 text-xl font-bold">
          {singing ? `🎵 ${currentSong}` : 'Idle'}
        </p>
        {singing && (
          <p className="text-slate-400 text-sm">{Math.floor(timeLeft / 60)}:{String(timeLeft % 60).padStart(2, '0')} remaining</p>
        )}
        <p className="text-xs text-slate-500 mt-2">{connected ? '● connected' : '● reconnecting'}</p>
      </div>
    </div>
  );
}

export default App;

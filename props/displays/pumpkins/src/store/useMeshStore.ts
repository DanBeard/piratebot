import { create } from 'zustand';
import type { MeshMessage } from '../types/mesh';

interface PumpkinState {
  connected: boolean;
  brokerUrl: string;
  singing: boolean;
  currentSong: string | null;
  setConnected: (connected: boolean) => void;
  setSinging: (singing: boolean, song?: string | null) => void;
  send: (msg: Omit<MeshMessage, 'timestamp'>) => void;
}

let ws: WebSocket | null = null;

function getBrokerUrl(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get('broker') || `ws://${window.location.hostname}:9001/ws`;
}

function connect(set: (fn: PumpkinState | Partial<PumpkinState>) => void, get: () => PumpkinState) {
  if (ws?.readyState === WebSocket.OPEN || ws?.readyState === WebSocket.CONNECTING) return;

  ws = new WebSocket(get().brokerUrl);

  ws.onopen = () => {
    set({ connected: true });
    get().send({
      topic: 'prop.state.announce',
      source: 'pumpkins',
      payload: {
        id: 'pumpkins',
        name: 'Singing Pumpkins',
        capabilities: ['scene.pumpkin.sing', 'scene.pumpkin.idle', 'audio.play'],
      },
    });
  };

  ws.onclose = () => {
    set({ connected: false });
    ws = null;
    setTimeout(() => connect(set, get), 1500);
  };

  ws.onerror = () => ws?.close();
}

export const useMeshStore = create<PumpkinState>((set) => ({
  connected: false,
  brokerUrl: getBrokerUrl(),
  singing: false,
  currentSong: null,

  setConnected: (connected: boolean) => set({ connected }),
  setSinging: (singing: boolean, song: string | null = null) => set({ singing, currentSong: song }),

  send: (msg) => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        ...msg,
        timestamp: Date.now() / 1000,
        meta: { seq: 0, session: 'halloween-2026', ...(msg.meta || {}) },
      }));
    }
  },
}));

connect(useMeshStore.setState, useMeshStore.getState);

export function onMeshMessage(handler: (msg: MeshMessage) => void) {
  const wrapped = (event: MessageEvent) => {
    try {
      handler(JSON.parse(event.data));
    } catch {
      // ignore
    }
  };
  const tryAttach = () => ws?.addEventListener('message', wrapped);
  tryAttach();
  ws?.addEventListener('open', tryAttach);
}

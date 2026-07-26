import { create } from 'zustand';
import type { MeshMessage } from '../types/mesh';

interface ShipState {
  connected: boolean;
  brokerUrl: string;
  lastCannonTime: number;
  thunderActive: boolean;
  setConnected: (connected: boolean) => void;
  send: (msg: Omit<MeshMessage, 'timestamp'>) => void;
}

let ws: WebSocket | null = null;

function getBrokerUrl(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get('broker') || `ws://${window.location.hostname}:9001/ws`;
}

function connect(set: (fn: ShipState | Partial<ShipState>) => void, get: () => ShipState) {
  if (ws?.readyState === WebSocket.OPEN || ws?.readyState === WebSocket.CONNECTING) return;

  ws = new WebSocket(get().brokerUrl);

  ws.onopen = () => {
    set({ connected: true });
    get().send({
      topic: 'prop.state.announce',
      source: 'garage_ship',
      payload: {
        id: 'garage_ship',
        name: 'Garage Ship Projection',
        capabilities: ['effects.cannon.fire', 'effects.thunder.clap', 'scene.garage'],
      },
    });
    get().send({
      topic: 'scene.start',
      source: 'garage_ship',
      payload: { scene: 'garage' },
    });
  };

  ws.onclose = () => {
    set({ connected: false });
    ws = null;
    setTimeout(() => connect(set, get), 1500);
  };

  ws.onerror = () => ws?.close();
}

export const useMeshStore = create<ShipState>((set) => ({
  connected: false,
  brokerUrl: getBrokerUrl(),
  lastCannonTime: 0,
  thunderActive: false,

  setConnected: (connected: boolean) => set({ connected }),

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
  ws?.addEventListener('message', wrapped);
  // Also listen on a re-created socket by re-registering after reconnect
  const originalConnect = ws?.onopen;
  ws?.addEventListener('open', () => {
    ws?.addEventListener('message', wrapped);
    if (originalConnect) originalConnect.call(ws as any, new Event('open'));
  });
}

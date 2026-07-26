import { create } from 'zustand';
import type { ActiveEffect, LogEntry, MeshMessage, PropAnnouncement } from '../types/mesh';

interface MeshState {
  connected: boolean;
  brokerUrl: string;
  session: string;
  scene: string;
  estop: boolean;
  manifest: Record<string, PropAnnouncement>;
  activeEffects: Record<string, ActiveEffect>;
  logs: LogEntry[];
  selectedTopics: Set<string>;

  setConnected: (connected: boolean) => void;
  setBrokerUrl: (url: string) => void;
  setSession: (session: string) => void;
  setScene: (scene: string) => void;
  setEstop: (estop: boolean) => void;
  addLog: (msg: MeshMessage) => void;
  handleAnnounce: (source: string, payload: PropAnnouncement) => void;
  handleHeartbeat: (source: string, payload: Record<string, unknown>) => void;
  handleEffectStart: (msg: MeshMessage) => void;
  handleEffectEnd: (source: string, topic: string) => void;
  clearLogs: () => void;
  sendMessage: (msg: Omit<MeshMessage, 'timestamp'>) => void;
  fire: (topic: string, payload?: Record<string, unknown>) => void;
  setSceneCommand: (scene: string) => void;
}

let ws: WebSocket | null = null;
let logIdCounter = 0;

function getBrokerUrl(): string {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get('broker');
  if (fromUrl) return fromUrl;
  return `ws://${window.location.hostname}:9001/ws`;
}

function connect(set: (fn: (state: MeshState) => Partial<MeshState>) => void, get: () => MeshState) {
  if (ws?.readyState === WebSocket.OPEN || ws?.readyState === WebSocket.CONNECTING) return;

  const url = get().brokerUrl;
  ws = new WebSocket(url);

  ws.onopen = () => {
    set(() => ({ connected: true }));
    const source = 'control_center';
    get().sendMessage({
      topic: 'prop.state.announce',
      source,
      payload: {
        id: source,
        name: 'Control Center',
        capabilities: ['scene.*'],
      },
    });
    get().sendMessage({
      topic: 'scene.start',
      source,
      payload: { scene: get().scene },
    });
  };

  ws.onmessage = (event) => {
    let msg: MeshMessage;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    get().addLog(msg);

    if (msg.topic === 'prop.state.announce' && msg.source) {
      get().handleAnnounce(msg.source, msg.payload as PropAnnouncement);
    }
    if (msg.topic === 'prop.state.heartbeat' && msg.source) {
      get().handleHeartbeat(msg.source, msg.payload);
    }
    if (msg.topic.startsWith('effects.')) {
      get().handleEffectStart(msg);
    }
    if (msg.topic === 'world.estop') {
      set(() => ({ estop: Boolean(msg.payload.active) }));
    }
    if (msg.topic.startsWith('scene.')) {
      const newScene = msg.payload.scene as string | undefined;
      if (newScene) set(() => ({ scene: newScene }));
    }
  };

  ws.onclose = () => {
    set(() => ({ connected: false }));
    ws = null;
    setTimeout(() => connect(set, get), 1500);
  };

  ws.onerror = () => {
    ws?.close();
  };
}

export const useMeshStore = create<MeshState>((set, get) => ({
  connected: false,
  brokerUrl: getBrokerUrl(),
  session: 'halloween-2026',
  scene: 'idle',
  estop: false,
  manifest: {},
  activeEffects: {},
  logs: [],
  selectedTopics: new Set(),

  setConnected: (connected) => set({ connected }),
  setBrokerUrl: (brokerUrl) => {
    set({ brokerUrl });
    connect(set, get);
  },
  setSession: (session) => set({ session }),
  setScene: (scene) => set({ scene }),
  setEstop: (estop) => set({ estop }),

  addLog: (msg) => {
    const entry: LogEntry = {
      id: `${Date.now()}-${++logIdCounter}`,
      time: new Date(msg.timestamp * 1000).toLocaleTimeString(),
      topic: msg.topic,
      source: msg.source,
      payload: JSON.stringify(msg.payload ?? {}),
    };
    set((state) => ({
      logs: [entry, ...state.logs].slice(0, 250),
    }));
  },

  handleAnnounce: (source, payload) => {
    set((state) => ({
      manifest: {
        ...state.manifest,
        [source]: { ...payload, id: payload.id || source },
      },
    }));
  },

  handleHeartbeat: (source, payload) => {
    if (!get().manifest[source]) return;
    set((state) => ({
      manifest: {
        ...state.manifest,
        [source]: { ...state.manifest[source], ...payload, last_seen: Date.now() },
      },
    }));
  },

  handleEffectStart: (msg) => {
    const key = `${msg.source}::${msg.topic}`;
    set((state) => ({
      activeEffects: {
        ...state.activeEffects,
        [key]: {
          topic: msg.topic,
          source: msg.source,
          started: Date.now(),
          duration_ms: msg.payload.duration_ms as number | undefined,
        },
      },
    }));
    const duration = (msg.payload.duration_ms as number) || 5000;
    setTimeout(() => {
      get().handleEffectEnd(msg.source, msg.topic);
    }, duration + 250);
  },

  handleEffectEnd: (source, topic) => {
    const key = `${source}::${topic}`;
    set((state) => {
      const { [key]: _, ...rest } = state.activeEffects;
      return { activeEffects: rest };
    });
  },

  clearLogs: () => set({ logs: [] }),

  sendMessage: (msg) => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          ...msg,
          timestamp: Date.now() / 1000,
          meta: { seq: 0, session: get().session, ...(msg.meta || {}) },
        })
      );
    }
  },

  fire: (topic, payload = {}) => {
    get().sendMessage({
      topic,
      source: 'control_center',
      payload,
    });
  },

  setSceneCommand: (scene) => {
    get().sendMessage({
      topic: scene === 'estop' ? 'scene.estop' : 'scene.start',
      source: 'control_center',
      payload: scene === 'estop' ? { active: true } : { scene },
    });
    if (scene === 'resume') {
      get().sendMessage({
        topic: 'scene.resume',
        source: 'control_center',
        payload: {},
      });
    }
    set({ scene });
  },
}));

// Auto-connect on import
connect(useMeshStore.setState, useMeshStore.getState);

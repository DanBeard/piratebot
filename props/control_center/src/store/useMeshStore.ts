import { create } from 'zustand';
import type { ActiveEffect, LogEntry, MeshMessage, PropAnnouncement, WorldView } from '../types/mesh';

interface MeshState {
  connected: boolean;
  brokerUrl: string;
  session: string;
  scene: string;
  estop: boolean;
  family_mode: boolean;
  manifest: Record<string, PropAnnouncement>;
  activeEffects: Record<string, ActiveEffect>;
  logs: LogEntry[];
  selectedTopics: Set<string>;
  world: WorldView;

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
  handleWorldUpdate: (msg: MeshMessage) => void;
  clearLogs: () => void;
  sendMessage: (msg: Omit<MeshMessage, 'timestamp'>) => void;
  fire: (topic: string, payload?: Record<string, unknown>) => void;
  setSceneCommand: (scene: string) => void;
  toggleFamilyMode: () => void;
  fireCannon: () => void;
  nextPumpkinSong: () => void;
  testPortraitLine: () => void;
}

let ws: WebSocket | null = null;
let logIdCounter = 0;

const DEFAULT_ZONES = ['sidewalk', 'front_yard', 'driveway', 'graveyard', 'sideyard'];

function emptyWorld(): WorldView {
  return {
    scene: 'idle',
    family_mode: false,
    audience_present: false,
    pumpkins_singing: false,
    portrait_speaking: false,
    estop: false,
    cannon_cooldown: false,
    fog_cooldown: false,
    thunder_cooldown: false,
    linger_alert: false,
    zones: DEFAULT_ZONES.map((zone) => ({ zone, occupied: false, count: 0, linger_s: 0 })),
  };
}

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
        capabilities: ['scene.*', 'director.*'],
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
    if (msg.topic.startsWith('world.')) {
      get().handleWorldUpdate(msg);
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
  family_mode: false,
  manifest: {},
  activeEffects: {},
  logs: [],
  selectedTopics: new Set(),
  world: emptyWorld(),

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

  handleWorldUpdate: (msg) => {
    const { topic, payload } = msg;
    set((state) => {
      const world = { ...state.world };
      const zones = new Map(world.zones.map((z) => [z.zone, { ...z }]));

      if (topic === 'world.audience.present') {
        world.audience_present = true;
      } else if (topic === 'world.audience.absent') {
        world.audience_present = false;
      } else if (topic === 'world.sun') {
        // no UI state yet
      } else if (topic === 'world.late_night') {
        // no UI state yet
      } else if (topic.startsWith('world.') && topic.endsWith('.occupied')) {
        const zone = topic.slice('world.'.length, -'.occupied'.length);
        const z = zones.get(zone) || { zone, occupied: false, count: 0, linger_s: 0 };
        z.occupied = true;
        z.count = (payload.count as number) || 1;
        zones.set(zone, z);
      } else if (topic.startsWith('world.') && topic.endsWith('.vacant')) {
        const zone = topic.slice('world.'.length, -'.vacant'.length);
        const z = zones.get(zone) || { zone, occupied: false, count: 0, linger_s: 0 };
        z.occupied = false;
        z.count = 0;
        z.linger_s = 0;
        zones.set(zone, z);
      }

      world.zones = DEFAULT_ZONES.map((zone) => zones.get(zone) || { zone, occupied: false, count: 0, linger_s: 0 });
      world.linger_alert = world.zones.some((z) => z.occupied && z.linger_s >= 180);
      return { world };
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
      topic: scene === 'estop' ? 'scene.estop' : scene === 'resume' ? 'scene.resume' : 'scene.start',
      source: 'control_center',
      payload: scene === 'estop' ? { active: true } : { scene },
    });
    set({ scene });
  },

  toggleFamilyMode: () => {
    const next = !get().family_mode;
    get().sendMessage({
      topic: next ? 'scene.family_mode' : 'scene.spooky_mode',
      source: 'control_center',
      payload: { active: next },
    });
    set({ family_mode: next });
  },

  fireCannon: () => {
    get().sendMessage({
      topic: 'director.cannon.fire',
      source: 'control_center',
      payload: { reason: 'manual' },
    });
  },

  nextPumpkinSong: () => {
    get().sendMessage({
      topic: 'director.pirate_button',
      source: 'control_center',
      payload: { action: 'next_song' },
    });
  },

  testPortraitLine: () => {
    get().sendMessage({
      topic: 'pirate.speak',
      source: 'control_center',
      payload: { category: 'greeting', interruptible: true },
    });
  },
}));

// Auto-connect on import
connect(useMeshStore.setState, useMeshStore.getState);

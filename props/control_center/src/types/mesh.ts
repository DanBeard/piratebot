export interface Timing {
  delay_ms?: number | null;
  at_ts?: number | null;
  expire_ms?: number | null;
}

export interface Meta {
  seq: number;
  session: string;
  codecs?: string[];
}

export interface MeshMessage {
  topic: string;
  source: string;
  target?: string | null;
  payload: Record<string, unknown>;
  timing?: Timing;
  meta?: Meta;
  timestamp: number;
}

export interface PropAnnouncement {
  id: string;
  name?: string;
  capabilities: string[];
  codecs?: string[];
  safety?: Record<string, number>;
  [key: string]: unknown;
}

export interface ActiveEffect {
  topic: string;
  source: string;
  started: number;
  duration_ms?: number;
}

export interface LogEntry {
  id: string;
  time: string;
  topic: string;
  source: string;
  payload: string;
}

export interface ZoneView {
  zone: string;
  occupied: boolean;
  count: number;
  linger_s: number;
}

export interface WorldView {
  scene: string;
  family_mode: boolean;
  audience_present: boolean;
  pumpkins_singing: boolean;
  portrait_speaking: boolean;
  estop: boolean;
  cannon_cooldown: boolean;
  fog_cooldown: boolean;
  thunder_cooldown: boolean;
  linger_alert: boolean;
  zones: ZoneView[];
}

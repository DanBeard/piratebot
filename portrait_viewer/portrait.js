/**
 * Portrait player: manifest-driven layered 2D pirate portrait with lip sync,
 * eye tracking, and CSS-driven emotes. Connects to the Python orchestrator
 * via WebSocket.
 *
 * The set of PNG layers, their sizes, and their positions are loaded from
 * assets/<set>/manifest.json. Swapping pirates or backgrounds only requires
 * generating a new asset set and changing the config.
 */

const WS_URL = `ws://${window.location.host}/ws`;
const MANIFEST_URL = 'assets/default/manifest.json';

// Rhubarb mouth cue -> mouth sprite file suffix.
const VISEME_SPRITES = {
  'X': 'mouth_rest.png',
  'H': 'mouth_rest.png',
  'B': 'mouth_rest.png',
  'A': 'mouth_ah.png',
  'D': 'mouth_ah.png',
  'C': 'mouth_ee.png',
  'E': 'mouth_ee.png',
  'F': 'mouth_f.png',
  'G': 'mouth_oh.png',
  'I': 'mouth_oh.png',
};

// Emotion -> CSS class on the avatar wrapper.
const EMOTES = {
  'neutral': '',
  'happy': 'emote-happy',
  'surprised': 'emote-surprised',
  'angry': 'emote-angry',
  'sad': 'emote-sad',
  'laugh': 'emote-laugh',
  'thinking': 'emote-thinking',
  'grumpy': 'emote-grumpy',
  'amused': 'emote-happy',
  'impressed': 'emote-surprised',
  'menacing': 'emote-angry',
  'dramatic': 'emote-surprised',
};

let manifest = null;
let assetBase = 'assets/default/';
let ws = null;
let reconnectTimeout = null;
let visemes = [];
let visemeIndex = 0;
let isTalking = false;
let gazeTarget = { x: 0.5, y: 0.5 };
let currentEmoteClass = '';
let idleDriftStart = performance.now();

const elements = {
  canvas: document.getElementById('canvas'),
  overlay: document.getElementById('overlay'),
  status: document.getElementById('status'),
  audio: document.getElementById('player'),
};

// Dynamically created layer elements.
const layers = {};

function setStatus(msg) {
  elements.status.textContent = msg;
  console.log('[portrait]', msg);
}

async function loadManifest() {
  setStatus('Loading asset manifest...');
  const res = await fetch(MANIFEST_URL);
  if (!res.ok) throw new Error(`Manifest fetch failed: ${res.status}`);
  manifest = await res.json();
  assetBase = manifest.files.background.replace('background.png', '');
  buildLayers();
  setStatus(`Loaded asset set: ${manifest.name}`);
}

function px(value) {
  return `${value * 100}%`;
}

function buildLayers() {
  const layout = manifest.layout;

  // Background
  const bg = createLayer('background', 'img');
  bg.src = fileUrl('background');

  // Body wrapper (for future tweening)
  const bodyWrap = document.createElement('div');
  bodyWrap.id = 'body-wrap';
  bodyWrap.className = 'layer-wrap';
  bodyWrap.style.width = '100%';
  bodyWrap.style.height = '100%';
  elements.canvas.insertBefore(bodyWrap, elements.overlay);
  const body = createLayer('body', 'img', bodyWrap);
  body.src = fileUrl('body');
  layers.bodyWrap = bodyWrap;

  // Head wrapper (for rotation/tilt tweening)
  const headWrap = document.createElement('div');
  headWrap.id = 'head-wrap';
  headWrap.className = 'layer-wrap';
  positionWrapper(headWrap, layout.head);
  elements.canvas.insertBefore(headWrap, elements.overlay);
  const head = createLayer('head', 'img', headWrap);
  head.src = fileUrl('head');
  layers.headWrap = headWrap;

  // Mouth
  const mouthWrap = document.createElement('div');
  mouthWrap.id = 'mouth-wrap';
  mouthWrap.className = 'layer-wrap';
  positionWrapper(mouthWrap, layout.mouth);
  elements.canvas.insertBefore(mouthWrap, elements.overlay);
  const mouth = createLayer('mouth', 'img', mouthWrap);
  mouth.id = 'mouth';
  mouth.style.opacity = '0';
  mouth.style.transition = 'opacity 0.08s';
  layers.mouth = mouth;

  // Eyes
  const eyesWrap = document.createElement('div');
  eyesWrap.id = 'eyes-wrap';
  eyesWrap.className = 'layer-wrap';
  elements.canvas.insertBefore(eyesWrap, elements.overlay);
  ['left', 'right'].forEach(side => {
    const cfg = layout.eyes[side];
    const eye = document.createElement('div');
    eye.className = 'eye';
    eye.id = `eye-${side}`;
    eye.style.left = px(cfg.center_x - cfg.width / 2);
    eye.style.top = px(cfg.center_y - cfg.height / 2);
    eye.style.width = px(cfg.width);
    eye.style.height = px(cfg.height);

    const white = document.createElement('img');
    white.className = 'eye-white';
    white.src = fileUrl(`eye_${side}`);
    eye.appendChild(white);

    const pupil = document.createElement('div');
    pupil.className = 'pupil';
    pupil.id = `pupil-${side}`;
    pupil.style.width = px(layout.pupil.width);
    pupil.style.height = px(layout.pupil.height);
    pupil.style.top = px(0.5 - layout.pupil.height / 2);
    pupil.style.left = px(0.5 - layout.pupil.width / 2);
    eye.appendChild(pupil);

    eyesWrap.appendChild(eye);
    layers[`eye_${side}`] = eye;
    layers[`pupil_${side}`] = pupil;
  });

  // Avatar wrapper groups body + head for emote transforms.
  const avatar = document.createElement('div');
  avatar.id = 'avatar';
  avatar.className = 'avatar';
  // Move body and head wrappers inside avatar so emotes can transform them together.
  avatar.appendChild(bodyWrap);
  avatar.appendChild(headWrap);
  avatar.appendChild(mouthWrap);
  avatar.appendChild(eyesWrap);
  elements.canvas.insertBefore(avatar, elements.overlay);
  layers.avatar = avatar;
}

function fileUrl(name) {
  return manifest.files[name];
}

function createLayer(id, tag, parent) {
  const el = document.createElement(tag);
  el.id = id;
  el.className = 'layer';
  const target = parent || elements.canvas;
  target.insertBefore(el, elements.overlay);
  layers[id] = el;
  return el;
}

function positionWrapper(wrap, cfg) {
  wrap.style.position = 'absolute';
  wrap.style.left = px(cfg.center_x - cfg.width / 2);
  wrap.style.top = px(cfg.center_y - cfg.width / 2); // square-ish anchor using width as proxy height
  wrap.style.width = px(cfg.width);
  wrap.style.height = px(cfg.width);
}

function connect() {
  setStatus(`Connecting to ${WS_URL}...`);
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setStatus('Connected to Captain Barnacle Bill');
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
  };

  ws.onmessage = (event) => {
    try {
      const cmd = JSON.parse(event.data);
      handleCommand(cmd);
    } catch (err) {
      console.error('Bad command:', event.data, err);
    }
  };

  ws.onclose = () => {
    setStatus('Disconnected — retrying...');
    reconnectTimeout = setTimeout(connect, 2000);
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
  };
}

function handleCommand(cmd) {
  switch (cmd.type) {
    case 'play_audio':
      playAudio(cmd.audio_url, cmd.visemes, cmd.emotion);
      break;
    case 'stop_audio':
      stopAudio();
      break;
    case 'set_gaze':
      setGaze(cmd.x, cmd.y);
      break;
    case 'set_expression':
      applyEmotion(cmd.expression);
      break;
    case 'play_animation':
      applyEmotion(cmd.animation);
      break;
    case 'reset':
      resetPortrait();
      break;
    default:
      console.warn('Unknown command:', cmd);
  }
}

function playAudio(audioUrl, newVisemes, emotion) {
  stopAudio();
  if (emotion) applyEmotion(emotion);

  elements.audio.src = audioUrl;
  elements.audio.load();

  visemes = (newVisemes || []).sort((a, b) => a.start - b.start);
  visemeIndex = 0;
  isTalking = true;
  layers.mouth.style.opacity = '1';

  elements.audio.play().catch((err) => {
    console.error('Audio play failed:', err);
    isTalking = false;
    layers.mouth.style.opacity = '0';
  });

  elements.audio.onended = () => {
    isTalking = false;
    layers.mouth.style.opacity = '0';
    applyEmotion('neutral');
  };
}

function stopAudio() {
  if (!elements.audio.paused) {
    elements.audio.pause();
    elements.audio.currentTime = 0;
  }
  elements.audio.onended = null;
  isTalking = false;
  visemes = [];
  visemeIndex = 0;
  if (layers.mouth) layers.mouth.style.opacity = '0';
}

function setGaze(x, y) {
  gazeTarget = { x: clamp(x, 0, 1), y: clamp(y, 0, 1) };
}

function applyEmotion(emotion) {
  const cls = EMOTES[emotion] || '';
  if (currentEmoteClass) {
    layers.avatar.classList.remove(currentEmoteClass);
  }
  currentEmoteClass = cls;
  if (cls) {
    layers.avatar.classList.add(cls);
  }

  elements.overlay.className = '';
  if (emotion === 'surprised' || emotion === 'impressed') {
    elements.overlay.classList.add('candlelight');
  }
}

function resetPortrait() {
  stopAudio();
  applyEmotion('neutral');
  setGaze(0.5, 0.5);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function updateEyes() {
  let targetX = gazeTarget.x;
  let targetY = gazeTarget.y;

  if (!isTalking && targetX === 0.5 && targetY === 0.5) {
    const t = (performance.now() - idleDriftStart) / 1000;
    targetX = 0.5 + Math.sin(t * 0.3) * 0.08;
    targetY = 0.5 + Math.cos(t * 0.2) * 0.05;
  }

  const maxPx = 8;
  const dx = (targetX - 0.5) * maxPx * 2;
  const dy = (targetY - 0.5) * maxPx * 2;

  if (layers.pupil_left) layers.pupil_left.style.transform = `translate(${dx}px, ${dy}px)`;
  if (layers.pupil_right) layers.pupil_right.style.transform = `translate(${dx}px, ${dy}px)`;
}

function updateMouth() {
  if (!isTalking || visemes.length === 0 || !layers.mouth) return;

  const t = elements.audio.currentTime;
  if (isNaN(t)) return;

  while (visemeIndex < visemes.length && t >= visemes[visemeIndex].end) {
    visemeIndex++;
  }

  const cue = visemes[visemeIndex];
  if (cue && t >= cue.start && t < cue.end) {
    const file = VISEME_SPRITES[cue.shape] || VISEME_SPRITES['X'];
    const src = assetBase + file;
    if (layers.mouth.src !== src) {
      layers.mouth.src = src;
    }
  } else if (cue && t < cue.start) {
    layers.mouth.src = assetBase + VISEME_SPRITES['X'];
  }
}

function loop() {
  updateEyes();
  updateMouth();
  requestAnimationFrame(loop);
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
    document.body.classList.add('fullscreen');
  } else {
    document.exitFullscreen().catch(() => {});
    document.body.classList.remove('fullscreen');
  }
}

document.addEventListener('click', toggleFullscreen);
document.addEventListener('keydown', (e) => {
  if (e.key === 'f' || e.key === 'F' || e.key === ' ') {
    toggleFullscreen();
  }
  if (e.key === 'q' || e.key === 'Escape') {
    stopAudio();
  }
});

loadManifest()
  .then(() => {
    connect();
    loop();
  })
  .catch((err) => {
    setStatus(`Failed to load manifest: ${err.message}`);
    console.error(err);
  });

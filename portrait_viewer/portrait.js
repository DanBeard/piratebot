/**
 * Portrait player: layered 2D pirate portrait with lip sync, eye tracking,
 * and CSS-driven emotes. Connects to the Python orchestrator via WebSocket.
 */

const WS_URL = `ws://${window.location.host}/ws`;

// Rhubarb mouth cue -> mouth sprite file.
const VISEME_SPRITES = {
  'X': 'assets/mouth_rest.png',
  'H': 'assets/mouth_rest.png',
  'B': 'assets/mouth_rest.png',
  'A': 'assets/mouth_ah.png',
  'D': 'assets/mouth_ah.png',
  'C': 'assets/mouth_ee.png',
  'E': 'assets/mouth_ee.png',
  'F': 'assets/mouth_f.png',
  'G': 'assets/mouth_oh.png',
  'I': 'assets/mouth_oh.png',  // fallback
};

// Emotion -> CSS class on body + overlay effect.
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

const elements = {
  mouth: document.getElementById('mouth'),
  body: document.getElementById('body'),
  leftEye: document.getElementById('eye-left'),
  rightEye: document.getElementById('eye-right'),
  leftPupil: document.getElementById('pupil-left'),
  rightPupil: document.getElementById('pupil-right'),
  overlay: document.getElementById('overlay'),
  status: document.getElementById('status'),
  audio: document.getElementById('player'),
};

let ws = null;
let reconnectTimeout = null;
let visemes = [];
let visemeIndex = 0;
let isTalking = false;
let gazeTarget = { x: 0.5, y: 0.5 };
let currentEmoteClass = '';
let idleDriftStart = performance.now();

// Prefer eye-white images if present; fallback to CSS circles.
let useImageEyes = false;

async function loadImageEyes() {
  try {
    const left = new Image();
    const right = new Image();
    left.src = 'assets/eye_left.png';
    right.src = 'assets/eye_right.png';
    await Promise.all([
      new Promise((res, rej) => { left.onload = res; left.onerror = rej; }),
      new Promise((res, rej) => { right.onload = res; right.onerror = rej; }),
    ]);
    useImageEyes = true;
    elements.leftEye.style.background = 'transparent';
    elements.rightEye.style.background = 'transparent';
    elements.leftEye.style.opacity = '1';
    elements.rightEye.style.opacity = '1';
  } catch (e) {
    // Fallback CSS eye whites.
    elements.leftEye.style.opacity = '1';
    elements.rightEye.style.opacity = '1';
  }
}

function setStatus(msg) {
  elements.status.textContent = msg;
  console.log('[portrait]', msg);
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
  elements.mouth.style.opacity = '1';

  elements.audio.play().catch((err) => {
    console.error('Audio play failed:', err);
    isTalking = false;
    elements.mouth.style.opacity = '0';
  });

  elements.audio.onended = () => {
    isTalking = false;
    elements.mouth.style.opacity = '0';
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
  elements.mouth.style.opacity = '0';
}

function setGaze(x, y) {
  gazeTarget = { x: clamp(x, 0, 1), y: clamp(y, 0, 1) };
}

function applyEmotion(emotion) {
  const cls = EMOTES[emotion] || '';
  if (currentEmoteClass) {
    elements.body.classList.remove(currentEmoteClass);
  }
  currentEmoteClass = cls;
  if (cls) {
    elements.body.classList.add(cls);
  }

  // Overlay effects
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

  // If no one is around, slow idle drift.
  if (!isTalking && targetX === 0.5 && targetY === 0.5) {
    const t = (performance.now() - idleDriftStart) / 1000;
    targetX = 0.5 + Math.sin(t * 0.3) * 0.08;
    targetY = 0.5 + Math.cos(t * 0.2) * 0.05;
  }

  const maxPx = 8; // maximum pupil offset in CSS pixels
  const dx = (targetX - 0.5) * maxPx * 2;
  const dy = (targetY - 0.5) * maxPx * 2;

  elements.leftPupil.style.transform = `translate(${dx}px, ${dy}px)`;
  elements.rightPupil.style.transform = `translate(${dx}px, ${dy}px)`;
}

function updateMouth() {
  if (!isTalking || visemes.length === 0) return;

  const t = elements.audio.currentTime;
  if (isNaN(t)) return;

  // Advance to current cue
  while (visemeIndex < visemes.length && t >= visemes[visemeIndex].end) {
    visemeIndex++;
  }

  const cue = visemes[visemeIndex];
  if (cue && t >= cue.start && t < cue.end) {
    const src = VISEME_SPRITES[cue.shape] || VISEME_SPRITES['X'];
    if (elements.mouth.src !== src) {
      elements.mouth.src = src;
    }
  } else if (cue && t < cue.start) {
    // Gap before next cue: rest
    elements.mouth.src = VISEME_SPRITES['X'];
  }
}

function loop() {
  updateEyes();
  updateMouth();
  requestAnimationFrame(loop);
}

// Fullscreen on click / key
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

loadImageEyes().then(() => {
  connect();
  loop();
});

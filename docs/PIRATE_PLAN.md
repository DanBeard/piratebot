# PirateBot 2026 Halloween Plan

## Goal

Build an interactive porch pirate that recognizes trick-or-treaters' costumes and responds with pre-approved, family-safe pirate voice lines, lip-sync, and eye contact. The 2026 target is a **"haunted painting" portrait** displayed on a framed monitor.

## Current status

The portrait-mode path is implemented and runnable. It only needs:
1. The cluster `parrotts` voice-cloning service restored.
2. A set of portrait PNG assets.
3. The porch PC (gaming PC with RTX 3080) back online.

The 3D Godot path is kept as a future option but is not the priority.

---

## Architecture

```
Porch PC (RTX 3080 / 64GB RAM)
  ├─ Webcam
  ├─ YOLOv8 person detection
  ├─ Moondream2 costume description
  ├─ PirateBot orchestrator (main.py)
  ├─ PortraitAvatarController (WebSocket + HTTP server)
  └─ Chrome kiosk showing the portrait player

Cluster services (when hardware available)
  ├─ parrotts — voice cloning + voice library
  └─ litellm — optional LLM for drafting lines (never live inference at showtime)
```

All speech is pre-generated from a **human-curated** voice corpus. There is no live text generation at showtime, so children cannot trick the pirate into saying something unsafe.

---

## Voice corpus safety model

| Stage | What happens | Who controls it |
|---|---|---|
| 1. Draft | `tools/expand_voice_lines.py` asks a local LLM for candidate lines | LLM, offline |
| 2. Curate | Human reviews, edits, and sets `approved: true` in `data/voice_lines.yaml` | Human |
| 3. Generate | `tools/migrate_to_parrotts.py` registers the pirate voice and batch-generates WAVs | Human-triggered |
| 4. Cache | `tools/mirror_parrotts_cache.py` downloads WAVs to porch PC | Human-triggered |
| 5. Visemes | `tools/batch_rhubarb.py` pre-computes lip-sync JSON | Automated |
| 6. Showtime | PirateBot only plays pre-approved audio + visemes | No new text generation |

---

## File inventory

### Core orchestration

| File | Purpose |
|---|---|
| `main.py` | Main loop: webcam → detection → VLM → line selection → avatar |
| `config.portrait.yaml` | Porch-PC config for portrait mode |
| `config.porch.yaml` | Porch-PC config for 3D/Godot mode |
| `scripts/setup-porch.sh` | Install deps, cache models, mirror voice cache |
| `scripts/run-portrait.sh` | Launch Chrome kiosk + orchestrator |
| `scripts/run-porch.sh` | Launch Godot + orchestrator (3D path) |

### Vision / detection

| File | Purpose |
|---|---|
| `services/yolo_detector.py` | YOLOv8 person detection + tracking |
| `services/moondream_vlm.py` | Moondream2 costume description |
| `services/appearance_cache.py` | Recognize returning visitors |
| `services/interaction_state.py` | Track arrive/interact/depart lifecycle |

### Voice / audio

| File | Purpose |
|---|---|
| `data/voice_lines.yaml` | Curated line library with approval flags |
| `services/parrotts_tts.py` | Search and fetch lines from cluster parrotts |
| `services/parrotts_vendor/` | Vendored HTTP client for parrotts |
| `tools/migrate_to_parrotts.py` | Register pirate + batch-generate WAVs |
| `tools/mirror_parrotts_cache.py` | Download WAVs to porch PC |
| `tools/batch_rhubarb.py` | Pre-compute visemes for all WAVs |
| `tools/expand_voice_lines.py` | LLM-assisted line drafting with approval gate |

### Portrait avatar (2026 primary path)

| File | Purpose |
|---|---|
| `services/portrait_avatar.py` | WebSocket + HTTP server for browser player |
| `portrait_viewer/index.html` | Browser player layout |
| `portrait_viewer/portrait.css` | Painting frame, layers, emotes |
| `portrait_viewer/portrait.js` | Lip sync, eye tracking, emotes, audio playback |

### 3D avatar (future)

| File | Purpose |
|---|---|
| `services/godot_avatar.py` | WebSocket client for Godot 3D avatar |
| `godot_project/` | Godot 4 placeholder scene + scripts |
| `docs/PIRATE_MODEL.md` | 3D model sourcing guide |

### Docs

| File | Purpose |
|---|---|
| `docs/PORTRAIT_BACKUP_PLAN.md` | Original portrait design rationale |
| `docs/PIRATE_PLAN.md` | This file |

---

## Pre-show checklist

### 1. Cluster: restore parrotts

```bash
ssh beard@192.168.0.2 "sudo kubectl scale deploy/parrotts --replicas=1 -n default"
```

Wait for the pod to start. The deployment now targets any `gpu=true` node.

### 2. Porch PC: install system deps

- NVIDIA drivers + CUDA
- Webcam
- Godot 4 (only if doing 3D path)
- Rhubarb lip-sync CLI
- Chrome
- uv

### 3. Porch PC: setup

```bash
cd ~/piratebot
./scripts/setup-porch.sh
```

### 4. Create portrait assets

Place PNGs in `portrait_viewer/assets/`:

| Required | Description |
|---|---|
| `background.png` | Stormy sea scene |
| `body.png` | Pirate portrait, no mouth/eyes |
| `mouth_rest.png` | Closed mouth |
| `mouth_ah.png` | Open "ah" |
| `mouth_ee.png` | Wide "ee" |
| `mouth_oh.png` | Rounded "oh" |
| `mouth_f.png` | F/V lip shape |

Optional:
| `eye_left.png`, `eye_right.png` | Eye-white images |
| `pupil.png` | Custom pupil sprite |

### 5. Build the voice corpus

Draft new lines:

```bash
uv run python tools/expand_voice_lines.py \
  --theme "comebacks when kids say the pirate is fake" \
  --category comebacks \
  --subcategory fake \
  --emotion grumpy \
  --count 20
```

Edit `data/voice_lines.yaml` and set `approved: true` on good lines.

Repeat for other themes: roasts, jokes, specific costumes, group greetings, time-of-night, weather.

### 6. Generate audio

```bash
kubectl port-forward -n default svc/parrotts 18003:8000
uv run python tools/migrate_to_parrotts.py
```

### 7. Cache audio + visemes on porch PC

```bash
uv run python tools/mirror_parrotts_cache.py --base-url http://localhost:18003
uv run python tools/batch_rhubarb.py
```

### 8. Run the show

```bash
./scripts/run-portrait.sh
```

---

## Physical prop notes

- 24–32" monitor, ideally portrait orientation.
- Thrift-store picture frame around the monitor.
- Backlight strip behind frame for a haunted glow.
- Optional fog machine triggered by detection events.

---

## Known issues / risks

| Risk | Mitigation |
|---|---|
| No free cluster GPU for parrotts | Can run parrotts locally on the porch PC's 3080, or scale down a cluster LLM pod temporarily |
| Portrait mouth alignment off | Photopea/Blender adjustment once assets are in place |
| Eye tracking looks robotic | Tune `maxPx` offset and add saccade drift in `portrait.js` |
| Kids try to make it say bad things | No live inference; only pre-approved lines |
| Audio latency | Use local WAV cache; network only for optional LLM drafting |

---

## Tools / debug

| File | Purpose |
|---|---|
| `tools/debug_interaction.py` | **(planned)** Text-based PirateBot simulator. Enter a costume description or kid line and see the chosen voice line + prop triggers without a webcam. |

---

## In-progress / next-up TODO list

1. **Wire broker perception fuser** — connect `PerceptionFuser` to the message bus and add experience/scene rules.
2. **Map old Helm story beats to new prop mesh** — translate 2021 ambiance/pumpkins/cannon beats into fuser rules.
3. **Zigbee2MQTT device mapping** — fill in friendly names in `props/broker/mesh_broker.py` once hardware is installed.
4. **Field test the broker + displays + ESP32 props** on a laptop without cameras or cluster.
5. **Restore `parrotts` cluster service** once GPU capacity is available.
6. **Generate portrait PNG assets** once the art pipeline is ready.
7. **Build `tools/debug_interaction.py`** — a text-based PirateBot walkthrough/simulator so we can enter costume descriptions or kid lines and see the chosen voice line + prop reactions without buying 1000 costumes.

---

## Future improvements (post-2026)

- Add a real 3D rigged pirate model and switch back to Godot.
- Add more emote layers (hat tip, coat flutter, lantern flicker).
- Add weather/time-of-night context to line selection.
- Add group-size detection (one kid vs. a crowd).
- Add audio input for kid heckles → safe heckler comeback lines.

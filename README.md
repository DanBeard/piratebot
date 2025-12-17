# PirateBot - Interactive Halloween Pirate Decoration

**"Arrr, nice mermaid costume, matey!"**

An AI-powered interactive pirate character for your Halloween porch. When trick-or-treaters approach, the pirate sees their costume, thinks of something clever to say, and speaks it aloud with lip-synced animation - all in under 10 seconds, running entirely on your local hardware.

---

## What It Does

1. **Sees** - A webcam detects when someone approaches
2. **Recognizes** - AI describes their costume ("a child dressed as a vampire")
3. **Thinks** - A pirate-personality AI generates a fun comment
4. **Speaks** - Text-to-speech with a gruff pirate voice
5. **Animates** - 3D pirate avatar moves its mouth in sync

**Example interactions:**
- *Kid in skeleton costume* → "Arrr, a mighty fine skeleton ye be! Ye'd fit right in on me ghost ship!"
- *Kid as a superhero* → "Shiver me timbers, a superhero! Even pirates need heroes sometimes!"
- *Group of kids* → "Blimey! A whole crew of scallywags! Welcome aboard!"

---

## Demo

*[Add video/GIF of PirateBot in action here]*

---

## Why Local?

Everything runs on YOUR computer:
- **Privacy**: Photos of kids stay on your machine, never uploaded
- **Speed**: No internet latency - responses in 3-7 seconds
- **Cost**: No API fees - run it all night for free
- **Reliability**: Works even if your internet goes down

---

## Hardware Requirements

### Minimum (Single GPU)
- NVIDIA RTX 3080 (10GB VRAM) or better
- 32GB RAM
- USB webcam
- Display/monitor for the pirate

### Recommended (Multi-GPU)
- 2-3 NVIDIA GPUs (RTX 3080 class)
- Distributes load: Detection + VLM + LLM on separate cards
- Faster response times, smoother operation

### What I Used
- 3x RTX 3080 10GB
- 64GB RAM
- Logitech C920 webcam
- 32" TV for pirate display

---

## Software Requirements

- **Linux** (tested on Fedora 37, Ubuntu 22.04)
- **Python 3.10+**
- **Godot 4.2+** (for 3D avatar)
- **Ollama** (LLM runtime)
- **NVIDIA drivers + CUDA**

---

## Quick Start

### 1. Setup Python Environment

```bash
cd piratebot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama + Model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download the pirate's brain
ollama pull llama3.2:3b

# Start Ollama server
ollama serve
```

### 3. Install TTS

```bash
# Best option (fast)
pip install kokoro-onnx

# Or fallback (works but lower quality)
pip install pyttsx3
```

### 4. Get a Pirate Model

Download from [RenderHub (free)](https://www.renderhub.com/maksim-bugrimov/pirate) or find another pirate model.

**Important**: The model needs "blend shapes" for lip-sync. See [3D Model Setup](#3d-model-setup) below.

### 5. Run It!

```bash
# Terminal 1: Start Ollama (if not running)
ollama serve

# Terminal 2: Start Godot (from godot_project folder)
godot

# Terminal 3: Start PirateBot
source venv/bin/activate
python main.py
```

---

## Configuration

All settings are in `config.yaml`:

```yaml
# Which webcam to use
webcam:
  device_id: 0        # Usually 0, check with: ls /dev/video*

# GPU assignments (for multi-GPU setups)
gpu:
  display: 0          # YOLO + TTS
  vlm: 1              # Moondream2
  llm: 2              # Ollama

# Pirate voice (try different ones!)
tts:
  voice: "am_adam"    # Male voices: am_adam, am_michael, bm_george
                      # Female voices: af_bella, af_nicole, af_sky

# How long to wait before commenting on the same person again
detector:
  cooldown_seconds: 30
```

---

## 3D Model Setup

The pirate needs a 3D model with **blend shapes** (also called "shape keys") for lip-sync.

### Required Blend Shapes

| Name | Purpose |
|------|---------|
| `viseme_aa` | "Ah" sound (jaw open) |
| `viseme_E` | "Ee" sound |
| `viseme_I` | "Ih" sound |
| `viseme_O` | "Oh" sound (rounded lips) |
| `viseme_U` | "Oo" sound |
| `viseme_rest` | Mouth closed |

### Adding Blend Shapes in Blender

1. Open your pirate model in Blender
2. Select the face/head mesh
3. Go to **Object Data Properties** → **Shape Keys**
4. Click **+** to add each viseme
5. Sculpt the mouth shape for each
6. Export as GLTF (.glb)

### Import into Godot

1. Copy `.glb` file to `godot_project/assets/pirate_model/`
2. Open Godot and import the project
3. Replace the placeholder in `scenes/pirate.tscn`
4. Connect the mesh to the PirateController script

---

## Customization

### Change the Pirate's Personality

Edit `config.yaml` under `prompts.system`:

```yaml
prompts:
  system: |
    You are Captain Barnacle Bill, a friendly pirate...
    # Add your own personality here!
```

### Add Idle Chatter

When no one is around, the pirate can say random things:

```yaml
idle:
  enabled: true
  phrases:
    - "Arrr, where be all the little scallywags tonight?"
    - "Any brave souls out there lookin' fer candy?"
  min_interval_seconds: 45
  max_interval_seconds: 120
```

### Change the Voice

Available Kokoro voices:
- **American Male**: `am_adam`, `am_michael`
- **American Female**: `af_bella`, `af_nicole`, `af_sarah`, `af_sky`
- **British Male**: `bm_george`, `bm_lewis`
- **British Female**: `bf_emma`, `bf_isabella`

---

## Testing Individual Parts

Each component can be tested separately:

```bash
# Test if Ollama is working
python -m services.ollama_llm

# Test webcam + person detection (shows live view)
python -m services.yolo_detector

# Test text-to-speech
python -m services.kokoro_tts

# Test vision model with an image
python -m services.moondream_vlm /path/to/costume-photo.jpg
```

---

## Troubleshooting

### "Ollama not available"
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start it
ollama serve

# Make sure model is downloaded
ollama list
ollama pull llama3.2:3b
```

### "CUDA out of memory"
- Close other GPU applications
- Use smaller models in config
- Distribute across GPUs (edit `gpu.*` in config.yaml)

### "Webcam not found"
```bash
# Find your webcam
ls /dev/video*

# Test it
ffplay /dev/video0

# Update config.yaml
webcam:
  device_id: 0  # Change to match your device
```

### Lip-sync not working
- Install Rhubarb: https://github.com/DanielSWolf/rhubarb-lip-sync/releases
- Add to PATH: `sudo mv rhubarb /usr/local/bin/`
- Without Rhubarb, falls back to simple mouth movement

### Pirate not responding
- Check WebSocket connection (Godot must be running first)
- Look at console output for errors
- Test components individually (see above)

---

## Project Structure

```
piratebot/
├── main.py              # Main program - ties everything together
├── config.yaml          # All your settings
├── README.md            # You are here!
├── CLAUDE.md            # Info for AI assistants
├── RESEARCH.md          # Technology alternatives explored
│
├── interfaces/          # Defines what each component must do
│   ├── detector.py      # Person detection interface
│   ├── vision_model.py  # Costume description interface
│   ├── language_model.py# Text generation interface
│   ├── tts_engine.py    # Speech synthesis interface
│   └── avatar_controller.py # Avatar control interface
│
├── services/            # Actual implementations
│   ├── yolo_detector.py # Uses YOLOv8
│   ├── moondream_vlm.py # Uses Moondream2
│   ├── ollama_llm.py    # Uses Ollama
│   ├── kokoro_tts.py    # Uses Kokoro TTS
│   └── godot_avatar.py  # Talks to Godot via WebSocket
│
├── godot_project/       # The 3D avatar
│   ├── project.godot
│   ├── scenes/pirate.tscn
│   └── scripts/*.gd
│
└── prompts/
    └── pirate_persona.txt  # The pirate's personality
```

---

## How It Works (Technical)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Webcam    │────▶│ YOLOv8       │────▶│ Person      │
│             │     │ Detection    │     │ Detected!   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐            │
                    │ Moondream2   │◀───────────┘
                    │ "Kid in      │
                    │  vampire     │
                    │  costume"    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Llama 3.2    │
                    │ (via Ollama) │
                    │ "Arrr! A     │
                    │  vampire!"   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Kokoro TTS   │
                    │ [audio.wav]  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Godot 3D     │
                    │ Avatar +     │
                    │ Lip-sync     │
                    └──────────────┘
```

---

## Credits & Thanks

- [Moondream2](https://github.com/vikhyat/moondream) - Tiny but mighty vision model
- [Ollama](https://ollama.com) - Makes running LLMs easy
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) - Blazing fast speech
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Real-time detection
- [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) - Mouth animation
- [Godot Engine](https://godotengine.org) - Free 3D engine

---

## License

MIT License - Use it for your Halloween decorations, haunted houses, or whatever!

---

## See Also

- `RESEARCH.md` - Deep dive into technology choices and alternatives
- `CLAUDE.md` - Context for AI assistants working on this code
- `config.yaml` - All configuration options

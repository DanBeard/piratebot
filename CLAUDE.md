# CLAUDE.md - AI Agent Context for PirateBot

This file provides context for AI assistants (Claude, etc.) working on this project.

## Project Overview

PirateBot is an interactive Halloween decoration featuring a 3D pirate character that:
1. Detects approaching trick-or-treaters via webcam (YOLOv8)
2. Captures their photo and describes their costume (Moondream2 VLM)
3. Selects a pre-generated pirate line via the cluster parrotts service
4. Plays the cloned-voice audio and displays a 3D avatar with lip-sync (Godot 4)

Vision + LLM run locally on consumer GPUs. Voice cloning + line generation
runs on the shared homelab `parrotts` service.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Orchestrator)                    │
├─────────────────────────────────────────────────────────────────┤
│  Interfaces (Abstract Base Classes)                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │IDetector │ │IVisionMdl│ │ILangModel│ │IAvatar   │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘            │
├───────┼────────────┼────────────┼────────────┼──────────────────┤
│  Services (Implementations)                                      │
│  ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌────────┐│
│  │YoloDet   │ │Moondream │ │OllamaLLM │ │GodotAvtr │ │Parrotts││
│  │(GPU 0)   │ │(GPU 1)   │ │(cluster) │ │(WS)      │ │TTS(HTTP)││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘│
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket (JSON)
                                    ▼
                    ┌───────────────────────────────┐
                    │   Godot 4 (3D Avatar Render)  │
                    │   - WebSocket Server          │
                    │   - Pirate Controller         │
                    │   - Lip-sync Animation        │
                    └───────────────────────────────┘
```

## Key Design Decisions

### 1. Interface-Based Architecture
All services implement abstract base classes in `interfaces/`. This allows:
- Easy swapping of implementations (e.g., replace Moondream with LLaVA)
- Mock implementations for testing
- Clear contracts between components

### 2. Multi-GPU Distribution
- **Local GPU 0**: Display, YOLO detection
- **Local GPU 1**: Moondream2 VLM (~5GB)
- **Cluster (k3s-1 1080 Ti)**: parrotts (Chatterbox voice cloning, ~3GB)
- **Cluster (k3s-8 RTX 3060)**: Ollama / litellm (Qwen3-14B)

Local devices configured via `CUDA_VISIBLE_DEVICES`. Cluster services
reached over the homelab network (`parrotts.default.svc.cluster.local`).

### 3. Python-First with Minimal Godot
- All business logic in Python
- Godot only handles 3D rendering
- Communication via WebSocket (JSON commands)
- Minimal GDScript - just enough to receive commands and animate

### 4. Lazy Loading
All models use lazy loading (`_ensure_loaded()`) to:
- Reduce startup time
- Allow selective initialization
- Enable warmup before first use

## File Structure

```
piratebot/
├── pyproject.toml              # Project config, dependencies (uv/pip)
├── main.py                     # Entry point, orchestration loop
├── config.yaml                 # All configuration in one place
├── interfaces/                 # Abstract base classes
│   ├── detector.py             # IDetector, Detection dataclass
│   ├── vision_model.py         # IVisionModel
│   ├── language_model.py       # ILanguageModel, GenerationConfig/Result
│   └── avatar_controller.py    # IAvatarController, Expression, Animation, Viseme
├── services/                   # Concrete implementations
│   ├── yolo_detector.py        # YOLOv8 with tracking
│   ├── moondream_vlm.py        # Moondream2 VLM
│   ├── ollama_llm.py           # Ollama REST API client
│   ├── parrotts_tts.py         # Cluster parrotts service adapter (only voice path)
│   ├── parrotts_vendor/        # Vendored parrotts HTTP client (~325 LoC)
│   └── godot_avatar.py         # WebSocket client for Godot
├── tools/
│   └── migrate_to_parrotts.py  # Seed parrotts from voice_lines.yaml
├── godot_project/              # Godot 4 project
│   ├── scripts/
│   │   ├── websocket_server.gd # Autoload, receives commands
│   │   └── pirate_controller.gd# Avatar control, lip-sync
│   └── scenes/
│       └── pirate.tscn         # Main scene (has placeholder model)
└── prompts/
    └── pirate_persona.txt      # System prompt for pirate character
```

## Common Tasks

### Adding a New VLM Implementation
1. Create `services/new_vlm.py`
2. Implement `IVisionModel` interface
3. Add to `services/__init__.py`
4. Update `main.py` to use new implementation

### Changing the Pirate Personality
Edit `config.yaml` under `prompts.system` or modify `prompts/pirate_persona.txt`.

### Adding New Expressions
1. Edit `EXPRESSION_SHAPES` in `godot_project/scripts/pirate_controller.gd`
2. Add corresponding blend shapes to the 3D model
3. Add to `Expression` enum in `interfaces/avatar_controller.py`

### Testing Without Hardware
Use `MockAvatarController` from `services/godot_avatar.py` - it logs all commands instead of sending them.

## Configuration Reference

Key `config.yaml` sections:
- `gpu.*`: GPU device assignments
- `webcam.*`: Camera settings
- `detector.*`: YOLO model and thresholds
- `vlm.*`: Moondream model and prompt
- `llm.*`: Ollama URL, model, generation params
- `parrotts.*`: Cluster parrotts TTS service — base_url, character, cache_dir.
  Run `python tools/migrate_to_parrotts.py` once to seed the line library.
- `avatar.*`: Godot WebSocket host/port
- `prompts.*`: System prompt and user template
- `idle.*`: Random idle phrases when no one is around

## Dependencies

Critical external dependencies:
- `ultralytics` - YOLOv8
- `transformers` + `torch` - Moondream2
- `aiohttp` / `httpx` - Ollama client
- `websockets` - Godot communication
- `kokoro-onnx` or `kokoro` - TTS (falls back to `pyttsx3`)
- Ollama (system service) - LLM runtime
- Rhubarb (optional CLI) - Lip-sync viseme generation

## Error Handling

Each service has:
- `warmup()` - Pre-load model, catch cold-start issues
- `cleanup()` - Release GPU memory
- `is_available()` / `is_loaded()` - Check readiness

The main loop catches exceptions per-detection to avoid crashing on single failures.

## Performance Targets

- **Total latency**: <15 seconds (typically 3-7 seconds)
- **Detection**: ~30-50ms
- **VLM**: ~1-2 seconds
- **LLM**: ~1-3 seconds
- **TTS**: ~0.3-1 second

## Known Limitations

1. **3D Model**: The placeholder needs to be replaced with a real pirate model with blend shapes
2. **Rhubarb dependency**: Lip-sync requires Rhubarb CLI installed; falls back to amplitude-based
3. **Single-person focus**: Currently processes one detection at a time
4. **Godot on Linux**: Needs X11/Wayland display; headless mode limited

## Future Improvements (Not Implemented)

- Voice cloning with XTTS-v2 (need 6-second pirate voice sample)
- Multi-person tracking and group comments
- Seasonal prompt variations
- Remote monitoring/statistics dashboard
- Pre-generated responses for common costumes (faster)

## Testing Commands

```bash
# Sync dependencies first
uv sync --extra full   # Full runtime (Linux)
uv sync --extra mac    # Mac voice generation

# Test LLM connection
uv run python -m services.ollama_llm

# Test webcam detection
uv run python -m services.yolo_detector

# Test TTS
uv run python -m services.kokoro_tts

# Test VLM with image
uv run python -m services.moondream_vlm /path/to/image.jpg

# Test avatar connection (needs Godot running)
uv run python -m services.godot_avatar

# Generate voice lines (Mac) - quick test with 2 lines
./quickstart_mac.sh --test

# Generate voice lines (Mac) - specific category
uv run python tools/generate_voice_lines_mac.py --category greetings

# Generate voice lines (Linux)
uv run python tools/generate_voice_lines.py --category greetings

# Expand voice lines with Ollama (dry-run shows what would happen)
uv run python tools/expand_voice_lines.py --dry-run
```

## References

- See `RESEARCH.md` for technology alternatives and trade-offs
- See `README.md` for user-facing setup instructions
- Plan document: `~/.claude/plans/wobbly-meandering-kahn.md`

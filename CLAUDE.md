# CLAUDE.md - AI Agent Context for PirateBot

This file provides context for AI assistants (Claude, etc.) working on this project.

## Project Overview

PirateBot is an interactive Halloween decoration featuring a 3D pirate character that:
1. Detects approaching trick-or-treaters via webcam (YOLOv8)
2. Captures their photo and describes their costume (Moondream2 VLM)
3. Generates personalized pirate-themed banter (Ollama LLM)
4. Speaks the response with a gruff voice (Kokoro TTS)
5. Displays a 3D avatar with lip-sync (Godot 4)

**All processing runs locally** on consumer GPUs - no cloud APIs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Orchestrator)                    │
├─────────────────────────────────────────────────────────────────┤
│  Interfaces (Abstract Base Classes)                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐│
│  │IDetector │ │IVisionMdl│ │ILangModel│ │ITTSEng  │ │IAvatar  ││
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘ └────┬────┘│
├───────┼────────────┼────────────┼────────────┼───────────┼──────┤
│  Services (Implementations)                                      │
│  ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐ ┌────┴────┐ ┌────┴────┐│
│  │YoloDet   │ │Moondream │ │OllamaLLM │ │KokoroTTS│ │GodotAvtr││
│  │(GPU 0)   │ │(GPU 1)   │ │(GPU 2)   │ │(GPU 0)  │ │(WS)     ││
│  └──────────┘ └──────────┘ └──────────┘ └─────────┘ └─────────┘│
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
The system distributes models across GPUs to fit in 10GB VRAM cards:
- **GPU 0**: Display, YOLO detection, TTS (~2GB)
- **GPU 1**: Moondream2 VLM (~5GB)
- **GPU 2**: Ollama LLM (~4GB)

Configure via `CUDA_VISIBLE_DEVICES` environment variable.

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
├── main.py                     # Entry point, orchestration loop
├── config.yaml                 # All configuration in one place
├── interfaces/                 # Abstract base classes
│   ├── detector.py             # IDetector, Detection dataclass
│   ├── vision_model.py         # IVisionModel
│   ├── language_model.py       # ILanguageModel, GenerationConfig/Result
│   ├── tts_engine.py           # ITTSEngine, TTSConfig/Result, Viseme
│   └── avatar_controller.py    # IAvatarController, Expression, Animation
├── services/                   # Concrete implementations
│   ├── yolo_detector.py        # YOLOv8 with tracking
│   ├── moondream_vlm.py        # Moondream2 VLM
│   ├── ollama_llm.py           # Ollama REST API client
│   ├── kokoro_tts.py           # Kokoro + Rhubarb lip-sync
│   └── godot_avatar.py         # WebSocket client for Godot
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
- `tts.*`: Voice selection, output directory
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
# Test LLM connection
python -m services.ollama_llm

# Test webcam detection
python -m services.yolo_detector

# Test TTS
python -m services.kokoro_tts

# Test VLM with image
python -m services.moondream_vlm /path/to/image.jpg

# Test avatar connection (needs Godot running)
python -m services.godot_avatar
```

## References

- See `RESEARCH.md` for technology alternatives and trade-offs
- See `README.md` for user-facing setup instructions
- Plan document: `~/.claude/plans/wobbly-meandering-kahn.md`

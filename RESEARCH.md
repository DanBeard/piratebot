# RESEARCH.md - Technology Research & Alternatives

This document captures all research done while building PirateBot, including the technologies evaluated, their trade-offs, and when you might want to switch to an alternative.

---

## Table of Contents

1. [3D Avatar / Rendering Engine](#1-3d-avatar--rendering-engine)
2. [Vision Language Models (VLM)](#2-vision-language-models-vlm)
3. [Language Models (LLM)](#3-language-models-llm)
4. [Text-to-Speech (TTS)](#4-text-to-speech-tts)
5. [Person Detection](#5-person-detection)
6. [Lip-Sync Technology](#6-lip-sync-technology)
7. [3D Pirate Model Sources](#7-3d-pirate-model-sources)
8. [All-in-One Solutions](#8-all-in-one-solutions)

---

## 1. 3D Avatar / Rendering Engine

### Current Choice: Godot 4

**Why we chose it:**
- Open source (MIT license)
- Native Linux support
- Lightweight compared to Unity/Unreal
- Good scripting (GDScript, C#, or GDExtension)
- Active community

**Limitations:**
- Less mature than Unity for character animation
- Smaller asset ecosystem
- Some lip-sync plugins have version compatibility issues

### Alternatives

#### Unity with ai-iris-avatar
- **Project**: https://github.com/Scthe/ai-iris-avatar
- **Pros**:
  - Battle-tested, complete LLM+TTS+Unity lip-sync system
  - Claims <4s response time
  - Includes Oculus LipSync (high quality)
  - Adobe Mixamo animation integration
- **Cons**:
  - Requires Windows/macOS (Oculus LipSync)
  - Not Linux-native
  - Unity licensing considerations
- **When to switch**: If you need higher quality lip-sync and are willing to use Windows

#### Open-LLM-VTuber (Live2D)
- **Project**: https://github.com/Open-LLM-VTuber/Open-LLM-VTuber
- **Pros**:
  - Complete solution (LLM+TTS+Avatar all included)
  - Linux, Windows, macOS support
  - Offline capable
  - Camera/screen vision support
  - Voice interruption capability
  - Long-term memory with Letta
- **Cons**:
  - 2D avatars (Live2D), not 3D
  - May be overkill for simple use case
- **When to switch**: If you want 2D VTuber style and want everything pre-integrated

#### VU-VRM (Browser-based)
- **Project**: https://github.com/Automattic/VU-VRM
- **Pros**:
  - Cross-platform (runs in browser)
  - Works as OBS browser source
  - Simple mic-based lip-sync
- **Cons**:
  - Limited control
  - Mic-based only (no viseme data)
- **When to switch**: If you want the simplest possible setup and don't need precise lip-sync

#### Live2D Python SDK
- **Project**: https://github.com/EasyLive2D/live2d-v2
- **Pros**:
  - Pure Python (no JavaScript/C++ bindings)
  - Uses PyOpenGL
- **Cons**:
  - Cubism 2.1 only (older format)
  - Performance needs improvement
  - Less documentation
- **When to switch**: If you want all-Python stack with 2D avatars

---

## 2. Vision Language Models (VLM)

### Current Choice: Moondream2

**Why we chose it:**
- Tiny: 1.86B parameters
- Fast: Optimized for edge/on-device inference
- License: Apache 2.0 (commercial OK)
- Good accuracy for its size
- Active development

**Benchmarks:**
- Fits in ~5GB VRAM
- Inference: 1-3 seconds on RTX 3080

**Limitations:**
- Less capable than larger models
- May miss subtle costume details
- Newer model, less battle-tested

### Alternatives

#### SmolVLM
- **Size**: 2B parameters
- **URL**: https://huggingface.co/blog/smolvlm
- **Pros**:
  - State-of-the-art for memory footprint
  - Optimized for edge deployment
  - Token compression (729→81 visual tokens)
- **Cons**:
  - Newer, less community support
- **When to switch**: If Moondream isn't accurate enough and you have similar VRAM

#### LLaVA / LLaVA-NeXT
- **Size**: 7B+ parameters
- **URL**: https://llava-vl.github.io/
- **Pros**:
  - More capable, better descriptions
  - Catches up to Gemini Pro on some benchmarks
  - Well-documented
- **Cons**:
  - Slower (7B vs 1.8B)
  - Needs more VRAM (~14GB for 7B)
- **When to switch**: If costume descriptions aren't detailed enough and you have a beefier GPU

#### FastVLM (Apple)
- **URL**: https://machinelearning.apple.com/research/fast-vision-language-models
- **Pros**:
  - 85x faster than LLaVA-OneVision
  - 5.2x faster than SmolVLM
  - Hybrid architecture for speed
- **Cons**:
  - Newer (CVPR 2025)
  - MLX-focused (Apple Silicon)
  - Less NVIDIA optimization
- **When to switch**: If on Apple Silicon or if official NVIDIA support improves

#### CogVLM / Qwen-VL
- **Size**: 7B-17B parameters
- **Pros**: High accuracy, broad capabilities
- **Cons**: Large, slower
- **When to switch**: If accuracy is paramount and latency isn't critical

### VLM Comparison Table

| Model | Size | Speed | VRAM | Best For |
|-------|------|-------|------|----------|
| Moondream2 | 1.86B | Very Fast | ~5GB | Edge, real-time |
| SmolVLM | 2B | Fast | ~5GB | Edge, balanced |
| LLaVA 7B | 7B | Medium | ~14GB | Accuracy |
| FastVLM | Varies | Fastest | Low | Apple Silicon |

---

## 3. Language Models (LLM)

### Current Choice: Ollama + Llama 3.2 3B

**Why we chose it:**
- Simple setup: `ollama pull llama3.2:3b`
- REST API (easy to integrate)
- 107+ tokens/sec on good GPU
- Q4 quantization for speed
- 3B fits easily in 10GB VRAM

**Limitations:**
- 3B has limited reasoning depth
- May repeat phrases
- Ollama single-instance limit (no concurrency)

### Alternatives

#### Mistral 7B
- **Setup**: `ollama pull mistral`
- **Pros**:
  - Better quality outputs
  - Same 107+ tokens/sec
  - More creative responses
- **Cons**:
  - Needs ~8GB VRAM (Q4)
- **When to switch**: If pirate responses feel repetitive or low quality

#### Phi-3 Mini
- **Size**: 3.8B parameters
- **Pros**:
  - Microsoft-optimized
  - Good balance of speed/quality
- **When to switch**: If Llama 3.2 3B isn't creative enough

#### Larger Models (Llama 3 8B, Mistral 7B)
- **When to switch**: If you have 24GB+ VRAM on the LLM GPU

#### vLLM / TensorRT-LLM
- **Pros**:
  - Higher throughput
  - Better concurrency
  - Production-grade
- **Cons**:
  - More complex setup
- **When to switch**: If serving multiple users simultaneously

### LLM Latency Optimization Tips

1. **Quantization**: Use Q4_0 for 60%+ speed boost
2. **Keep prompts short**: Fewer input tokens = faster
3. **Limit output**: `max_tokens: 100` for short responses
4. **Pre-warm model**: Avoid cold start (10-30s delay)
5. **Dedicated GPU**: Don't share with other models

---

## 4. Text-to-Speech (TTS)

### Current Choice: Kokoro TTS

**Why we chose it:**
- Tiny: 82M parameters
- Blazing fast: 90-210x real-time
- License: Apache 2.0
- 54 voices in 8 languages
- Runs on CPU or GPU

**Benchmarks:**
- RTX 4090: 210x real-time
- RTX 3090 Ti: 90x real-time
- CPU: Still usable (~10x real-time)

**Limitations:**
- No voice cloning (uses preset voices)
- Quality not as good as larger models
- Pirate accent relies on LLM text

### Alternatives

#### Coqui XTTS-v2
- **Size**: 467M parameters
- **URL**: https://huggingface.co/coqui/XTTS-v2
- **Pros**:
  - Voice cloning with 6-second sample
  - 17 languages
  - <200ms streaming latency
  - Cross-language cloning
- **Cons**:
  - Larger model (5x Kokoro)
  - Slightly slower
  - Original repo unmaintained (use fork)
- **When to switch**: If you want to clone a specific pirate voice from a sample

#### Bark
- **Pros**:
  - Very expressive
  - Handles non-speech ("Arrr!", laughs, etc.)
  - Unconstrained voice cloning
- **Cons**:
  - Slower than Kokoro/XTTS
  - Less predictable
- **When to switch**: If you want more expressive vocalizations

#### MeloTTS
- **Pros**: Lightweight, efficient
- **When to switch**: For embedded/low-resource devices

#### pyttsx3 (Fallback)
- **Pros**: Works everywhere, no GPU needed
- **Cons**: Robotic voice quality
- **When to switch**: Never by choice; it's our fallback

### Voice Cloning Setup (XTTS)

If you want a custom pirate voice:

1. Record or find a 6-15 second sample of desired voice
2. Install XTTS: `pip install coqui-tts`
3. Modify `services/kokoro_tts.py` to use:

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="Arrr!",
    speaker_wav="pirate_sample.wav",
    language="en",
    file_path="output.wav"
)
```

---

## 5. Person Detection

### Current Choice: YOLOv8-nano

**Why we chose it:**
- Ultrafast: ~30ms inference
- Pre-trained on COCO (includes "person" class)
- Simple API: `ultralytics` package
- Built-in tracking (persistent IDs)

**Model Options:**
| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| yolov8n | Fastest | Good | Real-time (our choice) |
| yolov8s | Fast | Better | Balanced |
| yolov8m | Medium | Best | When accuracy matters |

### Alternatives

#### MediaPipe
- **Pros**: Google-backed, face/pose estimation
- **Cons**: Overkill for just person detection
- **When to switch**: If you need pose estimation or facial landmarks

#### OpenCV HOG
- **Pros**: No GPU needed, simple
- **Cons**: Much slower, less accurate
- **When to switch**: CPU-only environments

---

## 6. Lip-Sync Technology

### Current Choice: Rhubarb Lip Sync + Godot Plugin

**Why we chose it:**
- Generates timing data from audio
- Outputs standard viseme codes (A, B, C, D, E, F, G, H, X)
- Works with any voice
- CLI tool (easy to integrate)

**Project**: https://github.com/DanielSWolf/rhubarb-lip-sync

**Limitations:**
- Not real-time (processes file after generation)
- Requires audio file (can't stream)
- Adds ~0.5-2s to pipeline

### Alternatives

#### Oculus LipSync
- **Pros**:
  - Real-time audio-driven
  - High quality
- **Cons**:
  - Windows/macOS only
  - Requires Unity/Unreal
- **When to switch**: If using Unity on Windows

#### Amplitude-Based (Fallback)
- **How it works**: Read audio dB, open mouth proportionally
- **Pros**: Simple, real-time, no dependencies
- **Cons**: Less accurate, not realistic
- **When to switch**: If Rhubarb is too slow or unavailable

#### Real-Time Viseme Generation
- Neural network approaches exist but are heavier
- **When to switch**: When real-time streaming TTS + lip-sync matures

---

## 7. 3D Pirate Model Sources

### Recommended: RenderHub Free Pirate Captain
- **URL**: https://www.renderhub.com/maksim-bugrimov/pirate
- **Format**: FBX/GLTF
- **Features**:
  - Humanoid rig (Unity/UE4 compatible)
  - PBR textures (4K)
  - 132K polygons
- **Limitation**: No blend shapes (need to add in Blender)

### Other Sources

| Source | Count | Notes |
|--------|-------|-------|
| TurboSquid | 80+ free | Mixed quality |
| CGTrader | 5,800+ | Many paid options |
| RigModels | Several | Free, various formats |
| Sketchfab | Many | Check licenses |

### Making Your Own

#### VRoid Studio (Free)
- Create custom 3D character
- Export as VRM
- Convert to GLTF for Godot
- Add pirate accessories in Blender

#### Ready Player Me
- Create avatar from photo
- Can add custom clothing
- Limited pirate options

### Blend Shape Requirements

For lip-sync, your model needs these blend shapes:

| Name | Purpose | Rhubarb Code |
|------|---------|--------------|
| viseme_aa | "Ah" jaw open | A, D |
| viseme_E | "Ee" sound | C, E |
| viseme_I | "Ih" sound | (variant) |
| viseme_O | "Oh" rounded | G |
| viseme_U | "Oo" sound | F |
| viseme_rest | Closed/neutral | B, H, X |

---

## 8. All-in-One Solutions

If building from scratch seems daunting, consider these integrated solutions:

### Open-LLM-VTuber
- **URL**: https://github.com/Open-LLM-VTuber/Open-LLM-VTuber
- **What it includes**:
  - LLM support (Ollama, OpenAI, Claude, etc.)
  - ASR (Whisper, FunASR, etc.)
  - TTS (Kokoro, Coqui, Bark, etc.)
  - Live2D avatar
  - Camera/screen vision
  - Long-term memory
- **Use case**: Add person detection layer on top

### TalkMateAI
- **URL**: https://github.com/kiranbaby14/TalkMateAI
- **What it includes**:
  - 3D avatar with lip-sync (TalkingHead)
  - Camera integration
  - Streaming responses
  - Native timing sync
- **Use case**: If you want 3D but less custom control

### ai-iris-avatar
- **URL**: https://github.com/Scthe/ai-iris-avatar
- **What it includes**:
  - Unity 3D avatar
  - Ollama LLM
  - TTS with DeepSpeed
  - Oculus LipSync
  - <4s response time
- **Use case**: Windows users who want fast setup

---

## Performance Summary

### Target Latency: <15 seconds (achieved: 3-7 seconds)

| Component | Time | Technology |
|-----------|------|------------|
| Person Detection | 30-50ms | YOLOv8-nano |
| Photo Capture | 100ms | OpenCV |
| VLM (costume desc) | 1-2s | Moondream2 |
| LLM (pirate banter) | 1-3s | Llama 3.2 3B |
| TTS | 0.3-1s | Kokoro |
| Lip-sync generation | 0.5-1s | Rhubarb |
| **Total** | **3-7s** | |

### VRAM Usage (Multi-GPU)

| GPU | Usage | Components |
|-----|-------|------------|
| GPU 0 | ~2GB | YOLO + Kokoro + Display |
| GPU 1 | ~5GB | Moondream2 |
| GPU 2 | ~4GB | Ollama (Llama 3.2 3B Q4) |

---

## Decision Tree: When to Change Technologies

```
Is response quality poor?
├─ Costume descriptions bad? → Switch to LLaVA 7B or SmolVLM
├─ Pirate jokes repetitive? → Switch to Mistral 7B or larger Llama
└─ Voice sounds robotic? → Switch to XTTS with voice cloning

Is it too slow?
├─ VLM taking >3s? → Ensure GPU assignment, try FastVLM
├─ LLM taking >5s? → Use smaller model, enable Q4
├─ TTS taking >2s? → Ensure Kokoro is on GPU
└─ Overall >15s? → Profile each stage, optimize bottleneck

Do you need different features?
├─ Want 2D avatar? → Open-LLM-VTuber
├─ Want voice cloning? → XTTS-v2
├─ Want real-time lip-sync? → Oculus LipSync (Windows)
└─ Want all-in-one? → ai-iris-avatar or TalkMateAI
```

---

## Research Sources

### Official Documentation
- [Ollama](https://ollama.com)
- [Moondream2](https://huggingface.co/vikhyatk/moondream2)
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M)
- [YOLOv8](https://docs.ultralytics.com/)
- [Rhubarb](https://github.com/DanielSWolf/rhubarb-lip-sync)
- [Godot](https://docs.godotengine.org/)

### Community Projects
- [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)
- [ai-iris-avatar](https://github.com/Scthe/ai-iris-avatar)
- [Godot Baked Lipsync](https://github.com/fbcosentino/godot-baked-lipsync)
- [TalkMateAI](https://github.com/kiranbaby14/TalkMateAI)

### Research Papers & Articles
- [VLMs in 2025](https://huggingface.co/blog/vlms-2025)
- [FastVLM (Apple)](https://machinelearning.apple.com/research/fast-vision-language-models)
- [SmolVLM](https://huggingface.co/blog/smolvlm)
- [Ollama Performance](https://markaicode.com/ollama-inference-speed-optimization/)

### Model Marketplaces
- [RenderHub](https://www.renderhub.com)
- [TurboSquid](https://www.turbosquid.com)
- [CGTrader](https://www.cgtrader.com)
- [Sketchfab](https://sketchfab.com)

---

*Last updated: December 2024*
*Research conducted for PirateBot Halloween decoration project*

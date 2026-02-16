# MOSS-TTS on Windows (RTX 3080) - Quickstart Guide

Generate high-quality pirate voice lines using MOSS-TTS on your Windows RTX 3080 box, then sync the audio back to the Mac for PirateBot.

## Prerequisites

- **GPU**: NVIDIA RTX 3080 (10 GB VRAM) - or any Ampere/Ada GPU
- **OS**: Windows 10/11
- **Python**: 3.12 (via conda or standalone installer)
- **Git**: [Git for Windows](https://git-scm.com/download/win)
- **Conda**: [Miniconda](https://docs.anaconda.com/miniconda/) (recommended) or Anaconda

## Step 1: Install Miniconda (if you don't have it)

Download and run the Miniconda installer from https://docs.anaconda.com/miniconda/

Open **Anaconda Prompt** (or any terminal with conda on PATH) for all remaining steps.

## Step 2: Clone PirateBot

```cmd
cd %USERPROFILE%\Projects
git clone https://github.com/YOUR_USER/piratebot.git
cd piratebot
```

## Step 3: Create a Conda Environment for MOSS-TTS

```cmd
conda create -n moss-tts python=3.12 -y
conda activate moss-tts
```

## Step 4: Install MOSS-TTS

```cmd
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .
```

This installs PyTorch 2.9.1+cu128, transformers 5.0.0, and all dependencies. The CUDA wheels are pre-built for Windows.

### Optional: FlashAttention 2

FlashAttention 2 speeds up inference on Ampere+ GPUs (RTX 3080 = sm_86, supported). On Windows this can be tricky to compile. Try:

```cmd
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[flash-attn]"
```

If it fails (common on Windows due to C++ build tools), skip it. The script auto-falls back to PyTorch SDPA which works well.

## Step 5: Install PirateBot's Dependencies

Still in the `piratebot` directory (go back up from MOSS-TTS):

```cmd
cd ..
pip install pyyaml chromadb sentence-transformers soundfile
```

These are the only piratebot dependencies needed for voice generation (not the full runtime).

## Step 6: Verify CUDA Works

```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"
```

Expected output:
```
CUDA: True, GPU: NVIDIA GeForce RTX 3080, VRAM: 10.0 GB
```

## Step 7: Quick Test (2 Lines)

```cmd
python tools\generate_voice_lines_moss.py --test
```

This generates 2 voice lines using the 1.7B model (~3.4 GB VRAM). First run downloads the model (~3.5 GB) which takes a few minutes.

You should see output like:
```
Loading MOSS-TTS model: OpenMOSS-Team/MOSS-TTS-Local
  Device: cuda
  Attention: sdpa
  ...
  Model loaded successfully!
Processing 2 voice lines with MOSS-TTS (1.7B)...
  [1/2] (50%) ETA: 5s | greet_001
  [2/2] (100%) ETA: 0s | greet_002
Generation complete: 2 generated, 0 skipped, 0 failed
```

Generated files land in `godot_project\assets\audio\greetings\`.

## Step 8: Choose Your Model

### MOSS-TTS-Local (1.7B) - Default, Recommended Start

```cmd
python tools\generate_voice_lines_moss.py --model 1.7B
```

- ~3.4 GB VRAM (bf16)
- Good quality, fast
- Supports voice cloning if you have `data\pirate_voice_sample.wav`

### MOSS-TTS (8B) - Best Quality

```cmd
python tools\generate_voice_lines_moss.py --model 8B
```

- ~16 GB at bf16 (won't fit in 10 GB)
- **You'll need quantization** - if MOSS-TTS supports automatic quantization, this will work; otherwise stick with 1.7B
- SOTA quality, rivals closed-source TTS

### MOSS-VoiceGenerator (1.7B) - No Reference Audio Needed

```cmd
python tools\generate_voice_lines_moss.py --model voicegen
```

- ~3.4 GB VRAM
- Creates a pirate voice purely from a text description
- Great for testing without any voice sample file
- Uses a built-in pirate voice prompt (gruff British, Geoffrey Rush style)

## Step 9: Voice Cloning (Optional)

If you have a pirate voice sample (6-10 seconds of gruff pirate speech), place it at:

```
data\pirate_voice_sample.wav
```

The `1.7B` and `8B` models will automatically use it for voice cloning. Without it, they generate with the model's default voice.

Alternatively, use `--model voicegen` which doesn't need reference audio at all - it designs the voice from a text description.

## Step 10: Generate All Lines

```cmd
:: Generate everything
python tools\generate_voice_lines_moss.py

:: Or just one category
python tools\generate_voice_lines_moss.py --category greetings

:: Force regenerate (overwrite existing)
python tools\generate_voice_lines_moss.py --force
```

Progress is saved to `data\.voice_gen_moss_progress.json`. If you interrupt with Ctrl+C, it resumes where it left off next time.

## Step 11: Verify

```cmd
python tools\generate_voice_lines_moss.py --verify
```

Expected:
```
Verification Results:
  Total lines: 50
  Valid: 50
  Missing audio: 0
  Missing visemes: 0
```

## Step 12: Sync Audio Back to Mac

From the Windows box, use any of these methods to get the generated audio to the Mac:

### Option A: Git (if you push from Windows)

```cmd
git add godot_project\assets\audio\
git commit -m "Add MOSS-TTS generated voice lines"
git push
```

Then on the Mac: `git pull`

### Option B: rsync via WSL or SSH

If you have SSH set up between the machines:

```bash
# From Mac, pulling from Windows (if Windows has SSH server)
rsync -avz windows-box:Projects/piratebot/godot_project/assets/audio/ godot_project/assets/audio/
```

### Option C: Shared folder / USB drive

Just copy the `godot_project\assets\audio\` folder.

## Troubleshooting

### "CUDA not available"

Make sure your NVIDIA drivers are up to date. Check with:
```cmd
nvidia-smi
```

If `nvidia-smi` works but PyTorch doesn't see CUDA, you may have a CPU-only PyTorch. Reinstall:
```cmd
pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchaudio==2.9.1
```

### "Out of memory" with 8B model

The 8B model at bf16 needs ~16 GB VRAM. Options:
- Use `--model 1.7B` instead (recommended for RTX 3080)
- Use `--device cpu` (very slow but works)

### FlashAttention won't install

This is common on Windows. Skip it - the script falls back to PyTorch SDPA automatically. No quality difference, just slightly slower.

### "ModuleNotFoundError: No module named 'services'"

Make sure you're running from the piratebot root directory:
```cmd
cd %USERPROFILE%\Projects\piratebot
python tools\generate_voice_lines_moss.py --test
```

### Ctrl+C doesn't stop cleanly

Press Ctrl+C once and wait a few seconds. It finishes the current line then stops. Progress is saved so you can resume.

### Model download is slow

First run downloads ~3.5 GB (1.7B) or ~16 GB (8B) from HuggingFace. If it's slow, you can set a mirror:
```cmd
set HF_ENDPOINT=https://hf-mirror.com
python tools\generate_voice_lines_moss.py --test
```

## A/B Comparison with Qwen3-TTS

To compare MOSS-TTS output with the existing Qwen3-TTS lines:

1. Generate a few lines with MOSS-TTS on Windows:
   ```cmd
   python tools\generate_voice_lines_moss.py --category greetings --force
   ```

2. Copy the WAVs to a comparison folder on Mac:
   ```bash
   mkdir -p data/comparison/moss
   cp godot_project/assets/audio/greetings/*.wav data/comparison/moss/
   ```

3. Compare with existing Qwen3-TTS output in the same audio directory.

Listen for: naturalness, gruffness, expressiveness, consistency across lines.

## File Reference

| File | Purpose |
|------|---------|
| `tools\generate_voice_lines_moss.py` | MOSS-TTS generation script |
| `data\voice_lines.yaml` | Voice line definitions (input) |
| `data\pirate_voice_sample.wav` | Reference audio for voice cloning (optional) |
| `data\.voice_gen_moss_progress.json` | Resume progress (auto-created) |
| `godot_project\assets\audio\**\*.wav` | Generated audio (output) |
| `godot_project\assets\audio\**\*.json` | Viseme timing data (output) |

#!/bin/bash
#
# PirateBot Mac Quickstart
# ========================
# One-command setup and voice line generation for Apple Silicon Macs.
#
# Usage:
#   ./quickstart_mac.sh              # Full setup and generation
#   ./quickstart_mac.sh --force      # Regenerate all audio
#   ./quickstart_mac.sh --skip-expand  # Skip text expansion
#   ./quickstart_mac.sh --category greetings  # Generate specific category
#
# Requirements:
#   - macOS with Apple Silicon (M1/M2/M3)
#   - Python 3.10+
#   - ~8GB free RAM for 1.7B model

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default options
SKIP_EXPAND=false
FORCE=false
CATEGORY=""
MODEL="1.7B"
VARIATIONS=2
TEST_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-expand)
            SKIP_EXPAND=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --variations)
            VARIATIONS="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=true
            SKIP_EXPAND=true
            shift
            ;;
        --help|-h)
            echo "PirateBot Mac Quickstart"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-expand      Skip voice line expansion (use existing 57 lines)"
            echo "  --force            Regenerate all audio files"
            echo "  --category NAME    Only generate specific category"
            echo "  --model SIZE       Model size: 1.7B (default) or 0.6B"
            echo "  --variations N     Variations per line for expansion (default: 2)"
            echo "  --test             Quick test: generate only 2 lines to verify setup"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${BLUE}PirateBot Mac Quickstart${NC}"
echo "========================"
echo ""

# Step 1: Check environment
echo -e "${BLUE}[1/7] Checking environment...${NC}"

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}Error: This script is for macOS only${NC}"
    exit 1
fi
OS_VERSION=$(sw_vers -productVersion)
echo -e "  ${GREEN}✓${NC} macOS $OS_VERSION"

# Check Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: Not running on Apple Silicon (detected: $ARCH)${NC}"
    echo -e "  MPS backend may not be available"
else
    # Try to get chip name
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    echo -e "  ${GREEN}✓${NC} $CHIP"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    echo "  Install with: brew install python@3.11"
    exit 1
fi
PY_VERSION=$(python3 --version)
echo -e "  ${GREEN}✓${NC} $PY_VERSION"

# Step 2: Virtual environment
echo ""
echo -e "${BLUE}[2/7] Setting up virtual environment...${NC}"

VENV_PATH=".venv-mac"

if [[ ! -d "$VENV_PATH" ]]; then
    echo "  Creating $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    echo -e "  ${GREEN}✓${NC} Created $VENV_PATH"
else
    echo -e "  ${GREEN}✓${NC} $VENV_PATH exists"
fi

# Activate venv
source "$VENV_PATH/bin/activate"
echo -e "  ${GREEN}✓${NC} Activated virtual environment"

# Install/upgrade pip
pip install --quiet --upgrade pip

# Install dependencies
echo "  Installing dependencies..."
pip install --quiet -r requirements-mac.txt

echo -e "  ${GREEN}✓${NC} Dependencies installed"

# Step 3: Check optional tools
echo ""
echo -e "${BLUE}[3/7] Checking optional tools...${NC}"

# Check Rhubarb
if command -v rhubarb &> /dev/null; then
    RHUBARB_VERSION=$(rhubarb --version 2>&1 | head -1)
    echo -e "  ${GREEN}✓${NC} Rhubarb: $RHUBARB_VERSION"
else
    echo -e "  ${YELLOW}⚠${NC} Rhubarb not found (fallback visemes will be used)"
    echo "    Install with: brew install rhubarb-lip-sync"
fi

# Check Ollama
OLLAMA_RUNNING=false
if command -v ollama &> /dev/null; then
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        OLLAMA_RUNNING=true
        echo -e "  ${GREEN}✓${NC} Ollama running"
    else
        echo -e "  ${YELLOW}⚠${NC} Ollama installed but not running"
        echo "    Start with: ollama serve"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Ollama not installed (text expansion unavailable)"
    echo "    Install with: brew install ollama"
fi

# Step 4: Check voice sample
echo ""
echo -e "${BLUE}[4/7] Checking voice sample...${NC}"

VOICE_SAMPLE="data/pirate_voice_sample.wav"
if [[ -f "$VOICE_SAMPLE" ]]; then
    echo -e "  ${GREEN}✓${NC} Voice sample: $VOICE_SAMPLE"
else
    echo -e "  ${YELLOW}⚠${NC} Voice sample not found: $VOICE_SAMPLE"
    echo "    Using default voice. Add a 6-10 second pirate voice sample for cloning."
fi

# Step 5: Voice line expansion (optional)
echo ""
echo -e "${BLUE}[5/7] Voice line expansion...${NC}"

YAML_PATH="data/voice_lines.yaml"
EXPANDED_YAML="data/voice_lines_expanded.yaml"

if [[ "$SKIP_EXPAND" == true ]]; then
    echo "  Skipping expansion (--skip-expand)"
    # Use original YAML
    YAML_PATH="data/voice_lines.yaml"
elif [[ "$OLLAMA_RUNNING" == true ]]; then
    # Check if already expanded
    if [[ -f "$EXPANDED_YAML" ]] && [[ "$FORCE" != true ]]; then
        echo -e "  ${GREEN}✓${NC} Using existing expanded lines: $EXPANDED_YAML"
        YAML_PATH="$EXPANDED_YAML"
    else
        echo "  Expanding voice lines with Ollama (this may take a few minutes)..."

        EXPAND_ARGS="--yaml $YAML_PATH --output $EXPANDED_YAML --variations $VARIATIONS"
        if [[ -n "$CATEGORY" ]]; then
            EXPAND_ARGS="$EXPAND_ARGS --category $CATEGORY"
        fi

        if python tools/expand_voice_lines.py $EXPAND_ARGS; then
            echo -e "  ${GREEN}✓${NC} Voice lines expanded"
            YAML_PATH="$EXPANDED_YAML"
        else
            echo -e "  ${YELLOW}⚠${NC} Expansion failed, using original lines"
            YAML_PATH="data/voice_lines.yaml"
        fi
    fi
else
    echo "  Ollama not available, using original 57 lines"
    YAML_PATH="data/voice_lines.yaml"
fi

# Count lines
LINE_COUNT=$(grep -c "text:" "$YAML_PATH" 2>/dev/null || echo "?")
echo "  Voice lines to process: $LINE_COUNT"

# Step 6: Generate audio
echo ""
echo -e "${BLUE}[6/7] Generating audio...${NC}"

# Build arguments
GEN_ARGS="--yaml $YAML_PATH --model $MODEL"
if [[ "$FORCE" == true ]]; then
    GEN_ARGS="$GEN_ARGS --force --reset-progress"
fi
if [[ -n "$CATEGORY" ]]; then
    GEN_ARGS="$GEN_ARGS --category $CATEGORY"
fi
if [[ "$TEST_MODE" == true ]]; then
    GEN_ARGS="$GEN_ARGS --test"
fi

# Show device info
echo "  Device: MPS (Apple Silicon)"
echo "  Dtype: float32 (required for voice cloning)"
echo "  Model: Qwen/Qwen3-TTS-12Hz-${MODEL}-Base"
echo ""
echo "  Generating audio (press Ctrl+C to pause, resume by running again)..."
echo ""

# Run generator
python tools/generate_voice_lines_mac.py $GEN_ARGS

# Step 7: Summary
echo ""
echo -e "${BLUE}[7/7] Summary${NC}"

# Count generated files
AUDIO_DIR="godot_project/assets/audio"
if [[ -d "$AUDIO_DIR" ]]; then
    AUDIO_COUNT=$(find "$AUDIO_DIR" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
    VISEME_COUNT=$(find "$AUDIO_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
else
    AUDIO_COUNT=0
    VISEME_COUNT=0
fi

echo "  Audio files: $AUDIO_COUNT"
echo "  Viseme files: $VISEME_COUNT"
echo ""
echo "  Audio location: $AUDIO_DIR/"
echo ""

# Check progress file
PROGRESS_FILE="data/.voice_gen_progress.json"
if [[ -f "$PROGRESS_FILE" ]]; then
    COMPLETED=$(python3 -c "import json; print(len(json.load(open('$PROGRESS_FILE')).get('completed_lines', [])))" 2>/dev/null || echo "?")
    FAILED=$(python3 -c "import json; print(len(json.load(open('$PROGRESS_FILE')).get('failed_lines', [])))" 2>/dev/null || echo "?")
    echo "  Progress: $COMPLETED completed, $FAILED failed"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
echo "Next steps:"
echo "  1. Test audio: afplay $AUDIO_DIR/greetings/greet_001.wav"
echo "  2. Add custom voice: copy 6-10s sample to $VOICE_SAMPLE"
echo "  3. Regenerate with voice: ./quickstart_mac.sh --force"
echo "  4. Run PirateBot: python main.py"
echo ""

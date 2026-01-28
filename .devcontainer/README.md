# Semantic CLI Development Containers

Three independent devcontainers for developing the Platform CLIs (audio, video, research) that **reuse the production Dockerfile**.

## Features

- 🚀 **Production parity**: Uses the same Dockerfile (`src/Dockerfile`) as production
- 🔧 **Live code editing**: Source code is mounted, changes reflect immediately
- 🎮 **GPU support**: NVIDIA GPU passthrough for CUDA workloads
- 📦 **Fast package installation**: Uses `uv` for ultra-fast Python package management
- ⚡ **Single-service builds**: Each devcontainer only builds its own service (not all CLIs)

## Directory Structure

```
.devcontainer/
├── audio/
│   ├── devcontainer.json    # Audio CLI devcontainer config
│   └── docker-compose.yml   # Standalone compose (only audio service)
├── video/
│   ├── devcontainer.json    # Video CLI devcontainer config
│   └── docker-compose.yml   # Standalone compose (only video service)
├── research/
│   ├── devcontainer.json    # Research CLI devcontainer config
│   └── docker-compose.yml   # Standalone compose (only research service)
├── reset-venvs.ps1          # Reset dev environment (Windows)
├── reset-venvs.sh           # Reset dev environment (Linux/macOS)
└── README.md

# Production Dockerfile used by all devcontainers:
src/Dockerfile
```

## Quick Start

### Prerequisites

1. Install the "Dev Containers" extension in VS Code
2. Ensure Docker is running with NVIDIA GPU support
3. Host directories are created automatically by the devcontainer

### Opening a Devcontainer

1. Open the Platform folder in VS Code
2. Press `Ctrl+Shift+P` → **"Dev Containers: Reopen in Container"**
3. Select the desired CLI configuration:
   - **Semantics Audio CLI** (worker-audio)
   - **Semantics Video CLI** (worker-video)
   - **Semantics Research CLI** (worker-research)
4. Wait for the initial build (first time only - subsequent starts are fast!)

### Using Docker Compose Directly

```bash
# Build and start a specific CLI (from devcontainer folder)
cd .devcontainer/audio
docker compose up -d

# Or from root, use production compose for all services
docker compose build
docker compose up -d worker-audio    # or worker-video, worker-research
docker compose exec worker-audio bash

# View all services
docker compose ps
```

## Available CLIs

Once inside a container:

```bash
# Audio container
audio-cli -h
audio-cli -i /workspaces/assets/sample.mp4 -o /workspaces/results/test_001_audio

# Video container
video-cli -h
video-cli -i /workspaces/assets/sample.mp4 -o /workspaces/results/test_001_video

# Research container
research-cli -h
```

## Host Data Structure

Workspaces are stored on the host for persistence:

```
.data/
└── platform/
    └── workspaces/   # Input/output data (mounted to /workspaces in container)
        ├── assets/   # Test input files
        └── results/  # CLI output
```

## Managing Dependencies

### Updating requirements

1. Edit the `requirements.txt` in the respective CLI folder (`src/platform/{cli}/`)
2. Rebuild the container to reinstall dependencies:

```bash
# From root folder
docker compose build worker-audio   # or worker-video, worker-research
```

### Refreshing lock files

Regenerate `requirements.lock` with updated versions by rebuilding with the refresh flag:

```bash
# Inside the container
SETUP_CLI_REFRESH_LOCK=1 /usr/local/bin/setup-cli-uv.sh /platform/<cli>/requirements.txt
```

### Clearing cached data

If you need to start fresh:

```powershell
# Windows PowerShell - from Platform directory
.\.devcontainer\reset-venvs.ps1
```

```bash
# Linux/macOS - from Platform directory
./.devcontainer/reset-venvs.sh
```

## Environment Variables

| Variable | Purpose | Used By |
|----------|---------|---------|
| `FORCE_REINSTALL=1` | Force reinstall all dependencies even if venvs exist | All CLIs |
| `SETUP_CLI_REFRESH_LOCK=1` | Delete and regenerate lock files | All CLIs |
| `TF_ENABLE_ONEDNN_OPTS=0` | Disable TensorFlow oneDNN optimizations | All CLIs |
| `TF_DISABLE_XLA=1` | Disable TensorFlow XLA | Video, Research |
| `UV_LINK_MODE=copy` | Fix for Windows→Linux filesystem issues | All CLIs |

## CLI-Specific Notes

### Audio CLI
- Uses NeMo for diarization
- Heavy dependencies: torch, torchaudio, faster-whisper, demucs

### Video CLI  
- Uses CLIP, DeepFace, YOLO for analysis
- Heavy dependencies: torch, ultralytics, deepface, easyocr

### Research CLI
- Uses Crawl4AI for web scraping
- Heavy dependencies: litellm, sentence-transformers

## Troubleshooting

### Container won't start with GPU

Make sure you have:
- NVIDIA Container Toolkit installed
- Docker configured to use the NVIDIA runtime
- Updated GPU drivers

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Dependencies out of sync

Rebuild the image to reinstall dependencies:

```bash
# From root folder
docker compose build worker-audio   # or worker-video, worker-research
```

### Permission issues with mounted volumes (Linux)

```bash
# On the host
sudo chown -R $(id -u):$(id -g) .data/
```

### "Exit code 1" when opening devcontainer

The most common cause is the `initializeCommand` failing on Windows. Ensure PowerShell is available and run manually:

```powershell
New-Item -ItemType Directory -Force -Path '.data/platform/workspaces/assets','.data/platform/workspaces/results'
```

## Architecture

The devcontainer setup uses:

1. **Production Dockerfile (`src/Dockerfile`)** - The same image definition used in production
2. **Standalone compose files** - Each CLI has its own `docker-compose.yml` that:
   - Defines only its specific service (no building other CLIs)
   - References the production Dockerfile
   - Adds dev-specific settings (tty, stdin_open, source mounts)
3. **Separate devcontainer.json files** - Each CLI has its own VS Code configuration

This architecture ensures:
- **Production parity**: Dev containers use the same Dockerfile as production
- **Fast builds**: Only the selected CLI is built (not all services)
- **Live editing**: Source files are mounted for immediate changes
- **Venv preservation**: Individual file mounts avoid overwriting the venv

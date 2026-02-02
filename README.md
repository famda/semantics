# Semantics CLI

A unified CLI toolkit for media intelligence, providing audio processing, video analysis, and web research capabilities.

## Overview

The Semantics CLI consists of three specialized command-line tools:

| CLI | Purpose |
|-----|---------|
| `semantics-audio` | Audio processing: transcription, diarization, noise reduction, emotion recognition |
| `semantics-video` | Video analysis: object detection, scene extraction, OCR, captioning |
| `semantics-research` | Web research: search, crawling, content extraction, structured data |

---

## Install

### Prerequisites

- Docker with NVIDIA GPU support (for GPU acceleration)

### Pull the Worker Images

Pre-built images are available on Docker Hub:

```bash
# Pull all worker images
docker pull famda/semantics:audio-latest
docker pull famda/semantics:video-latest
docker pull famda/semantics:research-latest

# Or pull a specific commit
docker pull famda/semantics:audio-abc1234
```

### Available Tags

| Tag Pattern | Description |
|-------------|-------------|
| `audio-latest`, `video-latest`, `research-latest` | Latest stable build from main branch |
| `audio-<sha>`, `video-<sha>`, `research-<sha>` | Specific commit builds (7-char SHA) |

---

## Quick Start with Docker Compose (Recommended)

The easiest way to get started is with Docker Compose. This keeps containers running and lets you execute commands interactively.

### 1. Create a Docker Compose File

Create a `docker-compose.yml` file in your project directory:

```yaml
x-cuda-support: &cuda-support
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu, utility, compute, video]

x-volumes: &volumes
  volumes:
    - ./.data/semantics:/workspaces

x-environment: &environment
  environment:
    - TF_ENABLE_ONEDNN_OPTS=0
    - TF_DISABLE_XLA=1
    - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

services:
  semantics-audio:
    image: famda/semantics:audio-latest
    tty: true
    stdin_open: true
    <<: [*environment, *volumes, *cuda-support]

  semantics-video:
    image: famda/semantics:video-latest
    tty: true
    stdin_open: true
    <<: [*environment, *volumes, *cuda-support]

  semantics-research:
    image: famda/semantics:research-latest
    tty: true
    stdin_open: true
    <<: [*environment, *volumes, *cuda-support]
```

### 2. Setup Your Workspace

```bash
# Create directories for input files and results
mkdir -p .data/semantics/assets
mkdir -p .data/semantics/results

# Copy your media files to the assets folder
cp your_video.mp4 .data/semantics/assets/
```

### 3. Start the Workers

```bash
# Start all workers in the background
docker compose up -d

# Or start only the worker you need
docker compose up -d semantics-audio
docker compose up -d semantics-video
docker compose up -d semantics-research
```

### 4. Interactive Shell Access

Enter a container shell for debugging or exploration:

```bash
# Enter audio worker
docker compose exec semantics-audio pwsh

# Enter video worker
docker compose exec semantics-video pwsh

# Enter research worker
docker compose exec semantics-research pwsh
```

### 5. Run Commands

From inside the container (after step 4), run these commands directly:

#### Audio: Transcribe and Identify Speakers

```bash
semantics-audio -i /workspaces/assets/your_video.mp4 -o /workspaces/results/audio_test -n -s -t -d
```

#### Video: Extract Scenes and Detect Objects

```bash
semantics-video -i /workspaces/assets/your_video.mp4 -o /workspaces/results/video_test --from-segments -s -eo
```

#### Research: Search and Download Web Content

```bash
semantics-research -o /workspaces/results/research_test -s 'AI agents 2026' --search-limit 5 --download
```

### 6. View Results

Results are saved to `.data/semantics/results/` on your host machine.

### 7. Stop Workers

```bash
# Stop all workers
docker compose down

# Stop a specific worker
docker compose stop semantics-audio
```

---

## Common Workflows

### Interview Transcription (Audio)

Denoise, separate vocals, transcribe, identify speakers, and detect emotions:

```bash
docker compose exec semantics-audio bash -lc "semantics-audio -i /workspaces/assets/interview.mp4 -o /workspaces/results/interview -n -s -t -d -em"
```

### Full Audio Analysis Pipeline

Run all audio modules including classification and scene detection:

```bash
docker compose exec semantics-audio bash -lc "semantics-audio -i /workspaces/assets/video.mp4 -o /workspaces/results/full_audio -e -n -s -v -t -d -ctc -c -em -se"
```

### Video Scene Analysis with Object Tracking

Extract scenes, detect objects (people by default), and save annotations:

```bash
docker compose exec semantics-video bash -lc "semantics-video -i /workspaces/assets/video.mp4 -o /workspaces/results/scenes --from-segments -s -eo --save-annotations"
```

### Web Research Pipeline

Search for a topic, download results, and extract structured content:

```bash
docker compose exec semantics-research bash -lc "semantics-research -o /workspaces/results/research -s 'machine learning trends' --search-limit 10 --download --structured"
```

### Deep Crawl a Website

Crawl a specific URL with depth and page limits:

```bash
docker compose exec semantics-research bash -lc "semantics-research -o /workspaces/results/crawl --download-url 'https://example.com/docs' --download-deep --download-max-depth 3 --download-max-pages 50 --structured"
```

---

## Alternative: Standalone Docker Run

If you prefer not to use Docker Compose, you can run containers directly:

### Audio Processing

```bash
mkdir -p ./assets ./results

docker run --rm --gpus all \
  -v "$(pwd)/assets:/workspaces/assets" \
  -v "$(pwd)/results:/workspaces/results" \
  famda/semantics:audio-latest \
  semantics-audio \
    -i /workspaces/assets/sample.mp4 \
    -o /workspaces/results/my_audio_run \
    -t -d
```

### Video Analysis

```bash
mkdir -p ./assets ./results

docker run --rm --gpus all \
  -v "$(pwd)/assets:/workspaces/assets" \
  -v "$(pwd)/results:/workspaces/results" \
  famda/semantics:video-latest \
  semantics-video \
    -i /workspaces/assets/sample.mp4 \
    -o /workspaces/results/my_video_run \
    --from-segments -s -eo
```

### Web Research

```bash
mkdir -p ./results

docker run --rm \
  -v "$(pwd)/results:/workspaces/results" \
  famda/semantics:research-latest \
  semantics-research \
    -o /workspaces/results/my_research_run \
    -s 'machine learning trends 2025' \
    --download
```

---

## Commands

### semantics-audio

Audio and speech processing toolkit.

| Flag | Description |
|------|-------------|
| `-i, --input PATH` | Input media file **(required)** |
| `-o, --output PATH` | Output folder path **(required)** |
| `-e, --enhance-audio` | Enhance audio quality |
| `-n, --denoise` | Denoise the audio file |
| `-s, --stem` | Enable source separation (extract vocals) |
| `-v, --vad` | Enable Voice Activity Detection |
| `-t, --transcribe` | Transcribe audio to text |
| `-te, --transcribe-experimental` | Ultra-fast transcription with CTC alignment |
| `-d, --diarize` | Enable speaker diarization |
| `-ctc, --ctc-align` | Enable CTC forced alignment (requires `-t` and `-d`) |
| `-c, --classify` | Enable audio classification |
| `-ct, --classify-timeline` | Enable timeline audio classification |
| `-em, --emotion` | Enable emotion recognition (requires `-t` or `-ctc`) |
| `-se, --scene` | Enable scene/chapter detection (requires `-t` or `-ctc`) |
| `--debug` | Enable verbose debug logging |
| `--config PATH` | Path to YAML config file |
| `-h, --help` | Show help message |

**Example:**
```bash
docker compose exec semantics-audio bash -lc "semantics-audio -i /workspaces/assets/video.mp4 -o /workspaces/results/audio_full -n -s -t -d"
```

---

### semantics-video

Video analysis and object detection toolkit.

| Flag | Description |
|------|-------------|
| `-i, --input PATH` | Input video file or YouTube URL **(required)** |
| `-o, --output PATH` | Output folder path **(required)** |
| `--from-frames` | Analyze from extracted video frames |
| `--from-clustering` | Analyze from keyframe/clustering on frames |
| `--from-segments` | Analyze from keyframes/segments **(one of these three is required)** |
| `-t, --tiles` | Enable video tiling |
| `-eo, --extract-objects` | Extract objects from the video |
| `-co, --cluster-objects` | Cluster the extracted objects |
| `-classes, --object-classes TEXT` | Object classes to extract (default: `person`) |
| `--save-annotations` | Persist detection crops and masks to disk |
| `-c, --captions` | Extract captions from the video |
| `-s, --scenes` | Enable scene extraction |
| `-ocr, --extract-text` | Enable text extraction (OCR) |
| `-cl, --classify` | Enable frame classification |
| `--download-resolution INT` | Max video height when downloading from URL |
| `--save-frames` | Save extracted frames to disk |
| `-fps, --frames-per-second INT` | Frames per second to analyze (default: 1) |
| `--debug` | Enable verbose debug logging |
| `--config PATH` | Path to YAML config file |
| `-h, --help` | Show help message |

> **Note:** You must specify one of `--from-frames`, `--from-clustering`, or `--from-segments`.

**Example:**
```bash
docker compose exec semantics-video bash -lc "semantics-video -i /workspaces/assets/video.mp4 -o /workspaces/results/video_full --from-segments -s -eo"
```

---

### semantics-research

Web research and content extraction toolkit.

| Flag | Description |
|------|-------------|
| `-i, --input PATH` | Input file for processing |
| `-o, --output PATH` | Output folder path **(required)** |
| `-s, --search TEXT` | Text query to research |
| `--search-limit INT` | Maximum number of web/video results |
| `--download` | Download/crawl search results (use with `-s`) |
| `--download-url URL` | Specific URL to crawl (alternative to `--download`) |
| `--download-deep` | Enable BFS deep crawling |
| `--download-max-depth INT` | Maximum traversal depth when deep crawling |
| `--download-max-pages INT` | Page budget when deep crawling |
| `--download-include-external` | Allow deep crawl to follow external domains |
| `--download-word-threshold INT` | Minimum word count for page materialization |
| `--structured` | Extract structured content from crawled pages |
| `--debug` | Enable verbose debug logging |
| `--config PATH` | Path to YAML config file |
| `-h, --help` | Show help message |

> **Note:** You must specify one of `-s` (search query), `--download-url`, or `-i` (input file).

**Example:**
```bash
docker compose exec semantics-research bash -lc "semantics-research -o /workspaces/results/research_full -s 'AI agents 2026' --search-limit 5 --download"
```

---

## Configuration

Each CLI supports YAML configuration files for advanced settings:

```bash
# Use a custom config file
docker compose exec semantics-audio bash -lc "semantics-audio -i /workspaces/assets/input.mp4 -o /workspaces/results/output --config /workspaces/assets/my_config.yml -t -d"
```

Default configuration files are located at:
- `configs/audio-config.yml`
- `configs/video-config.yml`
- `configs/research-config.yml`

---

## Output Structure

All CLIs write results to the specified output folder with organized subdirectories and structured data:

```
output_folder/
├── transcripts/        # Audio transcriptions (JSON, SRT, VTT)
├── diarization/        # Speaker diarization results
├── emotions/           # Emotion recognition data
├── scenes/             # Scene/chapter detection
├── objects/            # Detected objects and crops
├── frames/             # Extracted video frames
└── ... 
```

---

## License

MIT

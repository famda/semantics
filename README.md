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

## Quick Start

### Audio Processing

```bash
# Create local directories for input/output
mkdir -p ./assets ./results/my_audio_run

# Transcribe and diarize an audio/video file
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
# Create local directories for input/output
mkdir -p ./assets ./results/my_video_run

# Extract scenes and objects from a video
docker run --rm --gpus all \
  -v "$(pwd)/assets:/workspaces/assets" \
  -v "$(pwd)/results:/workspaces/results" \
  famda/semantics:video-latest \
  semantics-video \
    -i /workspaces/assets/sample.mp4 \
    -o /workspaces/results/my_video_run \
    -s -eo
```

### Web Research

```bash
# Create local output directory
mkdir -p ./results/my_research_run

# Search and crawl web content
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

```
Usage: semantics-audio [OPTIONS]

Options:
  -i, --input PATH              Input media file (required)
  -o, --output PATH             Output folder path (required)
  -e, --enhance-audio           Enhance audio quality
  -n, --denoise                 Denoise the audio file
  -s, --stem                    Enable source separation (extract vocals)
  -v, --vad                     Enable Voice Activity Detection
  -t, --transcribe              Transcribe audio to text
  -te, --transcribe-experimental  Ultra-fast transcription with CTC alignment
  -d, --diarize                 Enable speaker diarization
  -ctc, --ctc-align             Enable CTC forced alignment
  -c, --classify                Enable audio classification
  -ct, --classify-timeline      Enable timeline audio classification
  -em, --emotion                Enable emotion recognition
  -se, --scene                  Enable scene/chapter detection
  -su, --summarize              Summarize transcribed content
  -sed, --sentiment             Analyze sentiment in transcribed content
  --debug                       Enable verbose debug logging
  --config PATH                 Path to YAML config file
  -h, --help                    Show help message
```

**Example: Full audio pipeline**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/assets:/workspaces/assets" \
  -v "$(pwd)/results:/workspaces/results" \
  famda/semantics:audio-latest \
  semantics-audio \
    -i /workspaces/assets/interview.mp4 \
    -o /workspaces/results/interview_analysis \
    -n -s -t -d -c -em --debug
```

---

### semantics-video

Video analysis and object detection toolkit.

```
Usage: semantics-video [OPTIONS]

Options:
  -i, --input PATH              Input video file or YouTube URL (required)
  -o, --output PATH             Output folder path (required)
  -t, --tiles                   Enable video tiling
  -eo, --extract-objects        Extract objects from the video
  -co, --cluster-objects        Cluster the extracted objects
  -classes, --object-classes    Object classes to extract (default: person)
  --save-annotations            Persist detection crops and masks to disk
  -c, --captions                Extract captions from the video
  -s, --scenes                  Enable scene extraction
  -ocr, --extract-text          Enable text extraction (OCR)
  --download-resolution INT     Max video height when downloading from URL
  --from-frames                 Analyze from extracted video frames
  --from-clustering             Analyze from keyframe/clustering on frames
  --from-segments               Analyze from keyframes/segments
  --save-frames                 Save extracted frames to disk
  -fps, --frames-per-second     Frames per second to analyze (default: 1)
  --debug                       Enable verbose debug logging
  --config PATH                 Path to YAML config file
  -h, --help                    Show help message
```

**Example: Extract scenes and detect people**
```bash
docker run --rm --gpus all \
  -v "$(pwd)/assets:/workspaces/assets" \
  -v "$(pwd)/results:/workspaces/results" \
  famda/semantics:video-latest \
  semantics-video \
    -i /workspaces/assets/conference.mp4 \
    -o /workspaces/results/conference_analysis \
    -s -eo -classes person --save-annotations --debug
```

---

### semantics-research

Web research and content extraction toolkit.

```
Usage: semantics-research [OPTIONS]

Options:
  -i, --input PATH              Input file for processing
  -o, --output PATH             Output folder path (required)
  -s, --search TEXT             Text query to research
  --search-limit INT            Maximum number of web/video results
  --download                    Download/crawl search results
  --download-url URL            Specific URL to crawl
  --download-deep               Enable BFS deep crawling
  --download-max-depth INT      Maximum traversal depth when deep crawling
  --download-max-pages INT      Page budget when deep crawling
  --download-include-external   Allow deep crawl to follow external domains
  --download-word-threshold INT Minimum word count for page materialization
  --structured                  Extract structured content from crawled pages
  --debug                       Enable verbose debug logging
  --config PATH                 Path to YAML config file
  -h, --help                    Show help message
```

**Example: Deep crawl a website**
```bash
docker run --rm \
  -v "$(pwd)/results:/workspaces/results" \
  famda/semantics:research-latest \
  semantics-research \
    -o /workspaces/results/website_crawl \
    --download-url 'https://example.com/docs' \
    --download-deep \
    --download-max-depth 3 \
    --download-max-pages 100 \
    --structured --debug
```

---

## Configuration

Each CLI supports YAML configuration files for advanced settings:

```bash
# Use a custom config file
semantics-audio -i input.mp4 -o output/ --config my_config.yml
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

# parakeet

A skill for your OpenClaw agent that uses NVIDIA's Parakeet TDT models via the NeMo toolkit for local speech-to-text.

The default `parakeet-tdt-0.6b-v3` is a 600-million-parameter multilingual ASR model that is extremely accurate, automatically adds punctuation and adjusts capitalization, supports word-level timestamps, and has a whopping **~3380× realtime** inference speed.

Only needs ~2GB VRAM, but **you need a NVIDIA GPU**.

## Features

- **~3380× realtime** on GPU — transcribe 1 hour of audio in ~1 second
- **Automatic punctuation & capitalization** — clean, readable output with no post-processing
- **25 European languages** with automatic language detection
- **Word, segment, and char-level timestamps**
- **SRT/VTT subtitle export** with configurable line length
- **Batch processing** with ETA and per-file progress
- **URL/YouTube input** via yt-dlp
- **Long-form audio** up to 3 hours via chunked inference
- **Streaming output** — prints results as they arrive

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (RTX 3070 or better recommended)
- `ffmpeg` (optional, for format conversion)
- `yt-dlp` (optional, for URL/YouTube input)

## Setup

```bash
bash setup.sh
```

## Usage

See `SKILL.md` for full usage documentation and examples.

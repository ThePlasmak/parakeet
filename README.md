# parakeet

A skill for your OpenClaw agent that uses NVIDIA's Parakeet TDT models via the NeMo toolkit for local speech-to-text.

The default `parakeet-tdt-0.6b-v3` is a 600-million-parameter multilingual speech recognition model that has a whopping **~3380× realtime** inference speed, while being extremely accurate (with a 6.34% average WER).

To achieve its advertised performance, however, **you need a NVIDIA GPU**.

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
- Any modern CUDA-capable GPU (~2GB VRAM minimum; NVIDIA GPU recommended)
- `ffmpeg` (optional, for format conversion)
- `yt-dlp` (optional, for URL/YouTube input)

## Setup

```bash
bash setup.sh
```

## Usage

See `SKILL.md` for full usage documentation and examples.

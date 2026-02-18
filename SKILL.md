---
name: parakeet
description: "Local speech-to-text using NVIDIA Parakeet TDT (NeMo). 600M-param multilingual ASR with automatic punctuation/capitalization, word-level timestamps, and ~3380x realtime speed on GPU. Supports 25 European languages with auto-detection, long-form audio up to 3 hours, streaming output, SRT/VTT subtitles, batch processing, and URL/YouTube input."
version: 1.0.0
author: ThePlasmak
tags:
  [
    "audio",
    "transcription",
    "speech-to-text",
    "parakeet",
    "nemo",
    "nvidia",
    "ml",
    "cuda",
    "gpu",
    "subtitles",
    "multilingual",
    "asr",
    "timestamps",
  ]
platforms: ["linux", "wsl2"]
metadata:
  {
    "openclaw":
      {
        "emoji": "🦜",
        "requires":
          {
            "bins": ["python3"],
            "optionalBins": ["ffmpeg", "yt-dlp"],
          },
      },
  }
---

# 🦜 Parakeet — NVIDIA Speech-to-Text

Local speech-to-text using NVIDIA's Parakeet TDT models via the NeMo toolkit. The default `parakeet-tdt-0.6b-v3` is a 600-million-parameter multilingual ASR model that delivers **state-of-the-art accuracy** with **automatic punctuation and capitalization**, word-level timestamps, and an insane **~3380× realtime** inference speed on GPU.

Only needs **~2GB VRAM** to run — plenty of room on an RTX 3070 (8GB).

## When to Use

Use this skill when you need:

- **Best-accuracy transcription** — Parakeet v3 achieves 6.34% average WER on the Open ASR Leaderboard
- **Automatic punctuation & capitalization** — no post-processing needed, output is clean and readable
- **European language transcription** — 25 languages with automatic detection (no prompting required)
- **Word/segment/char timestamps** — precise timing at every granularity level
- **Long audio files** — up to 24 min with full attention, up to 3 hours with `--long-form` local attention
- **Streaming output** — see segments as they're transcribed with `--streaming`
- **Subtitle generation** — SRT and VTT formats with word-level line splitting
- **Batch processing** — glob patterns, directories, skip-existing support with ETA
- **URL/YouTube transcription** — auto-downloads via yt-dlp
- **Blazing speed** — ~3380× realtime means a 10-minute file transcribes in under 0.2 seconds on GPU

**Trigger phrases:**
"transcribe with parakeet", "nemo transcribe", "nvidia speech to text", "parakeet transcribe",
"transcribe this audio", "convert speech to text", "what did they say", "make a transcript",
"audio to text", "subtitle this video", "transcribe in [language]", "European language transcription",
"best accuracy transcription", "high accuracy speech to text", "transcribe long audio",
"transcribe lecture", "transcribe meeting", "word timestamps"

### 🦜 Parakeet vs 🗣️ faster-whisper — When to Use Which

| Feature | Parakeet | faster-whisper |
|---|---|---|
| **Accuracy** | ✅ Best (6.34% avg WER) | Good (distil: 7.08% WER) |
| **Speed** | ✅ ~3380× realtime | ~20× realtime |
| **Auto punctuation** | ✅ Built-in | ❌ Requires post-processing |
| **Languages** | 25 European | ✅ 99+ worldwide |
| **Diarization** | ❌ Not supported | ✅ pyannote speaker ID |
| **Chapters/search** | ❌ Not supported | ✅ Chapter detection, search |
| **Output formats** | text, JSON, SRT, VTT | ✅ text, JSON, SRT, VTT, ASS, LRC, TTML, CSV, TSV, HTML |
| **Translation** | ❌ Not supported | ✅ Any language → English |
| **VRAM usage** | ~2GB | ~1.5GB (distil) |
| **Long audio** | ✅ Up to 3 hours | Limited by VRAM |
| **Streaming** | ✅ Chunked inference | ✅ Segment streaming |
| **Noise handling** | Good (built-in robustness) | ✅ --denoise, --normalize |
| **Filler removal** | ❌ | ✅ --clean-filler |

**Rule of thumb:**
- **Parakeet** when you want: best accuracy, fastest speed, auto-punctuation, European languages
- **faster-whisper** when you need: diarization, 99+ languages, translation, chapters/search, more output formats, audio preprocessing

**⚠️ Agent guidance — keep invocations minimal:**

_CORE RULE: default command (`./scripts/transcribe audio.wav`) is the fastest path — add flags only when the user explicitly asks for that capability._

**Transcription:**

- Only add `--timestamps` if the user asks for word-level or segment-level timestamps (auto-enabled for srt, vtt, json formats)
- Only add `--format srt/vtt` if the user asks for subtitles/captions in that format
- Only add `--format json` if the user wants structured/programmatic output
- Only add `--long-form` if the audio is longer than 24 minutes
- Only add `--streaming` if the user wants live/progressive output for long files
- Only add `-l/--language CODE` if the user specifies a language (auto-detection is usually fine)
- Only add `--model` if the user wants a specific model variant (v2, 1.1b)
- Only add `--device cpu` if GPU is not available or user requests CPU
- Only add `--batch-size N` if the user reports OOM errors
- Only add `--max-words-per-line` or `--max-chars-per-line` for subtitle readability on long segments

**Batch processing:**

- Only add `--skip-existing` when resuming interrupted batch jobs
- For batch output, always use `-o <dir>` (directory, not file)
- ETA is shown automatically for batch jobs (no flag needed)

**Output format for agent relay:**

- **Text** (default) → safe to show directly to user for short files; summarise for long ones
- **JSON** → useful for programmatic post-processing; not ideal to paste in full to user
- **SRT/VTT** → always write to `-o` file; tell the user the output path, never paste raw subtitle content

**When NOT to use:**

- Need speaker diarization → use faster-whisper with `--diarize`
- Non-European languages (Asian, African, etc.) → use faster-whisper
- Need translation to English → use faster-whisper with `--translate`
- Cloud-only environments without local compute

## Quick Reference

| Task | Command | Notes |
|---|---|---|
| **Basic transcription** | `./scripts/transcribe audio.wav` | Auto-punctuated, GPU-accelerated |
| **Transcribe MP3** | `./scripts/transcribe audio.mp3` | Auto-converts via ffmpeg |
| **SRT subtitles** | `./scripts/transcribe audio.wav --format srt -o subs.srt` | Timestamps auto-enabled |
| **VTT subtitles** | `./scripts/transcribe audio.wav --format vtt -o subs.vtt` | WebVTT format |
| **Word timestamps** | `./scripts/transcribe audio.wav --timestamps --format json -o out.json` | Word/segment/char level |
| **JSON output** | `./scripts/transcribe audio.wav --format json -o result.json` | Full metadata + timestamps |
| **Long audio (>24 min)** | `./scripts/transcribe lecture.wav --long-form` | Local attention, up to ~3 hours |
| **Streaming output** | `./scripts/transcribe audio.wav --streaming` | Print segments as transcribed |
| **Specify language** | `./scripts/transcribe audio.wav -l fr` | Validate language (v3 auto-detects) |
| **YouTube/URL** | `./scripts/transcribe https://youtube.com/watch?v=...` | Auto-downloads via yt-dlp |
| **Batch process** | `./scripts/transcribe *.wav -o ./transcripts/` | Output to directory |
| **Batch with skip** | `./scripts/transcribe *.wav --skip-existing -o ./out/` | Resume interrupted batches |
| **English-only model** | `./scripts/transcribe audio.wav -m nvidia/parakeet-tdt-0.6b-v2` | v2 English-only |
| **Larger model** | `./scripts/transcribe audio.wav -m nvidia/parakeet-tdt-1.1b` | 1.1B param English model |
| **CPU mode** | `./scripts/transcribe audio.wav --device cpu` | If no GPU available |
| **Quiet mode** | `./scripts/transcribe audio.wav -q` | Suppress progress messages |
| **Subtitle word wrapping** | `./scripts/transcribe audio.wav --format srt --max-words-per-line 8 -o subs.srt` | Split long subtitle cues |
| **Char-based wrapping** | `./scripts/transcribe audio.wav --format srt --max-chars-per-line 42 -o subs.srt` | Character limit per line |
| **Show version** | `./scripts/transcribe --version` | Print NeMo version |
| **System check** | `./setup.sh --check` | Verify GPU, Python, NeMo, ffmpeg |
| **Upgrade NeMo** | `./setup.sh --update` | Upgrade without full reinstall |

## Model Selection

### Available Models

| Model | Params | Languages | Speed | Use Case |
|---|---|---|---|---|
| **`nvidia/parakeet-tdt-0.6b-v3`** | 600M | 25 EU languages | ~3380× RT | **Default, best multilingual** |
| `nvidia/parakeet-tdt-0.6b-v2` | 600M | English only | ~3380× RT | English-only, slightly better English WER |
| `nvidia/parakeet-tdt-1.1b` | 1.1B | English only | Slower | Maximum English accuracy |

### Supported Languages (v3)

Bulgarian (bg), Croatian (hr), Czech (cs), Danish (da), Dutch (nl), English (en), Estonian (et), Finnish (fi), French (fr), German (de), Greek (el), Hungarian (hu), Italian (it), Latvian (lv), Lithuanian (lt), Maltese (mt), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk), Slovenian (sl), Spanish (es), Swedish (sv), Russian (ru), Ukrainian (uk)

Language is **auto-detected** — no prompting or configuration needed. The model identifies the language from the audio content automatically.

### Accuracy Benchmarks (v3)

| Benchmark | WER |
|---|---|
| **Open ASR Leaderboard (avg)** | **6.34%** |
| LibriSpeech test-clean | 1.93% |
| LibriSpeech test-other | 3.59% |
| AMI | 11.31% |
| GigaSpeech | 9.59% |
| SPGI Speech | 3.97% |
| TED-LIUM v3 | 2.75% |
| VoxPopuli | 6.14% |

## Setup

### Linux / WSL2

```bash
# Full install (creates venv, installs NeMo + CUDA PyTorch)
./setup.sh

# Verify installation
./setup.sh --check

# Upgrade NeMo toolkit
./setup.sh --update
```

Requirements:
- Python 3.10+
- NVIDIA GPU with CUDA (strongly recommended; CPU works but is much slower)
- ffmpeg (recommended for mp3/m4a/mp4 format conversion; not needed for .wav/.flac)
- Optional: yt-dlp (for URL/YouTube input)

### Platform Support

| Platform | Acceleration | Notes |
|---|---|---|
| **Linux + NVIDIA GPU** | CUDA | ~3380× realtime 🚀 |
| **WSL2 + NVIDIA GPU** | CUDA | ~3380× realtime 🚀 |
| Linux (no GPU) | CPU | Functional but slow |
| macOS | ❌ | NeMo has limited macOS support |

> **VRAM:** The 600M model only needs ~2GB VRAM to load. An RTX 3070 (8GB) has plenty of headroom.

### GPU Support

The setup script auto-detects your GPU and installs PyTorch with CUDA. **Always use GPU if available.**

If setup didn't detect your GPU, manually install PyTorch with CUDA:

```bash
# For CUDA 12.x
uv pip install --python ./venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- **WSL2 users**: Ensure you have the [NVIDIA CUDA drivers for WSL](https://docs.nvidia.com/cuda/wsl-user-guide/) installed on Windows

## Usage

```bash
# Basic transcription (auto-punctuated, auto-capitalized)
./scripts/transcribe audio.wav

# Transcribe an MP3 (auto-converts to WAV via ffmpeg)
./scripts/transcribe recording.mp3

# SRT subtitles
./scripts/transcribe audio.wav --format srt -o subtitles.srt

# WebVTT subtitles
./scripts/transcribe audio.wav --format vtt -o subtitles.vtt

# Full JSON with word/segment/char timestamps
./scripts/transcribe audio.wav --timestamps --format json -o result.json

# Transcribe from YouTube URL
./scripts/transcribe https://youtube.com/watch?v=dQw4w9WgXcQ

# Long lecture (>24 min, up to 3 hours)
./scripts/transcribe lecture.wav --long-form

# Streaming mode (print segments as they're transcribed)
./scripts/transcribe audio.wav --streaming

# Batch process a directory
./scripts/transcribe ./recordings/ -o ./transcripts/

# Batch with glob, skip already-done files
./scripts/transcribe *.wav --skip-existing -o ./transcripts/

# Use English-only v2 model
./scripts/transcribe audio.wav --model nvidia/parakeet-tdt-0.6b-v2

# JSON output with metadata
./scripts/transcribe audio.wav --format json -o result.json

# Specify expected language (validation only; v3 auto-detects)
./scripts/transcribe audio.wav --language fr
```

## Options

```
Input:
  AUDIO                 Audio file(s), directory, glob pattern, or URL
                        Native: .wav, .flac (16kHz mono preferred)
                        Converts via ffmpeg: .mp3, .m4a, .mp4, .mkv, .ogg, .webm, .aac, .wma, .avi, .opus
                        URLs auto-download via yt-dlp (YouTube, direct links, etc.)

Model & Language:
  -m, --model NAME      NeMo ASR model (default: nvidia/parakeet-tdt-0.6b-v3)
                        Also: nvidia/parakeet-tdt-0.6b-v2 (English), nvidia/parakeet-tdt-1.1b (larger)
  -l, --language CODE   Expected language code, e.g. en, es, fr (v3 auto-detects if omitted)
                        Used for validation; does not force the model

Output Format:
  -f, --format FMT      text | json | srt | vtt (default: text)
  --timestamps          Enable word/segment/char timestamps (auto-enabled for srt, vtt, json)
  --max-words-per-line N  For SRT/VTT, split segments into sub-cues of at most N words
  --max-chars-per-line N  For SRT/VTT, split lines so each fits within N characters
                        Takes priority over --max-words-per-line when both are set
  -o, --output PATH     Output file or directory (directory for batch mode)

Long-Form & Streaming:
  --long-form           Enable local attention for audio >24 min (up to ~3 hours)
                        Changes attention model to rel_pos_local_attn with [256,256] context
  --streaming           Print segments as they are transcribed (chunked inference)

Inference:
  --batch-size N        Inference batch size (default: 32; reduce if OOM)

Batch Processing:
  --skip-existing       Skip files whose output already exists (batch mode)

Device:
  --device DEV          auto | cpu | cuda (default: auto)
  -q, --quiet           Suppress progress and status messages

Utility:
  --version             Print version info and exit
```

## Output Formats

### Text (default)

Clean, automatically punctuated and capitalized transcript:

```
Hello, welcome to the meeting. Today we'll discuss the quarterly results
and our plans for next quarter.
```

### JSON (`--format json`)

Full metadata including segments, word/char timestamps, and performance stats:

```json
{
  "file": "audio.wav",
  "text": "Hello, welcome to the meeting.",
  "duration": 600.5,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, welcome to the meeting.",
      "words": [
        {"word": "Hello,", "start": 0.0, "end": 0.4},
        {"word": "welcome", "start": 0.5, "end": 0.9},
        {"word": "to", "start": 0.95, "end": 1.1},
        {"word": "the", "start": 1.15, "end": 1.3},
        {"word": "meeting.", "start": 1.35, "end": 2.0}
      ]
    }
  ],
  "words": [...],
  "chars": [...],
  "stats": {
    "processing_time": 0.18,
    "realtime_factor": 3380.0
  }
}
```

### SRT (`--format srt`)

Standard subtitle format for video players:

```
1
00:00:00,000 --> 00:00:02,500
Hello, welcome to the meeting.

2
00:00:02,800 --> 00:00:05,200
Today we'll discuss the quarterly results.
```

### VTT (`--format vtt`)

WebVTT format for web video players:

```
WEBVTT

1
00:00:00.000 --> 00:00:02.500
Hello, welcome to the meeting.

2
00:00:02.800 --> 00:00:05.200
Today we'll discuss the quarterly results.
```

## Long-Form Audio

Parakeet v3 supports two attention modes:

| Mode | Max Duration | Flag | Notes |
|---|---|---|---|
| **Full attention** (default) | ~24 min | _(none)_ | Best accuracy, requires more VRAM |
| **Local attention** | ~3 hours | `--long-form` | Slightly reduced accuracy, much lower VRAM |

For audio over 24 minutes, use `--long-form`:

```bash
./scripts/transcribe long-lecture.wav --long-form
```

This changes the attention model to `rel_pos_local_attn` with a context window of [256, 256], allowing the model to process very long audio without running out of memory.

## URL & YouTube Input

Pass any URL as input — audio is downloaded automatically via yt-dlp:

```bash
# YouTube video
./scripts/transcribe https://youtube.com/watch?v=dQw4w9WgXcQ

# Direct audio URL
./scripts/transcribe https://example.com/podcast.mp3

# With options
./scripts/transcribe https://youtube.com/watch?v=... -l en --format srt -o subs.srt
```

Requires `yt-dlp` (checks PATH and `~/.local/share/pipx/venvs/yt-dlp/bin/yt-dlp`).

## Batch Processing

Process multiple files at once with glob patterns, directories, or multiple paths:

```bash
# All WAV files in current directory
./scripts/transcribe *.wav -o ./transcripts/

# Specific directory
./scripts/transcribe ./recordings/ -o ./transcripts/

# Resume interrupted batch
./scripts/transcribe *.wav --skip-existing -o ./transcripts/

# SRT subtitles for all files
./scripts/transcribe *.wav --format srt -o ./subtitles/
```

**ETA** is shown automatically for batch jobs after the first file completes.

## Audio Format Support

| Format | Support | Notes |
|---|---|---|
| `.wav` | ✅ Native | Preferred; 16kHz mono |
| `.flac` | ✅ Native | Lossless; works directly |
| `.mp3` | 🔄 Converts | Requires ffmpeg |
| `.m4a` | 🔄 Converts | Requires ffmpeg |
| `.mp4` | 🔄 Converts | Requires ffmpeg |
| `.ogg` | 🔄 Converts | Requires ffmpeg |
| `.webm` | 🔄 Converts | Requires ffmpeg |
| `.aac` | 🔄 Converts | Requires ffmpeg |
| `.wma` | 🔄 Converts | Requires ffmpeg |
| `.opus` | 🔄 Converts | Requires ffmpeg |
| `.mkv` | 🔄 Converts | Requires ffmpeg |

Non-native formats are automatically converted to 16kHz mono WAV using ffmpeg before transcription.

## License

The Parakeet TDT v3 model is released under **CC-BY-4.0** — free for commercial and non-commercial use.

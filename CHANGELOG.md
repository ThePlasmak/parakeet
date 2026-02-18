# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-02-18

### Initial Release

- Local speech-to-text using NVIDIA Parakeet TDT (NeMo) — `parakeet-tdt-0.6b-v3` by default
- ~3380× realtime inference speed on GPU with ~2GB VRAM usage
- Automatic punctuation and capitalization
- 25 European languages with automatic language detection
- Word, segment, and char-level timestamps
- SRT and VTT subtitle export with configurable line wrapping
- Batch processing with ETA and per-file progress reporting
- URL and YouTube input via yt-dlp
- Long-form audio support up to 3 hours via chunked inference
- Streaming output mode

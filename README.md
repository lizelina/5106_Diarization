# CPU-Based Pipelines for Long-form Audio Transcription and Diarization

This repository presents three pipelines for transcribing long-form audio and performing speaker diarization using CPU resources:

**WhisperX**: The original pipeline utilizing batch transcription with Whisper and forced alignment.

**WhisperX_Modified**: An optimized version of WhisperX that incorporates multiprocessing and equal-length audio splitting.

**VAD_WhisperX**: An enhanced pipeline that uses Voice Activity Detection (VAD) for audio splitting, combined with multiprocessing.

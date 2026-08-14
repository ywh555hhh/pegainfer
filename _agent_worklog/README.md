# Agent Worklog Archive

This branch is a durable worklog branch, not a PR-ready branch.

It stores the Higgs-Audio migration overlay, 4090 evidence, profiling outputs,
handoff notes, and pause/resume instructions so future agents can inspect the
work without relying on local machine state.

Primary entry points:

- `higgs-audio-migration/HIGGS_AUDIO_PAUSE_HANDOFF_2026-08-14.md`
- `higgs-audio-migration/RESUME_AFTER_SHUTDOWN_2026-08-14.md`
- `higgs-audio-migration/SHA256SUMS`
- `higgs-audio-migration/archive-20260814-native-codec-vocoder-pause/`
- `higgs-audio-migration/4090-results-native-codec-vocoder-20260814/`

Claim boundary at archive time:

- Supported: `native_codec_vocoder_waveform_bringup`
- Not claimed: strict parity, production serving, SOTA performance, or full
  pure-CUDA generation.

Sensitive SSH passwords and cloud credentials are intentionally not stored here.

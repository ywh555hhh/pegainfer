# Higgs-Audio Resume After Shutdown - 2026-08-14

This workspace is intentionally paused while upstream PR #864 is waiting for community feedback. Do not broaden the feature branch before the PR direction is settled.

## Durable Local State

- Repo: `/Users/yiweihan/Documents/muxi/pegainfer-higgs-one-step`
- Branch: `dev/higgs-audio`
- Head at pause: `946d0e12`
- Main handoff: `/Users/yiweihan/Documents/muxi/higgs-audio-migration/HIGGS_AUDIO_PAUSE_HANDOFF_2026-08-14.md`
- Pause archive: `/Users/yiweihan/Documents/muxi/higgs-audio-migration/archive-20260814-native-codec-vocoder-pause/`
- Latest native WAV evidence: `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results-native-codec-vocoder-20260814/`

## Verified Claim Boundary

Current claim: `native_codec_vocoder_waveform_bringup`.

Supported:

- Native retained continuation produced raw codec rows on RTX 4090.
- Native Rust + CUDA codec/vocoder consumed raw codec rows and wrote a WAV.
- `higgs_native_e2e_smoke` completed with `status=ok`.
- WAV evidence is 24 kHz mono, 40320 samples, 1.68 seconds.

Not supported:

- Strict parity.
- Production serving.
- SOTA or performance claims.
- Full pure-CUDA generation, because part of the audio-head continuation path still uses a Rust CPU helper.

## Before Resuming Development

1. Read `HIGGS_AUDIO_PAUSE_HANDOFF_2026-08-14.md`.
2. Run `cd /Users/yiweihan/Documents/muxi/higgs-audio-migration && shasum -a 256 -c SHA256SUMS`.
3. Run `cd /Users/yiweihan/Documents/muxi/pegainfer-higgs-one-step && git status --short --branch`.
4. Exclude dirty `pegainfer-kernels/third_party/*` submodule entries from any PR.
5. Sync/rebase with upstream main before preparing a clean review branch.
6. Use `$pegainfer-rule-dev` before making architecture, runtime, kernel, or claim changes.

## Next Useful Work

- Rename misleading `CudaSkeleton` wording before PR cleanup.
- Split the overlay into small reviewable commits.
- Decide with maintainers whether Qwen3 diagnostic APIs should stay shared or move local.
- Add waveform CPU/CUDA diff and reference quality metrics before any stronger audio E2E claim.
- Re-run 4090 validation before claiming fresh results.

## Verification Commands

```bash
cd /Users/yiweihan/Documents/muxi/higgs-audio-migration
shasum -a 256 -c SHA256SUMS

cd /Users/yiweihan/Documents/muxi/higgs-audio-migration/archive-20260814-native-codec-vocoder-pause
shasum -a 256 -c SHA256SUMS
```

Sensitive SSH passwords and cloud credentials are intentionally not stored in this archive.

# Higgs-Audio 4090 Migration Runbook

TL;DR: The current no-GPU host is safe to shut down after the durable migration
bundle is generated under `/Users/yiweihan/Documents/muxi/higgs-audio-migration`.
The next 4090 machine only needs a fresh PegaInfer checkout plus this overlay,
Higgs model/golden files, Rust/CUDA/protoc, and optional Nsight tools to resume
the native retained-KV trace validation.

Last touched: 2026-08

## Current Boundary

This branch is a native incremental audio-code-generation validation slice. It
does not claim native wav E2E, native codec/vocoder, production serving, or
strict trace parity.

What is ready locally:

- Higgs-Audio feature-gated model-line detection and fail-closed launch
  preflight.
- Higgs-owned audio-code generation state, delay-pattern handling, codec-input
  artifact shape, and trace JSON schema.
- A retained-KV continuation contract:
  `feedback_embedding -> final_normed_hidden -> audio logits -> sampled row`.
- A narrow Qwen3 diagnostic surface for embedding-fed retained decode hidden.
- Local non-CUDA contract gates, PR evidence renderer, diff classifier, overlay
  packager, and 4090 validation driver.

What still requires a real 4090:

- `pegainfer-server --features higgs-audio` Linux/CUDA build evidence.
- Runtime-Qwen3 retained continuation test.
- Official/HF incremental `past_key_values` reference trace generation.
- Native trace vs reference trace comparison.
- `nsys` end-to-end profile and `ncu` dominant-kernel profile.
- PR evidence rendered from the real 4090 gate summary.

## Durable Local Bundle

Generate the durable transfer bundle from the local repo:

```bash
cd /Users/yiweihan/Documents/muxi/pegainfer-higgs-one-step
mkdir -p /Users/yiweihan/Documents/muxi/higgs-audio-migration

python3 tools/higgs/classify_higgs_native_codegen_diff.py \
  --markdown-out /Users/yiweihan/Documents/muxi/higgs-audio-migration/higgs-native-codegen-diff.md

tools/higgs/prepare_higgs_4090_overlay.sh \
  --out-dir /Users/yiweihan/Documents/muxi/higgs-audio-migration \
  --label "$(git rev-parse --short HEAD)-4090"
```

Expected durable files:

- `SHA256SUMS`
- `higgs-native-codegen-diff.md`
- `pegainfer-higgs-native-codegen-overlay-<label>.tgz`
- `pegainfer-higgs-native-codegen-overlay-files-<label>.txt`
- `pegainfer-higgs-run-4090-<label>.sh`

The overlay intentionally excludes dirty
`pegainfer-kernels/third_party/{DeepGEMM,FlashMLA,flashinfer}` submodules.
Those are treated as local workspace pollution unless a separate kernel
submodule decision is made.

## New 4090 Host Requirements

Target assumptions:

- Linux with an NVIDIA RTX 4090 or compatible Ada GPU.
- CUDA toolchain with `nvcc`.
- Rust nightly or the repo-required Rust toolchain.
- `protoc` / `protobuf-compiler`.
- Python reference environment for golden/reference generation.
- `nsys` and `ncu` when profiling evidence is required.
- Durable paths:
  - repo: `/data/src/pegainfer`
  - results: `/data/results/pegainfer/higgs-audio`
  - Python env: `/data/venvs/ai-infra`

Environment knobs:

```bash
export PEGAINFER_CUDA_SM=89
export PEGAINFER_NVCC_JOBS=8
```

## Transfer And Apply Overlay

On the local machine:

```bash
cd /Users/yiweihan/Documents/muxi/higgs-audio-migration
scp pegainfer-higgs-native-codegen-overlay-<label>.tgz <4090-host>:/tmp/
scp pegainfer-higgs-run-4090-<label>.sh <4090-host>:/tmp/
```

Optional local integrity check before transfer:

```bash
cd /Users/yiweihan/Documents/muxi/higgs-audio-migration
LC_ALL=C shasum -a 256 -c SHA256SUMS
```

On the 4090 host:

```bash
mkdir -p /data/src
cd /data/src
git clone https://github.com/ywh555hhh/pegainfer.git pegainfer
cd /data/src/pegainfer
git fetch origin
git checkout dev/higgs-audio
git pull --ff-only

tar -xzf /tmp/pegainfer-higgs-native-codegen-overlay-<label>.tgz -C /data/src/pegainfer
```

If the branch already exists on the remote checkout, keep it fast-forwarded
before extracting the overlay. The overlay is for dirty/uncommitted validation
work; it should not be used to hide unrelated repo drift.

## Required Inputs

The 4090 run needs two Higgs inputs:

- `MODEL_DIR`: Higgs checkpoint/config directory.
- `GOLDEN`: one-step golden safetensors used by the existing Higgs diagnostic
  gate.

These files are intentionally not committed into the repo. Store them under a
durable data path on the 4090 host, for example:

```bash
export MODEL_DIR=/data/models/higgs-audio
export GOLDEN=/data/golden/higgs-audio/higgs-one-step-golden.safetensors
```

If an official/HF incremental trace already exists, pass it through
`--reference-trace`. Otherwise the validation driver can generate it with
`--generate-reference-trace`. Add `--forced-prefix` when you want the gate to
also run the common-prefix diagnostic that consumes HF sampled rows while still
recording native retained-KV/audio-head logits.

## Validation Command

Preferred one-command path on the 4090 host:

```bash
cd /data/src/pegainfer
export MODEL_DIR=/data/models/higgs-audio
export GOLDEN=/data/golden/higgs-audio/higgs-one-step-golden.safetensors
export RESULT_ROOT=/data/results/pegainfer/higgs-audio
export LABEL="$(git rev-parse --short HEAD)-4090"
export PEGAINFER_CUDA_SM=89
export PEGAINFER_NVCC_JOBS=8

bash /tmp/pegainfer-higgs-run-4090-<label>.sh
```

Equivalent explicit driver command:

```bash
tools/higgs/run_higgs_4090_native_codegen_validation.sh \
  --result-root "$RESULT_ROOT" \
  --label "$LABEL" \
  --model-dir "$MODEL_DIR" \
  --golden "$GOLDEN" \
  --generate-reference-trace \
  --forced-prefix \
  --profile \
  --sm "$PEGAINFER_CUDA_SM" \
  --nvcc-jobs "$PEGAINFER_NVCC_JOBS" \
  --install-protoc
```

The driver finishes by running
`tools/higgs/check_higgs_4090_evidence_bundle.py`. That checker is intentionally
stricter than the local no-GPU contract gate: it rejects `gpu_info=unavailable`,
requires runtime-Qwen3, native continuation, reference trace comparison, and
`nsys`/`ncu` profiling to be `ok`, and verifies the rendered PR evidence file.

## Expected 4090 Outputs

The important outputs live under:

```text
/data/results/pegainfer/higgs-audio/native-codegen/
```

Expected files:

- `higgs-server-check-<label>.txt`
- `higgs-native-codegen-contract-gate-<label>.txt`
- `higgs-native-continuation-trace-<label>.json`
- `higgs-native-continuation-codec-input-<label>.json`
- `higgs-reference-incremental-trace-<label>.json`
- `higgs-reference-incremental-codec-input-<label>.json`
- `higgs-codegen-trace-compare-<label>.txt`
- `higgs-native-forced-prefix-trace-<label>.json` when `--forced-prefix` is set
- `higgs-native-forced-prefix-codec-input-<label>.json` when `--forced-prefix` is set
- `higgs-forced-prefix-trace-compare-<label>.txt` when `--forced-prefix` is set
- `pr-evidence-<label>.md`
- `profiles/higgs-native-codegen-<label>.nsys-rep`
- `profiles/higgs-native-codegen-ncu-<label>.ncu-rep`

If `nsys` or `ncu` is missing, treat that as a profiling-evidence blocker and
record it in the PR limitations. Do not silently replace profiler evidence with
manual timing claims.

## Copy Evidence Back

After the 4090 run:

```bash
mkdir -p /Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results
scp -r <4090-host>:/data/results/pegainfer/higgs-audio/native-codegen \
  /Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/
```

Then update the PR evidence from the copied `pr-evidence-<label>.md` and keep
the claim boundary matched to the measured table.

To verify a copied result bundle locally:

```bash
cd /Users/yiweihan/Documents/muxi/pegainfer-higgs-one-step
python3 tools/higgs/check_higgs_4090_evidence_bundle.py \
  /Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/native-codegen/higgs-native-codegen-contract-gate-<label>.txt \
  --pr-evidence /Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/native-codegen/pr-evidence-<label>.md \
  --expected-label <label> \
  --expected-sm 89 \
  --expected-nvcc-jobs 8
```

## Known Blockers At Handoff

- The currently powered-on old machine has no GPU, so it cannot produce the
  required retained-KV, reference-trace, or profiler evidence.
- The previous old 4090 endpoint responded through `SSHPiper`, but recorded
  credentials no longer authenticated.
- The previous Seetacloud endpoint refused or closed SSH connections.
- The active goal therefore remains blocked on a fresh GPU endpoint, not on
  local repo preparation.

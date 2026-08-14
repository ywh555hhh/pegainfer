# Higgs-Audio Native Codegen Handoff

Date: 2026-08-14

Local repo:

- `/Users/yiweihan/Documents/muxi/pegainfer-higgs-one-step`
- branch: `dev/higgs-audio`
- base commit label used for this bundle: `946d0e12-4090`

Durable bundle:

- `higgs-native-codegen-diff.md`
- `pegainfer-higgs-native-codegen-overlay-946d0e12-4090.tgz`
- `pegainfer-higgs-native-codegen-overlay-files-946d0e12-4090.txt`
- `pegainfer-higgs-run-4090-946d0e12-4090.sh`
- `local-gate/native-codegen/higgs-native-codegen-contract-gate-local-nogpu-946d0e12.txt`
- `local-gate/native-codegen/pr-evidence-local-nogpu-946d0e12.md`

Current state:

- Local Rust formatting/check/test/syntax gates have passed in prior runs.
- A durable local no-GPU contract gate has been saved. Its claim boundary is
  `native_contract_only_no_native_decode`.
- The 4090 driver now ends with
  `tools/higgs/check_higgs_4090_evidence_bundle.py`, which intentionally
  rejects local/no-GPU evidence, missing reference trace comparison, and missing
  `nsys`/`ncu` profiles.
- Current branch has no shared `pegainfer-core` or shared `pegainfer-kernels`
  source semantic changes in the native-codegen slice.
- Dirty `pegainfer-kernels/third_party/{DeepGEMM,FlashMLA,flashinfer}` entries
  are intentionally excluded from the overlay and should not enter the PR.
- The current powered-on machine has no GPU, so real 4090 evidence is still
  pending.

Next 4090 resume command:

```bash
mkdir -p /data/src
cd /data/src
git clone https://github.com/ywh555hhh/pegainfer.git pegainfer
cd /data/src/pegainfer
git checkout dev/higgs-audio
tar -xzf /tmp/pegainfer-higgs-native-codegen-overlay-946d0e12-4090.tgz -C /data/src/pegainfer

export MODEL_DIR=/data/models/higgs-audio
export GOLDEN=/data/golden/higgs-audio/higgs-one-step-golden.safetensors
export RESULT_ROOT=/data/results/pegainfer/higgs-audio
export LABEL=946d0e12-4090
export PEGAINFER_CUDA_SM=89
export PEGAINFER_NVCC_JOBS=8

bash /tmp/pegainfer-higgs-run-4090-946d0e12-4090.sh
```

The runner should finish with:

```text
higgs 4090 evidence bundle: ok ...
```

If it fails with `runtime_qwen3='skipped'`, `trace_compare='skipped'`, or
`profile='skipped'`, the output is not PR-ready evidence yet.

Required before PR claim upgrade:

- Linux/4090 `cargo check --release -p pegainfer-server --features higgs-audio`.
- Runtime-Qwen3 retained continuation test.
- Official/HF incremental reference trace with full logits/top-k evidence.
- Native retained trace comparison against that reference trace.
- `nsys` and `ncu` artifacts or explicit profiler-tool blocker.
- PR evidence regenerated from the real 4090 summary.

Detailed runbook:

- `/Users/yiweihan/Documents/muxi/pegainfer-higgs-one-step/docs/models/higgs-audio/4090-migration-runbook.md`

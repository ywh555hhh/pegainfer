## Higgs-Audio native code-generation evidence

Scope: feature-gated Higgs-Audio bring-up for native incremental audio-code generation diagnostics.
Claim boundary: `native_contract_only_no_native_decode`. This does not claim native wav E2E, native codec/vocoder, production serving, strict trace parity without the 4090 trace-comparison table.
Supported claim: Native retained-continuation execution with forced common-prefix diagnostic evidence; strict semantic parity and profiling are still unresolved.

### Claim decision

- Runtime-Qwen3 retained bridge passed.
- Native retained continuation smoke emitted a retained-KV trace.
- Forced common-prefix replay ran the native retained-KV/audio-head path while consuming HF sampled rows.
- Official/HF free-running trace comparison has not passed.
- 4090 `nsys`/`ncu` profiling evidence has not been captured in this summary.

### Design boundary

- Higgs-Audio owns the multimodal continuation semantics: feedback embedding, audio head, sampling, delay pattern, raw codec rows, and trace rows.
- The Qwen3 touch is intentionally narrow: an embedding-fed retained-KV diagnostic step that returns final normed hidden before the text `lm_head`.
- The current slice does not require shared `pegainfer-core` or shared `pegainfer-kernels` semantic changes; Python remains reference/golden/profiling tooling only.
- Copying the Qwen3 decode body into Higgs would keep the diff model-local, but would duplicate KV, numeric policy, CUDA graph, and future Qwen3 decode fixes.

### Environment

| Field | Value |
| --- | --- |
| `commit` | `946d0e12` |
| `label` | `946d0e12-4090-debug7-forced-gate` |
| `sm` | `89` |
| `nvcc_jobs` | `8` |
| `gpu_info` | `NVIDIA GeForce RTX 4090, 595.71.05, 24564 MiB` |
| `cuda_version` | `nvcc: NVIDIA (R) Cuda compiler driver` |
| `nsys_version` | `NVIDIA Nsight Systems version 2025.3.1.0` |
| `ncu_version` | `NVIDIA (R) Nsight Compute Command Line Profiler` |
| `rustc_version` | `rustc 1.99.0-nightly (ba28ff76f 2026-08-13)` |
| `cargo_version` | `cargo 1.99.0-nightly (eb98b54bc 2026-08-11)` |
| `uname` | `Linux autodl-container-a9c346b8e9-f2562921 5.15.0-97-generic #107-Ubuntu SMP Wed Feb 7 13:26:48 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux` |

### Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Continuation contract | passed | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-continuation-contract-946d0e12-4090-debug7-forced-gate.txt` |
| Higgs lib tests | passed | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-lib-tests-946d0e12-4090-debug7-forced-gate.txt` |
| runtime-qwen3 retained bridge tests | passed | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-runtime-qwen3-tests-946d0e12-4090-debug7-forced-gate.txt` |
| Native retained continuation smoke | passed | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-native-continuation-smoke-946d0e12-4090-debug7-forced-gate.txt` |
| Official/HF trace comparison | failed | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-codegen-trace-compare-946d0e12-4090-debug7-forced-gate.txt` |
| Forced common-prefix smoke | passed | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-native-forced-prefix-smoke-946d0e12-4090-debug7-forced-gate.txt` |
| Forced common-prefix comparison | failed | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-forced-prefix-trace-compare-946d0e12-4090-debug7-forced-gate.txt` |
| nsys/ncu profiling | not run | `nsys=/data/results/pegainfer/higgs-audio/profiles/higgs-native-codegen-946d0e12-4090-debug7-forced-gate.nsys-rep` / `ncu=/data/results/pegainfer/higgs-audio/profiles/higgs-native-codegen-ncu-946d0e12-4090-debug7-forced-gate.ncu-rep` |
| nsys profile | not run | `/data/results/pegainfer/higgs-audio/profiles/higgs-native-codegen-946d0e12-4090-debug7-forced-gate.nsys-rep` |
| nsys stats | not run | `/data/results/pegainfer/higgs-audio/profiles/higgs-native-codegen-946d0e12-4090-debug7-forced-gate-stats_cuda_gpu_kern_sum.csv` |
| ncu profile | not run | `/data/results/pegainfer/higgs-audio/profiles/higgs-native-codegen-ncu-946d0e12-4090-debug7-forced-gate.log` |

### Artifacts

| Artifact | Path |
| --- | --- |
| `artifact_trace_json` | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-artifact-trace-946d0e12-4090-debug7-forced-gate.json` |
| `artifact_codec_json` | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-artifact-codec-input-946d0e12-4090-debug7-forced-gate.json` |
| `native_continuation_trace` | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-native-continuation-trace-946d0e12-4090-debug7-forced-gate.json` |
| `native_continuation_codec` | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-native-continuation-codec-input-946d0e12-4090-debug7-forced-gate.json` |
| `forced_prefix_trace` | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-native-forced-prefix-trace-946d0e12-4090-debug7-forced-gate.json` |
| `forced_prefix_codec` | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-native-forced-prefix-codec-input-946d0e12-4090-debug7-forced-gate.json` |
| `reference_trace` | `/data/results/pegainfer/higgs-audio/native-codegen/higgs-reference-incremental-trace-946d0e12-4090.json` |

### Remaining limitations

- Official/HF incremental reference trace comparison has not passed; sampled-row-only traces are not semantic parity evidence.
- Forced common-prefix replay improves diagnosis but still does not satisfy strict trace parity.
- 4090 nsys/ncu profiling evidence has not been fully captured.

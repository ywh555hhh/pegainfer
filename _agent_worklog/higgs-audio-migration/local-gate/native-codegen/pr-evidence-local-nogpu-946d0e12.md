## Higgs-Audio native code-generation evidence

Scope: feature-gated Higgs-Audio bring-up for native incremental audio-code generation diagnostics.
Claim boundary: `native_contract_only_no_native_decode`. This does not claim native wav E2E, native codec/vocoder, production serving, strict trace parity without the 4090 trace-comparison table.

### Design boundary

- Higgs-Audio owns the multimodal continuation semantics: feedback embedding, audio head, sampling, delay pattern, raw codec rows, and trace rows.
- The Qwen3 touch is intentionally narrow: an embedding-fed retained-KV diagnostic step that returns final normed hidden before the text `lm_head`.
- The current slice does not require shared `pegainfer-core` or shared `pegainfer-kernels` semantic changes; Python remains reference/golden/profiling tooling only.
- Copying the Qwen3 decode body into Higgs would keep the diff model-local, but would duplicate KV, numeric policy, CUDA graph, and future Qwen3 decode fixes.

### Environment

| Field | Value |
| --- | --- |
| `commit` | `946d0e12` |
| `label` | `local-nogpu-946d0e12` |
| `sm` | `89` |
| `nvcc_jobs` | `8` |
| `gpu_info` | `unavailable` |
| `cuda_version` | `unavailable` |
| `nsys_version` | `unavailable` |
| `ncu_version` | `unavailable` |
| `rustc_version` | `rustc 1.99.0-nightly (af3d95584 2026-07-09)` |
| `cargo_version` | `cargo 1.99.0-nightly (59800466c 2026-07-07)` |
| `uname` | `Darwin Uzuki.local 23.5.0 Darwin Kernel Version 23.5.0: Wed May  1 20:16:51 PDT 2024; root:xnu-10063.121.3~5/RELEASE_ARM64_T8103 arm64` |

### Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Continuation contract | passed | `/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/native-codegen/higgs-continuation-contract-local-nogpu-946d0e12.txt` |
| Higgs lib tests | passed | `/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/native-codegen/higgs-lib-tests-local-nogpu-946d0e12.txt` |
| runtime-qwen3 retained bridge tests | not run | `/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/native-codegen/higgs-runtime-qwen3-tests-local-nogpu-946d0e12.txt` |
| Native retained continuation smoke | not run | `/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/native-codegen/higgs-native-continuation-smoke-local-nogpu-946d0e12.txt` |
| Official/HF trace comparison | not run | `/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/native-codegen/higgs-codegen-trace-compare-local-nogpu-946d0e12.txt` |
| nsys/ncu profiling | not run | `nsys=/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/profiles/higgs-native-codegen-local-nogpu-946d0e12.nsys-rep` / `ncu=/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/profiles/higgs-native-codegen-ncu-local-nogpu-946d0e12.ncu-rep` |

### Artifacts

| Artifact | Path |
| --- | --- |
| `artifact_trace_json` | `/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/native-codegen/higgs-artifact-trace-local-nogpu-946d0e12.json` |
| `artifact_codec_json` | `/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/native-codegen/higgs-artifact-codec-input-local-nogpu-946d0e12.json` |
| `native_continuation_trace` | `/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/native-codegen/higgs-native-continuation-trace-local-nogpu-946d0e12.json` |
| `native_continuation_codec` | `/Users/yiweihan/Documents/muxi/higgs-audio-migration/local-gate/native-codegen/higgs-native-continuation-codec-input-local-nogpu-946d0e12.json` |
| `reference_trace` | `n/a` |

### Remaining limitations

- Native retained continuation smoke has not passed on the target Linux/4090 host.
- Official/HF incremental reference trace comparison has not passed; sampled-row-only traces are not semantic parity evidence.
- 4090 nsys/ncu profiling evidence has not been captured.

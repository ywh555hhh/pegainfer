# Higgs-Audio native-codegen diff classification

| Classification | Files |
| --- | --- |
| `doc-tool` | `docs/index.md`<br>`docs/models/higgs-audio/4090-migration-runbook.md`<br>`docs/models/higgs-audio/native-code-generation-plan.md`<br>`docs/models/higgs-audio/native-codegen-pr-notes.md`<br>`docs/models/higgs-audio/next-stage-todo.md`<br>`docs/models/higgs-audio/retained-decode-hidden-boundary.md`<br>`tools/higgs/check_higgs_4090_evidence_bundle.py`<br>`tools/higgs/check_higgs_incremental_trace_reference.py`<br>`tools/higgs/check_higgs_native_codegen_contract_summary.py`<br>`tools/higgs/classify_higgs_native_codegen_diff.py`<br>`tools/higgs/dump_higgs_incremental_trace_reference.py`<br>`tools/higgs/prepare_higgs_4090_overlay.sh`<br>`tools/higgs/render_higgs_native_codegen_pr_evidence.py`<br>`tools/higgs/run_higgs_4090_native_codegen_validation.sh`<br>`tools/higgs/run_higgs_native_codegen_contract_gate.sh` |
| `higgs-local` | `pegainfer-higgs-audio/Cargo.toml`<br>`pegainfer-higgs-audio/src/audio_codegen.rs`<br>`pegainfer-higgs-audio/src/bin/higgs_compare_codegen_trace.rs`<br>`pegainfer-higgs-audio/src/bin/higgs_continuation_contract_report.rs`<br>`pegainfer-higgs-audio/src/bin/higgs_native_continuation_smoke.rs`<br>`pegainfer-higgs-audio/src/bin/higgs_prepare_codegen_artifacts.rs`<br>`pegainfer-higgs-audio/src/bin/higgs_replay_hidden_codegen.rs`<br>`pegainfer-higgs-audio/src/continuation_contract.rs`<br>`pegainfer-higgs-audio/src/decode_trace.rs`<br>`pegainfer-higgs-audio/src/launch_preflight.rs`<br>`pegainfer-higgs-audio/src/lib.rs`<br>`pegainfer-higgs-audio/src/model_line.rs`<br>`pegainfer-higgs-audio/src/runtime_bridge.rs`<br>`pegainfer-higgs-audio/src/trace_compare.rs` |
| `qwen3-diagnostic` | `pegainfer-qwen3/src/batch_decode.rs`<br>`pegainfer-qwen3/src/executor.rs`<br>`pegainfer-qwen3/src/lib.rs` |
| `server-feature-wiring` | `Cargo.lock`<br>`pegainfer-server/Cargo.toml`<br>`pegainfer-server/src/main.rs` |
| `unexpected-third-party-dirty` | `pegainfer-kernels/third_party/DeepGEMM`<br>`pegainfer-kernels/third_party/FlashMLA`<br>`pegainfer-kernels/third_party/flashinfer` |

Review notes:
- `higgs-local` and `doc-tool` are the expected owners for this slice.
- `qwen3-diagnostic` is expected only for the narrow embedding-fed retained-hidden API.
- `server-feature-wiring` is expected only for feature-gated registration and config hints.
- `unexpected-third-party-dirty`, `unexpected-shared`, and `unexpected` require cleanup or explicit maintainer-facing rationale before PR.

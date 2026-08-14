# Higgs-Audio Next-Stage Todo

TL;DR: The next stage is to turn the current Higgs-Audio native code-generation
slice into reviewable evidence: keep runtime pure Rust + CUDA, keep shared/core
changes minimal, run the 4090 retained-KV trace gate, capture nsys/ncu profiles,
and use the resulting artifacts to decide what can honestly be claimed in the
next PR.

Last touched: 2026-08

## Goal

Complete the native incremental audio-code generation milestone up to a
maintainer-reviewable boundary.

This means proving:

- the Higgs model line can launch through the feature-gated server path;
- the retained prompt KV path can run repeated audio-code continuation steps on
  an RTX 4090 without rebuilding the full prompt;
- the native trace can be compared against an official/HF incremental
  `past_key_values` reference trace;
- correctness and profiling artifacts are reproducible from documented commands;
- the PR claim stays below native audio E2E until native codec/vocoder support
  exists.

## Todo List

### 1. Lock The Review Boundary

- [x] Re-read the current diff and classify every touched file as
  `higgs-local`, `qwen3-diagnostic`, `server-feature-wiring`, `doc/tool`, or
  `unexpected`.
- [x] Confirm there are no shared `pegainfer-core` or `pegainfer-kernels`
  semantic changes in this branch.
- [x] Keep the Qwen3 retained-hidden API documented as a narrow diagnostic
  tradeoff, not as a broad shared abstraction.
- [x] Write down the non-claims in the PR evidence: no native wav E2E, no native
  codec/vocoder, no production serving, no strict trace parity unless the 4090
  table proves it.

Acceptance:

- `git diff --stat` can be explained in one table.
- Any non-Higgs file has a specific rationale and a test/evidence owner.

Current diff classification:

| Area | Files | Classification | PR risk / action |
| --- | --- | --- | --- |
| Higgs model crate | `pegainfer-higgs-audio/Cargo.toml`, `src/lib.rs`, `src/runtime_bridge.rs`, new `audio_codegen`, `continuation_contract`, `decode_trace`, `launch_preflight`, `model_line`, `trace_compare`, and Higgs bins | `higgs-local` | Primary owner for this milestone. Keep claims to native incremental audio-code generation until 4090 evidence lands. |
| Qwen3 body access | `pegainfer-qwen3/src/batch_decode.rs`, `executor.rs`, `lib.rs` | `qwen3-diagnostic` | Intentional narrow API for embedding-fed retained decode hidden. Must stay documented as a diagnostic tradeoff, not broad serving API. |
| Server feature wiring | `pegainfer-server/Cargo.toml`, `pegainfer-server/src/main.rs`, matching `Cargo.lock` entries | `server-feature-wiring` | Acceptable only as feature-gated model-line registration and config hint. Launch remains fail-closed until evidence supports more. |
| Docs and validation tools | `docs/models/higgs-audio/*.md`, `tools/higgs/*native_codegen*`, reference trace checker/generator, PR evidence renderer | `doc/tool` | Python is allowed only as reference/golden/profiling tooling; it must not be invoked from shipped Rust runtime paths. |
| Existing decode-to-wav bring-up commit | `higgs_vocode_codes`, `codec_input`, `decode_session`, `delay_pattern`, `tools/higgs/*codec*`, `slow_higgs_fullprefill_e2e.py`, `run_higgs_audio_e2e_gate.sh` | `reference-sidecar / bring-up` | Useful for artifacts, but not native audio E2E evidence. `higgs_vocode_codes` is explicitly gated behind `reference-python-codec` because it launches a Python codec sidecar. Keep wording explicit if these remain in the branch. |
| Third-party submodules | `pegainfer-kernels/third_party/DeepGEMM`, `FlashMLA`, `flashinfer` | `unexpected dirty state` | Do not include in PR. Treat as local workspace pollution unless a separate kernel-submodule decision is made. |
| Shared core/kernels | `pegainfer-core`, `pegainfer-kernels` source files | `none` | No semantic shared core/kernel source change in the current uncommitted native-codegen slice. |

### 2. Finish Local Hygiene Before The 4090 Run

- [x] Run formatting and cheap static checks locally.
- [x] Run Higgs library tests locally.
- [x] Run shell/Python syntax checks for all Higgs validation scripts.
- [x] Ensure `docs/index.md` routes to every new Higgs doc.

Commands:

```bash
cargo fmt --all
cargo test --release -p pegainfer-higgs-audio --lib
cargo check --release -p pegainfer-higgs-audio --bins
bash -n tools/higgs/run_higgs_native_codegen_contract_gate.sh
bash -n tools/higgs/run_higgs_4090_native_codegen_validation.sh
python3 -m py_compile \
  tools/higgs/check_higgs_4090_evidence_bundle.py \
  tools/higgs/check_higgs_incremental_trace_reference.py \
  tools/higgs/check_higgs_native_codegen_contract_summary.py \
  tools/higgs/dump_higgs_incremental_trace_reference.py \
  tools/higgs/render_higgs_native_codegen_pr_evidence.py
git diff --check
```

Acceptance:

- Local checks pass or each failure is documented as Linux/CUDA-only.
- No unrelated submodule or generated-file churn is folded into the PR.

Local check result:

| Command | Result |
| --- | --- |
| `cargo fmt --all` | Pass |
| `cargo test --release -p pegainfer-higgs-audio --lib` | Pass, 71 tests |
| `cargo check --release -p pegainfer-higgs-audio --bins` | Pass |
| `cargo check --release -p pegainfer-higgs-audio --features reference-python-codec --bin higgs_vocode_codes` | Pass; Python-launching codec helper is explicit reference-only tooling |
| `bash -n tools/higgs/run_higgs_native_codegen_contract_gate.sh && bash -n tools/higgs/run_higgs_4090_native_codegen_validation.sh` | Pass |
| `python3 -m py_compile tools/higgs/check_higgs_4090_evidence_bundle.py tools/higgs/check_higgs_incremental_trace_reference.py tools/higgs/check_higgs_native_codegen_contract_summary.py tools/higgs/dump_higgs_incremental_trace_reference.py tools/higgs/render_higgs_native_codegen_pr_evidence.py` | Pass |
| `git diff --check` | Pass |
| `tools/higgs/run_higgs_native_codegen_contract_gate.sh --result-root /tmp/pegainfer-higgs-audio-local --label local-$(git rev-parse --short HEAD)` | Pass locally with `runtime_qwen3=skipped`, `native_continuation=skipped`, `trace_compare=skipped`, `profile=skipped` |
| `python3 tools/higgs/render_higgs_native_codegen_pr_evidence.py /tmp/pegainfer-higgs-audio-local/native-codegen/higgs-native-codegen-contract-gate-local-$(git rev-parse --short HEAD).txt --out /tmp/pegainfer-higgs-audio-local/native-codegen/pr-evidence-local-$(git rev-parse --short HEAD).md` | Pass; local evidence correctly lists Linux/4090 runtime/profile gaps as limitations |
| `python3 tools/higgs/render_higgs_native_codegen_pr_evidence.py <local-summary> --out /tmp/pegainfer-higgs-audio-local/native-codegen/pr-evidence-local-rerender.md` | Pass; rendered evidence includes explicit non-claims, Qwen3 diagnostic tradeoff, no shared core/kernel semantic change, and Python reference/tool boundary |
| `bash -n tools/higgs/prepare_higgs_4090_overlay.sh` | Pass |
| `tools/higgs/prepare_higgs_4090_overlay.sh --out-dir /tmp --label local-$(git rev-parse --short HEAD)-<time>` | Pass; generated a 31-file overlay, exact file list, and remote runner while excluding `pegainfer-kernels/third_party/*` |
| `docs/models/higgs-audio/native-codegen-pr-notes.md` | Added; PR body skeleton now records scope, design boundary, validation evidence slots, claim boundary, and maintainer-review question. The 4090 evidence table is still pending. |
| `tools/higgs/run_higgs_native_codegen_contract_gate.sh --result-root /tmp/pegainfer-higgs-audio-local --label stderr-capture-$(git rev-parse --short HEAD)` | Pass; gate logs now capture `cargo` stderr through `2>&1 | tee`, so local and 4090 evidence files include build output rather than only test stdout. |
| `python3 tools/higgs/classify_higgs_native_codegen_diff.py --markdown-out /tmp/higgs-native-codegen-diff.md` | Added; classifies the worktree into Higgs-local, Qwen3 diagnostic, server feature wiring, doc/tool, and unexpected buckets before PR. |
| `python3 tools/higgs/check_higgs_4090_evidence_bundle.py <summary> --pr-evidence <md> --expected-label <label> --expected-sm 89 --expected-nvcc-jobs 8` | Added; validates a PR-ready 4090 result bundle and intentionally rejects local/no-GPU evidence, missing reference trace comparison, or missing `nsys`/`ncu` profiles. |

### 3. Prepare The 4090 Environment

- [ ] Sync the branch to the RTX 4090 host.
- [ ] Install or verify `protoc`, CUDA toolchain, `nsys`, `ncu`, Rust nightly,
  and Python reference environment.
- [ ] Set deterministic build knobs.
- [ ] Confirm the server feature can compile with Higgs enabled.

Commands:

```bash
which protoc || apt-get update && apt-get install -y protobuf-compiler
export PEGAINFER_CUDA_SM=89
export PEGAINFER_NVCC_JOBS=8
rustc --version
cargo --version
nvidia-smi
which nsys || true
which ncu || true
cargo check --release -p pegainfer-server --features higgs-audio
```

Acceptance:

- Build metadata is captured in the gate summary.
- Missing profiling tools are treated as blockers for performance evidence, not
  silently ignored.

Current remote status:

| Target | Result | Next action |
| --- | --- | --- |
| Old 4090 at `42.123.114.169:32222` | SSH port responds with `SSHPiper`, but both recorded users reject the stored password. | Need refreshed credential or a different machine. |
| Seetacloud at `connect.nmb1.seetacloud.com:13096` | DNS resolves, but the port is currently refused / closes during SSH key exchange. | Need instance restarted or a fresh rental endpoint. |

Prepared local transfer artifacts:

- `/tmp/pegainfer-higgs-native-codegen-overlay-946d0e12.tgz`: overlay tarball
  containing the 30 changed/untracked files needed for the native-codegen slice,
  excluding dirty `pegainfer-kernels/third_party/*` submodules.
- `/tmp/pegainfer-higgs-native-codegen-overlay-files.txt`: exact overlay file
  list.
- `/tmp/pegainfer-higgs-run-4090-template.sh`: remote execution template. Run
  it from the remote repo root after extracting the overlay and setting
  `MODEL_DIR` and `GOLDEN`.
- `tools/higgs/prepare_higgs_4090_overlay.sh`: repo helper that regenerates the
  overlay tarball, exact file list, and remote runner while excluding dirty
  `pegainfer-kernels/third_party/*` submodules.

Reusable sync workflow once a 4090 host is available:

```bash
tools/higgs/prepare_higgs_4090_overlay.sh --out-dir /tmp --label "$(git rev-parse --short HEAD)-4090"
scp /tmp/pegainfer-higgs-native-codegen-overlay-$(git rev-parse --short HEAD)-4090.tgz <host>:/tmp/
scp /tmp/pegainfer-higgs-run-4090-$(git rev-parse --short HEAD)-4090.sh <host>:/tmp/
ssh <host> 'cd /data/src/pegainfer && tar -xzf /tmp/pegainfer-higgs-native-codegen-overlay-<label>.tgz && MODEL_DIR=/path/to/higgs GOLDEN=/path/to/higgs-one-step-golden.safetensors bash /tmp/pegainfer-higgs-run-4090-<label>.sh'
```

### 4. Generate The Reference Incremental Trace

- [ ] Generate the official/HF incremental trace with `use_cache=True` /
  `past_key_values`.
- [ ] Validate the reference trace schema before using it as golden evidence.
- [ ] Preserve provenance: model path, config hash or commit, package versions,
  prompt, step count, dtype, and device.

Commands:

```bash
tools/higgs/run_higgs_4090_native_codegen_validation.sh \
  --result-root /data/results/pegainfer/higgs-audio \
  --label "$(git rev-parse --short HEAD)" \
  --model-dir /path/to/higgs \
  --golden /path/to/higgs-one-step-golden.safetensors \
  --generate-reference-trace

python3 tools/higgs/check_higgs_incremental_trace_reference.py \
  /data/results/pegainfer/higgs-audio/native-codegen/higgs-reference-incremental-trace-$(git rev-parse --short HEAD).json \
  --expected-steps 9
```

Acceptance:

- The reference trace includes full logits/top-k evidence, not only sampled
  rows.
- Weak or malformed golden files fail before Rust comparison.

### 5. Run Native Retained-KV Code Generation

- [ ] Run the runtime-Qwen3 retained continuation test on the 4090.
- [ ] Run the native codegen contract gate with `--runtime-qwen3`.
- [ ] Confirm the native trace proves repeated retained-KV steps and does not
  rebuild the full prompt.

Commands:

```bash
export PEGAINFER_CUDA_SM=89
export PEGAINFER_NVCC_JOBS=8
cargo test --release -p pegainfer-higgs-audio --features runtime-qwen3 --lib runtime_bridge -- --nocapture

tools/higgs/run_higgs_native_codegen_contract_gate.sh \
  --result-root /data/results/pegainfer/higgs-audio \
  --label "$(git rev-parse --short HEAD)" \
  --runtime-qwen3 \
  --model-dir /path/to/higgs \
  --golden /path/to/higgs-one-step-golden.safetensors
```

Acceptance:

- A native trace JSON and codec-input JSON are written under
  `/data/results/pegainfer/higgs-audio/native-codegen/`.
- The gate summary records SM 89 and `PEGAINFER_NVCC_JOBS=8`.

### 6. Compare Native Trace Against Reference Trace

- [ ] Run the semantic trace comparison.
- [ ] Report more than cosine: first divergent step, argmax agreement, logits
  cosine, max/mean/p99 drift, argmax regret, and top-k overlap.
- [ ] Decide whether the evidence supports semantic parity, strict trace parity,
  or only bring-up.

Command:

```bash
cargo run --release -p pegainfer-higgs-audio --bin higgs_compare_codegen_trace -- \
  --reference /data/results/pegainfer/higgs-audio/native-codegen/higgs-reference-incremental-trace-$(git rev-parse --short HEAD).json \
  --actual /data/results/pegainfer/higgs-audio/native-codegen/higgs-native-continuation-trace-$(git rev-parse --short HEAD).json
```

Acceptance:

- The PR evidence table uses the measured result, not optimistic wording.
- If the first divergence is early or argmax agreement is weak, the claim stays
  at bring-up/diagnostic instead of native decode.

### 7. Capture nsys And ncu Evidence

- [ ] Run one end-to-end `nsys` profile for the native continuation smoke.
- [ ] Run one `ncu` profile for the dominant kernel or phase found by `nsys`.
- [ ] Record launch count, synchronization gaps, dominant kernels, and whether
  the path is performance-relevant yet.

Command:

```bash
tools/higgs/run_higgs_native_codegen_contract_gate.sh \
  --result-root /data/results/pegainfer/higgs-audio \
  --label "$(git rev-parse --short HEAD)-profile" \
  --runtime-qwen3 \
  --model-dir /path/to/higgs \
  --golden /path/to/higgs-one-step-golden.safetensors \
  --profile
```

Acceptance:

- Both `.nsys-rep` and `.ncu-rep` exist, or the missing tool/failure is recorded
  as a blocker.
- No optimization claim is made without correctness and E2E context.

### 8. Render PR Evidence

- [ ] Validate the gate summary.
- [ ] Render maintainer-facing PR evidence Markdown.
- [ ] Summarize artifacts and limitations in a compact table.

Commands:

```bash
python3 tools/higgs/check_higgs_native_codegen_contract_summary.py \
  /data/results/pegainfer/higgs-audio/native-codegen/higgs-native-codegen-contract-gate-$(git rev-parse --short HEAD).txt \
  --expected-label "$(git rev-parse --short HEAD)" \
  --expected-sm 89 \
  --expected-nvcc-jobs 8 \
  --check-files

python3 tools/higgs/render_higgs_native_codegen_pr_evidence.py \
  /data/results/pegainfer/higgs-audio/native-codegen/higgs-native-codegen-contract-gate-$(git rev-parse --short HEAD).txt \
  --out /data/results/pegainfer/higgs-audio/native-codegen/pr-evidence-$(git rev-parse --short HEAD).md
```

Acceptance:

- The PR evidence can be pasted with minimal explanation.
- Open maintainer questions are limited to architectural tradeoffs, especially
  the Qwen3 diagnostic API versus Higgs-local copied executor slice.

### 9. Decide The Next PR Boundary

- [ ] If trace and profiling evidence are clean, prepare a PR for native
  incremental audio-code generation foundation.
- [ ] If Qwen3 diagnostic touch is still too invasive, split into two PRs:
  first Higgs-local scaffolding and evidence tools, then Qwen3 retained-hidden
  diagnostic API.
- [ ] If semantic trace evidence is weak, keep the PR as bring-up only and file
  a follow-up issue for first-divergent-stage diagnosis.

Acceptance:

- The PR title uses standard terms, not local labels like "A-layer".
- The PR body clearly separates implemented code, measured evidence,
  limitations, and maintainers' requested design feedback.

## Completion Criteria

This todo list is complete only when all of the following are true:

- 4090 build and runtime-Qwen3 smoke pass.
- Official/HF incremental reference trace exists and passes the checker.
- Native retained-KV trace exists and passes or meaningfully reports the semantic
  comparison gate.
- nsys/ncu artifacts are captured or explicitly blocked by missing tooling.
- PR evidence Markdown is generated from real artifacts.
- The final claim boundary is updated in `native-code-generation-plan.md` and
  the PR text.

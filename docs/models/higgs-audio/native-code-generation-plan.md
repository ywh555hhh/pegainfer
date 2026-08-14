# Higgs-Audio Native Code Generation Plan

TL;DR: Next milestone is not native wav E2E. It is a native Rust + CUDA
incremental audio-code generation path: Higgs config detection, model-line
launch boundary, retained prompt KV, repeated audio-code steps, trace artifacts,
and reference-driven semantic gates, with Python kept outside the runtime path.

Last touched: 2026-08

## Claim Boundary

This milestone may claim:

- Higgs-Audio is recognized as a PegaInfer model line.
- Native one-step prefill/audio-head diagnostics remain available.
- A native retained-KV incremental audio-code generation loop is the target.
- Python tools are references, golden generators, or codec sidecars only.

This milestone must not claim:

- native wav E2E;
- native codec/vocoder;
- production-ready serving;
- strict trace parity unless the evidence table says so.

## Component State

| Component | State | Evidence | Next owner |
| --- | --- | --- | --- |
| Model-line detection | Implemented | `pegainfer-server` registers Higgs behind `--features higgs-audio`; `model_line` probes `higgs_multimodal_qwen3`. | Keep minimal until launch is real. |
| Launch preflight | Implemented | Reads `config.json`, manifest, and Higgs runtime load plan before failing closed. | Extend into real launch only after decode loop exists. |
| Native one-step prefill | Implemented as diagnostic/runtime bridge | `HiggsAudioRuntime` uses Qwen3 executor for prompt prefill and Higgs-owned audio head projection. | Preserve as the first correctness gate. |
| Retained prompt KV | Implemented for the diagnostic continuation path, 4090 verification pending | `prefill_prompt_session` retains the prompt under a Higgs session handle; each native continuation schedules, forwards, applies, or reverts exactly one retained KV step. | Verify repeated retention and recovery on Linux/4090. |
| Audio-code generation state | Implemented for model-local loop contract | `audio_codegen` owns per-step sampling rows, delay/raw rows, feedback embedding, trace updates, and a continuation driver contract keyed by the retained Higgs session id. A hidden-returning backend adapter projects final normed hidden through the Higgs-owned audio head before sampling. | Compare the real trace against the official/HF incremental reference. |
| Continuation stage contract | Implemented | `continuation_contract` pins feedback-embedding input, final-normed-hidden output, retained KV, and no full-prompt rebuild. | Keep the public boundary narrow while validation expands. |
| Multi-step audio-code decode | Implemented in the diagnostic native path, 4090 verification pending | The model-local driver consumes feedback embeddings; `higgs_native_continuation_smoke` requests eight retained Qwen3 continuation steps and records the returned hidden/audio-head/codegen path. | Run the smoke on Linux/4090 and compare the trace. |
| Decode trace artifact | Emitted by the native smoke path, reference comparison gate wired | `decode_trace` records prompt length, sampled rows, delayed/raw rows, done state, argmax, and optional full logits/top-k evidence; the native smoke writes a nine-step artifact from the prompt seed plus eight retained continuations. `higgs_compare_codegen_trace` compares a native trace against an official/HF incremental trace when one is supplied. | Produce/run the official/HF incremental trace on 4090 and run the comparison gate. |
| Final-hidden replay bridge | Implemented as bring-up tool | `higgs_replay_hidden_codegen` consumes a safetensors `final_hidden.bf16 [steps, 2560]` dump, applies the Higgs-owned audio head, and emits the same trace/codec artifacts. The hidden dump must come from the real backend to count as backend evidence. | Produce the hidden dump from embedding-fed retained-KV GPU continuation. |
| Codec / wav output | Reference sidecar only | `higgs_vocode_codes` can call a Python codec helper from a tool binary. | Do not count as native runtime evidence. |
| Profiling evidence | Partially captured on 4090 | `run_higgs_native_codegen_contract_gate.sh --profile` profiles the native continuation smoke automatically when `--runtime-qwen3`, `--model-dir`, and `--golden` are provided. `nsys` captured successfully; `ncu` is blocked on the current cloud host by `ERR_NVGPUCTRPERM`. | Rerun `ncu` on a profiler-friendly host. |
| Environment evidence | Gate summary wired, 4090 capture pending | The native codegen gate records GPU/driver/CUDA, nvcc, nsys, ncu, rustc, cargo, uname, commit, SM, and nvcc job count. Missing local tools are recorded as `unavailable` rather than guessed. | Use the 4090 summary as the PR evidence source. |
| PR evidence summary | Implemented | `render_higgs_native_codegen_pr_evidence.py` turns a gate summary into maintainer-facing Markdown with scope, claim boundary, supported-claim decision, environment, gates, artifacts, and limitations. | Render it from the 4090 summary and paste the relevant table into the PR without upgrading the generated supported-claim line by hand. |
| PR-ready evidence bundle check | Implemented, target run pending | `check_higgs_4090_evidence_bundle.py` verifies that the copied/result bundle is real 4090 evidence: GPU/CUDA metadata is present, runtime-Qwen3/native continuation/reference comparison/profile are all `ok`, trace/profile artifacts exist, and PR evidence has passed gates. | Run automatically at the end of the 4090 driver; use the same checker after copying results back. |
| Reference trace generator | Implemented, target run pending | `dump_higgs_incremental_trace_reference.py` uses Transformers Qwen3 `use_cache=True` / `past_key_values`, feeding fused audio-codebook embeddings per step, and emits the same trace schema consumed by `higgs_compare_codegen_trace`. `check_higgs_incremental_trace_reference.py` verifies schema/provenance/full-logits/top-k evidence before the Rust comparison runs. | Run it on 4090 or let the 4090 driver generate and check the trace. |
| 4090 validation driver | Implemented and exercised via lower-level gate | `run_higgs_4090_native_codegen_validation.sh` sequences optional HF reference-trace generation, server feature check, runtime-qwen3 native gate, optional reference-trace compare, optional forced-prefix diagnostics, optional profiling, and PR evidence rendering. The lower-level gate has produced the current 4090 evidence. | Use the full driver once strict PR-ready evidence is expected; use lower-level gate while strict compare/ncu are known blockers. |

## Todo List

- [x] Register a feature-gated Higgs-Audio `ModelLine` that probes
  `higgs_multimodal_qwen3` configs. The model-line surface is behind the
  Higgs crate's `server-line` feature so artifact-only checks do not compile
  the frontend/protobuf stack.
- [x] Keep server launch fail-closed until the real scheduler/decode path is
  wired, so detection cannot be mistaken for serving support.
- [x] Add a model-line-owned launch preflight that validates config, manifest,
  and runtime load-plan metadata before failing closed at the unsupported native
  decode stage.
- [ ] Replace the diagnostic-bin-only path with a model-line-owned launch path
  that builds the Higgs runtime and drives the real retained-KV decode loop.
- [x] Implement native retained-KV audio continuation: prompt prefill, audio
  logits, sampled codebook row, delay-pattern update, feedback embedding, and
  embedding-fed next-step decode.
- [x] Define a native decode trace artifact schema for each decode step:
  sampled codes, delayed rows, raw codec rows, logits summary, and done state.
- [ ] Keep Python out of all server/runtime paths. Python codec tools remain
  reference/bring-up helpers.
- [x] Add a Higgs-owned `AudioCodeGenerationSession` that composes the existing
  retained prompt session, delay-pattern state, codebook feedback embedding,
  and trace writer without exposing Qwen3 request ids to callers. The
  continuation backend receives the Higgs session id each step so a real backend
  must continue the retained KV state rather than acting as a stateless replay.
- [x] Add a model-local continuation driver contract for repeated steps:
  feedback embedding -> next audio prediction -> sampled codebook row ->
  delayed/raw rows -> trace.
- [x] Add a hidden-returning backend adapter for the real continuation shape:
  feedback embedding + retained session -> final normed hidden -> Higgs-owned
  audio head -> next audio prediction.
- [x] Add a runtime-facing hidden-step push boundary: once a backend returns
  final normed hidden for the retained session, `runtime_bridge` validates the
  Higgs session id, applies the Higgs-owned audio head, and records the
  code-generation trace step.
- [x] Add a runtime-facing `HiggsAudioCodegenSeed` that keeps the retained
  prompt session handle together with the seeded code-generation trace.
- [x] Add a testable continuation stage contract that rejects token-id/full
  prompt rebuild semantics.
- [x] Record the retained decode hidden boundary and implementation tradeoff in
  `retained-decode-hidden-boundary.md`.
- [x] Implement the first GPU-backed native continuation step after prompt
  prefill: feedback embedding as the next input embedding -> retained-KV Qwen3
  continuation hidden -> audio logits -> sampled codebook row.
- [x] Implement repeated retained-KV continuation for at least eight audio-code
  steps without rebuilding the full prompt.
- [x] Wire the decode trace artifact to the real retained-KV loop.
- [x] Add the reference-driven semantic gate command for official/HF
  incremental decode traces. It reports prompt/row equality, first divergent
  step, argmax agreement, logits cosine, max/mean/p99 drift, argmax regret, and
  top-k overlap. Missing full-logits/top-k trace evidence fails the semantic
  gate instead of being treated as parity.
- [x] Add the official/HF incremental `past_key_values` reference trace
  generator. It is a Python reference/golden tool only; it is not called from
  any Rust runtime path.
- [x] Add a reference trace checker that rejects missing provenance, wrong
  step count, sampled-row-only traces, missing full logits, missing top-k, and
  malformed argmax rows before semantic comparison.
- [x] Run the official/HF incremental trace generator on 4090 and run the trace
  comparison against the native retained-KV trace. The comparison is measured
  but not strict-pass: free-running native trace diverges at step 4 with logits
  cosine `0.99989086` and argmax agreement `51/72`.
- [x] Add and run forced common-prefix replay on 4090:
  `higgs_native_forced_prefix_smoke` consumes sampled rows from the HF trace
  while still running native retained-KV body/audio-head logits. This makes
  sampled/raw rows exact and improves cosine to `0.99999833`, argmax agreement
  to `64/72`; remaining mismatches are tied-logit strictness, not feedback-path
  divergence.
- [x] Add a tool/runtime command that writes the generated delayed/raw code rows
  and trace JSON from the native loop. A wav-producing Python sidecar may remain
  a separate reference tool, not the serving path.
- [x] Add a CPU-only artifact preparation tool for bring-up:
  sampled codebook rows -> native code-generation trace JSON + codec input JSON.
  This fixes artifact schema and review shape but is not native decode evidence.
- [x] Add a final-hidden replay bridge for backend bring-up:
  `final_hidden.bf16 [steps, 2560]` -> Higgs audio head -> codegen trace +
  codec artifacts. This isolates backend correctness from audio-head/delay/trace
  correctness without claiming that the hidden dump itself was native.
- [ ] Capture one `nsys` E2E profile and one `ncu` dominant-kernel profile on
  the 4090 path. `nsys` has been captured on the current cloud 4090; `ncu` is
  blocked by host performance-counter permissions (`ERR_NVGPUCTRPERM`) and
  needs a profiler-friendly host or provider-side setting change.
- [x] Record reproducibility metadata in the gate summary: GPU/driver/CUDA,
  nvcc, nsys, ncu, rustc, cargo, uname, commit, SM, and nvcc job count.
- [x] Add a PR evidence renderer that converts the gate summary into compact
  Markdown for the PR validation section.
- [x] Add a strict PR-ready 4090 evidence bundle checker. It intentionally
  rejects local/no-GPU summaries, missing reference trace comparison, and
  missing `nsys`/`ncu` profiles.
- [x] Add a 4090 validation driver that runs the server feature check, native
  codegen gate, optional trace compare/profile, PR evidence rendering, and the
  strict evidence-bundle check in one ordered command.

## Work Order

1. Land the current non-invasive boundary slice: server feature wiring,
   `ModelLine` detection, launch preflight, and native trace schema.
2. Build the Higgs-owned generation session in library code first, with CPU
   tests for delay, feedback embedding, done-state, and trace rows.
3. Add the continuation driver contract with a scripted backend so the real GPU
   backend has a precise per-step API to satisfy, including the retained Higgs
   session identity.
4. Choose the embedding-fed, hidden-returning backend route for retained-KV
   continuation: shared Qwen3 diagnostic API versus Higgs-local copied executor
   slice.
5. Connect one native continuation step to the existing runtime bridge and
   prove it does not full-prefill the prompt again. The code path now uses
   Qwen3's narrow embedding-fed retained hidden diagnostic API; Linux/4090 must
   still verify it executes and emits a trace row from the real backend.
6. Extend the path to an eight-step loop and write trace JSON from the real
   loop. The implementation and smoke command now exist; Linux/4090 execution
   remains the evidence gate.
7. Generate the official/HF incremental reference trace and run the wired
   comparison gate:
   `higgs_compare_codegen_trace --reference <official-trace> --actual <native-trace>`.
8. Only after correctness is stable, capture 4090 `nsys` and `ncu` evidence.

## Acceptance

The next milestone is accepted when the following are true:

```bash
cargo run --release -p pegainfer-higgs-audio --bin higgs_continuation_contract_report
tools/higgs/run_higgs_native_codegen_contract_gate.sh --result-root /data/results/pegainfer/higgs-audio --label "$(git rev-parse --short HEAD)"
python3 tools/higgs/check_higgs_native_codegen_contract_summary.py /data/results/pegainfer/higgs-audio/native-codegen/higgs-native-codegen-contract-gate-$(git rev-parse --short HEAD).txt --expected-label "$(git rev-parse --short HEAD)" --expected-sm 89 --expected-nvcc-jobs 8 --check-files
cargo check --release -p pegainfer-server --features higgs-audio
cargo test --release -p pegainfer-higgs-audio --lib
PEGAINFER_CUDA_SM=89 cargo test --release -p pegainfer-higgs-audio --features runtime-qwen3 --lib runtime_bridge -- --nocapture
cargo run --release -p pegainfer-higgs-audio --bin higgs_prepare_codegen_artifacts -- --sampled-codes-json /path/to/sampled-codes.json --prompt-tokens 12 --trace-out /data/results/pegainfer/higgs-audio/native-codegen/trace.json --codec-input-out /data/results/pegainfer/higgs-audio/native-codegen/codec-input.json
cargo run --release -p pegainfer-higgs-audio --bin higgs_replay_hidden_codegen -- --model-dir /path/to/higgs --hidden-safetensors /data/results/pegainfer/higgs-audio/native-codegen/final-hidden-rows.safetensors --seed-sampled-codes 1,101,201,301,401,501,601,701 --prompt-tokens 12 --trace-out /data/results/pegainfer/higgs-audio/native-codegen/replayed-trace.json --codec-input-out /data/results/pegainfer/higgs-audio/native-codegen/replayed-codec-input.json
```

4090/Linux environment preflight:

```bash
which protoc || apt-get update && apt-get install -y protobuf-compiler
export PEGAINFER_CUDA_SM=89
export PEGAINFER_NVCC_JOBS=8
cargo check --release -p pegainfer-server --features higgs-audio

tools/higgs/run_higgs_4090_native_codegen_validation.sh \
  --result-root /data/results/pegainfer/higgs-audio \
  --label "$(git rev-parse --short HEAD)" \
  --model-dir /path/to/higgs \
  --golden /path/to/higgs-one-step-golden.safetensors \
  --generate-reference-trace \
  --profile

tools/higgs/run_higgs_native_codegen_contract_gate.sh \
  --result-root /data/results/pegainfer/higgs-audio \
  --label "$(git rev-parse --short HEAD)" \
  --runtime-qwen3 \
  --model-dir /path/to/higgs \
  --golden /path/to/higgs-one-step-golden.safetensors

tools/higgs/run_higgs_native_codegen_contract_gate.sh \
  --result-root /data/results/pegainfer/higgs-audio \
  --label "$(git rev-parse --short HEAD)-trace" \
  --runtime-qwen3 \
  --model-dir /path/to/higgs \
  --golden /path/to/higgs-one-step-golden.safetensors \
  --reference-trace /path/to/official-incremental-trace.json

tools/higgs/run_higgs_native_codegen_contract_gate.sh \
  --result-root /data/results/pegainfer/higgs-audio \
  --label "$(git rev-parse --short HEAD)-profile" \
  --runtime-qwen3 \
  --model-dir /path/to/higgs \
  --golden /path/to/higgs-one-step-golden.safetensors \
  --profile

python3 tools/higgs/render_higgs_native_codegen_pr_evidence.py \
  /data/results/pegainfer/higgs-audio/native-codegen/higgs-native-codegen-contract-gate-<label>.txt \
  --out /data/results/pegainfer/higgs-audio/native-codegen/pr-evidence-<label>.md

python3 tools/higgs/check_higgs_incremental_trace_reference.py \
  /data/results/pegainfer/higgs-audio/native-codegen/higgs-reference-incremental-trace-<label>.json \
  --expected-steps 9
```

The contract gate always writes and checks a CPU-only artifact smoke:

- `higgs-artifact-sampled-<label>.json`
- `higgs-artifact-trace-<label>.json`
- `higgs-artifact-codec-input-<label>.json`
- `higgs-artifact-tool-<label>.txt`
- `higgs-native-continuation-smoke-<label>.txt` when `--runtime-qwen3`,
  `--model-dir`, and `--golden` are set
- `higgs-native-continuation-trace-<label>.json` when the native continuation
  smoke runs
- `higgs-native-continuation-codec-input-<label>.json` when the native
  continuation smoke runs
- `higgs-native-forced-prefix-trace-<label>.json` and
  `higgs-native-forced-prefix-codec-input-<label>.json` when forced
  common-prefix replay is run against an HF trace
- `higgs-forced-prefix-trace-compare-<label>.txt` for common-prefix semantic
  drift diagnosis
- `profiles/higgs-native-codegen-<label>.nsys-rep` and
  `profiles/higgs-native-codegen-ncu-<label>.ncu-rep` when `--profile` is set.
  If `--native-loop-cmd` is omitted, the gate profiles the same native
  continuation smoke used for correctness.
- `higgs-codegen-trace-compare-<label>.txt` when `--reference-trace` is set.
  This is the semantic trace gate and requires full logits/top-k evidence in
  both traces; sampled rows alone are not enough.
- `higgs-reference-incremental-trace-<label>.json` when the 4090 validation
  driver is run with `--generate-reference-trace`.

The summary also records reproducibility evidence:

- `gpu_info`
- `cuda_version`
- `nsys_version`
- `ncu_version`
- `rustc_version`
- `cargo_version`
- `uname`

`render_higgs_native_codegen_pr_evidence.py` renders the summary into a compact
PR validation section. It deliberately repeats the claim boundary and lists
skipped gates under limitations, so the PR does not imply native wav E2E,
production serving, or strict trace parity.

`run_higgs_4090_native_codegen_validation.sh` is the preferred target-host
entrypoint once the model directory, one-step golden, and optional official/HF
incremental trace are in place. Use the lower-level gate commands only when
bisecting a failure.
If an external reference trace already exists, pass `--reference-trace`.
Otherwise pass `--generate-reference-trace` so the driver creates one through
`dump_higgs_incremental_trace_reference.py` before running the Rust comparison
gate.
Every reference trace is checked by `check_higgs_incremental_trace_reference.py`
before comparison, so an external JSON must prove it is an incremental
`past_key_values` reference and must carry full logits/top-k evidence.

The checker validates the trace schema
`higgs-audio-native-codegen-trace-v1`, the codec schema
`higgs-audio-codec-input-v1`, and the expected smoke raw-code row. This proves
  the artifact shape only; it is still not native retained decode evidence.
When the native continuation smoke runs, the checker also validates a nine-step
trace: prompt seed plus eight embedding-fed retained-KV continuation steps.

Local macOS evidence for this contract slice:

- `cargo fmt --all`: passed.
- `cargo test --release -p pegainfer-higgs-audio --lib`: passed, 68 tests.
- `cargo metadata --locked --no-deps --format-version 1`: passed.
- `git diff --check`: passed.
- `python3 -m py_compile tools/higgs/check_higgs_native_codegen_contract_summary.py`: passed.
- `bash -n tools/higgs/run_higgs_native_codegen_contract_gate.sh`: passed.
- `tools/higgs/run_higgs_native_codegen_contract_gate.sh --help`: passed and
  documents automatic `--profile` behavior for the native continuation smoke.
- `cargo check --release -p pegainfer-higgs-audio --bins`: passed after adding
  `higgs_compare_codegen_trace`.
- `tools/higgs/run_higgs_native_codegen_contract_gate.sh --result-root /tmp/pegainfer-higgs-audio-gate --label session-id-contract`: passed with `claim_boundary=native_contract_only_no_native_decode`.
- `tools/higgs/run_higgs_native_codegen_contract_gate.sh --result-root /tmp/pegainfer-higgs-audio-gate --label native-profile-autocmd-local`: passed with `profile=skipped`; this validates that the gate summary/checker still work after adding automatic profile command construction.
- `tools/higgs/run_higgs_native_codegen_contract_gate.sh --result-root /tmp/pegainfer-higgs-audio-gate --label trace-compare-local`: passed with `trace_compare=skipped`.
- `cargo run --release -p pegainfer-higgs-audio --bin higgs_compare_codegen_trace -- --reference <artifact-trace> --actual <same-artifact-trace>`: intentionally failed because the CPU artifact trace lacks full logits/top-k evidence. This proves the semantic gate does not pass weak sampled-row-only traces.
- `tools/higgs/run_higgs_native_codegen_contract_gate.sh --result-root /tmp/pegainfer-higgs-audio-gate --label metadata-local`: passed and wrote environment metadata into the summary. On this local macOS host, GPU/profiler/CUDA values may be `unavailable`; the 4090 run is the authoritative environment evidence.
- `python3 tools/higgs/render_higgs_native_codegen_pr_evidence.py /tmp/pegainfer-higgs-audio-gate/native-codegen/higgs-native-codegen-contract-gate-metadata-local.txt`: passed and rendered a PR-ready Markdown evidence table with limitations for skipped 4090 gates.
- `bash -n tools/higgs/run_higgs_4090_native_codegen_validation.sh`: passed.
- `tools/higgs/run_higgs_4090_native_codegen_validation.sh --help`: passed and documents the one-command 4090 validation flow.
- `python3 -m py_compile tools/higgs/dump_higgs_incremental_trace_reference.py`: passed.
- `python3 -m py_compile tools/higgs/check_higgs_incremental_trace_reference.py`: passed.
- `tools/higgs/check_higgs_incremental_trace_reference.py --help`: passed.
- `python3 tools/higgs/check_higgs_native_codegen_contract_summary.py /tmp/pegainfer-higgs-audio-gate/native-codegen/higgs-native-codegen-contract-gate-session-id-contract.txt --expected-label session-id-contract --expected-sm 89 --expected-nvcc-jobs 8 --check-files`: passed.
- `cargo run --release -p pegainfer-higgs-audio --bin higgs_prepare_codegen_artifacts -- --sampled-codes-json /tmp/.../sampled.json --prompt-tokens 12 --trace-out /tmp/.../trace.json --codec-input-out /tmp/.../codec.json`: passed; generated trace schema `higgs-audio-native-codegen-trace-v1` and codec schema `higgs-audio-codec-input-v1`.
- `tools/higgs/run_higgs_native_codegen_contract_gate.sh --result-root /tmp/pegainfer-higgs-audio-gate --label artifact-tool`: passed with artifact smoke included and `claim_boundary=native_contract_only_no_native_decode`.
- `cargo check --release -p pegainfer-server --features higgs-audio`: blocked on this macOS host because `vllm-server` build script could not find `protoc`; rerun on Linux/4090 after installing `protobuf-compiler`.
- `PEGAINFER_CUDA_SM=89 cargo check --release -p pegainfer-higgs-audio --features runtime-qwen3 --lib`: blocked on this macOS host because `rdma-mummy-sys` needs Linux headers (`endian.h`, `linux/types.h`) and `pegainfer-kernels` nvcc workers failed on local CUDA compilation; rerun on Linux/4090.
- `PEGAINFER_CUDA_SM=89 cargo test --release -p pegainfer-higgs-audio --features runtime-qwen3 --lib runtime_bridge -- --nocapture`: blocked by the same macOS/Linux/CUDA build environment issues; this remains a required 4090 gate.
- `PEGAINFER_CUDA_SM=89 cargo run --release -p pegainfer-higgs-audio --features runtime-qwen3 --bin higgs_native_continuation_smoke -- --model-dir /path/to/higgs --golden /path/to/higgs-one-step-golden.safetensors --trace-out /data/results/pegainfer/higgs-audio/native-codegen/higgs-native-continuation-trace.json --codec-input-out /data/results/pegainfer/higgs-audio/native-codegen/higgs-native-continuation-codec-input.json`: blocked by the same macOS/Linux/CUDA build environment issues; this is the first real retained-continuation smoke and remains a required 4090 gate.

Runtime-boundary checks:

- `pegainfer-server --features higgs-audio` detects Higgs configs through
  `ModelLine`.
- Launch preflight validates the Higgs artifact/load-plan boundary before
  reporting the exact unsupported native decode stage.
- No server/runtime path launches Python or imports PyTorch, Transformers, or
  SGLang.
- Existing Python tools are documented as reference/golden/codec sidecars.

Native code-generation checks:

- The continuation contract report says input is `feedback_embedding`, output is
  `final_normed_hidden`, `retained_kv=true`, and `full_prompt_rebuild=false`.
- Prompt prefill retains KV under a Higgs-owned session handle.
- Each continuation backend call is keyed by that Higgs-owned session handle.
- The real backend shape returns final normed hidden; Higgs-owned code then
  projects it through the fused audio head before sampling codebook rows.
- `runtime_bridge` rejects a final-hidden push if the Higgs session id does not
  match the code-generation session id.
- Incremental decode runs at least eight native audio-code steps without
  rebuilding the full prompt every step.
- The output includes delayed code rows and raw codec rows.
- The decode trace is emitted by the same native loop being measured, not by a
  CPU-only replay of sampled rows.
- `higgs_prepare_codegen_artifacts` may be used to validate artifact shape from
  sampled rows during bring-up, but it must not be cited as native retained
  decode evidence.
- `higgs_replay_hidden_codegen` validates the audio-head/delay/trace half after a
  backend hidden dump. It does not prove that the hidden dump used retained KV,
  consumed a feedback embedding, or avoided a prompt rebuild; those facts must
  be recorded by the backend runtime trace.
- `higgs_native_continuation_smoke` is the first runtime evidence hook for that
  backend fact: it seeds retained prompt KV, derives feedback embedding from the
  fused codebook embedding, calls the embedding-fed retained Qwen3 hidden API,
  then emits the trace/codec artifacts from the resulting Higgs loop.

Correctness evidence:

- golden/reference provenance records implementation, commit/version,
  checkpoint, config, dtype, and device;
- prompt ids are exact;
- audio argmax agreement is reported;
- logits cosine is reported and should be at least `0.999` for the first gate;
- top-64 minimum overlap is reported and should be at least `58`;
- max, mean, and p99 drift are recorded;
- any semantic/strict-parity gap is explained with the first divergent stage.

Performance evidence:

- `nsys` records the native incremental path end to end;
- `ncu` records the dominant kernel or phase;
- GPU, SM, CUDA/driver, command, commit, and correctness result are recorded
  together with the timing.

## Review Notes

The current `ModelLine` registration is intentionally a boundary marker, not a
serving claim. It lets maintainers review config detection, feature wiring, and
CLI ownership separately from the larger retained-KV decode work.

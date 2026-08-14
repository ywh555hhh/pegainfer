# Higgs-Audio Retained Decode Hidden Boundary

TL;DR: Higgs-Audio native audio-code continuation needs an embedding-fed
retained-KV Qwen3 body step that returns the final normed hidden state. This
slice adds a narrow Qwen3 diagnostic API for that shape and wires Higgs to it;
the claim remains below native decode until the runtime-qwen3 gate passes on a
Linux/4090 host and reference trace metrics are recorded.

Last touched: 2026-08

## Requirement

The native audio-code loop needs this per-step contract:

```text
retained prompt KV
  + previous delayed code row
  + Higgs-owned retained session id
  -> fused codebook feedback embedding [hidden_size]
  -> one embedding-fed retained-KV body continuation step
  -> final RMSNorm hidden [hidden_size]
  -> Higgs fused audio head [8, 1026]
  -> sampled code row
```

The current Higgs crate owns feedback embedding and every item after
`final RMSNorm hidden`: audio-head projection, argmax/top-k diagnostics, delay
pattern, raw-code recovery, and trace rows. The missing native GPU piece is the
embedding-fed, hidden-returning retained-KV body continuation step.

The Higgs-local `HiddenStateContinuationBackend` contract preserves that split:
the backend returns only final normed hidden for a retained session, and
Higgs-owned code applies the fused audio head and delay/trace state afterward.

## Current Qwen3 Boundary

The closest existing Qwen3 paths are:

- `Qwen3Executor::prefill_last_hidden_bf16_retained_prompt`: returns final
  normed hidden for the prompt's last token and retains KV.
- `Qwen3Executor::execute_decode`: consumes retained KV and a token id, but
  returns sampled text-token results, not final hidden.
- `Qwen3Model::batch_decode`: starts by embedding token ids through the text
  embedding table. Higgs-Audio continuation already has a fused codebook
  feedback embedding vector, so token-id decode is the wrong input contract.
- `Qwen3Model::batch_decode`: internally leaves final normed hidden in
  `BatchDecodeBuffers::normed` immediately before `lm_head`, but this buffer and
  the model/lane types are crate-private.

Therefore the Higgs runtime cannot implement a real GPU-backed audio
continuation step through the text-token public Qwen3 decode API. A token-id
decode diagnostic would be insufficient evidence because it would test a
text-token embedding path, not the Higgs audio feedback path.

## Options

| Option | Shape | Pros | Cost / risk |
| --- | --- | --- | --- |
| Shared Qwen3 diagnostic API | Add an embedding-fed, hidden-returning retained decode method to `pegainfer-qwen3`. | Smallest code duplication; uses exact existing kernels and buffers. | Touches another model line; must justify why arbitrary input-embedding decode is stable and not Higgs-specific leakage. |
| Higgs-local copied executor slice | Copy the narrow Qwen3 decode path into `pegainfer-higgs-audio`, replace token-id embedding lookup with Higgs feedback embedding input, and return hidden before the text `lm_head`. | Keeps PR model-local and avoids changing Qwen3 public surface. | High maintenance cost; copied Qwen3 logic can drift from upstream numeric and scheduler fixes. |
| Shared reusable body-forward abstraction | Extract a generic body-continuation trait shared by Qwen3/Higgs. | Architecturally clean if multiple model lines need it. | Too broad for the foundation PR; larger maintainer review burden. |

## Current Decision

Use the narrow Qwen3 diagnostic API option for this slice:

- `pegainfer-qwen3` exposes
  `Qwen3Executor::decode_embedding_last_hidden_bf16_retained` through
  `pegainfer_qwen3::runtime`;
- the method is single-rank diagnostic surface, not a serving scheduler API;
- it consumes a caller-owned BF16 input embedding, schedules one retained decode
  step, runs the Qwen3 body eagerly, returns final normed hidden before
  `lm_head`, and applies a bookkeeping token only to advance the retained KV
  sequence ledger;
- `pegainfer-higgs-audio` owns the multimodal semantics: feedback embedding,
  audio head, sampling, delay pattern, codec rows, and trace rows.

This is an intentional Qwen3 touch. Copying the Qwen3 executor/model decode
slice into Higgs would keep the diff model-local, but it would duplicate CUDA
graph, KV metadata, numeric-policy, LoRA, and future Qwen3 decode fixes. The
narrow API keeps one owner for the Qwen3 body while avoiding shared core,
shared kernel, or broad frontend/runtime abstraction changes.

Do not treat this as production native decode evidence until the Linux/4090
runtime gate proves the path executes and emits trace rows from the retained-KV
backend.

## Acceptance For The Next Slice

- The first continuation step does not rebuild the full prompt.
- The backend call is keyed by the same Higgs-owned session id created by prompt
  prefill, so the implementation cannot silently become a stateless replay.
- The backend consumes `feedback_embedding` from the previous delayed row as the
  model input embedding, not as a token id.
- The backend returns final normed hidden from the retained-KV step, not text
  logits or sampled text tokens.
- Higgs audio head projects that hidden to `[8, 1026]` logits.
- The trace row is emitted by the real GPU-backed continuation step.
- Correctness evidence compares against an official/HF incremental
  `past_key_values` trace for the same prompt and fixed prefix.

Required 4090 gate:

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

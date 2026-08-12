# Higgs Audio One-Step Golden Gate

> **Status:** fork-branch bring-up slice. This is the artifact contract and first
> Rust gate for Higgs Audio; it is not yet a full PegaInfer runtime parity gate.

## Purpose

The first Higgs Audio contribution should make the correctness target explicit
before runtime code grows around it. The gate pins one zero-shot prompt and the
first audio-row logits:

```text
<|tts|><|text|>Hello from PegaInfer.<|audio|>
```

The fixture stores the prompt ids, final hidden state, full fused audio logits,
top-64 logprobs, and argmax ids. The shape that matters for the future runtime
gate is:

```text
audio_logits.f32: [1, 8, 1026]
```

## Reference Design

The generator uses SGLang-Omni's Higgs semantics without importing the full
serving stack:

- Prompt construction mirrors
  `sglang_omni/models/higgs_tts/text_tokenizer.py`.
- The transformer backbone is loaded through HuggingFace/Transformers Qwen3 from
  the pinned Higgs checkpoint `body.*` tensors.
- The fused audio head follows
  `sglang_omni/models/higgs_tts/modeling.py`:

```text
F.linear(final_hidden, tied.embedding.modality_embeddings.0.embedding.weight)
    .reshape(batch, 8, 1026)
```

This deliberately matches the current checkpoint, where there is no separate
`tied.head.*` tensor for audio. The fused modality embedding is the audio head.

## Generated Artifact

The fork branch carries a small derived fixture:

```text
test_data/higgs-one-step-audio-logits.safetensors
```

Fixture metadata records:

- model id `bosonai/higgs-tts-3-4b`
- revision `7556c17e05201fccd9c8cc120bc216dcc7b5d561`
- config/tokenizer/index hashes
- SGLang-Omni reference files used for prompt/head semantics
- 4090 memory observed during generation

The current fixture SHA-256 is:

```text
a9c23650c0e9a39ee2b314f1dead7c7d2fd8adfe77c312b198b6e2e6b3d91471
```

## Rust Gate

`pegainfer-higgs-audio` currently validates the fixture contract only:

- required metadata is present and pinned
- tensor names, dtypes, and shapes match the expected one-step schema
- the committed fixture hash matches the generator output

This is intentionally narrower than the future #395 gate. It prevents silent
fixture drift while the model crate is being built.

## Artifact Gate

The second slice adds config and manifest validation around the same fixture:

```bash
cargo run -p pegainfer-higgs-audio --bin higgs_artifact_check -- \
  --model-dir /data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --golden test_data/higgs-one-step-audio-logits.safetensors
```

This check is deliberately static. It proves the local checkpoint directory,
weight manifest, and committed golden agree before runtime code is allowed to
depend on them:

- `config.json`, `tokenizer.json`, and `model.safetensors.index.json` hashes must
  match fixture metadata.
- Higgs text config must match the pinned Qwen3 body shape: 36 layers, hidden
  2560, 32 query heads, 8 KV heads, head dim 128, intermediate 9728.
- Higgs audio config must be the discrete 8-codebook, 1026-vocab,
  delay-pattern modality used by the golden.
- The manifest must expose `body.*`, text embedding, and the fused modality
  embedding/head at `tied.embedding.modality_embeddings.0.embedding.weight`.
- The expected KV footprint is 144 KiB per position, or 1152 MiB for 8192
  positions in bf16.

## Checkpoint Header Gate

The fourth slice strengthens `higgs_artifact_check` with a header-only
safetensors validation pass. It reads only the safetensors length prefix and JSON
header, not the multi-GB tensor payload, so it can run as a fast loader
preflight.

The gate validates the required runtime surface:

- text embedding: `[151936, 2560]` BF16
- fused modality embedding/head: `[8208, 2560]` BF16
- body norm: `[2560]` BF16
- per-layer attention projections:
  - `q_proj`: `[4096, 2560]` BF16
  - `k_proj` / `v_proj`: `[1024, 2560]` BF16
  - `o_proj`: `[2560, 4096]` BF16
  - `q_norm` / `k_norm`: `[128]` BF16
- per-layer MLP projections:
  - `gate_proj` / `up_proj`: `[9728, 2560]` BF16
  - `down_proj`: `[2560, 9728]` BF16

This proves the checkpoint is not merely named correctly in
`model.safetensors.index.json`; the actual safetensors shard header must agree
with the Higgs/Qwen3 loader contract.

## Runtime Load Plan

The fifth slice adds `HiggsRuntimeLoadPlan`, a model-owned mapping from checkpoint
tensor names to the future loader slots:

- `tied.embedding.text_embedding.weight` -> `qwen3.embed_tokens`
- `body.norm.weight` -> `qwen3.norm`
- `body.layers.N.*` -> `qwen3.layers.N.*`
- `tied.embedding.modality_embeddings.0.embedding.weight` ->
  `higgs.fused_audio_head`

The plan currently covers 399 BF16 tensors: 398 Qwen3 backbone tensors plus the
single fused Higgs audio head. It is intentionally a pure metadata layer: it does
not allocate GPU memory or read tensor payloads yet. The next runtime slice
should make the GPU loader consume this plan directly.

## Qwen3 Body View

The sixth slice adds a bridge materializer:

```bash
cargo run -p pegainfer-higgs-audio --bin higgs_materialize_qwen3_body -- \
  --model-dir /data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --out-dir /data/results/pegainfer/higgs-audio/qwen3-body-view
```

It rewrites the single Higgs safetensors shard into a Qwen3-compatible view:

- `tied.embedding.text_embedding.weight` -> `model.embed_tokens.weight`
- `body.norm.weight` -> `model.norm.weight`
- `body.layers.N.*` -> `model.layers.N.*`

The materialized view contains 398 BF16 tensors and excludes the fused Higgs
audio head. This is a bridge, not the final loader design: it duplicates the
7.5 GiB body payload so the existing `pegainfer-qwen3-4b` loader can be used
unchanged while the Higgs-owned loader contract is still being shaped.

The safetensors header is padded so the payload start is aligned for `bf16`.
Without this, debug Rust aborts inside `DeviceMatrix::from_safetensors` when it
casts tensor payload bytes to `bf16`.

## Comparison Gate

The third slice defines the actual runtime parity contract:

```bash
cargo run -p pegainfer-higgs-audio --bin higgs_compare_one_step -- \
  --golden test_data/higgs-one-step-audio-logits.safetensors \
  --actual /path/to/pegainfer-higgs-one-step-actual.safetensors
```

The expected actual file uses the same tensor schema as the golden:

- prompt tensors and audio argmax/top-k ids are exact-match gates
- `final_hidden.bf16` is compared with max/mean absolute drift
- `audio_logits.f32` is compared with max/mean absolute drift
- `audio_top64.logprobs.f32` is compared with max/mean absolute drift

Initial tolerances are intentionally explicit and CLI-overridable:

```text
hidden_abs_tol=0.03125
hidden_mean_abs_tol=0.003
logits_abs_tol=0.05
logits_mean_abs_tol=0.005
top_logprobs_abs_tol=0.05
top_logprobs_mean_abs_tol=0.005
```

These are not a substitute for runtime calibration. They are the current
engineering boundary for the first actual-vs-golden gate; tighten or widen them
only after a measured PegaInfer dump exists and the drift source is understood.

When workspace-level git dependencies block the full root workspace on remote
hosts, run the Higgs-only isolated gate:

```bash
tools/higgs/run_isolated_higgs_checks.sh
```

That script builds a temporary one-member workspace containing only
`pegainfer-higgs-audio` and the committed fixture, then runs fmt, unit tests, and
`higgs_compare_one_step` self-comparison. It is a workaround for dependency
isolation only; it does not replace full workspace CI.

## 4090 Bring-Up Notes

The 4090-D host at `/data/src/pegainfer` was synchronized to fork commit
`1462955e` on branch `feat/higgs-audio-one-step-golden`.

Static model/golden validation passed on the 4090 host with the Python reference
environment:

```text
golden_sha256 a9c23650c0e9a39ee2b314f1dead7c7d2fd8adfe77c312b198b6e2e6b3d91471
config_sha256 match=True
tokenizer_json_sha256 match=True
model_index_sha256 match=True
model_type higgs_multimodal_qwen3
arch HiggsMultimodalQwen3ForConditionalGeneration
text_layers 36
hidden 2560
audio_codebooks 8
audio_vocab 1026
body_tensors 397
total_tensors 927
```

The Rust isolated Higgs gate and the real checkpoint header gate passed on the
4090 host:

```text
isolated Higgs tests: 14 passed
higgs_compare_one_step self-comparison: ok
higgs_artifact_check: ok
checkpoint headers: files=1 tensors=399 bf16=399
runtime load plan: tensors=399 shard_files=1 bf16_mib=7712 qwen3_backbone=398 higgs_head=1
```

The Qwen3 body view was also materialized from the real checkpoint on the 4090
host:

```text
out_dir: /data/results/pegainfer/higgs-audio/qwen3-body-view
tensors: 398
payload_mib: 7672
model.safetensors size: 8044982042 bytes
header_len: 45842
data_start_mod2: 0
```

Header probes from the generated view:

```text
model.embed_tokens.weight                 BF16 [151936, 2560]
model.layers.0.self_attn.q_proj.weight    BF16 [4096, 2560]
model.layers.35.mlp.down_proj.weight      BF16 [2560, 9728]
model.norm.weight                         BF16 [2560]
has original Higgs tensor names: false
```

With FlashInfer headers and its pinned `cccl`, `cutlass`, and `spdlog`
third-party trees restored, the existing Qwen3 runtime successfully loaded the
materialized Higgs body view and completed a minimal prefill + decode smoke:

```bash
PEGAINFER_CUDA_SM=89 PEGAINFER_NVCC_JOBS=8 \
cargo run --release -p pegainfer-qwen3-4b --bin qwen3_decode_context -- \
  --model-path /data/results/pegainfer/higgs-audio/qwen3-body-view \
  --contexts 1,16,128 \
  --iters 3 \
  --disable-cuda-graph
```

Observed release decode timings on RTX 4090 D:

```text
prompt_context,kv_len_during_decode,iters,avg_ms,p50_ms,p90_ms,min_ms,max_ms
1,2,3,9.6083,9.6299,9.6503,9.5446,9.6503
16,17,3,9.5781,9.5664,9.6115,9.5563,9.6115
128,129,3,9.7341,9.7195,9.7664,9.7163,9.7664
```

An NSYS CUDA/NVTX/CUBLAS profile for context 16 was captured at:

```text
/data/results/pegainfer/higgs-audio/profiles/qwen3-body-c16.nsys-rep
```

`nsys stats` shows the two profiled decode steps are dominated by cuBLAS GEMV
kernels:

```text
cuBLAS/internal gemvx kernels: 95.1% of captured GPU kernel time
FlashInfer BatchDecodeWithPagedKVCacheKernel: 1.0%
FusedAddRMSNormRoundKernel: 1.4%
AppendPagedKVCacheKernel: 0.6%
```

NCU is installed (`2025.1.0.0`) but cannot collect GPU performance counters on
this host:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

Environment notes:

- GPU is RTX 4090 D, driver 570.124.06, 24564 MiB VRAM.
- Rust nightly is installed under `/root/.cargo/bin`, observed as
  `rustc 1.99.0-nightly`.
- Cargo uses `rsproxy.cn` sparse registry for crates.io.
- CUDA toolkit is 12.8 (`nvcc V12.8.61`).
- FlashInfer was restored from the pinned submodule commit
  `d768c14e7cf5dd5df45a8a1de78ae815879f108a` using tarballs, because a direct
  submodule clone through `gh-proxy.com` stalled. The pinned internal dependency
  commits used were:
  - `NVIDIA/cccl`: `876867684f7fac130e0f5911236e0a92a970d4fd`
  - `NVIDIA/cutlass`: `b46b16d003484063bca4ed365e44095c4c6ed633`
  - `gabime/spdlog`: `c3aed4b68373955e1cc94307683d44dca1515d2b`
- Direct `git clone --filter=blob:none --no-checkout
  https://github.com/vllm-project/vllm.git` failed with GitHub transfer too
  slow.
- `https://gh-proxy.com/https://github.com/vllm-project/vllm.git` passed
  `git ls-remote HEAD`, and a scoped `git config --global url.<proxy>.insteadOf`
  was installed for the vLLM URL only.
- Even after that, workspace-level `cargo test -p pegainfer-higgs-audio` still
  spent more than five minutes updating the vLLM git dependency. Treat this as a
  workspace dependency isolation/cache issue, not a Higgs crate correctness
  failure.

## Technical Debt

- The branch commits a derived fixture from a research/non-commercial model. This
  is acceptable for a fork validation branch, but upstream needs an explicit
  maintainer decision before merging the fixture.
- The generator uses HuggingFace Qwen3 for the backbone and SGLang-Omni semantics
  for prompt/head logic. A later gate should compare directly against a pinned
  SGLang-Omni execution path when the server stack is practical to run.
- The Rust crate does not yet load Higgs weights or execute the transformer. It
  locks the golden/artifact contract and can materialize a Qwen3-compatible body
  view for the existing Qwen3 runtime.
- The fixture covers one prompt. Wider prompt-length coverage belongs in the next
  parity slice after loader/backbone code exists.
- The generator captures `final_hidden.bf16`, but the first Rust test does not
  compare PegaInfer hidden states yet; the comparator now defines the check, but
  no PegaInfer Higgs runtime dump exists yet.
- The Qwen3 body smoke proves that the Higgs `body.*` tensors can be loaded and
  executed by the existing Qwen3 runtime after alias materialization. It does not
  yet prove golden parity, because the current smoke uses synthetic Qwen3 token
  ids and does not apply the fused Higgs audio head.
- Nsight Compute counters are blocked on the current 4090 host because
  `RmProfilingAdminOnly=1`; NSYS works.
- Remote Rust execution can still be blocked by workspace-level git dependencies
  such as `vllm-project/vllm.git`; `gh-proxy.com` is usable for `ls-remote` on
  the 4090 host, but full Cargo fetch still needs either a prewarmed cache or
  workspace dependency isolation.

## Next Execution Slice

1. Add a Higgs-owned runtime path that reuses the Qwen3 body without duplicating
   the safetensors payload.
2. Run the golden prompt through PegaInfer with the committed prompt ids instead
   of synthetic Qwen3 token ids.
3. Expose or dump the final hidden vector from the Qwen3 body path.
4. Load/apply `tied.embedding.modality_embeddings.0.embedding.weight` as the
   fused Higgs audio head.
5. Dump PegaInfer `final_hidden.bf16`, `[8, 1026]` audio logits, top-64
   logprobs, and argmax ids into the comparator schema.
6. Compare PegaInfer final hidden and `[8, 1026]` audio logits against this
   fixture with calibrated bf16 tolerances.
7. Only after that, add delay-pattern, sampling, KV decode, and codec gates.

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
`1611334a` on branch `feat/higgs-audio-one-step-golden`.

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

Environment notes:

- GPU is RTX 4090 D, driver 570.124.06, 24564 MiB VRAM.
- Rust nightly is installed under `/root/.cargo/bin`, observed as
  `rustc 1.99.0-nightly`.
- Cargo uses `rsproxy.cn` sparse registry for crates.io.
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
  only locks the golden/artifact contract.
- The fixture covers one prompt. Wider prompt-length coverage belongs in the next
  parity slice after loader/backbone code exists.
- The generator captures `final_hidden.bf16`, but the first Rust test does not
  compare PegaInfer hidden states yet; the comparator now defines the check, but
  no PegaInfer runtime dump exists yet.
- Nsight Compute counters are blocked on the current 4090 host because
  `RmProfilingAdminOnly=1`; NSYS works.
- Remote Rust execution can still be blocked by workspace-level git dependencies
  such as `vllm-project/vllm.git`; `gh-proxy.com` is usable for `ls-remote` on
  the 4090 host, but full Cargo fetch still needs either a prewarmed cache or
  workspace dependency isolation.

## Next Execution Slice

1. Add `HiggsConfig` and tensor-manifest parsing.
2. Load the pinned checkpoint's `body.*`, text embedding, and fused modality
   embedding/head.
3. Reuse the current Qwen3 backbone operator path for zero-shot prefill.
4. Dump PegaInfer `final_hidden.bf16`, `[8, 1026]` audio logits, top-64
   logprobs, and argmax ids into the comparator schema.
5. Compare PegaInfer final hidden and `[8, 1026]` audio logits against this
   fixture with calibrated bf16 tolerances.
6. Only after that, add delay-pattern, sampling, KV decode, and codec gates.

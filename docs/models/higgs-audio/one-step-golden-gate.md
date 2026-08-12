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

## Technical Debt

- The branch commits a derived fixture from a research/non-commercial model. This
  is acceptable for a fork validation branch, but upstream needs an explicit
  maintainer decision before merging the fixture.
- The generator uses HuggingFace Qwen3 for the backbone and SGLang-Omni semantics
  for prompt/head logic. A later gate should compare directly against a pinned
  SGLang-Omni execution path when the server stack is practical to run.
- The Rust crate does not yet load Higgs weights or execute the transformer. It
  only locks the golden contract.
- The fixture covers one prompt. Wider prompt-length coverage belongs in the next
  parity slice after loader/backbone code exists.
- The generator captures `final_hidden.bf16`, but the first Rust test does not
  compare PegaInfer hidden states yet.
- Nsight Compute counters are blocked on the current 4090 host because
  `RmProfilingAdminOnly=1`; NSYS works.

## Next Execution Slice

1. Add `HiggsConfig` and tensor-manifest parsing.
2. Load the pinned checkpoint's `body.*`, text embedding, and fused modality
   embedding/head.
3. Reuse the current Qwen3 backbone operator path for zero-shot prefill.
4. Compare PegaInfer final hidden and `[8, 1026]` audio logits against this
   fixture with calibrated bf16 tolerances.
5. Only after that, add delay-pattern, sampling, KV decode, and codec gates.

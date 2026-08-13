# Higgs-Audio SGLang-Omni Trace Golden on RTX 4090

This directory preserves a rich, stage-by-stage Higgs-Audio one-step trace golden generated from the SGLang-Omni source-reference path on the SeetaCloud/AutoDL RTX 4090 machine on 2026-08-13.

This is intentionally larger than the normal one-step fixture. Use it for calibration and drift localization, not as the default lightweight CI fixture.

## Files

- `higgs-one-step-sglang-omni-trace-autodl.safetensors`: rich trace golden with prompt, embedding, per-layer, per-module, and audio-head intermediate tensors.
- `higgs-one-step-sglang-omni-trace-autodl-summary.json`: machine-readable manifest with metadata, tensor names, shapes, dtypes, SHA256, commit IDs, and GPU info.
- `higgs-one-step-sglang-omni-golden-autodl-trace-run.safetensors`: lightweight one-step golden generated in the same run as the trace.

## Remote Environment

- Host: `autodl-container-a9c346b8e9-f2562921`
- GPU: `NVIDIA GeForce RTX 4090, 595.71.05, 24564 MiB`
- PegaInfer commit: `4d845bb63c3cc97a46ffd4a82a632c1c28338f43`
- SGLang-Omni commit: `c6980be8ce680deb50e6b366065c9fbd679415d1`
- Torch: `2.12.1+cu130`
- Transformers: `5.12.1`
- Model revision: `7556c17e05201fccd9c8cc120bc216dcc7b5d561`

## SHA256

- Trace golden: `b56c1668cddd69c42d8332cb80c34114fa1b66497b95d281c6a068c6c0ec0a34`
- Same-run lightweight one-step golden: `6fc9ca849d8ae8c123346b1d2ea819191021640e04952df38cb09d668b51c1eb`
- Local summary JSON: `d0cfe20a7aa9494ea738589044df897f3fc7d3a974eb18f98bd58b29f2b748b7`

## Trace Coverage

- Tensor count: `481`
- Prompt tensors: `3`
- Embedding tensors: `2`
- Layer tensors: `468`
- Final hidden tensor: `1`
- Audio tensors: `7`

## Tensor Contract

Prompt:

- `prompt.input_ids_padded`: `[batch, seq]`, int64
- `prompt.attention_mask`: `[batch, seq]`, int64
- `prompt.lengths`: `[batch]`, int64

Embedding and hidden-state trace:

- `embedding.sequence_hidden.bf16`: `[batch, seq, hidden]`
- `embedding.last_hidden.bf16`: `[batch, hidden]`
- `layer.NN.sequence_hidden.bf16`: `[batch, seq, hidden]`
- `layer.NN.last_hidden.bf16`: `[batch, hidden]`
- `final_hidden.bf16`: `[batch, hidden]`

Per-layer module outputs, for `NN = 00..35`:

- `layer.NN.input_layernorm.output.bf16`: `[batch, seq, hidden]`
- `layer.NN.self_attn.q_proj.output.bf16`: `[batch, seq, 4096]`
- `layer.NN.self_attn.k_proj.output.bf16`: `[batch, seq, 1024]`
- `layer.NN.self_attn.v_proj.output.bf16`: `[batch, seq, 1024]`
- `layer.NN.self_attn.q_norm.output.bf16`: `[batch, seq, 32, 128]`
- `layer.NN.self_attn.k_norm.output.bf16`: `[batch, seq, 8, 128]`
- `layer.NN.self_attn.o_proj.output.bf16`: `[batch, seq, hidden]`
- `layer.NN.post_attention_layernorm.output.bf16`: `[batch, seq, hidden]`
- `layer.NN.mlp.gate_proj.output.bf16`: `[batch, seq, 9728]`
- `layer.NN.mlp.up_proj.output.bf16`: `[batch, seq, 9728]`
- `layer.NN.mlp.down_proj.output.bf16`: `[batch, seq, hidden]`

Audio-head trace:

- `audio_head.input_hidden.bf16`: `[batch, hidden]`
- `audio_head.flat_logits.f32`: `[batch, 8208]`
- `audio_logits.f32`: `[batch, 8, 1026]`
- `audio_logprobs.f32`: `[batch, 8, 1026]`
- `audio_top64.ids`: `[batch, 8, 64]`
- `audio_top64.logprobs.f32`: `[batch, 8, 64]`
- `audio_argmax.ids`: `[batch, 8]`

## Intended Use

Use this file to bisect numeric drift by stage:

1. Verify prompt IDs and masks first.
2. Compare `embedding.last_hidden.bf16` and `layer.NN.last_hidden.bf16` to find the first drifting layer.
3. If a layer drifts, compare that layer's q/k/v/norm/o/mlp tensors to locate the first drifting substage.
4. Compare `audio_head.input_hidden.bf16`, then `audio_head.flat_logits.f32`, then `audio_logits.f32` and top-k outputs.

Safe claim: this artifact gives a stage-by-stage SGLang-Omni source-reference Higgs-Audio one-step trace on RTX 4090 for downstream PegaInfer/OpenInfer calibration.

Unsafe claim: this proves full SGLang-Omni runtime end-to-end parity. It is still source-reference, not full server/runtime parity.

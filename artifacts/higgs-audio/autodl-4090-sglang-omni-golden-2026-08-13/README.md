# Higgs-Audio SGLang-Omni Golden Candidate on RTX 4090

This directory preserves a SGLang-Omni source-reference one-step Higgs-Audio golden generated on the SeetaCloud/AutoDL RTX 4090 machine on 2026-08-13.

## Files

- `higgs-one-step-sglang-omni-golden-autodl.safetensors`: generated source-reference golden candidate.
- `higgs-one-step-sglang-omni-golden-autodl-summary.json`: structured provenance and tensor comparison against the current PegaInfer fixture.

## Remote Environment

- Host: `autodl-container-a9c346b8e9-f2562921`
- GPU: `NVIDIA GeForce RTX 4090, 595.71.05, 24564 MiB`
- PegaInfer commit: `4d845bb63c3cc97a46ffd4a82a632c1c28338f43`
- SGLang-Omni commit: `c6980be8ce680deb50e6b366065c9fbd679415d1`
- Torch: `2.12.1+cu130`
- Transformers: `5.12.1`
- Model revision: `7556c17e05201fccd9c8cc120bc216dcc7b5d561`

## SHA256

- AutoDL generated golden: `0738d01dc38ba7652c0f6ed38965bd51dff82ab74a1e74eaf7360a36e9837c54`
- Current repo fixture: `bbaae8018759b7e8f26d2acfb1aefb5bee3e5099d47573bb4ce3c980b6096684`
- Local summary JSON: `dec09a9d679693904567a5820c107527022b9fe471ceda17412fc4d8b5ed9ce4`

## Comparison Against Current Fixture

- Prompt tensors match exactly:
  - `prompt.input_ids_padded = [[151667, 151672, 9707, 504, 393, 11188, 641, 802, 13, 151670]]`
  - `prompt.attention_mask = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]`
  - `prompt.lengths = [10]`
- Audio argmax matches exactly:
  - `audio_argmax.ids = [[244, 1024, 1024, 1024, 1024, 1024, 1024, 1024]]`
- Floating tensors are not strict-equal:
  - `audio_logits.f32`: `max_abs = 1.0`, `mean_abs = 0.04559576138854027`, `allclose_1e-4 = false`
  - `final_hidden.bf16`: `max_abs = 0.5`, `mean_abs = 0.0070122540928423405`, `allclose_1e-4 = false`
  - `audio_top64.logprobs.f32`: `max_abs = 1.5`, `mean_abs = 0.22973215579986572`, `allclose_1e-4 = false`

## Interpretation

This artifact is useful as a 4090/SGLang-Omni source-reference regeneration record, but it should not replace the committed fixture blindly. It proves the same prompt path and top-level generated audio token IDs as the current fixture, while also showing stack-dependent numeric drift in hidden states and logits between the CUDA 13 / torch 2.12.1 AutoDL environment and the earlier fixture environment.

Safe claim: SGLang-Omni source-reference regeneration on RTX 4090 produced the same one-step Higgs-Audio semantic argmax as the committed PegaInfer fixture.

Unsafe claim: This regeneration is strict tensor-identical to the committed fixture.

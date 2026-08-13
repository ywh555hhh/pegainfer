# Old RTX 4090 Golden-Trace-Driven Higgs-Audio Evidence

This directory preserves the old RTX 4090D PegaInfer/OpenInfer-side evidence generated after adding the rich Higgs-Audio trace comparator.

## Environment

- Host path: `/data/src/pegainfer`
- Branch: `feat/higgs-audio-one-step-golden`
- Latest comparator commit used for layer0 stage compare: `13651bfd`
- CUDA gate commit: `ef7b8d46`
- GPU: old RTX 4090D, SM89
- Model: `/data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561`

## Files

- `higgs-one-step-cuda-gate-old4090-trace-driven-ef7b8d4.txt`: PegaInfer CUDA one-step gate summary.
- `one-step-actual-vs-trace-old4090-ef7b8d4.{txt,json}`: lightweight actual vs AutoDL trace golden comparison.
- `higgs-prefill-layer-hidden-old4090-ef7b8d4.safetensors`: PegaInfer per-layer hidden actual dump.
- `prefill-layer-hidden-vs-trace-old4090-ef7b8d4.{txt,json}`: per-layer hidden actual vs AutoDL trace golden comparison.
- `higgs-layer0-stages-old4090-13651bf.safetensors`: PegaInfer layer0 stage actual dump.
- `layer0-stages-vs-trace-old4090-13651bf.{txt,json}`: layer0 stage actual vs AutoDL trace golden comparison using `--alias-set layer0-stage`.

## Results

- One-step CUDA gate passed:
  - `semantic_comparison=ok`
  - `session_semantic_comparison=ok`
  - `duplicate_request_id_guard=ok`
- One-step actual vs trace golden compared 8 common tensors:
  - prompt exact
  - argmax exact
  - first alert: `final_hidden.bf16`
- Per-layer hidden vs trace golden compared 41 tensors:
  - prompt exact
  - `embedding.last_hidden.bf16` exact
  - `layer.00.last_hidden.bf16` within alert thresholds
  - first alert: `layer.01.last_hidden.bf16`
- Layer0 stage vs trace golden compared 13 mapped stage tensors:
  - alerts: `0`
  - first alert: `none`
  - worst mean abs: `layer0.output_hidden.bf16 -> layer.00.sequence_hidden.bf16:0.002617`

## Interpretation

The old 4090 path is suitable for golden-trace-driven development. The current evidence says:

1. The lightweight semantic gate is still green, so the implementation has not regressed at the user-visible one-step semantic level.
2. The prompt and embedding path are exact.
3. Layer0 substage outputs match the AutoDL SGLang-Omni source-reference trace within the current thresholds.
4. The first layer-level drift worth investigating is `layer.01.last_hidden.bf16`.

Next development target: add or reuse a layer1 stage dump and compare it against the trace golden. Do not hack around drift by injecting trace tensors or hardcoding outputs; the trace is only an oracle for locating the first semantic/numeric divergence.

## SHA256

```text
6efc49efa59d400e880ab511482ebf86770309da2723a2e178333863977e494d  higgs-layer0-stages-old4090-13651bf.safetensors
b42337ad855e0ed055f5c7e40af68fe016c40e1f60335bfbdf2e8d743e4ab569  higgs-one-step-cuda-gate-old4090-trace-driven-ef7b8d4.txt
6066ba530c6a3e7dce6c31e7514e7f6cb673a3a2e22fed9ea4906fc3e7936790  higgs-prefill-layer-hidden-old4090-ef7b8d4.safetensors
d6932c51429641c2970ffe3fdc3699d3708bb86fce3e811599486b197f152c4e  layer0-stages-vs-trace-old4090-13651bf.json
b860082458890a3b7c6baec04d51ff2f0f193dd0041f88a9491d4f33ca505cb3  layer0-stages-vs-trace-old4090-13651bf.txt
f9a64a5a2961e98036681f70f508c801f760586adf664506122cef1e742d4ff3  one-step-actual-vs-trace-old4090-ef7b8d4.json
6647b9e4154961c200f18810608a99fbc28aa64277364cd34429a906ea0837a0  one-step-actual-vs-trace-old4090-ef7b8d4.txt
a3639507ad3a91b8184eea1684f81470374d5f050c3ab6087c96642d3a849cae  prefill-layer-hidden-vs-trace-old4090-ef7b8d4.json
51a74863d3d4fb27663165787022e02e8aaacd2cc847277f3a6694416cf1d3c1  prefill-layer-hidden-vs-trace-old4090-ef7b8d4.txt
```

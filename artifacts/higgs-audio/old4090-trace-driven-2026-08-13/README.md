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
- `higgs-layer1-stages-old4090-dev.safetensors`: PegaInfer layer1 stage actual dump generated with generalized `--layer-idx 1` support.
- `layer1-stages-vs-trace-old4090-dev-v2.{txt,json}`: layer1 stage actual vs AutoDL trace golden comparison using execution-ordered `--alias-set layer-stage`.
- `layer1-qk-norm-drift-old4090-dev.json`: q/k RMSNorm diagnostic recomputing HF-like norm variants from golden/actual projections plus checkpoint q/k norm weights.
- `higgs-layer{0,1}-stages-hf-old4090-dev.safetensors`: HF/SGLang-source layer-stage goldens with 17 stages, including RoPE, attention output, and SiLU input stages that the rich trace does not expose directly.
- `layer{0,1}-full-stages-vs-hf-old4090-dev.{txt,json}`: PegaInfer actual stage dumps compared against the full 17-stage HF/SGLang-source goldens.

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
- Layer1 stage vs trace golden compared 13 mapped stage tensors:
  - input hidden, input norm, q/k/v projections are within thresholds
  - first execution-order alert: `layer1.q_norm.bf16 -> layer.01.self_attn.q_norm.output.bf16`
  - additional alerts: `k_norm`, `gate_proj`, `up_proj`, `output_hidden`
  - worst mean abs: `layer1.gate_proj.bf16 -> layer.01.mlp.gate_proj.output.bf16:0.008645`
- Layer1 q/k RMSNorm recompute diagnostic:
  - `hf_like_bf16_mid` recomputed from actual q/k projections matches PegaInfer actual q/k norm exactly (`actual_kernel_mean=0.00000000` for both q and k)
  - the same formula recomputed from golden q/k projections matches SGLang/Transformers trace q/k norm exactly (`golden_formula_mean=0.00000000` for both q and k)
  - layer1 q/k norm drift is therefore explained by upstream q/k projection drift propagation, not by a q/k RMSNorm kernel semantics mismatch
- Full 17-stage HF/SGLang-source comparison:
  - layer0 compared 17 stages with `first_mean_abs_gt_0.003000=none`; worst mean abs remains `layer0.output_hidden.bf16:0.002617`
  - layer1 compared 17 stages with first mean alert at `layer1.q_norm.bf16`; q/k norm remains explained by the recompute diagnostic
  - layer1 attention output is stable (`mean_abs=0.000295`), while downstream MLP projection stages show larger propagated drift (`gate_proj mean_abs=0.008645`, `up_proj mean_abs=0.005325`)

## Interpretation

The old 4090 path is suitable for golden-trace-driven development. The current evidence says:

1. The lightweight semantic gate is still green, so the implementation has not regressed at the user-visible one-step semantic level.
2. The prompt and embedding path are exact.
3. Layer0 substage outputs match the AutoDL SGLang-Omni source-reference trace within the current thresholds.
4. The first layer-level drift worth investigating is `layer.01.last_hidden.bf16`.
5. Layer1 substage comparison initially narrowed the first execution-order alert to q/k RMSNorm (`layer1.q_norm.bf16`, then `layer1.k_norm.bf16`), after input hidden and q/k/v projections remain within thresholds.
6. The q/k RMSNorm recompute diagnostic rules out a q/k RMSNorm semantics mismatch: both golden and actual projections reproduce their corresponding q/k norm tensors exactly under the HF-like bf16-mid rounding formula.
7. Full-stage HF/SGLang-source goldens confirm layer0 is within tolerance across all 17 exposed stages and show layer1 attention output is not the dominant drift amplifier; the next useful target is projection/MLP numeric drift, not q/k norm or attention semantics.

Next development target: inspect whether layer0 output-hidden drift and layer1 MLP projection drift are acceptable BF16/cuBLAS accumulation drift or whether a more exact projection diagnostic is needed. Do not hack around drift by injecting trace tensors or hardcoding outputs; the trace is only an oracle for locating and explaining divergence.

## SHA256

```text
6efc49efa59d400e880ab511482ebf86770309da2723a2e178333863977e494d  higgs-layer0-stages-old4090-13651bf.safetensors
b6eee0ad297ec0b5f8750affaf2bd1edfb682a80d745d8c55597ec6760e631da  higgs-layer0-stages-hf-old4090-dev.safetensors
7a9f6380c695373fd896af59564a2c7aaeebbc9eb6b26744d916e8ef6f1b8b7c  higgs-layer1-stages-old4090-dev.safetensors
4be3f4dc29e170f8d1b54d255f531a2c65d31b34f7753f2aca65579a6dfcc342  higgs-layer1-stages-hf-old4090-dev.safetensors
b42337ad855e0ed055f5c7e40af68fe016c40e1f60335bfbdf2e8d743e4ab569  higgs-one-step-cuda-gate-old4090-trace-driven-ef7b8d4.txt
6066ba530c6a3e7dce6c31e7514e7f6cb673a3a2e22fed9ea4906fc3e7936790  higgs-prefill-layer-hidden-old4090-ef7b8d4.safetensors
60b0af25df2534e3a685d4f2c623ed9a812653f3ea4b76ea2a61cbba8afbc728  layer0-full-stages-vs-hf-old4090-dev.json
cc87358ac2802269a99917793da1a18806c5b288eb203c1e50ee2dba027ac7b3  layer0-full-stages-vs-hf-old4090-dev.txt
d6932c51429641c2970ffe3fdc3699d3708bb86fce3e811599486b197f152c4e  layer0-stages-vs-trace-old4090-13651bf.json
b860082458890a3b7c6baec04d51ff2f0f193dd0041f88a9491d4f33ca505cb3  layer0-stages-vs-trace-old4090-13651bf.txt
b40bbf91203d88869a77aa618f4ea43fadf6792771b71d3c524a649d8b943816  layer1-full-stages-vs-hf-old4090-dev.json
88debab6a9d4c9a29e4624c817f53a17294ba0d4affd9a411af9d28549d98683  layer1-full-stages-vs-hf-old4090-dev.txt
fc19d6cdf6b5f77f45fea35d2c7f82027ac18416b91e6827228ed748648448f1  layer1-stages-vs-trace-old4090-dev-v2.json
c26fc5f889c1ac0ef9f72454fb0ad4dc14112d117da0a588955a0591f5461e89  layer1-stages-vs-trace-old4090-dev-v2.txt
e0c205a771b1cf604d5631198fedd00111749b4dce017ca022a34eef66495961  layer1-qk-norm-drift-old4090-dev.json
f9a64a5a2961e98036681f70f508c801f760586adf664506122cef1e742d4ff3  one-step-actual-vs-trace-old4090-ef7b8d4.json
6647b9e4154961c200f18810608a99fbc28aa64277364cd34429a906ea0837a0  one-step-actual-vs-trace-old4090-ef7b8d4.txt
a3639507ad3a91b8184eea1684f81470374d5f050c3ab6087c96642d3a849cae  prefill-layer-hidden-vs-trace-old4090-ef7b8d4.json
51a74863d3d4fb27663165787022e02e8aaacd2cc847277f3a6694416cf1d3c1  prefill-layer-hidden-vs-trace-old4090-ef7b8d4.txt
```

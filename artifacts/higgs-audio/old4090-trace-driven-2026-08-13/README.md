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
- `layer{0,1}-projection-drift-old4090-dev.{txt,json}`: projection diagnostics recomputing q/k/v/o/gate/up/down from golden and actual inputs plus checkpoint weights.
- `higgs-layer{0,1}-stages-old4090-hfround.safetensors`: PegaInfer layer-stage actual dumps after switching Qwen3 fused-add-RMSNorm to HF-like norm rounding.
- `layer{0,1}-full-stages-vs-hf-old4090-hfround.{txt,json}`: full-stage comparisons after the HF-like fused-add-RMSNorm change.
- `layer1-residual-drift-old4090-hfround.{txt,json}`: residual/fused-add-RMSNorm diagnostic after the change, proving actual post-attention norm now matches the HF-like branch exactly.
- `one-step-actual-vs-trace-old4090-hfround.{txt,json}` and `prefill-layer-hidden-vs-trace-old4090-hfround.{txt,json}`: end-to-end and layer-hidden comparisons after the change.
- `higgs-one-step-cuda-gate-old4090-hfround.txt`: one-step CUDA gate after the change; semantic and session semantic gates remain green.
- `higgs-layer{32,33,34,35}-stages-hf-old4090-hfround.safetensors` and `higgs-layer{32,33,34,35}-stages-old4090-hfround.safetensors`: late-layer full-stage HF/SGLang-source goldens and PegaInfer actual dumps after the HF-like fused-add-RMSNorm change.
- `layer{32,33,34,35}-full-stages-vs-hf-old4090-hfround.{txt,json}`: late-layer full-stage comparisons used to separate real late-layer amplification from a stale rich-trace hidden-state alias.
- `prefill-layer-hidden-vs-trace-old4090-hfround-skip-final-alias.{txt,json}`: per-layer hidden comparison against the AutoDL rich trace after excluding the known bad `layer.35.{sequence,last}_hidden` aliases from schema v2.

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
- Projection recompute diagnostic:
  - layer1 actual-input recompute matches PegaInfer actual outputs closely: q/k/v/o projection recompute mean abs is <= `0.00000238`, gate/up/down is <= `0.00008011`
  - layer1 actual-vs-golden projection drift is much larger than the recompute residual: `gate_proj mean_abs=0.008645`, `up_proj mean_abs=0.005325`, `down_proj mean_abs=0.002628`
  - projection/storage boundaries are therefore explained by upstream input drift and BF16 linear amplification, not by a projection weight layout or GEMM call-site mismatch
- HF-like fused-add-RMSNorm change:
  - residual diagnostic before the change showed PegaInfer actual `post_attn_norm` matched `fused_round_formula` exactly, while HF/SGLang golden matched `hf_like_bf16_mid` exactly
  - after the change, layer1 actual `post_attn_norm` matches the HF-like branch exactly (`actual_vs_actual=0.00000000`) and no longer matches the old fused formula (`actual_vs_actual=0.00037608`)
  - layer0 output hidden improves from `mean_abs=0.002617` to `0.001837`
  - layer1 `post_attn_norm` improves from `0.001059` to `0.000648`, `gate_proj` from `0.008645` to `0.005716`, and `output_hidden` from `0.004883` to `0.003201`
  - one-step trace comparison improves: `final_hidden.mean_abs 0.007636 -> 0.006403`, `audio_logits.mean_abs 0.115132 -> 0.043190`, `audio_top64.logprobs.mean_abs 0.309766 -> 0.044132`, with `audio_argmax.ids` still exact
- Late-layer trace alignment:
  - fresh full-stage HF/SGLang-source goldens show PegaInfer late layers remain high-cosine: layer32/33/34/35 output-hidden cosine is `0.999994576`, `0.999994576`, `0.999990582`, and `0.999991179`
  - late-layer output-hidden mean drift grows gradually rather than jumping: layer32 `0.197028`, layer33 `0.249898`, layer34 `0.546417`, layer35 `1.036249`
  - three-way comparison showed the old AutoDL rich trace's `layer.35.last_hidden.bf16` is effectively the final normed hidden, not the raw layer35 decoder output: fresh HF stage vs AutoDL trace at layer35 has `mean_abs=174.776855`, while PegaInfer actual vs fresh HF stage is only `1.036249`
  - after excluding the bad final-layer hidden alias, per-layer hidden vs trace compares 40 tensors; the worst hidden drift is `layer.34.last_hidden.bf16:0.589288`, and `final_hidden.bf16` remains close (`mean_abs=0.006403`, cosine `0.999991894`)

## Interpretation

The old 4090 path is suitable for golden-trace-driven development. The current evidence says:

1. The lightweight semantic gate is still green, so the implementation has not regressed at the user-visible one-step semantic level.
2. The prompt and embedding path are exact.
3. Layer0 substage outputs match the AutoDL SGLang-Omni source-reference trace within the current thresholds.
4. The first layer-level drift worth investigating is `layer.01.last_hidden.bf16`.
5. Layer1 substage comparison initially narrowed the first execution-order alert to q/k RMSNorm (`layer1.q_norm.bf16`, then `layer1.k_norm.bf16`), after input hidden and q/k/v projections remain within thresholds.
6. The q/k RMSNorm recompute diagnostic rules out a q/k RMSNorm semantics mismatch: both golden and actual projections reproduce their corresponding q/k norm tensors exactly under the HF-like bf16-mid rounding formula.
7. Full-stage HF/SGLang-source goldens confirm layer0 is within tolerance across all 17 exposed stages and show layer1 attention output is not the dominant drift amplifier.
8. Projection recompute diagnostics rule out a projection weight-layout/GEMM call-site mismatch: the same checkpoint weights reproduce PegaInfer actual projection outputs from PegaInfer actual inputs with tiny residuals.
9. Residual/fused-add-RMSNorm diagnostics found and fixed a real Qwen3/HF semantic mismatch: PegaInfer had been preserving the BF16 residual-add boundary but not the HF RMSNorm mid-round-before-weight boundary.
10. The fix is partial but real: it improves layer0/layer1 stage parity and end-to-end trace metrics while keeping one-step semantic and session semantic gates green.
11. The previous `layer.35.last_hidden` giant drift was not a valid raw-layer oracle. The old rich trace schema labeled the final normed hidden as `layer.35.last_hidden`; future trace generation now skips that alias and stores the final normed hidden only as `final_hidden.bf16`.

Next development target: continue from the new first alert (`layer1.k_norm.bf16`, just above the `0.003` mean threshold) and investigate why small early-layer BF16 drift is gradually amplified through late MLP/residual stages while final norm/logits remain high-cosine and argmax-exact. Do not inject trace tensors or hardcode outputs; the trace is only an oracle for locating and explaining divergence.

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
523a59d5ba91c5dfc989a92da39b3f96251066f2bf9ff7ef2f0629bda6679fc5  layer0-projection-drift-old4090-dev.txt
c7d0b9477c472ca7c150a12adb49f98913df0d7d4922c28c3a61fb37d16fb94b  layer0-projection-drift-old4090-dev.json
d6932c51429641c2970ffe3fdc3699d3708bb86fce3e811599486b197f152c4e  layer0-stages-vs-trace-old4090-13651bf.json
b860082458890a3b7c6baec04d51ff2f0f193dd0041f88a9491d4f33ca505cb3  layer0-stages-vs-trace-old4090-13651bf.txt
b40bbf91203d88869a77aa618f4ea43fadf6792771b71d3c524a649d8b943816  layer1-full-stages-vs-hf-old4090-dev.json
88debab6a9d4c9a29e4624c817f53a17294ba0d4affd9a411af9d28549d98683  layer1-full-stages-vs-hf-old4090-dev.txt
ba5af654ae61d48c0fae0d754a17037ed03a9cf7d6af1bba4a26cf0a11c6f5a8  layer1-projection-drift-old4090-dev.txt
1e8a2afe559e9b66181eb7387ea756d3a79b65b6c36b6cf036e1a1ecb6751128  layer1-projection-drift-old4090-dev.json
fc19d6cdf6b5f77f45fea35d2c7f82027ac18416b91e6827228ed748648448f1  layer1-stages-vs-trace-old4090-dev-v2.json
c26fc5f889c1ac0ef9f72454fb0ad4dc14112d117da0a588955a0591f5461e89  layer1-stages-vs-trace-old4090-dev-v2.txt
e0c205a771b1cf604d5631198fedd00111749b4dce017ca022a34eef66495961  layer1-qk-norm-drift-old4090-dev.json
f9a64a5a2961e98036681f70f508c801f760586adf664506122cef1e742d4ff3  one-step-actual-vs-trace-old4090-ef7b8d4.json
6647b9e4154961c200f18810608a99fbc28aa64277364cd34429a906ea0837a0  one-step-actual-vs-trace-old4090-ef7b8d4.txt
a3639507ad3a91b8184eea1684f81470374d5f050c3ab6087c96642d3a849cae  prefill-layer-hidden-vs-trace-old4090-ef7b8d4.json
51a74863d3d4fb27663165787022e02e8aaacd2cc847277f3a6694416cf1d3c1  prefill-layer-hidden-vs-trace-old4090-ef7b8d4.txt
e77eecfcf5f1ba4e46d4fe7fb49327b1c6b188f42ad874e6f2d5c5d3d19dee0b  higgs-layer0-stages-old4090-hfround.safetensors
58c6ca3af6529b37fd3c7e9acace9f1a7c742f2316257188d4e8ad407f505d55  higgs-layer1-stages-old4090-hfround.safetensors
74dd4a65645f27da4af402ba4595ab32943fcb94d42fcd4eb0952124a9e992c7  higgs-one-step-cuda-gate-old4090-hfround.txt
271c567923f76043c9b3b0ef62fe8a1ec21c305c531068c4955246f1ff07a3b5  higgs-prefill-layer-hidden-old4090-hfround.safetensors
238ebb5c4c360957578718787126159460e309d7c979657ca41370f9507d8da4  layer0-full-stages-vs-hf-old4090-hfround.json
d0d7d17653a50ba8e1dc8fd2e0477d085e8a5903d6f78dfbe7b7e36bae9ffc50  layer0-full-stages-vs-hf-old4090-hfround.txt
c02a8dc70f48a91f7be3f22eefa46f8181a9e41281941e25b019f57e0426261e  layer1-full-stages-vs-hf-old4090-hfround.json
2a8fdb1f96a1767625858677d739029b7f798b1cf746387355589fdfcd455870  layer1-full-stages-vs-hf-old4090-hfround.txt
db0378a324b59c66ddf17af2dc1d3f2a0e34f84f91ce709d44ce033681207a41  layer1-residual-drift-old4090-hfround.json
ee37a7c16d21d8435fc23f6bced17fb7028f594810933c45fe39f2202d3852c3  layer1-residual-drift-old4090-hfround.txt
2a485f6b7c027e8860af162db8c1a80f707cdecd7adc5cf13070638afc65885c  one-step-actual-vs-trace-old4090-hfround.json
208c244c2581884475c1f5af5f25fc1ea953df92d98553b7d02d1c57d03dc690  one-step-actual-vs-trace-old4090-hfround.txt
c64c4c1fa89cfcfb9310db76182f058aa989a199a128e6e4bef0614d9fbe2af9  prefill-layer-hidden-vs-trace-old4090-hfround.json
b5da888db80cc61ec626c42fa2b2b6004ae3f5b16bb9455d973969d2af7fa338  prefill-layer-hidden-vs-trace-old4090-hfround.txt
3ea92a9ddc31a2919500959c686e6c96442bdb27eeae23a8dbb8e6083520b4c2  higgs-layer32-stages-hf-old4090-hfround.safetensors
ea81cdee5981b5eae0d0139d0489147c0e48760c4ea500d703cf5c7eff486730  higgs-layer32-stages-old4090-hfround.safetensors
a4dc69eddbdb8c013208abf69a2ba7a75ef4e76f9d9feec28ccfd8e3b7df1b27  higgs-layer33-stages-hf-old4090-hfround.safetensors
e8d064989e649f16ac24baf3b01440ae068139f3c9942f9cf843fb25db93dce4  higgs-layer33-stages-old4090-hfround.safetensors
accb95a4f1e0f41faef1668303de89fa2a8f4fc509b024425b81f526ab43d46c  higgs-layer34-stages-hf-old4090-hfround.safetensors
e211c816ea9f1997e549647121436d6270e0da9fc993f28ca7b070fab75a57c7  higgs-layer34-stages-old4090-hfround.safetensors
da49a5b4bcf8d82f0c3c26bf3f845ddb195d8f04de495f074919a0702be7807d  higgs-layer35-stages-hf-old4090-hfround.safetensors
37dfd70384edd6346bc62ac06bddc5344d48182b2eb4eb2c36d5536da72b6517  higgs-layer35-stages-old4090-hfround.safetensors
f282328ca9401ed544ee5f8cba30874fa65d5f62e3feae8d277e522a7188d927  layer32-full-stages-vs-hf-old4090-hfround.json
37153b9032e8208bc0063793be0a13df9e113fb75523aba164cccd038fe314d8  layer32-full-stages-vs-hf-old4090-hfround.txt
59a1fd1f570a507c9a9c648a0295454d4b82a0d27dcd80b3f3845cc54eb20e86  layer33-full-stages-vs-hf-old4090-hfround.json
6f81bb1f00b839074d44d37cf37d4703504c14d736065cbb5dad9500f8d0c0b9  layer33-full-stages-vs-hf-old4090-hfround.txt
7b0a87cbafc93c4423085ecad7c0594ea8bd620877dd69ab09354e04ed045509  layer34-full-stages-vs-hf-old4090-hfround.json
7bb286da19e7aa49d46018a271d808a50d2dd9e0056af0ecdb05d2ae2d863ddb  layer34-full-stages-vs-hf-old4090-hfround.txt
e3f9607c1278f00ce9e9e03acb99230f867ac199a4e941923815ee13e86a49cd  layer35-full-stages-vs-hf-old4090-hfround.json
aecb76d3125d74008e8f2138e4ed4b54b0722f0e618953763f6b6eeef88d5fa4  layer35-full-stages-vs-hf-old4090-hfround.txt
8cab16b9bfeaaf643ae76bca873c502ecf37df5e49403220c0d13933f2a796a8  prefill-layer-hidden-vs-trace-old4090-hfround-skip-final-alias.json
6157e74e8eb350b5efbc812501d27c7df40edb3134a9d54fed711b2dc04d0192  prefill-layer-hidden-vs-trace-old4090-hfround-skip-final-alias.txt
```

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

## Current 4090 Parity Answer

As of the latest 4090-D evidence, the answer is deliberately split:

| Question | Current evidence | Claim status |
| --- | --- | --- |
| Does the PegaInfer/OpenInfer-style Higgs bridge run against the committed golden? | Yes. `run_higgs_one_step_cuda_gate.sh` produced a CUDA bf16 actual dump and the semantic comparator passed for both the one-shot path and retained prompt-session smoke. | **Proven for one-step semantic parity**, not strict tensor equality. |
| Does SGLang-Omni source produce the same committed golden? | Yes. `run_higgs_sglang_omni_source_gate.sh` imports the real Higgs tokenizer/head source modules, regenerates the reference, and strict-compares it to the committed fixture. | **Proven for source-reference parity**. |
| Has the full SGLang-Omni Higgs model/runtime produced the same output in this environment? | No. The readiness probe imports `text_tokenizer.py`, `modeling.py`, and `hf_config.py`, but `sglang_omni.models.higgs_tts.model` fails because the `sglang` runtime dependency is not installed. | **Not proven**. Do not claim runtime parity yet. |

The strongest safe public wording today is:

```text
Built a fail-closed Higgs-Audio one-step golden gate for PegaInfer/OpenInfer:
SGLang-Omni Higgs tokenizer/head source semantics strict-match the committed
golden, and the PegaInfer CUDA bf16 one-step bridge matches the same fixture at
the semantic level on RTX 4090-D.
```

The unsafe wording is:

```text
Matched SGLang-Omni runtime output end-to-end.
```

That has not been shown yet. To make that claim, the next evidence must come
from an isolated SGLang-Omni runtime environment that can import and execute the
full Higgs model path, dump the same one-step tensors, and compare them against
the same committed fixture or a newly documented runtime fixture.

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

## SGLang-Omni Source Gate

The branch now has a separate source-reference gate for the part of
SGLang-Omni that is practical to import without installing the full `sglang`
server package. It imports the real SGLang-Omni Higgs tokenizer and fused-head
modules directly from a source checkout, regenerates the one-step reference, and
strict-compares that output against the committed golden:

```bash
tools/higgs/run_higgs_sglang_omni_source_gate.sh \
  --model-dir /data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --sglang-omni-src /data/src/sglang-omni \
  --label a86082d
```

Validated on the 4090-D host:

```text
repo=/data/src/pegainfer
commit=a86082d5
label=a86082d
sglang_omni_src=/data/src/sglang-omni
sglang_omni_commit=c6980be8
reference=/data/results/pegainfer/higgs-audio/actual/higgs-one-step-sglang-omni-src-reference-a86082d.safetensors
compare_log=/data/results/pegainfer/higgs-audio/actual/sglang-omni-src-reference-compare-a86082d.txt
readiness_log=/data/results/pegainfer/higgs-audio/actual/sglang-omni-import-readiness-a86082d.txt
sglang_omni_direct_imports=ok
sglang_omni_full_model_import=missing_sglang
source_reference_strict_comparison=ok
artifacts_nonempty=ok
```

The script writes a key-value summary and validates it with
`tools/higgs/check_higgs_sglang_omni_source_gate_summary.py`, including required
keys, strict-compare/readiness markers, expected source commit, and non-empty
artifact paths.

The generated reference metadata records:

```text
sglang_omni_source_dir=/data/src/sglang-omni
sglang_omni_source_commit=c6980be8
sglang_omni_direct_imports=text_tokenizer.py;modeling.py
sglang_omni_full_model_imported=false
```

This closes the prompt/head-source gap for the one-step fixture. It is still not
a full SGLang-Omni runtime parity result because `sglang_omni.models.higgs_tts.model`
depends on the uninstalled `sglang` package in the current 4090 environment.

## SGLang-Omni Runtime Readiness Probe

The branch also includes a lightweight import probe to keep the runtime parity
claim auditable:

```bash
tools/higgs/check_higgs_sglang_omni_imports.py \
  --sglang-omni-src /data/src/sglang-omni \
  --require-direct
```

The current 4090-D environment reports:

```text
sglang_omni_src=/data/src/sglang-omni
sglang_omni_commit=c6980be8
module.sglang_omni.models.higgs_tts.text_tokenizer=ok
module.sglang_omni.models.higgs_tts.modeling=ok
module.sglang_omni.models.higgs_tts.hf_config=ok
module.sglang_omni.models.higgs_tts.model=fail
module.sglang_omni.models.higgs_tts.model.reason=missing_sglang
direct_higgs_imports=ok
full_higgs_model_import=missing_sglang
```

This means the one-step golden can be strict-checked against SGLang-Omni's Higgs
source modules, but a full SGLang-Omni model/runtime comparison still needs an
isolated environment with the `sglang` serving dependency installed.

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
bbaae8018759b7e8f26d2acfb1aefb5bee3e5099d47573bb4ce3c980b6096684
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
single fused Higgs audio head. It can now produce the Qwen3 requested-name to
Higgs stored-name alias map consumed by the runtime bridge, so the body weights
can be read from the original checkpoint without materializing a renamed copy.

## Kernel Plan

`pegainfer-higgs-audio` now exports `kernel_plan()`, matching the review style
used by `pegainfer-qwen3-4b`. The descriptor is deliberately scoped to the
runtime surface that exists today:

- `artifact`: checkpoint header validation and Qwen3 tensor-alias planning.
- `prefill`: Qwen3-backed Higgs body prefill, CUDA bf16 fused audio head, and
  CPU top-k/argmax extraction for the diagnostic one-step gate.
- `golden`: strict tensor comparison and semantic runtime comparison.

The plan records that `qwen3_body_prefill` is served by the existing Qwen3
runtime (`CUDA + cuBLAS + FlashInfer`) through tensor-name aliases, while
`fused_audio_head` is a CUDA bf16 linear over
`tied.embedding.modality_embeddings.0.embedding.weight`. It intentionally does
not claim a Higgs decode/KV-cache phase yet; that belongs to the next runtime
slice once prefill/decode continuation is owned by the Higgs crate.

The one-step audio head now has a reusable `OneStepAudioPrediction` boundary:
CPU and CUDA bf16 paths both compute logits/top-k/argmax first, and the
safetensors writer consumes that prediction as a separate step. This keeps the
golden dump path intact while making the next runtime slice less file-output
centric.

`HiggsAudioRuntime` now exposes `prefill_audio_from_prompt_ids`, which runs a
raw token prompt through the aliased Qwen3 body and returns
`HiggsAudioPrefill { prompt_tokens, final_hidden_bf16, audio }`. The golden
dump path loads prompt tensors from the fixture and then calls this runtime API;
it is no longer the only way to obtain Higgs audio logits from the bridge. This
is still a one-shot diagnostic prefill path, not a retained KV-cache session.

The runtime prompt surface is still intentionally single-prompt: `PromptTensors::prompt_ids`
rejects multi-row `prompt.lengths`, non-binary attention masks, and mask sums that
do not match the recorded prompt length. Wider fixtures should use the comparator
schema first, then add an explicitly batched runtime surface instead of slipping
through the one-step path.

The next bridge slice adds a Higgs-owned prompt session surface:
`prefill_prompt_session(HiggsPromptSession::new(id), prompt_ids)`. It uses a new
Qwen3 `prefill_last_hidden_bf16_retained_prompt` entrypoint internally and
commits prompt KV with `max_output_tokens = 0`, so no generated text token is
registered. This is intentionally narrower than full decode continuation: it
proves Higgs can own a prompt KV lifecycle without exposing Qwen3's `RequestId`
as the primary API, while avoiding the incorrect shortcut of feeding Higgs
audio-codebook ids into Qwen3's text-token decode state.

## Qwen3 Runtime Bridge

The sixth slice originally added a bridge materializer that rewrites the single
Higgs safetensors shard into a Qwen3-compatible view:

```bash
cargo run -p pegainfer-higgs-audio --bin higgs_materialize_qwen3_body -- \
  --model-dir /data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --out-dir /data/results/pegainfer/higgs-audio/qwen3-body-view
```

That fallback view contains 398 BF16 tensors, excludes the fused Higgs audio
head, and duplicates the 7.5 GiB body payload. The safetensors header is padded
so the payload start is aligned for `bf16`; without this, debug Rust aborts
inside `DeviceMatrix::from_safetensors` when it casts tensor payload bytes.

The preferred bridge now avoids the payload copy. It writes only a Qwen3 config
view plus a tensor-alias manifest, then asks the Qwen3 runtime to load requested
`model.*` tensors from their stored Higgs names:

```bash
cargo run --release -p pegainfer-higgs-audio --bin higgs_materialize_qwen3_body -- \
  --model-dir /data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --out-dir /data/results/pegainfer/higgs-audio/qwen3-config-view-alias \
  --metadata-only
```

The alias map has 398 entries and keeps the fused Higgs audio head outside the
Qwen3 body loader:

- `model.embed_tokens.weight` -> `tied.embedding.text_embedding.weight`
- `model.norm.weight` -> `body.norm.weight`
- `model.layers.N.*` -> `body.layers.N.*`

`higgs_dump_one_step_actual` now prepares this config-only alias view
automatically next to `--out` when neither `--qwen3-config-dir` nor
`--qwen3-body-dir` is provided. `--qwen3-config-dir` remains available for
reusing a prebuilt alias view, and `--qwen3-body-dir` remains available for
fallback and bisecting.

## Comparison Gate

The third slice defines the actual runtime parity contract:

```bash
cargo run -p pegainfer-higgs-audio --bin higgs_compare_one_step -- \
  --mode strict \
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

The comparator also has a semantic mode for bf16 runtime bring-up:

```bash
cargo run -p pegainfer-higgs-audio --bin higgs_compare_one_step -- \
  --mode semantic \
  --golden test_data/higgs-one-step-audio-logits.safetensors \
  --actual /path/to/pegainfer-higgs-one-step-actual.safetensors
```

Semantic mode still prints the strict tensor drift report, but it gates on:

- prompt tensors exact
- `audio_argmax.ids` exact
- hidden cosine at least `0.9998`
- logits cosine at least `0.99999`
- max golden-logit regret for actual argmax at most `0.20`
- minimum top-64 overlap per Higgs codebook at least `40`

This is a runtime smoke/parity gate, not a replacement for strict golden parity.
It is useful because exact top-64 ordering is brittle under small bf16 drift,
while argmax stability, regret, cosine, and overlap expose the semantic behavior
more directly.

When workspace-level git dependencies block the full root workspace on remote
hosts, run the Higgs-only isolated gate:

```bash
tools/higgs/run_isolated_higgs_checks.sh
```

That script builds a temporary one-member workspace containing only
`pegainfer-higgs-audio` and the committed fixture, compiles the Higgs gate
summary validators, checks synthetic CUDA/source-gate summaries, then runs fmt,
unit tests, and `higgs_compare_one_step` self-comparison. It is a workaround for
dependency isolation only; it does not replace full workspace CI.

## 4090 Bring-Up Notes

The 4090-D host at `/data/src/pegainfer` was synchronized through fork commit
`a111fb0` on branch `feat/higgs-audio-one-step-golden`.

Static model/golden validation passed on the 4090 host with the Python reference
environment:

```text
golden_sha256 bbaae8018759b7e8f26d2acfb1aefb5bee3e5099d47573bb4ce3c980b6096684
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
isolated Higgs tests: 21 passed
higgs_compare_one_step self-comparison: ok
higgs_artifact_check: ok
checkpoint headers: files=1 tensors=399 bf16=399
runtime load plan: tensors=399 shard_files=1 bf16_mib=7712 qwen3_backbone=398 higgs_head=1
```

The earlier Qwen3 body view was materialized from the real checkpoint on the
4090 host as a fallback bridge:

```text
out_dir: /data/results/pegainfer/higgs-audio/qwen3-body-view
tensors: 398
payload_mib: 7672
model.safetensors size: 8044982042 bytes
header_len: 45842
data_start_mod2: 0
```

The preferred config-only alias view was then generated without copying the
7.5 GiB payload:

```text
out_dir: /data/results/pegainfer/higgs-audio/qwen3-config-view-alias
config.json: 306 bytes
generation_config.json: 29 bytes
higgs-qwen3-tensor-aliases.json: 35 KiB
aliases: 398
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

An NSYS profile for the Higgs-owned one-step actual dump was captured after the
auto alias-view path landed:

```text
/data/results/pegainfer/higgs-audio/profiles/higgs-one-step-actual-auto-a514447.nsys-rep
/data/results/pegainfer/higgs-audio/profiles/higgs-one-step-actual-auto-a514447-stats_cuda_gpu_kern_sum.csv
profile size: 144 KiB
profiled actual: /data/results/pegainfer/higgs-audio/actual/higgs-one-step-actual-cuda-bf16-auto-a514447-profiled.safetensors
profiled actual size: 44 KiB
gate semantic comparison: ok
```

The `nsys stats --report cuda_gpu_kern_sum` summary shows the one-step path is
dominated by bf16 projection GEMMs, with FlashInfer prefill and small custom
kernels contributing much less GPU kernel time:

```text
CUTLASS bf16 GEMM 16x16x128x2: 53.8% GPU kernel time, 180 launches
CUTLASS bf16 GEMM 16x16x128x1: 40.5% GPU kernel time, 72 launches
FlashInfer BatchPrefillWithPagedKVCacheKernel: 1.6%, 36 launches
prefill_qk_norm_rope_kernel: 0.7%, 36 launches
FusedAddRMSNormRoundKernel: 0.7%, 36 launches
FlashInfer RMSNormKernel: 0.7%, 37 launches
AppendPagedKVCacheKernel: 0.6%, 36 launches
silu_mul_kernel: 0.5%, 36 launches
```

## Runtime Actual Dump

The current branch also has a Higgs-owned one-step runtime bridge over the Qwen3 executor.
The default actual-dump path no longer requires users to pass a Qwen3 view path;
it writes a small config-only alias view next to `--out` and loads the original
Higgs checkpoint payload through tensor aliases:

The preferred CUDA repro entrypoint is:

```bash
tools/higgs/run_higgs_one_step_cuda_gate.sh \
  --model-dir /data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --label d2dcb92
```

That script runs the `runtime-qwen3` bin check, dumps the CUDA bf16 actual file
through the auto alias-view path, runs the semantic comparator, smoke-tests the
prompt-only retained KV session path, compares that session actual against the
same golden, asserts the persisted gate markers, verifies the generated files are
non-empty, writes a key-value gate summary, and records the small generated Qwen3
config view. It then parses that summary with
`tools/higgs/check_higgs_gate_summary.py` to verify the required keys, `ok`
markers, SM/NVCC settings, and non-empty artifact paths. Add `--profile` to
capture an NSYS report for the same actual-dump path.

The script was validated on the 4090-D host at `a514447` after the retained
prompt-session bridge, `HiggsAudioRuntime` API surface, duplicate
request-id guard, persisted session-smoke log, explicit persisted-marker
assertions, non-empty artifact assertions, and key-value gate summary landed,
producing the complete gate artifact set:

```text
actual:      /data/results/pegainfer/higgs-audio/actual/higgs-one-step-actual-cuda-bf16-auto-a514447.safetensors
session:     /data/results/pegainfer/higgs-audio/actual/higgs-one-step-session-cuda-bf16-auto-a514447.safetensors
compare_log: /data/results/pegainfer/higgs-audio/actual/semantic-compare-auto-a514447.txt
smoke_log:   /data/results/pegainfer/higgs-audio/actual/higgs-prompt-session-smoke-a514447.txt
session_log: /data/results/pegainfer/higgs-audio/actual/semantic-compare-session-auto-a514447.txt
summary:     /data/results/pegainfer/higgs-audio/actual/higgs-one-step-cuda-gate-a514447.txt
semantic comparison: ok
session semantic comparison: ok
duplicate_request_id_guard: ok
artifacts_nonempty: ok
auto view:
  config.json 306 bytes
  generation_config.json 29 bytes
  higgs-qwen3-tensor-aliases.json 34933 bytes
```

The persisted key-value summary for that run is:

```text
status=ok
repo=/data/src/pegainfer
commit=a514447d
label=a514447
model_dir=/data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561
golden=/data/src/pegainfer/test_data/higgs-one-step-audio-logits.safetensors
sm=89
nvcc_jobs=8
actual=/data/results/pegainfer/higgs-audio/actual/higgs-one-step-actual-cuda-bf16-auto-a514447.safetensors
session_actual=/data/results/pegainfer/higgs-audio/actual/higgs-one-step-session-cuda-bf16-auto-a514447.safetensors
compare_log=/data/results/pegainfer/higgs-audio/actual/semantic-compare-auto-a514447.txt
session_smoke_log=/data/results/pegainfer/higgs-audio/actual/higgs-prompt-session-smoke-a514447.txt
session_compare_log=/data/results/pegainfer/higgs-audio/actual/semantic-compare-session-auto-a514447.txt
auto_view=/data/results/pegainfer/higgs-audio/actual/higgs-qwen3-config-view
semantic_comparison=ok
session_semantic_comparison=ok
duplicate_request_id_guard=ok
artifacts_nonempty=ok
```

The summary checker also passed on the 4090-D run:

```text
higgs gate summary: ok commit=a514447d label=a514447 sm=89
```

After the Higgs-owned `HiggsPromptSession` handle replaced Qwen3 `RequestId` as
the primary prompt-session API, the full CUDA gate was rerun on the same 4090-D
host at `d2dcb92`:

```text
status=ok
commit=d2dcb922
label=d2dcb92
actual=/data/results/pegainfer/higgs-audio/actual/higgs-one-step-actual-cuda-bf16-auto-d2dcb92.safetensors
session_actual=/data/results/pegainfer/higgs-audio/actual/higgs-one-step-session-cuda-bf16-auto-d2dcb92.safetensors
semantic_comparison=ok
session_semantic_comparison=ok
duplicate_request_id_guard=ok
artifacts_nonempty=ok
higgs gate summary: ok commit=d2dcb922 label=d2dcb92 sm=89
```

The session smoke retains the prompt KV under Higgs session id `1`, emits the
same one-step audio prediction, and drops the session explicitly. Internally this
still maps to the backing Qwen3 request id, but the primary runtime surface is now
a Higgs-owned prompt-session lifecycle and does not register an invalid generated
text token.

The script was validated on the 4090-D host at `7d5ab1d` after the prompt-id
prefill bridge split and produced:

```text
actual:      /data/results/pegainfer/higgs-audio/actual/higgs-one-step-actual-cuda-bf16-auto-7d5ab1d.safetensors
compare_log: /data/results/pegainfer/higgs-audio/actual/semantic-compare-auto-7d5ab1d.txt
semantic comparison: ok
auto view:
  config.json 306 bytes
  generation_config.json 29 bytes
  higgs-qwen3-tensor-aliases.json 34933 bytes
```

The underlying actual-dump command remains:

```bash
PEGAINFER_CUDA_SM=89 PEGAINFER_NVCC_JOBS=8 \
cargo run --release -p pegainfer-higgs-audio --features runtime-qwen3 \
  --bin higgs_dump_one_step_actual -- \
  --model-dir /data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --golden /data/src/pegainfer/test_data/higgs-one-step-audio-logits.safetensors \
  --out /data/results/pegainfer/higgs-audio/actual/higgs-one-step-actual-cuda-bf16-auto-a111fb0.safetensors
```

The 4090 run at `a111fb0` produced the expected actual file and automatic config
view:

```text
higgs one-step actual dump: ok
  out: /data/results/pegainfer/higgs-audio/actual/higgs-one-step-actual-cuda-bf16-auto-a111fb0.safetensors
  audio_head_backend: CudaBf16
  prompt_tokens: 10
  hidden_values: 2560
  audio_logits: 8208
/data/results/pegainfer/higgs-audio/actual/higgs-one-step-actual-cuda-bf16-auto-a111fb0.safetensors: 44 KiB
auto config view:
  config.json: 306 bytes
  generation_config.json: 29 bytes
  higgs-qwen3-tensor-aliases.json: 34933 bytes
```

Strict actual-vs-golden comparison does **not** pass yet:

```text
prompt.input_ids_padded          pass=true
prompt.attention_mask            pass=true
prompt.lengths                   pass=true
final_hidden.bf16                pass=false max_abs=0.500000 mean_abs=0.007480 p99_abs=0.062500
audio_logits.f32                 pass=false max_abs=1.000000 mean_abs=0.112908 p99_abs=0.500000
audio_top64.ids                  pass=false exact_mismatch=426
audio_top64.logprobs.f32         pass=false mean_abs=0.089786 p99_abs=0.500000
audio_argmax.ids                 pass=true
```

The useful interpretation is narrower than "pass" but still strong:

- Prompt tensors are exact, so the runtime is replaying the intended Higgs
  one-step prompt.
- `final_hidden.bf16` has high directional agreement with the Transformers
  golden (`cos=0.999990642`), but the absolute drift is larger than the current
  hidden tolerance.
- All 8 audio argmax ids match. The top-1 audio code for every codebook is
  stable even though top-64 ordering is tie/noise sensitive.
- Top-64 overlap has minimum `49` and mean `55.75`; exact
  top-64 id equality is too brittle for the current bf16 runtime path.

A diagnostic script captures these checks and the audio-head dtype attribution:

```bash
tools/accuracy/analyze_higgs_one_step_actual.py \
  --model-dir /data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --golden /data/results/pegainfer/higgs-audio/golden/higgs-one-step-golden-18137c4.safetensors \
  --actual /data/results/pegainfer/higgs-audio/actual/higgs-one-step-actual-cuda-bf16-auto-a111fb0.safetensors
```

The corrected RTX 4090 D run should be read as a semantic-pass, strict-fail
boundary:

```text
final_hidden.bf16: max=0.500000 mean=0.007480 p99=0.062500 rmse=0.020339 cosine=0.999990642
audio_logits.f32:  max=1.000000 mean=0.112908 p99=0.500000 rmse=0.224297 cosine=0.999997616
audio_argmax.ids:  exact=true for all 8 codebooks
top64_overlap:     min=49 mean=55.75
```

This proves the golden audio head is CUDA bf16 `F.linear`, and the Rust actual
writer now defaults to the same CUDA bf16 audio-head contract. The older CPU fp32
fallback remains available as a diagnostic backend, but it is no longer the
default actual path. The remaining strict drift is accumulated bf16/runtime
numerical drift across the Qwen3 body, not a prompt, embedding, or RoPE reference
bug.

The semantic comparison mode is expected to pass on this CUDA bf16 actual dump:

```text
higgs one-step strict comparison: passed=false diagnostic_only=true
higgs one-step semantic comparison:
  prompt_exact=true argmax_exact=true hidden_cosine=0.999990821 hidden_cosine_min=0.999800026
  logits_cosine=0.999997616 logits_cosine_min=0.999989986 max_argmax_regret=0.000000 argmax_regret_tol=0.200000
  top64_min_overlap=49 top64_mean_overlap=55.75 top64_min_overlap_tol=40
higgs one-step semantic comparison: ok
```

## Layer Drift Diagnostic

The branch now includes a layer-hidden diagnostic loop for the strict drift root
cause:

```bash
/data/venvs/ai-infra/bin/python tools/accuracy/dump_higgs_layer_hidden_golden.py \
  --snapshot-dir /data/models/higgs-audio/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --out /data/results/pegainfer/higgs-audio/layer-drift/higgs-layer-hidden-golden.safetensors

PEGAINFER_CUDA_SM=89 PEGAINFER_NVCC_JOBS=8 \
cargo run --release -p pegainfer-higgs-audio --features runtime-qwen3 \
  --bin higgs_dump_prefill_layer_hidden -- \
  --qwen3-body-dir /data/results/pegainfer/higgs-audio/qwen3-body-view \
  --golden /data/results/pegainfer/higgs-audio/golden/higgs-one-step-golden-18137c4.safetensors \
  --out /data/results/pegainfer/higgs-audio/layer-drift/higgs-layer-hidden-actual.safetensors

/data/venvs/ai-infra/bin/python tools/accuracy/compare_higgs_layer_hidden.py \
  --golden /data/results/pegainfer/higgs-audio/layer-drift/higgs-layer-hidden-golden.safetensors \
  --actual /data/results/pegainfer/higgs-audio/layer-drift/higgs-layer-hidden-actual.safetensors
```

The 4090 run produced this key attribution:

```text
prompt_exact=True
embedding.last_hidden.bf16   max_abs=0.000000 mean_abs=0.000000 p99_abs=0.000000 rmse=0.000000 cosine=1.000000358
layer.00.last_hidden.bf16    max_abs=0.250000 mean_abs=0.002617 p99_abs=0.031250 rmse=0.007614 cosine=0.999993920
layer.01.last_hidden.bf16    max_abs=1.000000 mean_abs=0.004883 p99_abs=0.031250 rmse=0.021483 cosine=0.999992847
layer.14.last_hidden.bf16    max_abs=4.000000 mean_abs=0.028061 p99_abs=0.125000 rmse=0.095240 cosine=0.999987006
layer.35.last_hidden.bf16    max_abs=32.000000 mean_abs=1.028503 p99_abs=4.000000 rmse=1.772655 cosine=0.999993205
final_hidden.bf16            max_abs=0.500000 mean_abs=0.007480 p99_abs=0.062500 rmse=0.020339 cosine=0.999990642
summary:
  first_mean_abs_gt_0.003000=layer.01.last_hidden.bf16
  first_cosine_lt_0.999800000=none
  worst_mean_abs=layer.35.last_hidden.bf16:1.028503
  worst_cosine=layer.14.last_hidden.bf16:0.999987006
```

This rules out prompt construction, tokenizer ids, embedding load, and Qwen3 body
view aliasing as the source of the remaining strict drift. It also shows that
the corrected golden reduces layer 0 from a suspicious failure boundary to a
within-tolerance stage: mean absolute drift stays below `0.003`, and no compared
layer drops below cosine `0.9998`.

The layer-0 stage dump further rules out the earlier RoPE suspicion:

```text
layer0.q_norm_rope.bf16      mean_abs=0.001261 cosine=0.999998450
layer0.k_norm_rope.bf16      mean_abs=0.001323 cosine=0.999999285
layer0.output_hidden.bf16    mean_abs=0.002617 cosine=0.999993920
summary:
  compared=17
  first_mean_abs_gt_0.003000=none
  first_cosine_lt_0.999800000=none
```

The earlier RoPE divergence was a false-positive in the HuggingFace golden
diagnostic: `Qwen3Model` was created on the meta device and then `to_empty()` was
used, which left non-persistent rotary buffers uninitialized. The generator now
rebuilds `Qwen3RotaryEmbedding` on the real device after `to_empty()`, before
loading the Higgs body weights.

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

## Contribution Summary

This branch now has a concrete Higgs-Audio bring-up milestone suitable for an
upstream issue update:

- Built a one-step Higgs-Audio golden pipeline with prompt tensors, final hidden,
  fused 8-codebook audio logits, top-64 ids/logprobs, and argmax ids.
- Added a Higgs-owned one-step runtime bridge over the Qwen3 executor, backed by
  tensor-name aliases so Higgs body weights load directly from the original
  checkpoint without a 7.5 GiB renamed copy.
- Added a Higgs kernel plan descriptor covering artifact, prefill, and golden
  phases so reviewers can see the current backend/runtime boundary explicitly.
- Split one-step audio prediction from safetensors writing so future decode or
  serving paths can reuse logits/top-k/argmax without going through a dump file.
- Added a prompt-id runtime bridge entrypoint that returns Higgs prefill hidden
  state plus audio prediction before any golden-file writer is involved.
- Added a prompt-only retained KV bridge for Higgs sessions, backed by Qwen3's
  paged KV lifecycle and explicit request-id drop.
- Identified and fixed a HuggingFace/meta-device RoPE buffer bug in the golden
  loader; the suspected CUDA RoPE failure was a false-positive.
- Added strict and semantic comparison modes. Corrected 4090 run passes semantic
  parity with exact 8-codebook argmax, zero argmax regret, hidden cosine
  `0.999990821`, and logits cosine `0.999997616`.
- Kept the committed one-prompt fixture contract strict while allowing the
  comparator schema to validate same-shape multi-prompt golden/actual pairs,
  including prompt length and attention-mask consistency; the one-step runtime
  prompt API remains explicitly single-prompt.
- Added layer-hidden and layer-0 stage diagnostics. Layer 0 is now within the
  mean drift threshold, while strict parity remains open because bf16/runtime
  drift accumulates across 36 Qwen3 layers.

## Technical Debt

- The branch commits a derived fixture from a research/non-commercial model. This
  is acceptable for a fork validation branch, but upstream needs an explicit
  maintainer decision before merging the fixture.
- The generator uses HuggingFace Qwen3 for the backbone. Prompt/head logic is now
  strict-checked against directly imported SGLang-Omni `text_tokenizer.py` and
  `modeling.py`, but a later gate should still compare against a pinned full
  SGLang-Omni execution path once the `sglang` runtime package is practical to
  run in the 4090 environment.
- The Rust crate does not yet have a fully Higgs-owned GPU loader. It locks the
  golden/artifact contract and can dump an actual one-step file through the
  existing Qwen3 runtime bridge, now backed by tensor-name aliases instead of a
  copied body-view payload.
- The committed fixture covers one prompt. Wider prompt-length coverage belongs
  in the next parity slice after loader/backbone code exists; the comparator
  schema is already separated from the fixed fixture contract so same-shape
  multi-prompt golden/actual files can be compared without changing the
  committed one-prompt artifact check. It also checks prompt lengths and binary
  attention-mask sums before comparing logits.
- The Qwen3 body smoke and auto alias-backed actual dump prove that Higgs `body.*`
  tensors can be loaded and executed by the existing Qwen3 runtime without
  rewriting the checkpoint payload. The actual dump now uses the real golden
  prompt and fused Higgs audio head, but strict hidden/logit/top-64 parity is
  not yet proven.
- Nsight Compute counters are blocked on the current 4090 host because
  `RmProfilingAdminOnly=1`; NSYS works.
- Remote Rust execution can still be blocked by workspace-level git dependencies
  such as `vllm-project/vllm.git`; `gh-proxy.com` is usable for `ls-remote` on
  the 4090 host, but full Cargo fetch still needs either a prewarmed cache or
  workspace dependency isolation.

## Next Execution Slice

1. Decide whether upstream wants semantic parity as the first Higgs-Audio gate,
   or whether strict hidden/logit/top-64 parity is required before review.
2. If strict parity is required, add deeper stage probes at later layers
   (`14`, `32`, `35`) where accumulated drift is largest, instead of continuing
   to focus on layer 0.
3. Replace the one-step bridge with a fuller Higgs-owned runtime surface:
   prefill/decode continuation, KV-cache ownership, and multi-prompt fixtures.
4. Broaden the fixture beyond one prompt: longer text, multiple prompt lengths,
   delay-pattern coverage, and at least one decode/KV-cache continuation.
5. After correctness gates are accepted, profile the Higgs-owned runtime path
   with NSYS; use NCU only on hosts where NVIDIA performance counters are
   unlocked.

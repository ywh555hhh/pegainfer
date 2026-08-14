# Higgs-Audio Native Codegen PR Notes

TL;DR: Use this as the PR body skeleton for the native incremental
audio-code-generation slice. It keeps the maintainer reading path short:
scope, design boundary, validation evidence, artifacts, limitations, and the
one architectural tradeoff that needs review. Fill the 4090 table from the
generated `pr-evidence-<label>.md`; do not upgrade the claim boundary without
the matching runtime trace/profile evidence.

Last touched: 2026-08

## Proposed Scope

This PR introduces a feature-gated Higgs-Audio bring-up path for native
incremental audio-code generation diagnostics.

Included:

- Higgs-Audio model-line registration behind `--features higgs-audio`.
- Fail-closed launch preflight for `higgs_multimodal_qwen3` configs.
- Higgs-owned code-generation state: delay pattern, feedback embedding, audio
  head projection, sampled rows, raw codec rows, and trace JSON.
- A retained-KV continuation contract:
  `feedback_embedding -> final_normed_hidden -> audio logits -> sampled row`.
- A narrow Qwen3 diagnostic surface that consumes an already materialized input
  embedding and returns the final normed hidden before text `lm_head`.
- Reference/golden/profiling tools for official/HF incremental trace generation,
  semantic trace comparison, `nsys`/`ncu` capture, and PR evidence rendering.

Not included:

- Native wav E2E.
- Native codec/vocoder runtime.
- Production serving.
- Strict trace parity unless the 4090 comparison table proves it.
- Shared `pegainfer-core` or shared `pegainfer-kernels` semantic changes.

## Design Boundary

Higgs-Audio owns the multimodal logic after final hidden:

- feedback embedding from the previous delayed code row;
- fused audio head projection;
- sampling/top-k diagnostics;
- delay-pattern update and raw-code recovery;
- codec-input JSON and trace rows.

Qwen3 owns the text-backbone body execution. The PR intentionally adds only a
narrow diagnostic API:

```text
retained RequestKv
  + caller-owned BF16 input embedding
  -> one retained decode body step
  -> final normed hidden before text lm_head
```

The alternative is copying the Qwen3 decode body into `pegainfer-higgs-audio`.
That would keep the diff more model-local, but would duplicate KV metadata,
numeric policy, CUDA graph assumptions, LoRA hooks, and future Qwen3 decode
fixes. The current shape keeps one owner for the Qwen3 body and keeps Higgs
multimodal semantics local.

This is the main maintainer-review question:

> Is this narrow Qwen3 diagnostic API acceptable for Higgs-Audio bring-up, or
> would the project prefer a Higgs-local copied executor slice despite the
> maintenance cost?

## Validation Evidence

Paste the generated 4090 evidence section here:

```bash
python3 tools/higgs/render_higgs_native_codegen_pr_evidence.py \
  /data/results/pegainfer/higgs-audio/native-codegen/higgs-native-codegen-contract-gate-<label>.txt \
  --out /data/results/pegainfer/higgs-audio/native-codegen/pr-evidence-<label>.md
```

Minimum expected gates before claiming native incremental code generation:

| Gate | Required evidence |
| --- | --- |
| Server feature check | `cargo check --release -p pegainfer-server --features higgs-audio` on Linux/4090. |
| Runtime retained bridge | `cargo test --release -p pegainfer-higgs-audio --features runtime-qwen3 --lib runtime_bridge -- --nocapture`. |
| Native continuation smoke | `higgs_native_continuation_smoke` writes a 9-step trace: one prompt seed plus eight retained continuation steps. |
| HF reference trace | `higgs-reference-incremental-trace-<label>.json` passes `check_higgs_incremental_trace_reference.py` and contains full logits/top-k evidence. |
| Semantic trace compare | `higgs_compare_codegen_trace` reports first divergent step, argmax agreement, logits cosine, max/mean/p99 drift, argmax regret, and top-k overlap. |
| Forced common-prefix trace | `higgs_native_forced_prefix_smoke` consumes sampled rows from the HF trace while still running native retained-KV body/audio-head logits. This separates feedback-path divergence from strict numeric/top-k drift. |
| Profiling | `profiles/higgs-native-codegen-<label>.nsys-rep` and `profiles/higgs-native-codegen-ncu-<label>.ncu-rep` exist, or missing tools are called out as blockers. |
| PR evidence | `pr-evidence-<label>.md` generated from the actual gate summary. |

Current 4090 evidence from labels `946d0e12-4090-debug5` and
`946d0e12-4090-debug6` supports a conservative claim:

- Native retained continuation executes on RTX 4090 and emits a 9-step trace.
- Free-running native trace diverges from the HF incremental trace at step 4:
  cosine `0.99989086`, argmax agreement `51/72`.
- Forced common-prefix replay makes sampled/raw rows exact and improves logits
  cosine to `0.99999833`, argmax agreement to `64/72`, with remaining mismatches
  dominated by tied reference logits.
- `nsys` profiling was captured; `ncu` was blocked by host performance-counter
  permissions (`ERR_NVGPUCTRPERM`).

Do not present this as strict trace parity. The useful maintainer discussion is
whether tied-logit argmax/top-k strictness should block this diagnostic slice,
or whether the current common-prefix evidence is enough for the foundation PR
claim boundary.

## Reproduction Commands

On the 4090 host:

```bash
export PEGAINFER_CUDA_SM=89
export PEGAINFER_NVCC_JOBS=8

cargo check --release -p pegainfer-server --features higgs-audio
cargo test --release -p pegainfer-higgs-audio --features runtime-qwen3 --lib runtime_bridge -- --nocapture

tools/higgs/run_higgs_4090_native_codegen_validation.sh \
  --result-root /data/results/pegainfer/higgs-audio \
  --label "$(git rev-parse --short HEAD)" \
  --model-dir /path/to/higgs \
  --golden /path/to/higgs-one-step-golden.safetensors \
  --generate-reference-trace \
  --profile
```

For a temporary remote checkout that cannot receive the full dirty worktree,
prepare and transfer an overlay from the local machine:

```bash
tools/higgs/prepare_higgs_4090_overlay.sh --out-dir /tmp --label "$(git rev-parse --short HEAD)-4090"
```

## Claim Boundary

Use the lowest claim supported by the rendered evidence:

| Evidence state | Claim |
| --- | --- |
| Local contract/artifact checks only | Bring-up scaffolding; no native retained decode claim. |
| 4090 runtime bridge + native smoke pass, no reference compare | Native retained-continuation execution, semantic parity unproven. |
| 4090 native trace compared against full-logits HF incremental trace | Native incremental audio-code generation diagnostic path, with the measured semantic-parity limits. |
| 4090 trace compare plus `nsys`/`ncu` artifacts | Native incremental audio-code generation diagnostic path with correctness and profiling evidence. |

Do not claim native audio E2E until native model execution, retained-KV decode,
codec/vocoder, and serving lifecycle are all connected and exercised without a
Python runtime dependency.

## PR Checklist

- [ ] Rebase on upstream `main`.
- [ ] Confirm dirty `pegainfer-kernels/third_party/*` submodules are not part of
  the PR.
- [ ] Paste the generated 4090 evidence table.
- [ ] Keep the generated `Supported claim` line unchanged unless the underlying
  gate summary changes.
- [ ] Keep Qwen3 changes described as a diagnostic surface.
- [ ] Mention that Python tools are reference/golden/profiling helpers only.
- [ ] Keep open questions to the Qwen3 diagnostic API tradeoff and any measured
  trace drift that needs maintainer input.

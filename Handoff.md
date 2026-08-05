# Higgs Audio Handoff

Generated: 2026-08-05

This document records the current verified state of the Higgs Audio work in `openinfer-project/pegainfer` and the user's fork `ywh555hhh/pegainfer`. It is intentionally fact-based. Where a statement comes only from an issue or PR description, it is labeled as such and should be revalidated before implementation decisions depend on it.

## Suggested Skills

- `triage`: use before posting issue or PR comments, changing labels, or moving stale items through the tracker.
- `ai-inference-infra`: use for the backbone, KV-cache, decode-loop, sampling, and correctness-gate work.
- `autodl`: use when setting up the RTX 5090 machine, CUDA toolchain, model cache, and long-running validation sessions.
- `diagnose`: use when a correctness gate, CUDA build, or runtime parity check fails.

## Verified Repository State

- Fork: `ywh555hhh/pegainfer`.
- Parent repository: `openinfer-project/pegainfer`.
- Fork default branch: `main`.
- Fork `main` commit from GitHub branch list: `927c7c12596c51b2f02c61069f31eeaba312da40`.
- Upstream `main` commit used for this handoff branch: `57ee51f71e45b3745050fd8c39c6dd88b4fbde5e`.
- GitHub compare result for `openinfer-project/pegainfer:main...ywh555hhh:main`: `status=behind`, `ahead_by=0`, `behind_by=355`.
- GitHub compare result for `openinfer-project/pegainfer:main...ywh555hhh:feat/higgs-audio-v1`: `status=diverged`, `ahead_by=6`, `behind_by=290`.
- Existing fork branch for the Higgs work: `feat/higgs-audio-v1` at `21abb93f257ad29c619c774dea0910b3fb4d9cd1`.
- This handoff branch is separate from the Higgs implementation branch: `draft/higgs-audio-handoff-20260805`.

## Verified PR State

- PR: `openinfer-project/pegainfer#409`.
- URL: `https://github.com/openinfer-project/pegainfer/pull/409`.
- Title: `feat(higgs-audio): crate skeleton + config/weights + backbone forward + parity gate`.
- Author: `ywh555hhh`.
- Head: `ywh555hhh/pegainfer:feat/higgs-audio-v1`.
- Base: `openinfer-project/pegainfer:main`.
- State: open.
- Draft: true.
- Mergeability from GitHub: `mergeable=CONFLICTING`, `mergeStateStatus=DIRTY`.
- Label: `stale`.
- Closing reference: PR #409 closes issue #408.
- CI shown on PR: `CPU checks` succeeded on 2026-06-16.
- Last activity on the PR is the stale bot comment from 2026-08-01: "This pull request has been inactive for 14 days. It will be closed after another 30 days unless there is new activity."

## Verified Issue State

All of the following issues are open and currently have labels `higgs-audio` and `stale`.

| Issue | Title | Assignee State | Last Updated |
| --- | --- | --- | --- |
| #336 | `[Model] Higgs Audio v3 roadmap` | assigned to `ywh555hhh` | 2026-07-19 |
| #395 | `higgs-audio: crate skeleton + config/weights + backbone parity` | unassigned | 2026-07-19 |
| #396 | `higgs-audio: delay-pattern state machine (pure logic, CPU-testable)` | unassigned | 2026-07-19 |
| #397 | `higgs-audio: multi-codebook embedding + head + sampling` | unassigned | 2026-07-19 |
| #398 | `higgs-audio: decode loop + KV integration (end-to-end tokens)` | unassigned | 2026-07-19 |
| #399 | `higgs-audio: codec decode side (vocoder) -> wav` | unassigned | 2026-07-19 |
| #400 | `higgs-audio: e2e bring-up + correctness fixture + docs` | unassigned | 2026-07-19 |
| #408 | `higgs-audio: crate skeleton + config/weights + backbone forward + parity gate` | assigned to `ywh555hhh` | 2026-07-19 |

Stale workflow facts verified from `.github/workflows/stale.yml`:

- Items are marked stale after 14 days of inactivity.
- Items are closed after another 30 days of inactivity.
- `remove-stale-when-updated: true`.
- `exempt-issue-labels: roadmap`.
- Issue #336 is a roadmap by content, but currently does not have the `roadmap` label.

## Issue Content Summary

This section summarizes issue text. It is not independent verification of model facts or golden artifact availability.

### #336 Parent Roadmap

- Proposed owner: `@ywh555hhh`.
- Intended crate: `pegainfer-higgs-audio`.
- Intended feature gate: `higgs-audio`.
- Intended docs path: `docs/models/higgs-audio/`.
- V1 contract says Higgs should be model-local work: one feature-gated crate with its own decode loop.
- V1 contract says the Qwen3 crate is a reference, not a base class or shared contract.
- V1 contract says existing text-model decode paths should not be widened to fit the 1-to-8 codebook delay pattern.
- V1 contract says shared changes should be additive and introduced only for concrete blockers.
- V1 target is single-request bring-up on one RTX 4090. A 5090 is acceptable as a validation host, but do not convert that into a performance claim without a dedicated benchmark.
- #336 states that correctness goldens for A-E are captured. That statement has not been independently verified in this handoff.
- #336 lists one open decision: codec route for #399, choosing among Rust kernels, ONNX, or Python sidecar.

### Sub-Issue Dependency Order

- #395: A, crate skeleton + weights + backbone parity.
- #396: B, delay-pattern state machine. It is CPU-testable and should not require the GPU machine.
- #397: C, multi-codebook embedding + head + sampling. Depends on A and B.
- #398: D, Higgs-local decode loop + KV integration. Depends on C.
- #399: E, codec decode side/vocoder to wav. Independent of D, but route decision is required.
- #400: F, e2e bring-up + correctness fixture + docs. Depends on D and E.
- Roadmap order from #336: `A -> (B || A-config) -> C -> D || E -> F`.

## Current Implementation Evidence

PR #409 changed these files:

- `Cargo.lock`
- `Cargo.toml`
- `openinfer-higgs-audio/Cargo.toml`
- `openinfer-higgs-audio/src/backbone.rs`
- `openinfer-higgs-audio/src/config.rs`
- `openinfer-higgs-audio/src/lib.rs`
- `openinfer-higgs-audio/src/weights.rs`
- `openinfer-higgs-audio/tests/backbone_parity.rs`

The PR body says it includes:

- crate skeleton and feature wiring,
- Higgs config parsing,
- `body.*` weight-name mapping,
- Qwen3-isomorphic text backbone forward,
- a backbone parity gate,
- Mac-side checks: `cargo fmt --all --check`, `cargo metadata --no-deps`, `cargo build -p openinfer-higgs-audio`, and `cargo test --lib`.

Those claimed checks were not re-run in this handoff.

Local merge probe:

- A local merge attempt of `ywh555hhh/feat/higgs-audio-v1` into current upstream `main` produced content conflicts in `Cargo.toml` and `Cargo.lock`.
- The merge probe showed the new `openinfer-higgs-audio/*` files as additions.
- The visible root cause is that current upstream uses `pegainfer-*` crate naming, while PR #409 was authored against the older `openinfer-*` naming.
- The implementation branch still contains paths and imports such as `openinfer-higgs-audio`, `openinfer_core`, `openinfer_kernels`, and `openinfer_kv_cache`.
- The tests in the PR use `OPENINFER_TEST_MODEL_PATH`; current project convention appears to use `PEGAINFER_TEST_MODEL_PATH` in active model tests.
- The merge-conflict result alone does not prove the PR will compile after resolving `Cargo.toml` and `Cargo.lock`. The code also needs naming migration and API revalidation against current `pegainfer-core`, `pegainfer-qwen3`, and `pegainfer-kv-cache`.

## Recommended Next Steps

1. Preserve the existing `feat/higgs-audio-v1` branch. Do not force-push over it until the migration path is clear.
2. Prevent tracker loss:
   - Add `roadmap` label to #336 if maintainers agree.
   - Add a factual update to #409 and relevant issues to remove `stale`.
   - If using the `triage` skill to post comments, prepend the required AI triage disclaimer.
3. Recreate or rebase #409 on current upstream `main`:
   - rename `openinfer-higgs-audio` to `pegainfer-higgs-audio`,
   - update crate name and root workspace membership,
   - update workspace dependencies from `openinfer-*` to `pegainfer-*`,
   - update Rust imports to `pegainfer_core`, `pegainfer_kernels`, and `pegainfer_kv_cache`,
   - update test environment names to current project convention,
   - regenerate `Cargo.lock`.
4. Run CPU-side validation before using a GPU:
   - `cargo fmt --all --check`,
   - `cargo metadata --no-deps`,
   - no-feature build for the Higgs crate,
   - unit tests that do not require model weights or CUDA.
5. Implement #396 separately if possible:
   - it is CPU-only by scope,
   - it reduces risk before #397/#398,
   - it should compare against SGLang-Omni delay-pattern behavior once the reference input/output is available.
6. Use a 5090 only after the branch builds:
   - set `HF_HOME` to a data disk path before downloading models,
   - set or verify `PEGAINFER_CUDA_SM=120`,
   - build the migrated Higgs crate with CUDA features,
   - run backbone parity,
   - then proceed to #397/#398 golden gates.
7. Resolve #399 before committing to the final V1 shape:
   - Rust + CUDA kernels is most aligned with the project but largest in scope,
   - ONNX is a middle-ground route,
   - Python sidecar is the fastest v1 route but may be considered temporary or less acceptable by maintainers.

## Open Decisions And Risks

- #399 codec route is not decided. This is the main implementation-scope decision.
- Golden artifact locations for A-E were not verified in this handoff.
- Model weight access for `bosonai/higgs-audio-v3-tts-4b` was not verified.
- #336 checklist references #395 as A, while PR #409 closes #408. The tracker linkage should be reconciled so A does not remain visually incomplete after a merge.
- Current fork `main` is 355 commits behind upstream. New implementation work should start from upstream `main` or a fresh fork branch based on upstream `main`.
- A 5090 is useful for CUDA validation, but the original roadmap's V1 target says one RTX 4090. Document any 5090-only result as a validation result, not the roadmap's claimed baseline.

## Minimal 5090 Setup Notes

- Use a CUDA image with `nvcc`.
- Put Hugging Face cache and model weights on the data disk, not the system disk.
- Recommended environment:

```bash
export HF_HOME=/root/autodl-tmp/cache/
export CUDA_HOME=/usr/local/cuda
export PEGAINFER_CUDA_SM=120
```

- Use `tmux` for long builds and parity runs.
- Do not commit model weights, raw audio, tokens with license restrictions, or local credentials.


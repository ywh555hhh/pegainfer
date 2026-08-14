# Higgs-Audio Native Codegen 4090 Status - 2026-08-14

## Scope

This evidence is for the PegaInfer Higgs-Audio native incremental audio-code generation diagnostic path on RTX 4090.

It does not claim native wav E2E, native codec/vocoder, production serving, or strict trace parity.

## Machine

- Host: `connect.nmb1.seetacloud.com:13096`
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `595.71.05`
- CUDA toolkit: `/usr/local/cuda-13.0`
- SM: `89`
- Rust: `rustc 1.99.0-nightly (ba28ff76f 2026-08-13)`
- Repo: `/data/src/pegainfer`
- Commit base: `946d0e12`
- Overlay: `/Users/yiweihan/Documents/muxi/higgs-audio-migration/pegainfer-higgs-native-codegen-overlay-946d0e12-4090.tgz`

## Result Summary

- `cargo check --release -p pegainfer-server --features higgs-audio`: passed earlier on this host.
- Higgs lib tests: passed, `71 passed`.
- Runtime-Qwen3 retained bridge tests: passed, `6 passed`.
- Native retained continuation smoke: passed on label `946d0e12-4090-debug4`.
- HF incremental reference trace comparison: failed strict parity.
- `nsys`: captured successfully on label `946d0e12-4090-debug4-profile`.
- `ncu`: blocked by platform performance-counter permission, `ERR_NVGPUCTRPERM`.

## Evidence Tooling Update

After this run, the Higgs native-codegen gate was updated so failed semantic comparison and blocked `ncu` profiling are recorded in the summary instead of aborting before summary creation.

Expected structured states for this host:

```text
trace_compare=failed
profile=partial
nsys_profile=ok
ncu_profile=blocked:nvgpuctrperm
```

The strict PR-ready evidence checker remains strict: it still rejects this bundle until `trace_compare=ok` and `ncu_profile=ok`. This is intentional; the relaxed summary checker is for auditable experiment logging, not for upgrading the claim.

## Native Smoke Evidence

Command result:

```text
higgs native continuation smoke: ok
prompt_tokens: 10
session_id: 1
continuation_steps: 8
last_continuation_step: 8
trace_steps: 9
raw_codec_rows: 2
```

Artifacts:

- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/native-codegen/higgs-native-continuation-smoke-946d0e12-4090-debug4.txt`
- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/native-codegen/higgs-native-continuation-trace-946d0e12-4090-debug4.json`
- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/native-codegen/higgs-native-continuation-codec-input-946d0e12-4090-debug4.json`

SHA256:

```text
536d77500f3d97e0bf372a5e5825d1431e1fe0a70272042a0995747742c705b7  higgs-native-continuation-trace-946d0e12-4090-debug4.json
```

## Strict Trace Compare Evidence

The native trace runs, but strict semantic parity with the HF incremental reference is not established.

Key metrics:

```text
prompt_tokens_match=true
steps_compared=9
first_divergent_step=Some(4)
sampled_rows_exact=false
raw_rows_exact=false
generation_done_exact=true
argmax_agreement=0.708333
argmax_matches=51/72
full_logits_available=true
logits_cosine=Some(0.99989086)
logits_max_abs=Some(12.981487)
logits_mean_abs=Some(0.6785689)
logits_p99_abs=Some(5.54644)
max_argmax_regret=Some(6.75)
topk_min_overlap=Some(10)
topk_mean_overlap=Some(49.72222)
```

First-divergence diagnosis:

```text
step 0..3 sampled codes: exact match
step 4 cb0: reference token 21, native token 775
step 4 cb2: reference token 337, native token 420
```

The first divergent logits are tie-sensitive in the HF reference:

```text
step 4 cb0 reference: token 21 = 59.25, token 775 = 59.25
step 4 cb0 native:    token 775 = 59.2906, token 21 = 59.2216

step 4 cb2 reference: token 337 = 75.0, token 420 = 75.0, token 704 = 75.0
step 4 cb2 native:    token 420 = 75.2798, token 337 = 75.1428, token 704 = 75.0330
```

This suggests the first strict mismatch is likely dominated by tie-breaking / small numeric perturbation near equal logits. Later drift, such as step 5 cb2, is already downstream of the step 4 sampled-code branch and should not be treated as an independent root cause without a forced-common-prefix replay.

## Forced Common Prefix Evidence

After the first native trace comparison, a forced-prefix diagnostic path was added:

- It still runs the native retained-KV Qwen3 body and Higgs audio head for every step.
- It forces the Higgs code-generation state to consume sampled rows from the HF reference trace.
- The resulting trace records forced sampled/raw rows plus native logits/top-k for the same common prefix.

4090 label: `946d0e12-4090-debug6`

Forced-prefix smoke:

```text
higgs native forced-prefix smoke: ok
prompt_tokens: 10
continuation_steps: 8
trace_steps: 9
raw_codec_rows: 2
```

Forced-prefix trace comparison:

```text
prompt_tokens_match=true
steps_compared=9
first_divergent_step=Some(4)
sampled_rows_exact=true
raw_rows_exact=true
generation_done_exact=true
argmax_agreement=0.888889
argmax_matches=64/72
full_logits_available=true
logits_cosine=Some(0.99999833)
logits_max_abs=Some(1.0)
logits_mean_abs=Some(0.12511392)
logits_p99_abs=Some(0.5)
max_argmax_regret=Some(0.0)
topk_min_overlap=Some(44)
topk_mean_overlap=Some(58.555557)
```

Compared with the free-running native trace, forced-prefix replay improves cosine from `0.99989086` to `0.99999833` and argmax agreement from `51/72` to `64/72`, while making sampled/raw rows exact. This narrows the remaining issue to strict numeric/top-k/tie-breaking behavior under a shared prefix, not a broken retained-KV feedback path.

Remaining forced-prefix argmax mismatches are tie-sensitive in the reference logits:

```text
step 4 cb0: reference 21 = 59.25, 775 = 59.25; native picks 775
step 4 cb2: reference 337 = 75.0, 420 = 75.0, 704 = 75.0; native picks 420
step 5 cb2: reference 86 = 74.0, 524 = 74.0, 733 = 74.0, 812 = 74.0; native picks 812
step 5 cb4: reference 133 = 118.5, 449 = 118.5; native picks 449
step 6 cb0: reference 483 = 57.0, 499 = 57.0; native picks 499
step 7 cb5: reference 356 = 115.5, 870 = 115.5; native picks 870
step 8 cb4: reference 597 = 119.0, 953 = 119.0; native picks 953
step 8 cb5: reference 49 = 114.0, 316 = 114.0; native picks 316
```

Artifacts:

- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results-debug6/higgs-native-forced-prefix-smoke-946d0e12-4090-debug6.txt`
- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results-debug6/higgs-native-forced-prefix-trace-946d0e12-4090-debug6.json`
- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results-debug6/higgs-native-forced-prefix-codec-input-946d0e12-4090-debug6.json`
- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results-debug6/higgs-forced-prefix-trace-compare-946d0e12-4090-debug6.txt`

Integrated gate evidence:

- `946d0e12-4090-debug7-forced-gate` reran the lower-level contract gate with `--reference-trace --forced-prefix`.
- The generated summary records `trace_compare=failed`, `forced_prefix=ok`, `forced_prefix_compare=failed`, and `profile=skipped`.
- Local rendered PR evidence from this summary uses the claim: native retained-continuation execution with forced common-prefix diagnostic evidence; strict semantic parity and profiling are still unresolved.

Integrated artifacts:

- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results-debug7/higgs-native-codegen-contract-gate-946d0e12-4090-debug7-forced-gate.txt`
- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results-debug7/pr-evidence-946d0e12-4090-debug7-forced-gate-local-rendered.md`

Artifacts:

- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/native-codegen/higgs-reference-incremental-trace-946d0e12-4090.json`
- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/native-codegen/higgs-codegen-trace-compare-946d0e12-4090-debug4.txt`

SHA256:

```text
7bdbc8d411bba2686bf4615c1ad999ac73e41b93a6bdbfb2c89013708c4e1b6e  higgs-codegen-trace-compare-946d0e12-4090-debug4.txt
```

## Profiling Evidence

`nsys` captured a native continuation smoke run successfully.

Artifacts:

- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/profiles/higgs-native-codegen-946d0e12-4090-debug4-profile.nsys-rep`
- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/profiles/higgs-native-codegen-946d0e12-4090-debug4-profile.sqlite`
- `/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results/profiles/higgs-native-codegen-946d0e12-4090-debug4-profile-stats_cuda_gpu_kern_sum.csv`

SHA256:

```text
523c91a0613d74a7acd26cf3b40026d84028da0d7870743881a2ae83d22801d1  higgs-native-codegen-946d0e12-4090-debug4-profile.nsys-rep
18e9f3e4c4d5a49586fe920cc0bb6f4a6863f15caf1ee04e420537fc7e61da53  higgs-native-codegen-946d0e12-4090-debug4-profile-stats_cuda_gpu_kern_sum.csv
```

Top `nsys` CUDA kernel summary begins with:

```text
18.7%, 300858486 ns, 2000 instances, cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8
10.8%, 173478458 ns, 356 instances, ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
10.6%, 170082436 ns, 1314 instances, cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8
```

`ncu` did not produce a usable report on this cloud host:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

For the next 4090 host, prefer a machine where NVIDIA performance counters are enabled by the provider or where the host driver can set `NVreg_RestrictProfilingToAdminUsers=0`.

## Reproduction Command

```bash
export RUSTUP_TOOLCHAIN=nightly
export PATH=/root/.cargo/bin:/usr/local/cuda-13.0/bin:/root/autodl-tmp/venvs/higgs-omni/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export PEGAINFER_CUDA_SM=89
export PEGAINFER_NVCC_JOBS=8

cd /data/src/pegainfer
tar -xzf /tmp/pegainfer-higgs-native-codegen-overlay-946d0e12-4090.tgz -C /data/src/pegainfer

tools/higgs/run_higgs_native_codegen_contract_gate.sh \
  --result-root /data/results/pegainfer/higgs-audio \
  --label 946d0e12-4090-debug4 \
  --runtime-qwen3 \
  --model-dir /root/autodl-tmp/models/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --golden /root/autodl-tmp/results/higgs-one-step-sglang-omni-golden-autodl.safetensors \
  --reference-trace /data/results/pegainfer/higgs-audio/native-codegen/higgs-reference-incremental-trace-946d0e12-4090.json \
  --profile
```

Because strict compare currently fails before profiling, use a separate profile run without `--reference-trace`:

```bash
tools/higgs/run_higgs_native_codegen_contract_gate.sh \
  --result-root /data/results/pegainfer/higgs-audio \
  --label 946d0e12-4090-debug4-profile \
  --runtime-qwen3 \
  --model-dir /root/autodl-tmp/models/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --golden /root/autodl-tmp/results/higgs-one-step-sglang-omni-golden-autodl.safetensors \
  --profile
```

## Next Decisions

1. Keep the output-budget fix: Qwen3 retained prefill uses a bookkeeping token, so Higgs continuation must reserve `steps + 1` output slots.
2. Do not claim strict parity yet. The cosine is high, but first divergence at step 4 and argmax agreement `51/72` are not strict enough.
3. Decide with maintainers whether strict argmax/top-k parity across tied logits is required, or whether common-prefix logits cosine/regret plus exact sampled/raw-row forcing is the right diagnostic acceptance criterion.
4. On a profiler-friendly host, rerun `ncu` for the top kernels identified by `nsys`.

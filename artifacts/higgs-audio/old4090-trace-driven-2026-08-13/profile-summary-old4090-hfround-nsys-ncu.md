# Old4090 Higgs-Audio NSYS/NCU Profile Summary

## Scope

- Host: old RTX 4090D, SM89
- Runtime label: `old4090-hfround-nsys`
- Gate: `tools/higgs/run_higgs_one_step_cuda_gate.sh --profile`
- NSYS: `--trace=cuda,nvtx,cublas --cuda-graph-trace=node`
- NCU attempt: `--set basic` on the dominant CUTLASS `128x2` BF16 GEMM kernel

## Semantic Gate

- One-step semantic comparison: ok
- Session semantic comparison: ok
- Duplicate request id guard: ok
- Strict trace drift remains diagnostic-only:
  - `final_hidden.bf16 mean_abs=0.006735`
  - `audio_logits.f32 mean_abs=0.040235`
  - `audio_argmax.ids` exact

## NSYS Kernel Summary

| Kernel family | Time | Instances | Avg ns | Interpretation |
|---|---:|---:|---:|---|
| CUTLASS BF16 GEMM `16x16_128x2_tn_align8` | 53.7% | 180 | 29,712.6 | Dominant projection path |
| CUTLASS BF16 GEMM `16x16_128x1_tn_align8` | 40.6% | 72 | 56,092.0 | Second GEMM variant, likely larger/alternate projection shape |
| FlashInfer paged prefill attention | 1.6% | 36 | 4,551.1 | Not the current bottleneck |
| QK norm + RoPE | 0.7% | 36 | 1,997.3 | Small kernel-time share |
| HF-round fused add + RMSNorm | 0.7% | 36 | 1,867.6 | Small kernel-time share |
| FlashInfer RMSNorm | 0.7% | 37 | 1,765.2 | Small kernel-time share |

The two CUTLASS GEMM kernel families account for `94.3%` of measured GPU kernel time in the one-step profile. This supports focusing performance work on projection/GEMM behavior, not attention or norm kernels, while keeping trace-driven numeric gates as the correctness guard.

## NCU Status

NCU is installed (`2025.1.0.0`) but kernel metric collection failed with:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

This means the current machine can run NSYS and produce kernel ranking evidence, but cannot yet produce NCU roofline/occupancy/stall metrics without enabling NVIDIA performance counters or moving to an image/host where the counters are accessible.

## Next Decision

Do not change runtime GEMM code yet. The projection recompute diagnostics show actual PegaInfer projection outputs are reproducible from actual inputs plus checkpoint weights, and NSYS says GEMM dominates runtime. The next useful step is either:

1. enable NCU performance counters and collect roofline/occupancy for the dominant CUTLASS GEMM kernels, or
2. build a controlled cuBLAS algorithm/math-mode experiment that is gated by full one-step trace metrics before any runtime change is kept.

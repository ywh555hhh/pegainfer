pub struct KernelPlan {
    pub model: &'static str,
    pub phases: &'static [KernelPhase],
}

pub struct KernelPhase {
    pub name: &'static str,
    pub ops: &'static [KernelOp],
}

pub struct KernelOp {
    pub id: &'static str,
    pub rust: &'static str,
    pub backend: &'static str,
    pub notes: &'static str,
}

pub static KERNEL_PLAN: KernelPlan = KernelPlan {
    model: "higgs-audio",
    phases: &[
        KernelPhase {
            name: "artifact",
            ops: &[
                KernelOp {
                    id: "checkpoint_header_gate",
                    rust: "weights::HiggsWeightManifest::from_model_dir",
                    backend: "safetensors header",
                    notes: "validates Higgs checkpoint tensor names, dtypes, and shapes without reading payloads",
                },
                KernelOp {
                    id: "qwen3_alias_plan",
                    rust: "load_plan::HiggsRuntimeLoadPlan::qwen3_tensor_aliases",
                    backend: "metadata",
                    notes: "maps Higgs body.* tensors onto Qwen3 requested tensor names without a 7.5 GiB payload copy",
                },
            ],
        },
        KernelPhase {
            name: "prefill",
            ops: &[
                KernelOp {
                    id: "qwen3_body_prefill",
                    rust: "runtime_bridge::HiggsAudioRuntime::prefill_audio_from_prompt_ids -> Qwen3Executor::prefill_last_hidden_bf16",
                    backend: "Qwen3 runtime: CUDA + cuBLAS + FlashInfer",
                    notes: "runs the Higgs text/body checkpoint through the existing Qwen3 prefill path via tensor-name aliases",
                },
                KernelOp {
                    id: "qwen3_prompt_session_prefill",
                    rust: "runtime_bridge::HiggsAudioRuntime::prefill_prompt_session -> Qwen3Executor::prefill_last_hidden_bf16_retained_prompt",
                    backend: "Qwen3 runtime: CUDA + cuBLAS + FlashInfer + paged KV",
                    notes: "retains prompt KV under a Higgs-owned session handle without registering a generated text token",
                },
                KernelOp {
                    id: "fused_audio_head",
                    rust: "one_step_actual::compute_one_step_audio_prediction_gpu_bf16 -> ops::linear",
                    backend: "CUDA bf16 linear",
                    notes: "projects the final hidden state with tied.embedding.modality_embeddings.0.embedding.weight into 8x1026 audio logits",
                },
                KernelOp {
                    id: "audio_topk_argmax",
                    rust: "one_step_actual::audio_topk_and_argmax",
                    backend: "CPU",
                    notes: "diagnostic one-step gate extracts top-64 and argmax ids from the fused audio logits",
                },
            ],
        },
        KernelPhase {
            name: "decode",
            ops: &[
                KernelOp {
                    id: "delay_pattern_state",
                    rust: "delay_pattern::DelayPatternState::step_from_sampled_codes",
                    backend: "CPU logic",
                    notes: "applies Higgs multi-codebook ramp-up BOC masking, cb0 EOC wind-down, done state, and delayed-row retention",
                },
                KernelOp {
                    id: "fused_codebook_feedback_embedding",
                    rust: "codebook_embedding::fused_codebook_embedding_cpu",
                    backend: "CPU logic",
                    notes: "sums per-codebook rows from tied.embedding.modality_embeddings.0.embedding.weight using codebook-local ids plus codebook offsets",
                },
                KernelOp {
                    id: "decode_session_collect",
                    rust: "decode_session::HiggsDecodeSession::step_from_sampled_codes",
                    backend: "CPU logic",
                    notes: "collects emitted delayed rows, exposes last feedback codes, and de-delays the accumulated code matrix for codec input",
                },
            ],
        },
        KernelPhase {
            name: "vocoder",
            ops: &[
                KernelOp {
                    id: "codec_input_prepare",
                    rust: "codec_input::codec_input_from_rows",
                    backend: "CPU logic",
                    notes: "de-delays generated Higgs rows and clamps BOC/EOC codec sentinels before vocoder decode",
                },
                KernelOp {
                    id: "python_codec_sidecar",
                    rust: "bin::higgs_vocode_codes -> tools/higgs/vocode_higgs_codes.py",
                    backend: "Python sidecar",
                    notes: "temporary route through Higgs-owned local codec wrapper to produce 24 kHz wav while Rust-native codec decode is scoped",
                },
            ],
        },
        KernelPhase {
            name: "golden",
            ops: &[
                KernelOp {
                    id: "strict_comparison",
                    rust: "compare::compare_one_step_files",
                    backend: "CPU",
                    notes: "exact prompt/argmax checks plus absolute drift diagnostics for hidden, logits, and top-64 logprobs",
                },
                KernelOp {
                    id: "semantic_comparison",
                    rust: "compare::compare_one_step_semantic_files",
                    backend: "CPU",
                    notes: "runtime bring-up gate using prompt exactness, argmax exactness, cosine, regret, and top-64 overlap",
                },
                KernelOp {
                    id: "reference_audio_e2e_gate",
                    rust: "tools/higgs/run_higgs_audio_e2e_gate.sh -> tools/higgs/request_higgs_audio.py",
                    backend: "SGLang-Omni HTTP reference + CPU wav validation",
                    notes: "requests /v1/audio/speech, persists a real 24 kHz wav artifact, and optionally replays captured Higgs code rows through the codec sidecar",
                },
                KernelOp {
                    id: "slow_fullprefill_reference_e2e",
                    rust: "tools/higgs/slow_higgs_fullprefill_e2e.py",
                    backend: "Transformers Qwen3 full-prefill reference + Python codec sidecar",
                    notes: "non-production bring-up path that composes text prompt embeddings, generated audio-code feedback embeddings, delayed-code reversal, and local codec decode into a real wav without touching runtime crates",
                },
            ],
        },
    ],
};

pub fn kernel_plan() -> &'static KernelPlan {
    &KERNEL_PLAN
}

#[cfg(test)]
mod tests {
    use super::kernel_plan;

    #[test]
    fn higgs_kernel_plan_names_current_phases() {
        let phase_names: Vec<_> = kernel_plan()
            .phases
            .iter()
            .map(|phase| phase.name)
            .collect();
        assert_eq!(
            phase_names,
            ["artifact", "prefill", "decode", "vocoder", "golden"]
        );
    }

    #[test]
    fn higgs_kernel_plan_records_runtime_backends() {
        let ops: Vec<_> = kernel_plan()
            .phases
            .iter()
            .flat_map(|phase| phase.ops.iter())
            .collect();

        assert!(ops.iter().any(|op| {
            op.id == "qwen3_body_prefill"
                && op.backend == "Qwen3 runtime: CUDA + cuBLAS + FlashInfer"
        }));
        assert!(ops.iter().any(|op| {
            op.id == "qwen3_prompt_session_prefill" && op.notes.contains("retains prompt KV")
        }));
        assert!(
            ops.iter()
                .any(|op| op.id == "fused_audio_head" && op.backend == "CUDA bf16 linear")
        );
        assert!(ops.iter().any(|op| {
            op.id == "delay_pattern_state"
                && op.backend == "CPU logic"
                && op.notes.contains("cb0 EOC wind-down")
        }));
        assert!(ops.iter().any(|op| {
            op.id == "fused_codebook_feedback_embedding"
                && op.backend == "CPU logic"
                && op.notes.contains("codebook offsets")
        }));
        assert!(ops.iter().any(|op| {
            op.id == "decode_session_collect"
                && op.backend == "CPU logic"
                && op.notes.contains("de-delays")
        }));
        assert!(ops.iter().any(|op| {
            op.id == "codec_input_prepare"
                && op.backend == "CPU logic"
                && op.notes.contains("clamps")
        }));
        assert!(ops.iter().any(|op| {
            op.id == "python_codec_sidecar"
                && op.backend == "Python sidecar"
                && op.notes.contains("24 kHz wav")
        }));
        assert!(
            ops.iter()
                .any(|op| op.id == "semantic_comparison" && op.backend == "CPU")
        );
        assert!(ops.iter().any(|op| {
            op.id == "reference_audio_e2e_gate"
                && op.backend == "SGLang-Omni HTTP reference + CPU wav validation"
                && op.notes.contains("real 24 kHz wav")
        }));
        assert!(ops.iter().any(|op| {
            op.id == "slow_fullprefill_reference_e2e"
                && op.backend == "Transformers Qwen3 full-prefill reference + Python codec sidecar"
                && op.notes.contains("without touching runtime crates")
        }));
    }

    #[test]
    fn higgs_kernel_plan_keeps_alias_copy_boundary_visible() {
        let alias_op = kernel_plan()
            .phases
            .iter()
            .flat_map(|phase| phase.ops.iter())
            .find(|op| op.id == "qwen3_alias_plan")
            .expect("qwen3 alias op should be in the plan");

        assert!(alias_op.notes.contains("without a 7.5 GiB payload copy"));
    }
}

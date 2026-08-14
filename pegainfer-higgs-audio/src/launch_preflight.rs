use std::path::Path;

use anyhow::Context;
use anyhow::Result;

use crate::config::HiggsConfig;
use crate::load_plan::HiggsRuntimeLoadPlan;
use crate::load_plan::LoadPlanSummary;
use crate::weights::HiggsWeightManifest;
use crate::weights::ManifestSummary;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiggsLaunchPreflight {
    pub manifest: ManifestSummary,
    pub load_plan: LoadPlanSummary,
    pub device_ordinal: usize,
}

impl HiggsLaunchPreflight {
    pub fn unsupported_launch_message(&self) -> String {
        format!(
            "Higgs Audio native server launch is not implemented yet: preflight ok \
             (body_tensors={}, qwen3_backbone_tensors={}, higgs_head_tensors={}, \
             bf16_mib={}, device_ordinal={}); current support stops at native \
             one-step prefill and retained prompt-session diagnostics; next \
             milestone is native incremental audio code generation",
            self.manifest.body_tensors,
            self.load_plan.qwen3_backbone_tensors,
            self.load_plan.higgs_head_tensors,
            self.load_plan.bf16_bytes / 1024 / 1024,
            self.device_ordinal
        )
    }
}

pub fn preflight_launch(
    model_dir: impl AsRef<Path>,
    device_ordinal: usize,
) -> Result<HiggsLaunchPreflight> {
    let model_dir = model_dir.as_ref();
    let config = HiggsConfig::from_model_dir(model_dir)
        .with_context(|| format!("preflight Higgs config in {}", model_dir.display()))?;
    let manifest = HiggsWeightManifest::from_model_dir(model_dir)
        .with_context(|| format!("preflight Higgs manifest in {}", model_dir.display()))?;
    let manifest_summary = manifest.validate_for_config(&config)?;
    let load_plan = HiggsRuntimeLoadPlan::from_manifest(&config, &manifest)?;
    Ok(HiggsLaunchPreflight {
        manifest: manifest_summary,
        load_plan: load_plan.summary(),
        device_ordinal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EXPECTED_ARCHITECTURE;
    use crate::config::EXPECTED_AUDIO_ENCODER_TYPE;
    use crate::config::EXPECTED_HEAD_DIM;
    use crate::config::EXPECTED_INTERMEDIATE_SIZE;
    use crate::config::EXPECTED_MODEL_TYPE;
    use crate::config::EXPECTED_NUM_ATTENTION_HEADS;
    use crate::config::EXPECTED_NUM_KV_HEADS;
    use crate::config::EXPECTED_NUM_LAYERS;
    use crate::config::EXPECTED_ROPE_THETA;
    use crate::config::EXPECTED_TEXT_VOCAB_SIZE;
    use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
    use crate::one_step_golden::HIDDEN_SIZE;
    use crate::one_step_golden::NUM_CODEBOOKS;
    use crate::weights::BODY_NORM;
    use crate::weights::FUSED_MODALITY_EMBEDDING;
    use crate::weights::TEXT_EMBEDDING;

    fn minimal_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": [EXPECTED_ARCHITECTURE],
            "audio_token_id": -100,
            "model_type": EXPECTED_MODEL_TYPE,
            "text_config": {
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": EXPECTED_INTERMEDIATE_SIZE,
                "num_hidden_layers": EXPECTED_NUM_LAYERS,
                "num_attention_heads": EXPECTED_NUM_ATTENTION_HEADS,
                "num_key_value_heads": EXPECTED_NUM_KV_HEADS,
                "head_dim": EXPECTED_HEAD_DIM,
                "vocab_size": EXPECTED_TEXT_VOCAB_SIZE,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 32768,
                "eos_token_id": 151643,
                "tie_word_embeddings": true,
                "rope_parameters": {"rope_theta": EXPECTED_ROPE_THETA}
            },
            "audio_encoder_config": {
                "encoder_type": EXPECTED_AUDIO_ENCODER_TYPE,
                "num_codebooks": NUM_CODEBOOKS,
                "vocab_size": CODEBOOK_VOCAB_SIZE,
                "out_dim": HIDDEN_SIZE,
                "tie_word_embeddings": true,
                "use_delay_pattern": true
            }
        })
    }

    fn minimal_manifest() -> serde_json::Value {
        let mut weight_map = serde_json::Map::new();
        weight_map.insert(
            TEXT_EMBEDDING.to_string(),
            serde_json::json!("model.safetensors"),
        );
        weight_map.insert(
            FUSED_MODALITY_EMBEDDING.to_string(),
            serde_json::json!("model.safetensors"),
        );
        weight_map.insert(
            BODY_NORM.to_string(),
            serde_json::json!("model.safetensors"),
        );
        for layer in 0..EXPECTED_NUM_LAYERS {
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                weight_map.insert(
                    format!("body.layers.{layer}.{suffix}"),
                    serde_json::json!("model.safetensors"),
                );
            }
        }
        serde_json::json!({
            "metadata": {"total_size": "1"},
            "weight_map": weight_map,
        })
    }

    #[test]
    fn preflight_reads_model_dir_contract_without_payloads() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::to_vec_pretty(&minimal_config()).unwrap(),
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&minimal_manifest()).unwrap(),
        )
        .unwrap();

        let preflight = preflight_launch(tmp.path(), 3).unwrap();

        assert_eq!(
            preflight.manifest.body_tensors,
            1 + EXPECTED_NUM_LAYERS * 11
        );
        assert_eq!(preflight.load_plan.higgs_head_tensors, 1);
        assert_eq!(preflight.device_ordinal, 3);
        assert!(
            preflight
                .unsupported_launch_message()
                .contains("preflight ok")
        );
    }
}

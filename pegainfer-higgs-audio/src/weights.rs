use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::config::HiggsConfig;
use crate::one_step_golden::{CODEBOOK_VOCAB_SIZE, HIDDEN_SIZE, NUM_CODEBOOKS};

pub const TEXT_EMBEDDING: &str = "tied.embedding.text_embedding.weight";
pub const FUSED_MODALITY_EMBEDDING: &str = "tied.embedding.modality_embeddings.0.embedding.weight";
pub const BODY_NORM: &str = "body.norm.weight";

#[derive(Debug, Clone)]
pub struct HiggsWeightManifest {
    pub total_size: Option<u64>,
    pub weight_map: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestSummary {
    pub total_tensors: usize,
    pub body_tensors: usize,
    pub decoder_only_tensors: usize,
    pub has_text_embedding: bool,
    pub has_fused_modality_embedding: bool,
    pub has_separate_audio_head: bool,
}

impl HiggsWeightManifest {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("model.safetensors.index.json");
        let value: Value = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse {}", path.display()))?;
        Self::from_json(&value)
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let total_size = value
            .get("metadata")
            .and_then(|metadata| metadata.get("total_size"))
            .and_then(|total| match total {
                Value::Number(n) => n.as_u64(),
                Value::String(s) => s.parse().ok(),
                _ => None,
            });
        let raw = value
            .get("weight_map")
            .and_then(Value::as_object)
            .context("index missing weight_map")?;
        let mut weight_map = HashMap::with_capacity(raw.len());
        for (key, value) in raw {
            let file = value
                .as_str()
                .with_context(|| format!("weight_map entry {key} is not a string"))?;
            weight_map.insert(key.clone(), file.to_string());
        }
        Ok(Self {
            total_size,
            weight_map,
        })
    }

    pub fn summary(&self) -> ManifestSummary {
        ManifestSummary {
            total_tensors: self.weight_map.len(),
            body_tensors: self
                .weight_map
                .keys()
                .filter(|name| name.starts_with("body."))
                .count(),
            decoder_only_tensors: self
                .weight_map
                .keys()
                .filter(|name| is_decoder_only_tensor(name))
                .count(),
            has_text_embedding: self.weight_map.contains_key(TEXT_EMBEDDING),
            has_fused_modality_embedding: self.weight_map.contains_key(FUSED_MODALITY_EMBEDDING),
            has_separate_audio_head: self.weight_map.keys().any(|name| {
                name.starts_with("tied.head.modality") || name.starts_with("tied.head.audio")
            }),
        }
    }

    pub fn validate_for_config(&self, config: &HiggsConfig) -> Result<ManifestSummary> {
        let summary = self.summary();
        if !summary.has_text_embedding {
            bail!("Higgs manifest missing {TEXT_EMBEDDING}");
        }
        if !summary.has_fused_modality_embedding {
            bail!("Higgs manifest missing {FUSED_MODALITY_EMBEDDING}");
        }
        if summary.has_separate_audio_head {
            bail!(
                "Higgs manifest unexpectedly contains a separate audio head; current checkpoint ties the fused modality head"
            );
        }
        let required = required_body_tensors(config);
        let missing: Vec<_> = required
            .iter()
            .filter(|name| !self.weight_map.contains_key(*name))
            .cloned()
            .collect();
        if !missing.is_empty() {
            bail!(
                "Higgs manifest missing {} required body tensor(s): {:?}",
                missing.len(),
                &missing[..missing.len().min(8)]
            );
        }
        if summary.body_tensors != required.len() {
            bail!(
                "Higgs manifest body tensor count mismatch: expected {}, got {}",
                required.len(),
                summary.body_tensors
            );
        }
        Ok(summary)
    }
}

pub fn required_body_tensors(config: &HiggsConfig) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert(BODY_NORM.to_string());
    for layer in 0..config.text.num_hidden_layers {
        let prefix = format!("body.layers.{layer}");
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
            names.insert(format!("{prefix}.{suffix}"));
        }
    }
    names
}

pub fn fused_modality_shape() -> [usize; 2] {
    [NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE, HIDDEN_SIZE]
}

fn is_decoder_only_tensor(name: &str) -> bool {
    name.starts_with("tied.embedding.modality_embeddings.0.model.quantizer")
        || name.starts_with("tied.embedding.modality_embeddings.0.model.fc2")
        || name.starts_with("tied.embedding.modality_embeddings.0.model.acoustic_decoder")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        EXPECTED_ARCHITECTURE, EXPECTED_AUDIO_ENCODER_TYPE, EXPECTED_HEAD_DIM,
        EXPECTED_INTERMEDIATE_SIZE, EXPECTED_MODEL_TYPE, EXPECTED_NUM_ATTENTION_HEADS,
        EXPECTED_NUM_KV_HEADS, EXPECTED_NUM_LAYERS, EXPECTED_ROPE_THETA, EXPECTED_TEXT_VOCAB_SIZE,
    };

    fn config() -> HiggsConfig {
        HiggsConfig::from_json(&serde_json::json!({
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
        }))
        .unwrap()
    }

    fn manifest_json(include_fused_head: bool) -> Value {
        let cfg = config();
        let mut weight_map = serde_json::Map::new();
        weight_map.insert(
            TEXT_EMBEDDING.to_string(),
            serde_json::json!("model.safetensors"),
        );
        if include_fused_head {
            weight_map.insert(
                FUSED_MODALITY_EMBEDDING.to_string(),
                serde_json::json!("model.safetensors"),
            );
        }
        for name in required_body_tensors(&cfg) {
            weight_map.insert(name, serde_json::json!("model.safetensors"));
        }
        serde_json::json!({"metadata": {"total_size": "8489763794"}, "weight_map": weight_map})
    }

    #[test]
    fn validates_required_higgs_manifest_surface() {
        let cfg = config();
        let manifest = HiggsWeightManifest::from_json(&manifest_json(true)).unwrap();
        let summary = manifest.validate_for_config(&cfg).unwrap();
        assert_eq!(summary.body_tensors, 397);
        assert_eq!(summary.total_tensors, 399);
        assert_eq!(fused_modality_shape(), [8208, 2560]);
    }

    #[test]
    fn rejects_manifest_without_fused_modality_head_weight() {
        let cfg = config();
        let manifest = HiggsWeightManifest::from_json(&manifest_json(false)).unwrap();
        let err = manifest.validate_for_config(&cfg).unwrap_err().to_string();
        assert!(err.contains(FUSED_MODALITY_EMBEDDING));
    }
}

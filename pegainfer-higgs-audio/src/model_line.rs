//! Higgs Audio's [`ModelLine`] implementation.
//!
//! This slice registers Higgs as a recognizable model family while the native
//! retained-KV audio decode loop is still under construction. Launch fails
//! closed so server detection and CLI ownership can land without claiming
//! native audio serving.

use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::CliError;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;
use pegainfer_frontend::model_line::ServePlan;

use crate::config::EXPECTED_ARCHITECTURE;
use crate::config::EXPECTED_MODEL_TYPE;
use crate::config::HiggsConfig;
use crate::launch_preflight::preflight_launch;

pub static MODEL_LINE: HiggsAudioLine = HiggsAudioLine;

pub struct HiggsAudioLine;

impl ModelLine for HiggsAudioLine {
    fn name(&self) -> &'static str {
        "Higgs Audio"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
        let architecture = config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .and_then(|items| items.first())
            .and_then(serde_json::Value::as_str);
        if model_type != Some(EXPECTED_MODEL_TYPE) || architecture != Some(EXPECTED_ARCHITECTURE) {
            return Err(format!(
                "model_type {model_type:?} / architecture {architecture:?} is not a Higgs Audio identity"
            ));
        }
        HiggsConfig::from_json(config)
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &["device_ordinal"]
    }

    fn serve_plan(&self, _ctx: &LaunchContext<'_>) -> Result<ServePlan, CliError> {
        Ok(ServePlan::default())
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        let preflight = preflight_launch(ctx.model_path, ctx.shared.device_ordinal)?;
        anyhow::bail!("{}", preflight.unsupported_launch_message())
    }
}

#[cfg(test)]
mod tests {
    use pegainfer_frontend::model_line::parse_for_line;

    use super::*;
    use crate::config::EXPECTED_AUDIO_ENCODER_TYPE;
    use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
    use crate::one_step_golden::HIDDEN_SIZE;
    use crate::one_step_golden::NUM_CODEBOOKS;

    fn minimal_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": [EXPECTED_ARCHITECTURE],
            "audio_token_id": -100,
            "model_type": EXPECTED_MODEL_TYPE,
            "text_config": {
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": crate::config::EXPECTED_INTERMEDIATE_SIZE,
                "num_hidden_layers": crate::config::EXPECTED_NUM_LAYERS,
                "num_attention_heads": crate::config::EXPECTED_NUM_ATTENTION_HEADS,
                "num_key_value_heads": crate::config::EXPECTED_NUM_KV_HEADS,
                "head_dim": crate::config::EXPECTED_HEAD_DIM,
                "vocab_size": crate::config::EXPECTED_TEXT_VOCAB_SIZE,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 32768,
                "eos_token_id": 151643,
                "tie_word_embeddings": true,
                "rope_parameters": {"rope_theta": crate::config::EXPECTED_ROPE_THETA}
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

    #[test]
    fn probe_accepts_higgs_audio_identity() {
        MODEL_LINE
            .probe(&minimal_config())
            .expect("Higgs Audio config should probe");
    }

    #[test]
    fn probe_rejects_qwen3_identity() {
        let config =
            serde_json::json!({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]});
        let reason = MODEL_LINE.probe(&config).unwrap_err();
        assert!(reason.contains("Higgs Audio"), "{reason}");
    }

    #[test]
    fn accepts_device_ordinal_as_owned_shared_arg() {
        parse_for_line(&MODEL_LINE, &["pegainfer", "--device-ordinal", "0"])
            .expect("Higgs Audio should accept its single-GPU device selector");
    }

    #[test]
    fn rejects_qwen3_only_runtime_flags() {
        let error = parse_for_line(&MODEL_LINE, &["pegainfer", "--no-prefix-cache"]).unwrap_err();
        assert!(
            error.to_string().contains("is not used by Higgs Audio"),
            "{error}"
        );
    }
}

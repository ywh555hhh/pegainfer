use std::path::Path;

use anyhow::{Context, Result};
use half::bf16;
use pegainfer_core::weight_loader::TensorNameAliases;
use pegainfer_qwen3_4b::runtime::Qwen3Executor;

use crate::config::HiggsConfig;
use crate::load_plan::HiggsRuntimeLoadPlan;
use crate::one_step_actual::{
    OneStepActualSummary, load_fused_audio_head_bf16, load_prompt_from_golden,
    write_one_step_actual, write_one_step_actual_with_gpu_audio_head,
};
use crate::weights::HiggsWeightManifest;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioHeadBackend {
    CudaBf16,
    CpuFp32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HiggsRuntimeSource<'a> {
    Qwen3BodyView { qwen3_body_dir: &'a Path },
    Qwen3ConfigAlias { qwen3_config_dir: &'a Path },
}

pub struct HiggsOneStepRuntime {
    executor: Qwen3Executor,
    audio_head: Vec<bf16>,
    audio_head_backend: AudioHeadBackend,
    device_ordinal: usize,
}

impl HiggsOneStepRuntime {
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        source: HiggsRuntimeSource<'_>,
        audio_head_backend: AudioHeadBackend,
        device_ordinal: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let executor = load_qwen3_executor(model_dir, source, device_ordinal)?;
        let audio_head = load_fused_audio_head_bf16(model_dir)?;
        Ok(Self {
            executor,
            audio_head,
            audio_head_backend,
            device_ordinal,
        })
    }

    pub fn dump_one_step_actual(
        &mut self,
        golden: impl AsRef<Path>,
        out: impl AsRef<Path>,
    ) -> Result<OneStepActualSummary> {
        let prompt = load_prompt_from_golden(golden)?;
        let prompt_ids = prompt.prompt_ids()?;
        let hidden = self
            .executor
            .prefill_last_hidden_bf16(prompt_ids)?
            .hidden_bf16;
        match self.audio_head_backend {
            AudioHeadBackend::CudaBf16 => write_one_step_actual_with_gpu_audio_head(
                out,
                &prompt,
                &hidden,
                &self.audio_head,
                self.device_ordinal,
            ),
            AudioHeadBackend::CpuFp32 => {
                write_one_step_actual(out, &prompt, &hidden, &self.audio_head)
            }
        }
    }
}

fn load_qwen3_executor(
    model_dir: &Path,
    source: HiggsRuntimeSource<'_>,
    device_ordinal: usize,
) -> Result<Qwen3Executor> {
    match source {
        HiggsRuntimeSource::Qwen3BodyView { qwen3_body_dir } => {
            let qwen3_body_dir = path_str(qwen3_body_dir, "qwen3 body dir")?;
            Qwen3Executor::from_runtime(qwen3_body_dir, false, &[device_ordinal])
        }
        HiggsRuntimeSource::Qwen3ConfigAlias { qwen3_config_dir } => {
            let qwen3_config_dir = path_str(qwen3_config_dir, "qwen3 config dir")?;
            let model_dir_str = path_str(model_dir, "model dir")?;
            Qwen3Executor::from_runtime_with_weight_source(
                qwen3_config_dir,
                Some(model_dir_str),
                qwen3_tensor_name_aliases(model_dir)?,
                false,
                &[device_ordinal],
            )
        }
    }
}

fn qwen3_tensor_name_aliases(model_dir: &Path) -> Result<TensorNameAliases> {
    let config = HiggsConfig::from_model_dir(model_dir)?;
    let manifest = HiggsWeightManifest::from_model_dir(model_dir)?;
    let plan = HiggsRuntimeLoadPlan::from_manifest(&config, &manifest)?;
    Ok(TensorNameAliases::new(
        plan.qwen3_tensor_aliases()?.into_iter().collect(),
    ))
}

fn path_str<'a>(path: &'a Path, label: &str) -> Result<&'a str> {
    path.to_str()
        .with_context(|| format!("{label} must be valid UTF-8"))
}

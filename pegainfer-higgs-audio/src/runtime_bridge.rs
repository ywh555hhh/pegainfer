use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use half::bf16;
use pegainfer_core::weight_loader::TensorNameAliases;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;

use crate::audio_codegen::AudioCodeGenerationSession;
use crate::audio_codegen::AudioCodeGenerationStep;
use crate::audio_codegen::AudioCodegenSessionId;
use crate::config::HiggsConfig;
use crate::decode_trace::AudioLogitsSummary;
use crate::load_plan::HiggsRuntimeLoadPlan;
use crate::materialize_qwen3::write_qwen3_config_view;
use crate::one_step_actual::OneStepActualSummary;
use crate::one_step_actual::OneStepAudioPrediction;
use crate::one_step_actual::PromptTensors;
use crate::one_step_actual::compute_one_step_audio_prediction;
use crate::one_step_actual::compute_one_step_audio_prediction_gpu_bf16;
use crate::one_step_actual::load_fused_audio_head_bf16;
use crate::one_step_actual::load_prompt_from_golden;
use crate::one_step_actual::write_one_step_actual_prediction;
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
    AutoConfigAlias { qwen3_config_dir: &'a Path },
}

/// Higgs Audio runtime surface backed by the existing Qwen3 executor.
///
/// The current implementation owns prefill and prompt-session smoke paths. Full
/// audio decode continuation is intentionally not exposed until the Higgs crate
/// owns the audio-codebook feedback semantics.
pub struct HiggsAudioRuntime {
    executor: Qwen3Executor,
    audio_head: Vec<bf16>,
    audio_head_backend: AudioHeadBackend,
    device_ordinal: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HiggsAudioPrefill {
    pub prompt_tokens: usize,
    pub final_hidden_bf16: Vec<bf16>,
    pub audio: OneStepAudioPrediction,
}

/// Compatibility alias for early one-step gate callers.
pub type HiggsOneStepRuntime = HiggsAudioRuntime;

/// Compatibility alias for early one-step gate callers.
pub type HiggsOneStepPrefill = HiggsAudioPrefill;

/// Higgs-owned handle for a retained prompt KV session.
///
/// The backing executor currently stores the session under a Qwen3 request id,
/// but callers should treat this as a Higgs Audio session id. Audio-codebook
/// continuation is intentionally not exposed through this handle yet.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct HiggsPromptSession {
    request_id: RequestId,
}

impl HiggsPromptSession {
    pub fn new(id: u64) -> Self {
        Self {
            request_id: RequestId::new(id),
        }
    }

    pub fn id(self) -> u64 {
        self.request_id.get()
    }

    fn request_id(self) -> RequestId {
        self.request_id
    }
}

impl From<RequestId> for HiggsPromptSession {
    fn from(request_id: RequestId) -> Self {
        Self { request_id }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HiggsPromptSessionPrefill {
    pub session: HiggsPromptSession,
    pub prompt_tokens: usize,
    pub final_hidden_bf16: Vec<bf16>,
    pub audio: OneStepAudioPrediction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HiggsAudioCodegenSeed {
    pub session: HiggsPromptSession,
    pub codegen: AudioCodeGenerationSession,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HiggsContinuationInput {
    pub session: HiggsPromptSession,
    pub feedback_embedding: Vec<f32>,
}

impl HiggsPromptSessionPrefill {
    pub fn into_audio_codegen_session(self) -> Result<AudioCodeGenerationSession> {
        let mut session = AudioCodeGenerationSession::new_with_id(
            AudioCodegenSessionId::new(self.session.id()),
            self.prompt_tokens,
        );
        session.push_prediction(&self.audio)?;
        Ok(session)
    }
}

impl HiggsAudioRuntime {
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
        let prefill = self.prefill_audio_from_prompt_ids(&prompt_ids)?;
        write_one_step_actual_prediction(out, &prompt, &prefill.final_hidden_bf16, &prefill.audio)
    }

    pub fn prefill_audio_from_prompt(
        &mut self,
        prompt: &PromptTensors,
    ) -> Result<HiggsAudioPrefill> {
        let prompt_ids = prompt.prompt_ids()?;
        self.prefill_audio_from_prompt_ids(&prompt_ids)
    }

    pub fn prefill_audio_from_prompt_ids(
        &mut self,
        prompt_ids: &[u32],
    ) -> Result<HiggsAudioPrefill> {
        let hidden = self
            .executor
            .prefill_last_hidden_bf16(prompt_ids.to_vec())?
            .hidden_bf16;
        let audio = match self.audio_head_backend {
            AudioHeadBackend::CudaBf16 => compute_one_step_audio_prediction_gpu_bf16(
                &hidden,
                &self.audio_head,
                self.device_ordinal,
            )?,
            AudioHeadBackend::CpuFp32 => {
                compute_one_step_audio_prediction(&hidden, &self.audio_head)?
            }
        };
        Ok(HiggsAudioPrefill {
            prompt_tokens: prompt_ids.len(),
            final_hidden_bf16: hidden,
            audio,
        })
    }

    pub fn prefill_prompt_session_from_prompt_ids(
        &mut self,
        request_id: RequestId,
        prompt_ids: &[u32],
    ) -> Result<HiggsPromptSessionPrefill> {
        self.prefill_prompt_session(request_id.into(), prompt_ids)
    }

    pub fn prefill_prompt_session(
        &mut self,
        session: HiggsPromptSession,
        prompt_ids: &[u32],
    ) -> Result<HiggsPromptSessionPrefill> {
        self.prefill_prompt_session_with_output_budget(session, prompt_ids, 1)
    }

    pub fn prefill_prompt_session_with_output_budget(
        &mut self,
        session: HiggsPromptSession,
        prompt_ids: &[u32],
        max_output_tokens: usize,
    ) -> Result<HiggsPromptSessionPrefill> {
        let retained = self
            .executor
            .prefill_last_hidden_bf16_retained_prompt_with_max_output_tokens(
                session.request_id(),
                prompt_ids.to_vec(),
                max_output_tokens,
            )?;
        let audio = match self.audio_head_backend {
            AudioHeadBackend::CudaBf16 => compute_one_step_audio_prediction_gpu_bf16(
                &retained.hidden_bf16,
                &self.audio_head,
                self.device_ordinal,
            )?,
            AudioHeadBackend::CpuFp32 => {
                compute_one_step_audio_prediction(&retained.hidden_bf16, &self.audio_head)?
            }
        };
        Ok(HiggsPromptSessionPrefill {
            session: retained.request_id.into(),
            prompt_tokens: prompt_ids.len(),
            final_hidden_bf16: retained.hidden_bf16,
            audio,
        })
    }

    pub fn start_audio_codegen_session(
        &mut self,
        session: HiggsPromptSession,
        prompt_ids: &[u32],
    ) -> Result<HiggsAudioCodegenSeed> {
        self.start_audio_codegen_session_with_output_budget(session, prompt_ids, 1)
    }

    pub fn start_audio_codegen_session_with_output_budget(
        &mut self,
        session: HiggsPromptSession,
        prompt_ids: &[u32],
        max_output_tokens: usize,
    ) -> Result<HiggsAudioCodegenSeed> {
        let prefill =
            self.prefill_prompt_session_with_output_budget(session, prompt_ids, max_output_tokens)?;
        let session = prefill.session;
        let codegen = prefill.into_audio_codegen_session()?;
        Ok(HiggsAudioCodegenSeed { session, codegen })
    }

    pub fn continue_audio_codegen_step(
        &mut self,
        input: HiggsContinuationInput,
        codegen: &mut AudioCodeGenerationSession,
    ) -> Result<AudioCodeGenerationStep> {
        validate_continuation_input(input.session, input.feedback_embedding.len(), codegen)?;

        let input_embedding_bf16: Vec<bf16> = input
            .feedback_embedding
            .iter()
            .map(|&value| bf16::from_f32(value))
            .collect();
        let hidden = self
            .executor
            .decode_embedding_last_hidden_bf16_retained(
                input.session.request_id(),
                input_embedding_bf16,
            )?
            .hidden_bf16;
        self.push_audio_codegen_hidden_step_cpu(input.session, codegen, &hidden)
    }

    pub fn continue_audio_codegen_step_from_feedback(
        &mut self,
        session: HiggsPromptSession,
        codegen: &mut AudioCodeGenerationSession,
        fused_embedding: &[bf16],
    ) -> Result<Option<AudioCodeGenerationStep>> {
        let feedback_embedding = match codegen.feedback_embedding_cpu(fused_embedding)? {
            Some(feedback_embedding) => feedback_embedding,
            None => return Ok(None),
        };
        self.continue_audio_codegen_step(
            HiggsContinuationInput {
                session,
                feedback_embedding,
            },
            codegen,
        )
        .map(Some)
    }

    pub fn continue_audio_codegen_step_from_feedback_with_sampled_codes(
        &mut self,
        session: HiggsPromptSession,
        codegen: &mut AudioCodeGenerationSession,
        fused_embedding: &[bf16],
        sampled_codes: Vec<u32>,
    ) -> Result<Option<AudioCodeGenerationStep>> {
        let feedback_embedding = match codegen.feedback_embedding_cpu(fused_embedding)? {
            Some(feedback_embedding) => feedback_embedding,
            None => return Ok(None),
        };
        validate_continuation_input(session, feedback_embedding.len(), codegen)?;
        let input_embedding_bf16: Vec<bf16> = feedback_embedding
            .iter()
            .map(|&value| bf16::from_f32(value))
            .collect();
        let hidden = self
            .executor
            .decode_embedding_last_hidden_bf16_retained(session.request_id(), input_embedding_bf16)?
            .hidden_bf16;
        let prediction = compute_one_step_audio_prediction(&hidden, &self.audio_head)?;
        validate_codegen_session(session, codegen)?;
        codegen
            .push_sampled_codes(
                sampled_codes,
                AudioLogitsSummary::from_prediction(&prediction),
            )
            .map(Some)
    }

    pub fn push_audio_codegen_hidden_step_cpu(
        &mut self,
        session: HiggsPromptSession,
        codegen: &mut AudioCodeGenerationSession,
        final_hidden_bf16: &[bf16],
    ) -> Result<AudioCodeGenerationStep> {
        push_audio_codegen_hidden_step_with_head(
            session,
            codegen,
            final_hidden_bf16,
            &self.audio_head,
        )
    }

    pub fn drop_prompt_session(&mut self, session: impl Into<HiggsPromptSession>) -> Result<()> {
        self.executor.drop_request(session.into().request_id())
    }
}

fn validate_continuation_input(
    session: HiggsPromptSession,
    feedback_embedding_len: usize,
    codegen: &AudioCodeGenerationSession,
) -> Result<()> {
    if codegen.id() != AudioCodegenSessionId::new(session.id()) {
        bail!(
            "Higgs Audio codegen session mismatch: input session={}, codegen session={}",
            session.id(),
            codegen.id().get()
        );
    }
    let expected = crate::one_step_golden::HIDDEN_SIZE;
    if feedback_embedding_len != expected {
        bail!(
            "Higgs Audio feedback embedding len mismatch: expected {expected}, got {feedback_embedding_len}"
        );
    }
    Ok(())
}

fn push_audio_codegen_hidden_step_with_head(
    session: HiggsPromptSession,
    codegen: &mut AudioCodeGenerationSession,
    final_hidden_bf16: &[bf16],
    audio_head: &[bf16],
) -> Result<AudioCodeGenerationStep> {
    validate_codegen_session(session, codegen)?;
    codegen.push_final_normed_hidden_cpu(final_hidden_bf16, audio_head)
}

fn validate_codegen_session(
    session: HiggsPromptSession,
    codegen: &AudioCodeGenerationSession,
) -> Result<()> {
    if codegen.id() != AudioCodegenSessionId::new(session.id()) {
        bail!(
            "Higgs Audio codegen session mismatch: input session={}, codegen session={}",
            session.id(),
            codegen.id().get()
        );
    }
    Ok(())
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
            load_qwen3_executor_from_alias_config(model_dir, qwen3_config_dir, device_ordinal)
        }
        HiggsRuntimeSource::AutoConfigAlias { qwen3_config_dir } => {
            prepare_qwen3_config_view(model_dir, qwen3_config_dir)?;
            load_qwen3_executor_from_alias_config(model_dir, qwen3_config_dir, device_ordinal)
        }
    }
}

fn load_qwen3_executor_from_alias_config(
    model_dir: &Path,
    qwen3_config_dir: &Path,
    device_ordinal: usize,
) -> Result<Qwen3Executor> {
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

fn prepare_qwen3_config_view(model_dir: &Path, qwen3_config_dir: &Path) -> Result<()> {
    let config = HiggsConfig::from_model_dir(model_dir)?;
    let manifest = HiggsWeightManifest::from_model_dir(model_dir)?;
    let plan = HiggsRuntimeLoadPlan::from_manifest(&config, &manifest)?;
    write_qwen3_config_view(qwen3_config_dir, &config, &plan)?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
    use crate::one_step_golden::HIDDEN_SIZE;
    use crate::one_step_golden::NUM_CODEBOOKS;

    fn seeded_prefill(session_id: u64, prompt_tokens: usize) -> HiggsPromptSessionPrefill {
        let mut logits = vec![0.0f32; NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE];
        for codebook in 0..NUM_CODEBOOKS {
            logits[codebook * CODEBOOK_VOCAB_SIZE + codebook] = 1.0;
        }
        HiggsPromptSessionPrefill {
            session: HiggsPromptSession::new(session_id),
            prompt_tokens,
            final_hidden_bf16: vec![bf16::from_f32(0.0); HIDDEN_SIZE],
            audio: OneStepAudioPrediction::from_logits(logits),
        }
    }

    #[test]
    fn prompt_prefill_seed_can_start_audio_codegen_trace() {
        let prefill = seeded_prefill(7, 16);
        let codegen = prefill.into_audio_codegen_session().unwrap();

        assert_eq!(codegen.prompt_tokens(), 16);
        assert_eq!(codegen.steps(), 1);
        assert_eq!(codegen.trace().steps[0].sampled_codes[0], 0);
        assert_eq!(codegen.delayed_codes().len(), 1);
    }

    #[test]
    fn codegen_seed_keeps_higgs_session_with_seeded_trace() {
        let prefill = seeded_prefill(11, 32);
        let session = prefill.session;
        let seed = HiggsAudioCodegenSeed {
            session,
            codegen: prefill.into_audio_codegen_session().unwrap(),
        };

        assert_eq!(seed.session.id(), 11);
        assert_eq!(seed.codegen.prompt_tokens(), 32);
        assert_eq!(seed.codegen.steps(), 1);
    }

    #[test]
    fn continuation_entry_rejects_wrong_codegen_session_before_executor() {
        let prefill = seeded_prefill(13, 32);
        let codegen = prefill.into_audio_codegen_session().unwrap();

        let err = validate_continuation_input(HiggsPromptSession::new(14), HIDDEN_SIZE, &codegen)
            .unwrap_err()
            .to_string();

        assert!(err.contains("codegen session mismatch"));
    }

    #[test]
    fn continuation_entry_rejects_wrong_feedback_embedding_len_before_executor() {
        let prefill = seeded_prefill(15, 32);
        let codegen = prefill.into_audio_codegen_session().unwrap();

        let err =
            validate_continuation_input(HiggsPromptSession::new(15), HIDDEN_SIZE - 1, &codegen)
                .unwrap_err()
                .to_string();

        assert!(err.contains("feedback embedding len mismatch"));
    }

    #[test]
    fn runtime_can_push_hidden_step_after_backend_returns_final_hidden() {
        let prefill = seeded_prefill(19, 32);
        let mut codegen = prefill.into_audio_codegen_session().unwrap();
        let mut audio_head =
            vec![bf16::from_f32(0.0); NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE];
        for codebook in 0..NUM_CODEBOOKS {
            let code = 30 + codebook as u32;
            let row = codebook * CODEBOOK_VOCAB_SIZE + code as usize;
            audio_head[row * HIDDEN_SIZE] = bf16::from_f32(1.0);
        }

        let step = push_audio_codegen_hidden_step_with_head(
            HiggsPromptSession::new(19),
            &mut codegen,
            &vec![bf16::from_f32(1.0); HIDDEN_SIZE],
            &audio_head,
        )
        .unwrap();

        assert_eq!(step.sampled_codes, vec![30, 31, 32, 33, 34, 35, 36, 37]);
        assert_eq!(codegen.trace().steps.len(), 2);
    }

    #[test]
    fn runtime_rejects_hidden_step_for_wrong_codegen_session() {
        let prefill = seeded_prefill(20, 32);
        let mut codegen = prefill.into_audio_codegen_session().unwrap();
        let audio_head =
            vec![bf16::from_f32(0.0); NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE];

        let err = push_audio_codegen_hidden_step_with_head(
            HiggsPromptSession::new(21),
            &mut codegen,
            &vec![bf16::from_f32(1.0); HIDDEN_SIZE],
            &audio_head,
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("codegen session mismatch"));
    }
}

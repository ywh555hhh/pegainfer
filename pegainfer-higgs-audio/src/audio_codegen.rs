use anyhow::Result;
use anyhow::bail;
use half::bf16;

use crate::codebook_embedding::FusedCodebookShape;
use crate::decode_session::HiggsDecodeSession;
use crate::decode_trace::AudioLogitsSummary;
use crate::decode_trace::DecodeTrace;
use crate::one_step_actual::OneStepAudioPrediction;
use crate::one_step_actual::compute_one_step_audio_prediction;
use crate::one_step_golden::HIDDEN_SIZE;
use crate::one_step_golden::NUM_CODEBOOKS;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AudioCodegenSessionId(u64);

impl AudioCodegenSessionId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioCodeGenerationSession {
    id: AudioCodegenSessionId,
    prompt_tokens: usize,
    decode: HiggsDecodeSession,
    trace: DecodeTrace,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioCodeGenerationStep {
    pub step: usize,
    pub sampled_codes: Vec<u32>,
    pub delayed_codes: Option<Vec<u32>>,
    pub raw_codes: Vec<Vec<u32>>,
    pub generation_done: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioCodeGenerationRun {
    pub steps: Vec<AudioCodeGenerationStep>,
    pub raw_codes: Vec<Vec<u32>>,
    pub generation_done: bool,
}

pub trait AudioContinuationBackend {
    fn next_prediction(
        &mut self,
        session_id: AudioCodegenSessionId,
        step: usize,
        feedback_embedding: Option<&[f32]>,
    ) -> Result<OneStepAudioPrediction>;
}

pub trait HiddenStateContinuationBackend {
    fn next_final_normed_hidden(
        &mut self,
        session_id: AudioCodegenSessionId,
        step: usize,
        feedback_embedding: Option<&[f32]>,
    ) -> Result<Vec<bf16>>;
}

pub struct HiddenStateAudioHeadBackend<'a, B> {
    inner: B,
    audio_head: &'a [bf16],
}

impl<'a, B> HiddenStateAudioHeadBackend<'a, B> {
    pub fn new(inner: B, audio_head: &'a [bf16]) -> Self {
        Self { inner, audio_head }
    }

    pub fn into_inner(self) -> B {
        self.inner
    }
}

impl<B> AudioContinuationBackend for HiddenStateAudioHeadBackend<'_, B>
where
    B: HiddenStateContinuationBackend,
{
    fn next_prediction(
        &mut self,
        session_id: AudioCodegenSessionId,
        step: usize,
        feedback_embedding: Option<&[f32]>,
    ) -> Result<OneStepAudioPrediction> {
        let final_hidden =
            self.inner
                .next_final_normed_hidden(session_id, step, feedback_embedding)?;
        if final_hidden.len() != HIDDEN_SIZE {
            bail!(
                "continuation final normed hidden len mismatch: expected {HIDDEN_SIZE}, got {}",
                final_hidden.len()
            );
        }
        compute_one_step_audio_prediction(&final_hidden, self.audio_head)
    }
}

impl AudioCodeGenerationSession {
    pub fn new(prompt_tokens: usize) -> Self {
        Self::new_with_id(AudioCodegenSessionId::new(0), prompt_tokens)
    }

    pub fn new_with_id(id: AudioCodegenSessionId, prompt_tokens: usize) -> Self {
        Self {
            id,
            prompt_tokens,
            decode: HiggsDecodeSession::new_higgs_v3(),
            trace: DecodeTrace::new(prompt_tokens),
        }
    }

    pub fn id(&self) -> AudioCodegenSessionId {
        self.id
    }

    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    pub fn steps(&self) -> usize {
        self.trace.steps.len()
    }

    pub fn generation_done(&self) -> bool {
        self.decode.generation_done()
    }

    pub fn delayed_codes(&self) -> &[Vec<u32>] {
        self.decode.delayed_codes()
    }

    pub fn raw_codes(&self, allow_short: bool) -> Result<Vec<Vec<u32>>> {
        self.decode.raw_codes(allow_short)
    }

    pub fn feedback_embedding_cpu(&self, fused_embedding: &[bf16]) -> Result<Option<Vec<f32>>> {
        self.decode.feedback_embedding_cpu(fused_embedding)
    }

    pub fn feedback_embedding_cpu_with_shape(
        &self,
        fused_embedding: &[bf16],
        shape: FusedCodebookShape,
    ) -> Result<Option<Vec<f32>>> {
        self.decode
            .feedback_embedding_cpu_with_shape(fused_embedding, shape)
    }

    pub fn trace(&self) -> &DecodeTrace {
        &self.trace
    }

    pub fn push_prediction(
        &mut self,
        prediction: &OneStepAudioPrediction,
    ) -> Result<AudioCodeGenerationStep> {
        let sampled_codes = sampled_codes_from_prediction(prediction)?;
        self.push_sampled_codes(
            sampled_codes,
            AudioLogitsSummary::from_prediction(prediction),
        )
    }

    pub fn push_final_normed_hidden_cpu(
        &mut self,
        final_hidden: &[bf16],
        audio_head: &[bf16],
    ) -> Result<AudioCodeGenerationStep> {
        if final_hidden.len() != HIDDEN_SIZE {
            bail!(
                "continuation final normed hidden len mismatch: expected {HIDDEN_SIZE}, got {}",
                final_hidden.len()
            );
        }
        let prediction = compute_one_step_audio_prediction(final_hidden, audio_head)?;
        self.push_prediction(&prediction)
    }

    pub fn push_sampled_codes(
        &mut self,
        sampled_codes: Vec<u32>,
        logits: AudioLogitsSummary,
    ) -> Result<AudioCodeGenerationStep> {
        if sampled_codes.len() != NUM_CODEBOOKS {
            bail!(
                "sampled code row has {} codebooks, expected {NUM_CODEBOOKS}",
                sampled_codes.len()
            );
        }
        if logits.argmax.len() != NUM_CODEBOOKS {
            bail!(
                "logits summary argmax has {} codebooks, expected {NUM_CODEBOOKS}",
                logits.argmax.len()
            );
        }

        let step = self.trace.steps.len();
        let decode = self.decode.step_from_sampled_codes(&sampled_codes)?;
        let raw_codes = self.decode.raw_codes(true)?;
        self.trace.push_step(
            step,
            sampled_codes.clone(),
            decode.emitted_codes.clone(),
            raw_codes.clone(),
            decode.generation_done,
            logits,
        )?;

        Ok(AudioCodeGenerationStep {
            step,
            sampled_codes,
            delayed_codes: decode.emitted_codes,
            raw_codes,
            generation_done: decode.generation_done,
        })
    }

    pub fn run_continuation_steps(
        &mut self,
        max_steps: usize,
        fused_embedding: &[bf16],
        backend: &mut impl AudioContinuationBackend,
    ) -> Result<AudioCodeGenerationRun> {
        let mut steps = Vec::new();
        for _ in 0..max_steps {
            if self.generation_done() {
                break;
            }
            let feedback_embedding = self.feedback_embedding_cpu(fused_embedding)?;
            let prediction =
                backend.next_prediction(self.id, self.steps(), feedback_embedding.as_deref())?;
            steps.push(self.push_prediction(&prediction)?);
        }
        Ok(AudioCodeGenerationRun {
            steps,
            raw_codes: self.raw_codes(true)?,
            generation_done: self.generation_done(),
        })
    }
}

pub fn sampled_codes_from_prediction(prediction: &OneStepAudioPrediction) -> Result<Vec<u32>> {
    prediction.validate()?;
    prediction
        .argmax
        .iter()
        .enumerate()
        .map(|(codebook, &id)| {
            u32::try_from(id).map_err(|_| {
                anyhow::anyhow!("argmax for codebook {codebook} must be non-negative, got {id}")
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delay_pattern::BOC_ID;
    use crate::delay_pattern::EOC_ID;
    use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
    use crate::one_step_golden::HIDDEN_SIZE;

    fn logits_for_argmax(row: &[u32]) -> OneStepAudioPrediction {
        let mut logits = vec![0.0f32; NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE];
        for (codebook, &code) in row.iter().enumerate() {
            logits[codebook * CODEBOOK_VOCAB_SIZE + code as usize] = 1.0;
        }
        OneStepAudioPrediction::from_logits(logits)
    }

    #[test]
    fn code_generation_session_records_delay_raw_trace_and_feedback() {
        let mut session = AudioCodeGenerationSession::new(12);

        let first = session
            .push_prediction(&logits_for_argmax(&[1, 101, 201, 301, 401, 501, 601, 701]))
            .unwrap();
        assert_eq!(first.step, 0);
        assert_eq!(
            first.delayed_codes.as_ref().unwrap(),
            &[1, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID]
        );
        assert!(first.raw_codes.is_empty());

        for row in [
            [2, 102, 202, 302, 402, 502, 602, 702],
            [3, 103, 203, 303, 403, 503, 603, 703],
            [4, 104, 204, 304, 404, 504, 604, 704],
            [5, 105, 205, 305, 405, 505, 605, 705],
            [6, 106, 206, 306, 406, 506, 606, 706],
            [7, 107, 207, 307, 407, 507, 607, 707],
            [EOC_ID, 108, 208, 308, 408, 508, 608, 708],
        ] {
            session.push_prediction(&logits_for_argmax(&row)).unwrap();
        }

        assert_eq!(session.steps(), 8);
        assert_eq!(
            session.raw_codes(false).unwrap(),
            vec![vec![1, 102, 203, 304, 405, 506, 607, 708]]
        );
        assert_eq!(session.trace().steps.len(), 8);
        assert!(!session.trace().steps[7].generation_done);
    }

    #[test]
    fn feedback_embedding_uses_last_delayed_row() {
        let shape = FusedCodebookShape {
            num_codebooks: NUM_CODEBOOKS,
            vocab_size: 1026,
            hidden_size: 1,
        };
        let mut weight = vec![bf16::from_f32(0.0); shape.num_codebooks * shape.vocab_size];
        weight[3] = bf16::from_f32(5.0);
        for codebook in 1..NUM_CODEBOOKS {
            weight[codebook * shape.vocab_size + BOC_ID as usize] = bf16::from_f32(7.0);
        }

        let mut session = AudioCodeGenerationSession::new(4);
        let logits = AudioLogitsSummary {
            argmax: vec![3, 4, 5, 6, 7, 8, 9, 10],
            top_ids: None,
            logits: None,
            top64_min_overlap_with_previous: None,
            logits_l2_norm: 1.0,
        };
        session
            .push_sampled_codes(vec![3, 4, 5, 6, 7, 8, 9, 10], logits)
            .unwrap();

        assert_eq!(
            session
                .feedback_embedding_cpu_with_shape(&weight, shape)
                .unwrap(),
            Some(vec![54.0])
        );
    }

    struct ScriptedBackend {
        rows: Vec<Vec<u32>>,
        seen_sessions: Vec<AudioCodegenSessionId>,
        seen_feedback: Vec<Option<Vec<f32>>>,
    }

    impl AudioContinuationBackend for ScriptedBackend {
        fn next_prediction(
            &mut self,
            session_id: AudioCodegenSessionId,
            step: usize,
            feedback_embedding: Option<&[f32]>,
        ) -> Result<OneStepAudioPrediction> {
            self.seen_sessions.push(session_id);
            self.seen_feedback
                .push(feedback_embedding.map(<[f32]>::to_vec));
            Ok(logits_for_argmax(&self.rows[step - 1]))
        }
    }

    struct ScriptedHiddenBackend {
        hidden: Vec<Vec<bf16>>,
        seen_sessions: Vec<AudioCodegenSessionId>,
        seen_feedback: Vec<Option<Vec<f32>>>,
    }

    impl HiddenStateContinuationBackend for ScriptedHiddenBackend {
        fn next_final_normed_hidden(
            &mut self,
            session_id: AudioCodegenSessionId,
            step: usize,
            feedback_embedding: Option<&[f32]>,
        ) -> Result<Vec<bf16>> {
            self.seen_sessions.push(session_id);
            self.seen_feedback
                .push(feedback_embedding.map(<[f32]>::to_vec));
            Ok(self.hidden[step - 1].clone())
        }
    }

    #[test]
    fn continuation_driver_consumes_feedback_and_records_real_loop_trace() {
        let mut session =
            AudioCodeGenerationSession::new_with_id(AudioCodegenSessionId::new(42), 12);
        session
            .push_prediction(&logits_for_argmax(&[1, 101, 201, 301, 401, 501, 601, 701]))
            .unwrap();

        let mut fused_embedding =
            vec![bf16::from_f32(0.0); NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE];
        for codebook in 0..NUM_CODEBOOKS {
            let code = if codebook == 0 { 1 } else { BOC_ID };
            let row = codebook * CODEBOOK_VOCAB_SIZE + code as usize;
            fused_embedding[row * HIDDEN_SIZE + codebook] = bf16::from_f32(1.0);
        }
        let mut backend = ScriptedBackend {
            rows: vec![
                vec![2, 102, 202, 302, 402, 502, 602, 702],
                vec![3, 103, 203, 303, 403, 503, 603, 703],
            ],
            seen_sessions: Vec::new(),
            seen_feedback: Vec::new(),
        };

        let run = session
            .run_continuation_steps(2, &fused_embedding, &mut backend)
            .unwrap();

        assert_eq!(run.steps.len(), 2);
        assert_eq!(session.steps(), 3);
        assert_eq!(session.trace().steps[2].sampled_codes[0], 3);
        assert_eq!(
            backend.seen_sessions,
            vec![
                AudioCodegenSessionId::new(42),
                AudioCodegenSessionId::new(42)
            ]
        );
        assert_eq!(backend.seen_feedback.len(), 2);
        let first_feedback = backend.seen_feedback[0].as_ref().unwrap();
        assert_eq!(first_feedback.len(), HIDDEN_SIZE);
        assert_eq!(
            &first_feedback[..NUM_CODEBOOKS],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
        assert_ne!(
            backend.seen_feedback[0], backend.seen_feedback[1],
            "second continuation step must consume the updated delayed row"
        );
    }

    #[test]
    fn hidden_backend_adapter_projects_final_hidden_through_higgs_audio_head() {
        let mut session =
            AudioCodeGenerationSession::new_with_id(AudioCodegenSessionId::new(77), 12);
        session
            .push_prediction(&logits_for_argmax(&[1, 101, 201, 301, 401, 501, 601, 701]))
            .unwrap();

        let mut fused_embedding =
            vec![bf16::from_f32(0.0); NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE];
        for codebook in 0..NUM_CODEBOOKS {
            let code = if codebook == 0 { 1 } else { BOC_ID };
            let row = codebook * CODEBOOK_VOCAB_SIZE + code as usize;
            fused_embedding[row * HIDDEN_SIZE + codebook] = bf16::from_f32(1.0);
        }

        let mut audio_head =
            vec![bf16::from_f32(0.0); NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE];
        for codebook in 0..NUM_CODEBOOKS {
            let code = 10 + codebook as u32;
            let row = codebook * CODEBOOK_VOCAB_SIZE + code as usize;
            audio_head[row * HIDDEN_SIZE] = bf16::from_f32(1.0);
        }
        let hidden_backend = ScriptedHiddenBackend {
            hidden: vec![vec![bf16::from_f32(1.0); HIDDEN_SIZE]],
            seen_sessions: Vec::new(),
            seen_feedback: Vec::new(),
        };
        let mut backend = HiddenStateAudioHeadBackend::new(hidden_backend, &audio_head);

        let run = session
            .run_continuation_steps(1, &fused_embedding, &mut backend)
            .unwrap();
        let hidden_backend = backend.into_inner();

        assert_eq!(run.steps.len(), 1);
        assert_eq!(
            run.steps[0].sampled_codes,
            vec![10, 11, 12, 13, 14, 15, 16, 17]
        );
        assert_eq!(
            hidden_backend.seen_sessions,
            vec![AudioCodegenSessionId::new(77)]
        );
        assert_eq!(hidden_backend.seen_feedback.len(), 1);
        assert_eq!(
            &hidden_backend.seen_feedback[0].as_ref().unwrap()[..NUM_CODEBOOKS],
            &[1.0; NUM_CODEBOOKS]
        );
    }

    #[test]
    fn session_can_push_final_hidden_through_higgs_audio_head() {
        let mut session =
            AudioCodeGenerationSession::new_with_id(AudioCodegenSessionId::new(88), 12);
        session
            .push_prediction(&logits_for_argmax(&[1, 101, 201, 301, 401, 501, 601, 701]))
            .unwrap();

        let final_hidden = vec![bf16::from_f32(1.0); HIDDEN_SIZE];
        let mut audio_head =
            vec![bf16::from_f32(0.0); NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE];
        for codebook in 0..NUM_CODEBOOKS {
            let code = 20 + codebook as u32;
            let row = codebook * CODEBOOK_VOCAB_SIZE + code as usize;
            audio_head[row * HIDDEN_SIZE] = bf16::from_f32(1.0);
        }

        let step = session
            .push_final_normed_hidden_cpu(&final_hidden, &audio_head)
            .unwrap();

        assert_eq!(step.sampled_codes, vec![20, 21, 22, 23, 24, 25, 26, 27]);
        assert_eq!(session.trace().steps[1].sampled_codes, step.sampled_codes);
    }
}

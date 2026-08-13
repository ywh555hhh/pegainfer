use anyhow::Result;
use half::bf16;

use crate::codebook_embedding::FusedCodebookShape;
use crate::codebook_embedding::fused_codebook_embedding_cpu_with_shape;
use crate::delay_pattern::DelayPatternState;
use crate::delay_pattern::reverse_delay_pattern;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiggsDecodeSession {
    delay: DelayPatternState,
    delayed_codes: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiggsDecodeStep {
    pub emitted_codes: Option<Vec<u32>>,
    pub generation_done: bool,
    pub delayed_rows: usize,
}

impl HiggsDecodeSession {
    pub fn new(num_codebooks: usize) -> Result<Self> {
        Ok(Self {
            delay: DelayPatternState::new(num_codebooks)?,
            delayed_codes: Vec::new(),
        })
    }

    pub fn new_higgs_v3() -> Self {
        Self {
            delay: DelayPatternState::new_higgs_v3(),
            delayed_codes: Vec::new(),
        }
    }

    pub fn step_from_sampled_codes(&mut self, sampled_codes: &[u32]) -> Result<HiggsDecodeStep> {
        let step = self.delay.step_from_sampled_codes(sampled_codes)?;
        if let Some(codes) = step.codes.as_ref() {
            self.delayed_codes.push(codes.clone());
        }
        Ok(HiggsDecodeStep {
            emitted_codes: step.codes,
            generation_done: step.generation_done,
            delayed_rows: self.delayed_codes.len(),
        })
    }

    pub fn generation_done(&self) -> bool {
        self.delay.generation_done()
    }

    pub fn delayed_codes(&self) -> &[Vec<u32>] {
        &self.delayed_codes
    }

    pub fn feedback_codes(&self) -> Option<&[u32]> {
        self.delay.last_codes()
    }

    pub fn feedback_embedding_cpu(&self, fused_embedding: &[bf16]) -> Result<Option<Vec<f32>>> {
        self.feedback_embedding_cpu_with_shape(fused_embedding, FusedCodebookShape::higgs_v3())
    }

    pub fn feedback_embedding_cpu_with_shape(
        &self,
        fused_embedding: &[bf16],
        shape: FusedCodebookShape,
    ) -> Result<Option<Vec<f32>>> {
        self.feedback_codes()
            .map(|codes| fused_codebook_embedding_cpu_with_shape(codes, fused_embedding, shape))
            .transpose()
    }

    pub fn raw_codes(&self, allow_short: bool) -> Result<Vec<Vec<u32>>> {
        reverse_delay_pattern(&self.delayed_codes, allow_short)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook_embedding::fused_codebook_embedding_cpu_with_shape;
    use crate::delay_pattern::BOC_ID;
    use crate::delay_pattern::EOC_ID;

    #[test]
    fn session_accumulates_delayed_rows_and_recovers_raw_codes_after_winddown() {
        let mut session = HiggsDecodeSession::new(3).unwrap();

        for sampled in [
            [1, 101, 201],
            [4, 2, 202],
            [7, 5, 3],
            [EOC_ID, 8, 6],
            [900, 901, 9],
        ] {
            session.step_from_sampled_codes(&sampled).unwrap();
        }

        assert!(session.generation_done());
        assert_eq!(
            session.delayed_codes(),
            &[
                vec![1, BOC_ID, BOC_ID],
                vec![4, 2, BOC_ID],
                vec![7, 5, 3],
                vec![EOC_ID, 8, 6],
                vec![900, 901, 9],
            ]
        );
        assert_eq!(
            session.raw_codes(false).unwrap(),
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        );
    }

    #[test]
    fn session_skips_rows_after_it_was_already_done() {
        let mut session = HiggsDecodeSession::new(2).unwrap();

        session.step_from_sampled_codes(&[1, 11]).unwrap();
        session.step_from_sampled_codes(&[2, 12]).unwrap();
        let done = session.step_from_sampled_codes(&[EOC_ID, 13]).unwrap();
        assert!(done.generation_done);
        assert_eq!(done.emitted_codes, Some(vec![EOC_ID, 13]));
        assert_eq!(session.delayed_codes().len(), 3);

        let overrun = session.step_from_sampled_codes(&[3, 14]).unwrap();
        assert_eq!(overrun.emitted_codes, None);
        assert!(overrun.generation_done);
        assert_eq!(session.delayed_codes().len(), 3);
    }

    #[test]
    fn session_exposes_last_feedback_codes_and_embedding() {
        let mut session = HiggsDecodeSession::new(2).unwrap();
        assert_eq!(session.feedback_codes(), None);

        session.step_from_sampled_codes(&[1, 2]).unwrap();
        assert_eq!(session.feedback_codes(), Some([1, BOC_ID].as_slice()));

        let shape = FusedCodebookShape {
            num_codebooks: 2,
            vocab_size: 1026,
            hidden_size: 1,
        };
        let mut weight = vec![bf16::from_f32(0.0); shape.num_codebooks * shape.vocab_size];
        weight[1] = bf16::from_f32(5.0);
        weight[shape.vocab_size + BOC_ID as usize] = bf16::from_f32(7.0);

        let via_session = session
            .feedback_embedding_cpu_with_shape(&weight, shape)
            .unwrap()
            .expect("feedback codes exist");
        let direct = fused_codebook_embedding_cpu_with_shape(&[1, BOC_ID], &weight, shape).unwrap();
        assert_eq!(via_session, direct);
    }
}

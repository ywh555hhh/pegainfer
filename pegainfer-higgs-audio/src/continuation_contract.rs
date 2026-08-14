use anyhow::Result;
use anyhow::bail;

use crate::one_step_golden::HIDDEN_SIZE;
use crate::one_step_golden::NUM_CODEBOOKS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuationInputKind {
    FeedbackEmbedding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuationOutputKind {
    FinalNormedHidden,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuationStepContract {
    pub input: ContinuationInputKind,
    pub output: ContinuationOutputKind,
    pub input_hidden_size: usize,
    pub output_hidden_size: usize,
    pub codebooks: usize,
    pub retained_kv: bool,
    pub full_prompt_rebuild: bool,
}

impl ContinuationStepContract {
    pub const fn higgs_v3() -> Self {
        Self {
            input: ContinuationInputKind::FeedbackEmbedding,
            output: ContinuationOutputKind::FinalNormedHidden,
            input_hidden_size: HIDDEN_SIZE,
            output_hidden_size: HIDDEN_SIZE,
            codebooks: NUM_CODEBOOKS,
            retained_kv: true,
            full_prompt_rebuild: false,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.input != ContinuationInputKind::FeedbackEmbedding {
            bail!("Higgs continuation input must be feedback embedding, not token id");
        }
        if self.output != ContinuationOutputKind::FinalNormedHidden {
            bail!("Higgs continuation output must be final normed hidden");
        }
        if self.input_hidden_size != HIDDEN_SIZE || self.output_hidden_size != HIDDEN_SIZE {
            bail!(
                "Higgs continuation hidden size mismatch: input={}, output={}, expected={HIDDEN_SIZE}",
                self.input_hidden_size,
                self.output_hidden_size
            );
        }
        if self.codebooks != NUM_CODEBOOKS {
            bail!(
                "Higgs continuation codebook count mismatch: got {}, expected {NUM_CODEBOOKS}",
                self.codebooks
            );
        }
        if !self.retained_kv {
            bail!("Higgs continuation must use retained KV");
        }
        if self.full_prompt_rebuild {
            bail!("Higgs continuation must not rebuild the full prompt");
        }
        Ok(())
    }
}

pub fn higgs_v3_continuation_contract() -> ContinuationStepContract {
    ContinuationStepContract::higgs_v3()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higgs_contract_is_embedding_fed_hidden_returning_retained_decode() {
        let contract = higgs_v3_continuation_contract();

        contract.validate().unwrap();
        assert_eq!(contract.input, ContinuationInputKind::FeedbackEmbedding);
        assert_eq!(contract.output, ContinuationOutputKind::FinalNormedHidden);
        assert!(contract.retained_kv);
        assert!(!contract.full_prompt_rebuild);
    }

    #[test]
    fn contract_rejects_full_prompt_rebuild() {
        let mut contract = higgs_v3_continuation_contract();
        contract.full_prompt_rebuild = true;

        let err = contract.validate().unwrap_err().to_string();

        assert!(err.contains("must not rebuild"));
    }
}

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use serde_json::Value;

use crate::codec_input::CodeRowsLayout;
use crate::codec_input::codec_input_from_rows;
use crate::delay_pattern::DelayPatternState;
use crate::one_step_actual::OneStepAudioPrediction;
use crate::one_step_golden::NUM_CODEBOOKS;

pub const TRACE_SCHEMA: &str = "higgs-audio-native-codegen-trace-v1";

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeTrace {
    pub prompt_tokens: usize,
    pub steps: Vec<DecodeTraceStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeTraceStep {
    pub step: usize,
    pub sampled_codes: Vec<u32>,
    pub delayed_codes: Option<Vec<u32>>,
    pub raw_codes: Vec<Vec<u32>>,
    pub generation_done: bool,
    pub logits: AudioLogitsSummary,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioLogitsSummary {
    pub argmax: Vec<i64>,
    pub top_ids: Option<Vec<Vec<i64>>>,
    pub logits: Option<Vec<Vec<f32>>>,
    pub top64_min_overlap_with_previous: Option<usize>,
    pub logits_l2_norm: f32,
}

impl DecodeTrace {
    pub fn new(prompt_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            steps: Vec::new(),
        }
    }

    pub fn push_step(
        &mut self,
        step: usize,
        sampled_codes: Vec<u32>,
        delayed_codes: Option<Vec<u32>>,
        raw_codes: Vec<Vec<u32>>,
        generation_done: bool,
        logits: AudioLogitsSummary,
    ) -> Result<()> {
        if sampled_codes.len() != NUM_CODEBOOKS {
            bail!(
                "sampled code row has {} codebooks, expected {NUM_CODEBOOKS}",
                sampled_codes.len()
            );
        }
        if let Some(delayed) = delayed_codes.as_ref()
            && delayed.len() != NUM_CODEBOOKS
        {
            bail!(
                "delayed code row has {} codebooks, expected {NUM_CODEBOOKS}",
                delayed.len()
            );
        }
        for (row_idx, row) in raw_codes.iter().enumerate() {
            if row.len() != NUM_CODEBOOKS {
                bail!(
                    "raw code row {row_idx} has {} codebooks, expected {NUM_CODEBOOKS}",
                    row.len()
                );
            }
        }
        self.steps.push(DecodeTraceStep {
            step,
            sampled_codes,
            delayed_codes,
            raw_codes,
            generation_done,
            logits,
        });
        Ok(())
    }

    pub fn to_json_value(&self) -> Value {
        serde_json::json!({
            "schema": TRACE_SCHEMA,
            "prompt_tokens": self.prompt_tokens,
            "steps": self.steps.iter().map(DecodeTraceStep::to_json_value).collect::<Vec<_>>(),
        })
    }
}

impl DecodeTraceStep {
    fn to_json_value(&self) -> Value {
        serde_json::json!({
            "step": self.step,
            "sampled_codes": self.sampled_codes,
            "delayed_codes": self.delayed_codes,
            "raw_codes": self.raw_codes,
            "generation_done": self.generation_done,
            "logits": self.logits.to_json_value(),
        })
    }
}

impl AudioLogitsSummary {
    pub fn from_prediction(prediction: &OneStepAudioPrediction) -> Self {
        let l2_sq = prediction
            .logits
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        Self {
            argmax: prediction.argmax.clone(),
            top_ids: Some(
                prediction
                    .top_ids
                    .chunks_exact(crate::one_step_golden::TOP_K)
                    .map(|row| row.to_vec())
                    .collect(),
            ),
            logits: Some(
                prediction
                    .logits
                    .chunks_exact(crate::one_step_golden::CODEBOOK_VOCAB_SIZE)
                    .map(|row| row.to_vec())
                    .collect(),
            ),
            top64_min_overlap_with_previous: None,
            logits_l2_norm: l2_sq.sqrt(),
        }
    }

    fn to_json_value(&self) -> Value {
        serde_json::json!({
            "argmax": self.argmax,
            "top_ids": self.top_ids,
            "logits": self.logits,
            "top64_min_overlap_with_previous": self.top64_min_overlap_with_previous,
            "logits_l2_norm": self.logits_l2_norm,
        })
    }
}

pub fn write_decode_trace_json(path: impl AsRef<Path>, trace: &DecodeTrace) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(&trace.to_json_value())
        .context("serialize Higgs decode trace JSON")?;
    std::fs::write(path.as_ref(), bytes)
        .with_context(|| format!("write {}", path.as_ref().display()))
}

/// Build a trace from already-sampled rows. This is a CPU reference harness for
/// the trace contract; the production decode loop should push the same records
/// from native retained-KV steps.
pub fn trace_sampled_code_rows(prompt_tokens: usize, rows: &[Vec<u32>]) -> Result<DecodeTrace> {
    let mut session = DelayPatternState::new_higgs_v3();
    let mut delayed_rows = Vec::new();
    let mut trace = DecodeTrace::new(prompt_tokens);
    for (step, sampled) in rows.iter().enumerate() {
        let state = session.step_from_sampled_codes(sampled)?;
        if let Some(delayed) = state.codes.as_ref() {
            delayed_rows.push(delayed.clone());
        }
        let raw_codes = codec_input_from_rows(&delayed_rows, CodeRowsLayout::Delayed)
            .unwrap_or_else(|_| Vec::new());
        trace.push_step(
            step,
            sampled.clone(),
            state.codes,
            raw_codes,
            state.generation_done,
            AudioLogitsSummary {
                argmax: sampled.iter().map(|&code| i64::from(code)).collect(),
                top_ids: None,
                logits: None,
                top64_min_overlap_with_previous: None,
                logits_l2_norm: 0.0,
            },
        )?;
    }
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delay_pattern::BOC_ID;
    use crate::delay_pattern::EOC_ID;

    #[test]
    fn trace_records_incremental_delayed_and_raw_rows() {
        let rows = vec![
            vec![1, 101, 201, 301, 401, 501, 601, 701],
            vec![2, 102, 202, 302, 402, 502, 602, 702],
            vec![3, 103, 203, 303, 403, 503, 603, 703],
            vec![4, 104, 204, 304, 404, 504, 604, 704],
            vec![5, 105, 205, 305, 405, 505, 605, 705],
            vec![6, 106, 206, 306, 406, 506, 606, 706],
            vec![7, 107, 207, 307, 407, 507, 607, 707],
            vec![EOC_ID, 108, 208, 308, 408, 508, 608, 708],
        ];

        let trace = trace_sampled_code_rows(12, &rows).unwrap();

        assert_eq!(trace.prompt_tokens, 12);
        assert_eq!(trace.steps.len(), 8);
        assert_eq!(
            trace.steps[0].delayed_codes.as_ref().unwrap(),
            &[1, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID]
        );
        assert_eq!(
            trace.steps[7].raw_codes,
            vec![vec![1, 102, 203, 304, 405, 506, 607, 708]]
        );
    }

    #[test]
    fn trace_json_has_stable_schema() {
        let trace = trace_sampled_code_rows(
            1,
            &[
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                vec![EOC_ID, 9, 10, 11, 12, 13, 14, 15],
            ],
        )
        .unwrap();
        let json = trace.to_json_value();

        assert_eq!(json["schema"], TRACE_SCHEMA);
        assert_eq!(json["prompt_tokens"], 1);
        assert_eq!(json["steps"][0]["step"], 0);
        assert_eq!(json["steps"][0]["logits"]["argmax"][0], 1);
    }

    #[test]
    fn trace_rejects_wrong_sampled_width() {
        let error = DecodeTrace::new(1)
            .push_step(
                0,
                vec![1],
                None,
                Vec::new(),
                false,
                AudioLogitsSummary {
                    argmax: vec![1],
                    top_ids: None,
                    logits: None,
                    top64_min_overlap_with_previous: None,
                    logits_l2_norm: 0.0,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("expected"));
    }
}

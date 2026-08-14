use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde_json::Value;

use crate::decode_trace::TRACE_SCHEMA;
use crate::one_step_golden::TOP_K;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodegenTraceTolerances {
    pub logits_cosine_min: f32,
    pub logits_max_abs_tol: f32,
    pub logits_mean_abs_tol: f32,
    pub logits_p99_abs_tol: f32,
    pub argmax_regret_tol: f32,
    pub topk_min_overlap: usize,
}

impl Default for CodegenTraceTolerances {
    fn default() -> Self {
        Self {
            logits_cosine_min: 0.999,
            logits_max_abs_tol: 0.50,
            logits_mean_abs_tol: 0.05,
            logits_p99_abs_tol: 0.25,
            argmax_regret_tol: 0.20,
            topk_min_overlap: 58,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodegenTraceComparison {
    pub prompt_tokens_match: bool,
    pub steps_compared: usize,
    pub first_divergent_step: Option<usize>,
    pub sampled_rows_exact: bool,
    pub raw_rows_exact: bool,
    pub generation_done_exact: bool,
    pub argmax_positions: usize,
    pub argmax_matches: usize,
    pub argmax_agreement: f32,
    pub logits_cosine: Option<f32>,
    pub logits_max_abs: Option<f32>,
    pub logits_mean_abs: Option<f32>,
    pub logits_p99_abs: Option<f32>,
    pub max_argmax_regret: Option<f32>,
    pub topk_min_overlap: Option<usize>,
    pub topk_mean_overlap: Option<f32>,
    pub full_logits_available: bool,
    pub topk_available: bool,
    pub tolerances: CodegenTraceTolerances,
}

impl CodegenTraceComparison {
    pub fn passed(&self) -> bool {
        self.prompt_tokens_match
            && self.sampled_rows_exact
            && self.raw_rows_exact
            && self.generation_done_exact
            && self.argmax_positions > 0
            && self.argmax_matches == self.argmax_positions
            && self.full_logits_available
            && self.topk_available
            && self.logits_cosine.unwrap_or(0.0) >= self.tolerances.logits_cosine_min
            && self.logits_max_abs.unwrap_or(f32::INFINITY) <= self.tolerances.logits_max_abs_tol
            && self.logits_mean_abs.unwrap_or(f32::INFINITY) <= self.tolerances.logits_mean_abs_tol
            && self.logits_p99_abs.unwrap_or(f32::INFINITY) <= self.tolerances.logits_p99_abs_tol
            && self.max_argmax_regret.unwrap_or(f32::INFINITY) <= self.tolerances.argmax_regret_tol
            && self.topk_min_overlap.unwrap_or(0) >= self.tolerances.topk_min_overlap
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Trace {
    prompt_tokens: usize,
    steps: Vec<Step>,
}

#[derive(Debug, Clone, PartialEq)]
struct Step {
    step: usize,
    sampled_codes: Vec<u32>,
    raw_codes: Vec<Vec<u32>>,
    generation_done: bool,
    argmax: Vec<i64>,
    top_ids: Option<Vec<Vec<i64>>>,
    logits: Option<Vec<Vec<f32>>>,
}

pub fn compare_codegen_trace_files(
    reference: impl AsRef<Path>,
    actual: impl AsRef<Path>,
    tolerances: CodegenTraceTolerances,
) -> Result<CodegenTraceComparison> {
    let reference = load_trace(reference.as_ref())?;
    let actual = load_trace(actual.as_ref())?;
    compare_codegen_traces(&reference, &actual, tolerances)
}

fn compare_codegen_traces(
    reference: &Trace,
    actual: &Trace,
    tolerances: CodegenTraceTolerances,
) -> Result<CodegenTraceComparison> {
    ensure!(
        reference.steps.len() == actual.steps.len(),
        "trace step count mismatch: reference {} actual {}",
        reference.steps.len(),
        actual.steps.len()
    );

    let mut first_divergent_step = None;
    let mut sampled_rows_exact = true;
    let mut raw_rows_exact = true;
    let mut generation_done_exact = true;
    let mut argmax_positions = 0usize;
    let mut argmax_matches = 0usize;
    let mut reference_logits_all = Vec::new();
    let mut actual_logits_all = Vec::new();
    let mut topk_overlaps = Vec::new();
    let mut max_argmax_regret = 0.0f32;
    let mut full_logits_available = true;
    let mut topk_available = true;

    for (reference_step, actual_step) in reference.steps.iter().zip(&actual.steps) {
        if reference_step.step != actual_step.step {
            bail!(
                "trace step id mismatch: reference {} actual {}",
                reference_step.step,
                actual_step.step
            );
        }

        let mut step_diverged = false;
        if reference_step.sampled_codes != actual_step.sampled_codes {
            sampled_rows_exact = false;
            step_diverged = true;
        }
        if reference_step.raw_codes != actual_step.raw_codes {
            raw_rows_exact = false;
            step_diverged = true;
        }
        if reference_step.generation_done != actual_step.generation_done {
            generation_done_exact = false;
            step_diverged = true;
        }

        ensure!(
            reference_step.argmax.len() == actual_step.argmax.len(),
            "argmax width mismatch at step {}: reference {} actual {}",
            reference_step.step,
            reference_step.argmax.len(),
            actual_step.argmax.len()
        );
        for (reference_argmax, actual_argmax) in
            reference_step.argmax.iter().zip(&actual_step.argmax)
        {
            argmax_positions += 1;
            if reference_argmax == actual_argmax {
                argmax_matches += 1;
            } else {
                step_diverged = true;
            }
        }

        match (&reference_step.logits, &actual_step.logits) {
            (Some(reference_logits), Some(actual_logits)) => {
                ensure!(
                    reference_logits.len() == actual_logits.len(),
                    "logits codebook count mismatch at step {}",
                    reference_step.step
                );
                for (codebook, (reference_row, actual_row)) in
                    reference_logits.iter().zip(actual_logits).enumerate()
                {
                    ensure!(
                        reference_row.len() == actual_row.len(),
                        "logits vocab mismatch at step {} codebook {}",
                        reference_step.step,
                        codebook
                    );
                    let actual_argmax = usize::try_from(actual_step.argmax[codebook])
                        .context("actual argmax must be non-negative")?;
                    ensure!(
                        actual_argmax < reference_row.len(),
                        "actual argmax {} out of reference logits range {} at step {} codebook {}",
                        actual_argmax,
                        reference_row.len(),
                        reference_step.step,
                        codebook
                    );
                    let best_reference = reference_row
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    max_argmax_regret =
                        max_argmax_regret.max(best_reference - reference_row[actual_argmax]);
                    reference_logits_all.extend(reference_row);
                    actual_logits_all.extend(actual_row);
                }
            }
            _ => {
                full_logits_available = false;
            }
        }

        match (&reference_step.top_ids, &actual_step.top_ids) {
            (Some(reference_top), Some(actual_top)) => {
                ensure!(
                    reference_top.len() == actual_top.len(),
                    "top-k codebook count mismatch at step {}",
                    reference_step.step
                );
                for (reference_row, actual_row) in reference_top.iter().zip(actual_top) {
                    topk_overlaps.push(topk_overlap(reference_row, actual_row));
                }
            }
            _ => {
                topk_available = false;
            }
        }

        if step_diverged && first_divergent_step.is_none() {
            first_divergent_step = Some(reference_step.step);
        }
    }

    let (logits_cosine, logits_max_abs, logits_mean_abs, logits_p99_abs) = if full_logits_available
    {
        let stats = diff_stats(&reference_logits_all, &actual_logits_all)?;
        (
            Some(cosine_similarity(
                &reference_logits_all,
                &actual_logits_all,
            )?),
            Some(stats.max_abs),
            Some(stats.mean_abs),
            Some(stats.p99_abs),
        )
    } else {
        (None, None, None, None)
    };
    let (topk_min_overlap, topk_mean_overlap) = if topk_available && !topk_overlaps.is_empty() {
        let min = *topk_overlaps.iter().min().unwrap();
        let mean = topk_overlaps.iter().sum::<usize>() as f32 / topk_overlaps.len() as f32;
        (Some(min), Some(mean))
    } else {
        (None, None)
    };

    Ok(CodegenTraceComparison {
        prompt_tokens_match: reference.prompt_tokens == actual.prompt_tokens,
        steps_compared: reference.steps.len(),
        first_divergent_step,
        sampled_rows_exact,
        raw_rows_exact,
        generation_done_exact,
        argmax_positions,
        argmax_matches,
        argmax_agreement: if argmax_positions == 0 {
            0.0
        } else {
            argmax_matches as f32 / argmax_positions as f32
        },
        logits_cosine,
        logits_max_abs,
        logits_mean_abs,
        logits_p99_abs,
        max_argmax_regret: full_logits_available.then_some(max_argmax_regret),
        topk_min_overlap,
        topk_mean_overlap,
        full_logits_available,
        topk_available,
        tolerances,
    })
}

pub fn ensure_codegen_trace_comparison_passed(comparison: &CodegenTraceComparison) -> Result<()> {
    if comparison.passed() {
        return Ok(());
    }
    bail!(
        "Higgs codegen trace comparison failed: prompt_tokens_match={} first_divergent_step={:?} sampled_rows_exact={} raw_rows_exact={} generation_done_exact={} argmax_agreement={:.6} full_logits_available={} logits_cosine={:?} logits_max_abs={:?} logits_mean_abs={:?} logits_p99_abs={:?} max_argmax_regret={:?} topk_available={} topk_min_overlap={:?}",
        comparison.prompt_tokens_match,
        comparison.first_divergent_step,
        comparison.sampled_rows_exact,
        comparison.raw_rows_exact,
        comparison.generation_done_exact,
        comparison.argmax_agreement,
        comparison.full_logits_available,
        comparison.logits_cosine,
        comparison.logits_max_abs,
        comparison.logits_mean_abs,
        comparison.logits_p99_abs,
        comparison.max_argmax_regret,
        comparison.topk_available,
        comparison.topk_min_overlap
    );
}

fn load_trace(path: &Path) -> Result<Trace> {
    let value: Value = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))?;
    trace_from_json(&value, path.display().to_string())
}

fn trace_from_json(value: &Value, label: String) -> Result<Trace> {
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some(TRACE_SCHEMA),
        "{label} must use schema {TRACE_SCHEMA}"
    );
    let prompt_tokens = usize_field(value, "prompt_tokens")?;
    let steps_value = value
        .get("steps")
        .and_then(Value::as_array)
        .with_context(|| format!("{label} missing array field steps"))?;
    let steps = steps_value
        .iter()
        .enumerate()
        .map(|(idx, step)| parse_step(step, idx))
        .collect::<Result<Vec<_>>>()?;
    Ok(Trace {
        prompt_tokens,
        steps,
    })
}

fn parse_step(value: &Value, idx: usize) -> Result<Step> {
    let logits = value
        .get("logits")
        .with_context(|| format!("step {idx} missing logits object"))?;
    Ok(Step {
        step: usize_field(value, "step")?,
        sampled_codes: u32_vec_field(value, "sampled_codes")?,
        raw_codes: nested_u32_vec_field(value, "raw_codes")?,
        generation_done: value
            .get("generation_done")
            .and_then(Value::as_bool)
            .with_context(|| format!("step {idx} missing bool generation_done"))?,
        argmax: i64_vec_field(logits, "argmax")?,
        top_ids: optional_nested_i64_vec_field(logits, "top_ids")?,
        logits: optional_nested_f32_vec_field(logits, "logits")?,
    })
}

fn usize_field(value: &Value, key: &str) -> Result<usize> {
    let raw = value
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing integer field {key}"))?;
    usize::try_from(raw).with_context(|| format!("{key} does not fit usize"))
}

fn u32_vec_field(value: &Value, key: &str) -> Result<Vec<u32>> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("missing array field {key}"))?;
    values
        .iter()
        .map(|value| {
            let raw = value
                .as_u64()
                .with_context(|| format!("{key} entry is not u32"))?;
            u32::try_from(raw).with_context(|| format!("{key} entry does not fit u32"))
        })
        .collect()
}

fn nested_u32_vec_field(value: &Value, key: &str) -> Result<Vec<Vec<u32>>> {
    let rows = value
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("missing array field {key}"))?;
    rows.iter()
        .map(|row| {
            row.as_array()
                .context("expected nested array")?
                .iter()
                .map(|value| {
                    let raw = value.as_u64().context("nested value is not u32")?;
                    u32::try_from(raw).context("nested value does not fit u32")
                })
                .collect()
        })
        .collect()
}

fn i64_vec_field(value: &Value, key: &str) -> Result<Vec<i64>> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("missing array field logits.{key}"))?;
    values
        .iter()
        .map(|value| {
            value
                .as_i64()
                .with_context(|| format!("{key} entry is not i64"))
        })
        .collect()
}

fn optional_nested_i64_vec_field(value: &Value, key: &str) -> Result<Option<Vec<Vec<i64>>>> {
    match value.get(key) {
        Some(Value::Null) | None => Ok(None),
        Some(Value::Array(rows)) => rows
            .iter()
            .map(|row| {
                row.as_array()
                    .context("expected nested top_ids array")?
                    .iter()
                    .map(|value| value.as_i64().context("top_ids entry is not i64"))
                    .collect()
            })
            .collect::<Result<Vec<_>>>()
            .map(Some),
        Some(_) => bail!("logits.{key} must be null or nested array"),
    }
}

fn optional_nested_f32_vec_field(value: &Value, key: &str) -> Result<Option<Vec<Vec<f32>>>> {
    match value.get(key) {
        Some(Value::Null) | None => Ok(None),
        Some(Value::Array(rows)) => rows
            .iter()
            .map(|row| {
                row.as_array()
                    .context("expected nested logits array")?
                    .iter()
                    .map(|value| {
                        let raw = value.as_f64().context("logits entry is not f32")?;
                        Ok(raw as f32)
                    })
                    .collect()
            })
            .collect::<Result<Vec<_>>>()
            .map(Some),
        Some(_) => bail!("logits.{key} must be null or nested array"),
    }
}

fn topk_overlap(reference: &[i64], actual: &[i64]) -> usize {
    let reference: BTreeSet<_> = reference.iter().take(TOP_K).copied().collect();
    actual
        .iter()
        .take(TOP_K)
        .filter(|id| reference.contains(id))
        .count()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32> {
    ensure!(
        left.len() == right.len(),
        "cosine length mismatch: {} vs {}",
        left.len(),
        right.len()
    );
    ensure!(!left.is_empty(), "cosine requires at least one value");
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (&left, &right) in left.iter().zip(right) {
        let left = left as f64;
        let right = right as f64;
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    ensure!(left_norm > 0.0, "left vector has zero norm");
    ensure!(right_norm > 0.0, "right vector has zero norm");
    Ok((dot / (left_norm.sqrt() * right_norm.sqrt())) as f32)
}

#[derive(Debug, Clone, Copy)]
struct DiffStats {
    max_abs: f32,
    mean_abs: f32,
    p99_abs: f32,
}

fn diff_stats(reference: &[f32], actual: &[f32]) -> Result<DiffStats> {
    ensure!(
        reference.len() == actual.len(),
        "diff length mismatch: {} vs {}",
        reference.len(),
        actual.len()
    );
    ensure!(
        !reference.is_empty(),
        "diff stats require at least one value"
    );
    let mut diffs: Vec<_> = reference
        .iter()
        .zip(actual)
        .map(|(reference, actual)| (*reference - *actual).abs())
        .collect();
    diffs.sort_by(f32::total_cmp);
    let max_abs = *diffs.last().unwrap();
    let mean_abs = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let p99_idx = ((diffs.len() - 1) as f32 * 0.99).ceil() as usize;
    Ok(DiffStats {
        max_abs,
        mean_abs,
        p99_abs: diffs[p99_idx],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_trace::AudioLogitsSummary;
    use crate::decode_trace::DecodeTrace;

    #[test]
    fn compares_matching_trace_with_logits_and_topk() {
        let trace = trace_with_logits(12);
        let comparison = compare_codegen_traces(
            &trace,
            &trace,
            CodegenTraceTolerances {
                logits_cosine_min: 0.999,
                logits_max_abs_tol: 0.0,
                logits_mean_abs_tol: 0.0,
                logits_p99_abs_tol: 0.0,
                argmax_regret_tol: 0.0,
                topk_min_overlap: 4,
            },
        )
        .unwrap();

        assert!(comparison.passed());
        assert_eq!(comparison.steps_compared, 1);
        assert_eq!(comparison.argmax_agreement, 1.0);
        assert_eq!(comparison.topk_min_overlap, Some(4));
    }

    #[test]
    fn missing_logits_keeps_semantic_gate_unproved() {
        let trace = trace_without_logits();
        let comparison =
            compare_codegen_traces(&trace, &trace, CodegenTraceTolerances::default()).unwrap();

        assert!(!comparison.passed());
        assert!(!comparison.full_logits_available);
        assert!(!comparison.topk_available);
    }

    #[test]
    fn detects_first_divergent_sampled_step() {
        let reference = trace_with_logits(12);
        let mut actual = trace_with_logits(12);
        actual.steps[0].sampled_codes[0] = 42;

        let comparison =
            compare_codegen_traces(&reference, &actual, CodegenTraceTolerances::default()).unwrap();

        assert_eq!(comparison.first_divergent_step, Some(0));
        assert!(!comparison.sampled_rows_exact);
    }

    fn trace_with_logits(prompt_tokens: usize) -> Trace {
        let mut trace = DecodeTrace::new(prompt_tokens);
        trace
            .push_step(
                0,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                Some(vec![1, 2, 3, 4, 5, 6, 7, 8]),
                vec![vec![1, 2, 3, 4, 5, 6, 7, 8]],
                false,
                AudioLogitsSummary {
                    argmax: vec![1, 2, 2, 2, 2, 2, 2, 2],
                    top_ids: Some(vec![
                        vec![1, 2, 3, 4],
                        vec![2, 3, 4, 5],
                        vec![2, 3, 4, 5],
                        vec![2, 3, 4, 5],
                        vec![2, 3, 4, 5],
                        vec![2, 3, 4, 5],
                        vec![2, 3, 4, 5],
                        vec![2, 3, 4, 5],
                    ]),
                    logits: Some(vec![
                        vec![0.0, 2.0, 1.0],
                        vec![0.0, 1.0, 2.0],
                        vec![0.0, 1.0, 2.0],
                        vec![0.0, 1.0, 2.0],
                        vec![0.0, 1.0, 2.0],
                        vec![0.0, 1.0, 2.0],
                        vec![0.0, 1.0, 2.0],
                        vec![0.0, 1.0, 2.0],
                    ]),
                    top64_min_overlap_with_previous: None,
                    logits_l2_norm: 1.0,
                },
            )
            .unwrap();
        trace_from_json(&trace.to_json_value(), "test".to_string()).unwrap()
    }

    fn trace_without_logits() -> Trace {
        let mut trace = DecodeTrace::new(12);
        trace
            .push_step(
                0,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                None,
                Vec::new(),
                false,
                AudioLogitsSummary {
                    argmax: vec![1, 2, 3, 4, 5, 6, 7, 8],
                    top_ids: None,
                    logits: None,
                    top64_min_overlap_with_previous: None,
                    logits_l2_norm: 0.0,
                },
            )
            .unwrap();
        trace_from_json(&trace.to_json_value(), "test".to_string()).unwrap()
    }
}

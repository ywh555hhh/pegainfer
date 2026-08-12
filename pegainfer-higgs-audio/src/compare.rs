use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use crate::one_step_golden::validate_required_tensors;

pub const PROMPT_INPUT_IDS: &str = "prompt.input_ids_padded";
pub const PROMPT_ATTENTION_MASK: &str = "prompt.attention_mask";
pub const PROMPT_LENGTHS: &str = "prompt.lengths";
pub const FINAL_HIDDEN_BF16: &str = "final_hidden.bf16";
pub const AUDIO_LOGITS_F32: &str = "audio_logits.f32";
pub const AUDIO_TOP64_IDS: &str = "audio_top64.ids";
pub const AUDIO_TOP64_LOGPROBS_F32: &str = "audio_top64.logprobs.f32";
pub const AUDIO_ARGMAX_IDS: &str = "audio_argmax.ids";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneStepTolerances {
    pub hidden_abs_tol: f32,
    pub hidden_mean_abs_tol: f32,
    pub logits_abs_tol: f32,
    pub logits_mean_abs_tol: f32,
    pub top_logprobs_abs_tol: f32,
    pub top_logprobs_mean_abs_tol: f32,
}

impl Default for OneStepTolerances {
    fn default() -> Self {
        Self {
            hidden_abs_tol: 0.03125,
            hidden_mean_abs_tol: 0.003,
            logits_abs_tol: 0.05,
            logits_mean_abs_tol: 0.005,
            top_logprobs_abs_tol: 0.05,
            top_logprobs_mean_abs_tol: 0.005,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorComparison {
    pub name: &'static str,
    pub elements: usize,
    pub exact_mismatches: usize,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub rmse: f32,
    pub p99_abs: f32,
    pub abs_tol: f32,
    pub mean_abs_tol: f32,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OneStepComparison {
    pub tensors: Vec<TensorComparison>,
}

impl OneStepComparison {
    pub fn passed(&self) -> bool {
        self.tensors.iter().all(|tensor| tensor.passed)
    }
}

pub fn compare_one_step_files(
    golden: impl AsRef<Path>,
    actual: impl AsRef<Path>,
    tolerances: OneStepTolerances,
) -> Result<OneStepComparison> {
    let golden_path = golden.as_ref();
    let actual_path = actual.as_ref();
    let golden_bytes =
        std::fs::read(golden_path).with_context(|| format!("read {}", golden_path.display()))?;
    let actual_bytes =
        std::fs::read(actual_path).with_context(|| format!("read {}", actual_path.display()))?;
    let golden_st =
        SafeTensors::deserialize(&golden_bytes).context("parse Higgs golden safetensors")?;
    let actual_st =
        SafeTensors::deserialize(&actual_bytes).context("parse Higgs actual safetensors")?;
    compare_one_step_safetensors(&golden_st, &actual_st, tolerances)
}

pub fn compare_one_step_safetensors(
    golden: &SafeTensors,
    actual: &SafeTensors,
    tolerances: OneStepTolerances,
) -> Result<OneStepComparison> {
    validate_required_tensors(golden, "golden")?;
    validate_required_tensors(actual, "actual")?;

    let tensors = vec![
        compare_i64_exact(golden, actual, PROMPT_INPUT_IDS)?,
        compare_i64_exact(golden, actual, PROMPT_ATTENTION_MASK)?,
        compare_i64_exact(golden, actual, PROMPT_LENGTHS)?,
        compare_bf16(
            golden,
            actual,
            FINAL_HIDDEN_BF16,
            tolerances.hidden_abs_tol,
            tolerances.hidden_mean_abs_tol,
        )?,
        compare_f32(
            golden,
            actual,
            AUDIO_LOGITS_F32,
            tolerances.logits_abs_tol,
            tolerances.logits_mean_abs_tol,
        )?,
        compare_i64_exact(golden, actual, AUDIO_TOP64_IDS)?,
        compare_f32(
            golden,
            actual,
            AUDIO_TOP64_LOGPROBS_F32,
            tolerances.top_logprobs_abs_tol,
            tolerances.top_logprobs_mean_abs_tol,
        )?,
        compare_i64_exact(golden, actual, AUDIO_ARGMAX_IDS)?,
    ];

    Ok(OneStepComparison { tensors })
}

pub fn ensure_comparison_passed(comparison: &OneStepComparison) -> Result<()> {
    if comparison.passed() {
        return Ok(());
    }
    let failing: Vec<_> = comparison
        .tensors
        .iter()
        .filter(|tensor| !tensor.passed)
        .map(|tensor| tensor.name)
        .collect();
    bail!("Higgs one-step comparison failed for tensor(s): {failing:?}");
}

fn compare_i64_exact(
    golden: &SafeTensors,
    actual: &SafeTensors,
    name: &'static str,
) -> Result<TensorComparison> {
    let golden = tensor(golden, name)?;
    let actual = tensor(actual, name)?;
    ensure!(golden.dtype() == Dtype::I64, "{name} golden must be I64");
    ensure!(actual.dtype() == Dtype::I64, "{name} actual must be I64");
    let golden_values = i64_values(golden)?;
    let actual_values = i64_values(actual)?;
    ensure!(
        golden_values.len() == actual_values.len(),
        "{name} element count mismatch: golden {} actual {}",
        golden_values.len(),
        actual_values.len()
    );

    let diffs: Vec<f32> = golden_values
        .iter()
        .zip(&actual_values)
        .map(|(golden, actual)| (*golden - *actual).unsigned_abs() as f32)
        .collect();
    let exact_mismatches = diffs.iter().filter(|diff| **diff != 0.0).count();
    let stats = stats(&diffs);
    Ok(TensorComparison {
        name,
        elements: diffs.len(),
        exact_mismatches,
        max_abs: stats.max_abs,
        mean_abs: stats.mean_abs,
        rmse: stats.rmse,
        p99_abs: stats.p99_abs,
        abs_tol: 0.0,
        mean_abs_tol: 0.0,
        passed: exact_mismatches == 0,
    })
}

fn compare_f32(
    golden: &SafeTensors,
    actual: &SafeTensors,
    name: &'static str,
    abs_tol: f32,
    mean_abs_tol: f32,
) -> Result<TensorComparison> {
    let golden = tensor(golden, name)?;
    let actual = tensor(actual, name)?;
    ensure!(golden.dtype() == Dtype::F32, "{name} golden must be F32");
    ensure!(actual.dtype() == Dtype::F32, "{name} actual must be F32");
    compare_float_values(
        name,
        &f32_values(golden)?,
        &f32_values(actual)?,
        abs_tol,
        mean_abs_tol,
    )
}

fn compare_bf16(
    golden: &SafeTensors,
    actual: &SafeTensors,
    name: &'static str,
    abs_tol: f32,
    mean_abs_tol: f32,
) -> Result<TensorComparison> {
    let golden = tensor(golden, name)?;
    let actual = tensor(actual, name)?;
    ensure!(golden.dtype() == Dtype::BF16, "{name} golden must be BF16");
    ensure!(actual.dtype() == Dtype::BF16, "{name} actual must be BF16");
    compare_float_values(
        name,
        &bf16_values(golden)?,
        &bf16_values(actual)?,
        abs_tol,
        mean_abs_tol,
    )
}

fn compare_float_values(
    name: &'static str,
    golden: &[f32],
    actual: &[f32],
    abs_tol: f32,
    mean_abs_tol: f32,
) -> Result<TensorComparison> {
    ensure!(
        golden.len() == actual.len(),
        "{name} element count mismatch: golden {} actual {}",
        golden.len(),
        actual.len()
    );
    let mut non_finite = 0usize;
    let diffs: Vec<f32> = golden
        .iter()
        .zip(actual)
        .map(|(golden, actual)| {
            let diff = (*golden - *actual).abs();
            if diff.is_finite() {
                diff
            } else {
                non_finite += 1;
                f32::INFINITY
            }
        })
        .collect();
    let stats = stats(&diffs);
    Ok(TensorComparison {
        name,
        elements: diffs.len(),
        exact_mismatches: non_finite,
        max_abs: stats.max_abs,
        mean_abs: stats.mean_abs,
        rmse: stats.rmse,
        p99_abs: stats.p99_abs,
        abs_tol,
        mean_abs_tol,
        passed: non_finite == 0 && stats.max_abs <= abs_tol && stats.mean_abs <= mean_abs_tol,
    })
}

#[derive(Debug, Clone, Copy)]
struct FloatStats {
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    p99_abs: f32,
}

fn stats(diffs: &[f32]) -> FloatStats {
    if diffs.is_empty() {
        return FloatStats {
            max_abs: 0.0,
            mean_abs: 0.0,
            rmse: 0.0,
            p99_abs: 0.0,
        };
    }
    let mut sorted = diffs.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let sum: f64 = diffs.iter().map(|value| f64::from(*value)).sum();
    let sum_sq: f64 = diffs
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum();
    let p99_idx = ((sorted.len() as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    FloatStats {
        max_abs: *sorted.last().expect("non-empty"),
        mean_abs: (sum / diffs.len() as f64) as f32,
        rmse: (sum_sq / diffs.len() as f64).sqrt() as f32,
        p99_abs: sorted[p99_idx],
    }
}

fn tensor<'a>(st: &'a SafeTensors, name: &str) -> Result<TensorView<'a>> {
    st.tensor(name)
        .with_context(|| format!("safetensors missing tensor {name}"))
}

fn i64_values(tensor: TensorView<'_>) -> Result<Vec<i64>> {
    bytes_to_chunks(tensor.data(), 8, "i64")?;
    Ok(tensor
        .data()
        .chunks_exact(8)
        .map(|bytes| {
            i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        })
        .collect())
}

fn f32_values(tensor: TensorView<'_>) -> Result<Vec<f32>> {
    bytes_to_chunks(tensor.data(), 4, "f32")?;
    Ok(tensor
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn bf16_values(tensor: TensorView<'_>) -> Result<Vec<f32>> {
    bytes_to_chunks(tensor.data(), 2, "bf16")?;
    Ok(tensor
        .data()
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        .collect())
}

fn bytes_to_chunks(bytes: &[u8], chunk: usize, dtype: &str) -> Result<()> {
    ensure!(
        bytes.len().is_multiple_of(chunk),
        "{dtype} tensor byte length {} is not divisible by {chunk}",
        bytes.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../test_data/higgs-one-step-audio-logits.safetensors"
    );

    #[test]
    fn golden_compares_equal_to_itself() {
        let comparison =
            compare_one_step_files(GOLDEN, GOLDEN, OneStepTolerances::default()).unwrap();
        assert!(comparison.passed());
        assert_eq!(comparison.tensors.len(), 8);
        for tensor in comparison.tensors {
            assert_eq!(tensor.max_abs, 0.0, "{}", tensor.name);
            assert_eq!(tensor.mean_abs, 0.0, "{}", tensor.name);
        }
    }

    #[test]
    fn comparator_rejects_logit_drift_beyond_tolerance() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut bytes = std::fs::read(GOLDEN).unwrap();
        let first_logit = tensor_data_offset(&bytes, AUDIO_LOGITS_F32);
        bytes[first_logit + 3] ^= 0x40;
        std::fs::write(tmp.path(), bytes).unwrap();

        let comparison =
            compare_one_step_files(GOLDEN, tmp.path(), OneStepTolerances::default()).unwrap();
        let logits = comparison
            .tensors
            .iter()
            .find(|tensor| tensor.name == AUDIO_LOGITS_F32)
            .unwrap();
        assert!(!logits.passed);
        assert!(!comparison.passed());
        let err = ensure_comparison_passed(&comparison)
            .unwrap_err()
            .to_string();
        assert!(err.contains(AUDIO_LOGITS_F32));
    }

    fn tensor_data_offset(bytes: &[u8], name: &str) -> usize {
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        let start = header[name]["data_offsets"][0].as_u64().unwrap() as usize;
        8 + header_len + start
    }
}

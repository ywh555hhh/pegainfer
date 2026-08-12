use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use half::bf16;
use memmap2::Mmap;
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorView, View},
};

use crate::compare::{
    AUDIO_ARGMAX_IDS, AUDIO_LOGITS_F32, AUDIO_TOP64_IDS, AUDIO_TOP64_LOGPROBS_F32,
    FINAL_HIDDEN_BF16, PROMPT_ATTENTION_MASK, PROMPT_INPUT_IDS, PROMPT_LENGTHS,
};
use crate::one_step_golden::{
    CODEBOOK_VOCAB_SIZE, HIDDEN_SIZE, NUM_CODEBOOKS, TOP_K, validate_required_tensors,
};
use crate::weights::{FUSED_MODALITY_EMBEDDING, fused_modality_shape};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTensors {
    pub input_ids_padded: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub lengths: Vec<i64>,
}

impl PromptTensors {
    pub fn prompt_ids(&self) -> Result<Vec<u32>> {
        let len = usize::try_from(
            *self
                .lengths
                .first()
                .context("prompt lengths tensor is empty")?,
        )
        .context("prompt length must be non-negative")?;
        ensure!(
            len <= self.input_ids_padded.len(),
            "prompt length {len} exceeds padded ids length {}",
            self.input_ids_padded.len()
        );
        self.input_ids_padded[..len]
            .iter()
            .map(|value| {
                u32::try_from(*value).with_context(|| format!("prompt id {value} is not u32"))
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OneStepActualSummary {
    pub output_path: std::path::PathBuf,
    pub prompt_tokens: usize,
    pub hidden_values: usize,
    pub audio_logits: usize,
}

#[derive(Clone)]
struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

pub fn load_prompt_from_golden(path: impl AsRef<Path>) -> Result<PromptTensors> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("parse Higgs golden safetensors")?;
    validate_required_tensors(&st, "golden")?;
    Ok(PromptTensors {
        input_ids_padded: i64_values(tensor(&st, PROMPT_INPUT_IDS)?)?,
        attention_mask: i64_values(tensor(&st, PROMPT_ATTENTION_MASK)?)?,
        lengths: i64_values(tensor(&st, PROMPT_LENGTHS)?)?,
    })
}

pub fn load_fused_audio_head_bf16(model_dir: impl AsRef<Path>) -> Result<Vec<bf16>> {
    let model_dir = model_dir.as_ref();
    let index_path = model_dir.join("model.safetensors.index.json");
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&index_path).with_context(|| format!("read {}", index_path.display()))?,
    )
    .with_context(|| format!("parse {}", index_path.display()))?;
    let shard = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .and_then(|weight_map| weight_map.get(FUSED_MODALITY_EMBEDDING))
        .and_then(serde_json::Value::as_str)
        .with_context(|| format!("index missing {FUSED_MODALITY_EMBEDDING}"))?;
    let shard_path = model_dir.join(shard);
    let file = std::fs::File::open(&shard_path)
        .with_context(|| format!("open {}", shard_path.display()))?;
    let mmap =
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", shard_path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parse {}", shard_path.display()))?;
    let tensor = st
        .tensor(FUSED_MODALITY_EMBEDDING)
        .with_context(|| format!("missing tensor {FUSED_MODALITY_EMBEDDING}"))?;
    ensure!(
        tensor.dtype() == Dtype::BF16,
        "{FUSED_MODALITY_EMBEDDING} must be BF16"
    );
    ensure!(
        tensor.shape() == fused_modality_shape(),
        "{FUSED_MODALITY_EMBEDDING} shape mismatch: expected {:?}, got {:?}",
        fused_modality_shape(),
        tensor.shape()
    );
    bf16_values(tensor)
}

pub fn write_one_step_actual(
    output_path: impl AsRef<Path>,
    prompt: &PromptTensors,
    final_hidden: &[bf16],
    audio_head: &[bf16],
) -> Result<OneStepActualSummary> {
    ensure!(
        final_hidden.len() == HIDDEN_SIZE,
        "final hidden len mismatch: expected {HIDDEN_SIZE}, got {}",
        final_hidden.len()
    );
    ensure!(
        audio_head.len() == NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE,
        "audio head len mismatch: expected {}, got {}",
        NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE,
        audio_head.len()
    );

    let logits = audio_logits(final_hidden, audio_head);
    let (top_ids, top_logprobs, argmax) = audio_topk_and_argmax(&logits);
    write_one_step_actual_tensors(
        output_path,
        prompt,
        final_hidden,
        &logits,
        &top_ids,
        &top_logprobs,
        &argmax,
    )
}

#[cfg(feature = "runtime-qwen3")]
pub fn write_one_step_actual_with_gpu_audio_head(
    output_path: impl AsRef<Path>,
    prompt: &PromptTensors,
    final_hidden: &[bf16],
    audio_head: &[bf16],
    device_ordinal: usize,
) -> Result<OneStepActualSummary> {
    ensure!(
        final_hidden.len() == HIDDEN_SIZE,
        "final hidden len mismatch: expected {HIDDEN_SIZE}, got {}",
        final_hidden.len()
    );
    ensure!(
        audio_head.len() == NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE,
        "audio head len mismatch: expected {}, got {}",
        NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE,
        audio_head.len()
    );

    let logits = audio_logits_gpu_bf16(final_hidden, audio_head, device_ordinal)?;
    let (top_ids, top_logprobs, argmax) = audio_topk_and_argmax(&logits);
    write_one_step_actual_tensors(
        output_path,
        prompt,
        final_hidden,
        &logits,
        &top_ids,
        &top_logprobs,
        &argmax,
    )
}

#[cfg(feature = "runtime-qwen3")]
fn audio_logits_gpu_bf16(
    final_hidden: &[bf16],
    audio_head: &[bf16],
    device_ordinal: usize,
) -> Result<Vec<f32>> {
    let ctx = pegainfer_core::tensor::DeviceContext::new_with_device(device_ordinal).with_context(
        || format!("create CUDA context for audio head on device {device_ordinal}"),
    )?;
    let hidden = pegainfer_core::tensor::DeviceVec::from_host(&ctx, final_hidden)
        .context("copy final hidden to GPU for Higgs audio head")?;
    let head = pegainfer_core::tensor::DeviceMatrix::from_host(
        &ctx,
        audio_head,
        NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE,
        HIDDEN_SIZE,
    )
    .context("copy fused Higgs audio head to GPU")?;
    let logits = pegainfer_core::ops::linear(&ctx, &hidden, &head)
        .context("run Higgs fused audio head as CUDA bf16 linear")?;
    logits
        .to_host(&ctx)
        .context("copy Higgs audio logits from GPU")
}

fn write_one_step_actual_tensors(
    output_path: impl AsRef<Path>,
    prompt: &PromptTensors,
    final_hidden: &[bf16],
    logits: &[f32],
    top_ids: &[i64],
    top_logprobs: &[f32],
    argmax: &[i64],
) -> Result<OneStepActualSummary> {
    ensure!(
        logits.len() == NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE,
        "audio logits len mismatch: expected {}, got {}",
        NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE,
        logits.len()
    );
    ensure!(
        top_ids.len() == NUM_CODEBOOKS * TOP_K,
        "top id len mismatch: expected {}, got {}",
        NUM_CODEBOOKS * TOP_K,
        top_ids.len()
    );
    ensure!(
        top_logprobs.len() == NUM_CODEBOOKS * TOP_K,
        "top logprob len mismatch: expected {}, got {}",
        NUM_CODEBOOKS * TOP_K,
        top_logprobs.len()
    );
    ensure!(
        argmax.len() == NUM_CODEBOOKS,
        "argmax len mismatch: expected {NUM_CODEBOOKS}, got {}",
        argmax.len()
    );
    let output_path = output_path.as_ref();
    let tensors = BTreeMap::from([
        (
            PROMPT_INPUT_IDS.to_string(),
            owned_i64(
                &[1, prompt.input_ids_padded.len()],
                &prompt.input_ids_padded,
            ),
        ),
        (
            PROMPT_ATTENTION_MASK.to_string(),
            owned_i64(&[1, prompt.attention_mask.len()], &prompt.attention_mask),
        ),
        (
            PROMPT_LENGTHS.to_string(),
            owned_i64(&[prompt.lengths.len()], &prompt.lengths),
        ),
        (
            FINAL_HIDDEN_BF16.to_string(),
            owned_bf16(&[1, HIDDEN_SIZE], final_hidden),
        ),
        (
            AUDIO_LOGITS_F32.to_string(),
            owned_f32(&[1, NUM_CODEBOOKS, CODEBOOK_VOCAB_SIZE], logits),
        ),
        (
            AUDIO_TOP64_IDS.to_string(),
            owned_i64(&[1, NUM_CODEBOOKS, TOP_K], top_ids),
        ),
        (
            AUDIO_TOP64_LOGPROBS_F32.to_string(),
            owned_f32(&[1, NUM_CODEBOOKS, TOP_K], top_logprobs),
        ),
        (
            AUDIO_ARGMAX_IDS.to_string(),
            owned_i64(&[1, NUM_CODEBOOKS], argmax),
        ),
    ]);
    let metadata = HashMap::from([(
        "fixture_kind".to_string(),
        "higgs-one-step-audio-logits-actual".to_string(),
    )]);
    safetensors::serialize_to_file(tensors, Some(metadata), output_path)
        .with_context(|| format!("write {}", output_path.display()))?;

    Ok(OneStepActualSummary {
        output_path: output_path.to_path_buf(),
        prompt_tokens: prompt.prompt_ids()?.len(),
        hidden_values: final_hidden.len(),
        audio_logits: logits.len(),
    })
}

fn audio_logits(final_hidden: &[bf16], audio_head: &[bf16]) -> Vec<f32> {
    let hidden: Vec<f32> = final_hidden.iter().map(|value| value.to_f32()).collect();
    audio_head
        .chunks_exact(HIDDEN_SIZE)
        .map(|row| {
            row.iter()
                .zip(&hidden)
                .map(|(weight, hidden)| weight.to_f32() * hidden)
                .sum()
        })
        .collect()
}

fn audio_topk_and_argmax(logits: &[f32]) -> (Vec<i64>, Vec<f32>, Vec<i64>) {
    let mut top_ids = Vec::with_capacity(NUM_CODEBOOKS * TOP_K);
    let mut top_logprobs = Vec::with_capacity(NUM_CODEBOOKS * TOP_K);
    let mut argmax = Vec::with_capacity(NUM_CODEBOOKS);
    for row in logits.chunks_exact(CODEBOOK_VOCAB_SIZE) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let logsumexp = max
            + row
                .iter()
                .map(|value| (*value - max).exp())
                .sum::<f32>()
                .ln();
        let mut indexed: Vec<_> = row.iter().copied().enumerate().collect();
        indexed.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        argmax.push(indexed[0].0 as i64);
        for (idx, value) in indexed.into_iter().take(TOP_K) {
            top_ids.push(idx as i64);
            top_logprobs.push(value - logsumexp);
        }
    }
    (top_ids, top_logprobs, argmax)
}

fn owned_i64(shape: &[usize], values: &[i64]) -> OwnedTensor {
    OwnedTensor {
        dtype: Dtype::I64,
        shape: shape.to_vec(),
        data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    }
}

fn owned_f32(shape: &[usize], values: &[f32]) -> OwnedTensor {
    OwnedTensor {
        dtype: Dtype::F32,
        shape: shape.to_vec(),
        data: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    }
}

fn owned_bf16(shape: &[usize], values: &[bf16]) -> OwnedTensor {
    OwnedTensor {
        dtype: Dtype::BF16,
        shape: shape.to_vec(),
        data: values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect(),
    }
}

fn tensor<'a>(st: &'a SafeTensors, name: &str) -> Result<TensorView<'a>> {
    st.tensor(name)
        .with_context(|| format!("safetensors missing tensor {name}"))
}

fn i64_values(tensor: TensorView<'_>) -> Result<Vec<i64>> {
    ensure!(tensor.dtype() == Dtype::I64, "tensor must be I64");
    ensure!(
        tensor.data().len().is_multiple_of(8),
        "I64 tensor byte length {} is not divisible by 8",
        tensor.data().len()
    );
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

fn bf16_values(tensor: TensorView<'_>) -> Result<Vec<bf16>> {
    ensure!(tensor.dtype() == Dtype::BF16, "tensor must be BF16");
    ensure!(
        tensor.data().len().is_multiple_of(2),
        "BF16 tensor byte length {} is not divisible by 2",
        tensor.data().len()
    );
    Ok(tensor
        .data()
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{OneStepTolerances, compare_one_step_files};

    #[test]
    fn prompt_ids_respect_recorded_length() {
        let prompt = PromptTensors {
            input_ids_padded: vec![10, 20, 30, 0],
            attention_mask: vec![1, 1, 1, 0],
            lengths: vec![3],
        };
        assert_eq!(prompt.prompt_ids().unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn actual_writer_emits_comparator_schema() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let prompt = PromptTensors {
            input_ids_padded: vec![1; 10],
            attention_mask: vec![1; 10],
            lengths: vec![10],
        };
        let hidden = vec![bf16::from_f32(0.0); HIDDEN_SIZE];
        let mut head = vec![bf16::from_f32(0.0); NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE * HIDDEN_SIZE];
        for codebook in 0..NUM_CODEBOOKS {
            let row = codebook * CODEBOOK_VOCAB_SIZE;
            head[row * HIDDEN_SIZE] = bf16::from_f32(1.0);
        }
        write_one_step_actual(tmp.path(), &prompt, &hidden, &head).unwrap();
        let err = compare_one_step_files(tmp.path(), tmp.path(), OneStepTolerances::default())
            .err()
            .map(|err| err.to_string());
        assert_eq!(err, None);
    }

    #[test]
    fn topk_uses_local_audio_token_ids() {
        let mut logits = vec![0.0; NUM_CODEBOOKS * CODEBOOK_VOCAB_SIZE];
        logits[5] = 10.0;
        logits[CODEBOOK_VOCAB_SIZE + 7] = 11.0;
        let (top_ids, _top_logprobs, argmax) = audio_topk_and_argmax(&logits);
        assert_eq!(argmax[0], 5);
        assert_eq!(argmax[1], 7);
        assert_eq!(top_ids[0], 5);
        assert_eq!(top_ids[TOP_K], 7);
    }
}

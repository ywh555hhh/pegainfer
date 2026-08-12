use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use half::bf16;

use crate::compare::{FINAL_HIDDEN_BF16, PROMPT_ATTENTION_MASK, PROMPT_INPUT_IDS, PROMPT_LENGTHS};
use crate::one_step_actual::{PromptTensors, owned_bf16, owned_i64};
use crate::one_step_golden::HIDDEN_SIZE;

pub const NUM_LAYERS: usize = 36;

#[derive(Debug, Clone, PartialEq)]
pub struct LayerHiddenDumpSummary {
    pub output_path: PathBuf,
    pub prompt_tokens: usize,
    pub layers: usize,
    pub hidden_values_per_layer: usize,
}

pub fn layer_hidden_tensor_name(layer_idx: usize) -> String {
    format!("layer.{layer_idx:02}.last_hidden.bf16")
}

pub fn write_layer_hidden_dump(
    output_path: impl AsRef<Path>,
    prompt: &PromptTensors,
    layer_hidden: &[Vec<bf16>],
    final_normed: &[bf16],
) -> Result<LayerHiddenDumpSummary> {
    ensure!(
        layer_hidden.len() == NUM_LAYERS,
        "expected {NUM_LAYERS} layer snapshots, got {}",
        layer_hidden.len()
    );
    ensure!(
        final_normed.len() == HIDDEN_SIZE,
        "final normed hidden len mismatch: expected {HIDDEN_SIZE}, got {}",
        final_normed.len()
    );

    let mut tensors = BTreeMap::from([
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
            owned_bf16(&[1, HIDDEN_SIZE], final_normed),
        ),
    ]);
    for (layer_idx, hidden) in layer_hidden.iter().enumerate() {
        ensure!(
            hidden.len() == HIDDEN_SIZE,
            "layer {layer_idx} hidden len mismatch: expected {HIDDEN_SIZE}, got {}",
            hidden.len()
        );
        tensors.insert(
            layer_hidden_tensor_name(layer_idx),
            owned_bf16(&[1, HIDDEN_SIZE], hidden),
        );
    }

    let output_path = output_path.as_ref();
    let metadata = HashMap::from([(
        "fixture_kind".to_string(),
        "higgs-prefill-layer-hidden-actual".to_string(),
    )]);
    safetensors::serialize_to_file(tensors, Some(metadata), output_path)
        .with_context(|| format!("write {}", output_path.display()))?;

    Ok(LayerHiddenDumpSummary {
        output_path: output_path.to_path_buf(),
        prompt_tokens: prompt.prompt_ids()?.len(),
        layers: layer_hidden.len(),
        hidden_values_per_layer: HIDDEN_SIZE,
    })
}

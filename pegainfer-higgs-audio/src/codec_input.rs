use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use serde_json::Value;

use crate::delay_pattern::reverse_delay_pattern;
use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
use crate::one_step_golden::NUM_CODEBOOKS;

pub const CODEC_AUDIO_VOCAB_SIZE: u32 = (CODEBOOK_VOCAB_SIZE - 2) as u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeRowsLayout {
    Delayed,
    Raw,
}

pub fn load_code_rows_json(
    path: impl AsRef<Path>,
    layout: CodeRowsLayout,
) -> Result<Vec<Vec<u32>>> {
    let path = path.as_ref();
    let value: Value = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))?;
    code_rows_from_json_value(&value, layout)
}

pub fn code_rows_from_json_value(value: &Value, layout: CodeRowsLayout) -> Result<Vec<Vec<u32>>> {
    let rows_value = match value {
        Value::Array(_) => value,
        Value::Object(map) => {
            let key = match layout {
                CodeRowsLayout::Delayed => "delayed_codes",
                CodeRowsLayout::Raw => "raw_codes",
            };
            map.get(key)
                .with_context(|| format!("code rows JSON object missing key {key:?}"))?
        }
        other => bail!("code rows JSON must be an array or object, got {other:?}"),
    };
    parse_rows(rows_value)
}

pub fn codec_input_from_rows(rows: &[Vec<u32>], layout: CodeRowsLayout) -> Result<Vec<Vec<u32>>> {
    let raw = match layout {
        CodeRowsLayout::Delayed => reverse_delay_pattern(rows, false)?,
        CodeRowsLayout::Raw => rows.to_vec(),
    };
    validate_rows(&raw, "raw codec input")?;
    Ok(raw
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|code| {
                    if code >= CODEC_AUDIO_VOCAB_SIZE {
                        0
                    } else {
                        code
                    }
                })
                .collect()
        })
        .collect())
}

pub fn write_codec_input_json(path: impl AsRef<Path>, rows: &[Vec<u32>]) -> Result<()> {
    validate_rows(rows, "codec input")?;
    let value = serde_json::json!({
        "schema": "higgs-audio-codec-input-v1",
        "num_codebooks": NUM_CODEBOOKS,
        "codebook_size": CODEBOOK_VOCAB_SIZE,
        "codec_audio_vocab_size": CODEC_AUDIO_VOCAB_SIZE,
        "raw_codes": rows,
    });
    let bytes = serde_json::to_vec_pretty(&value).context("serialize Higgs codec input JSON")?;
    std::fs::write(path.as_ref(), bytes)
        .with_context(|| format!("write {}", path.as_ref().display()))
}

fn parse_rows(value: &Value) -> Result<Vec<Vec<u32>>> {
    let rows = value
        .as_array()
        .context("code rows must be a two-dimensional array")?;
    let mut out = Vec::with_capacity(rows.len());
    for (row_idx, row_value) in rows.iter().enumerate() {
        let row = row_value
            .as_array()
            .with_context(|| format!("code row {row_idx} must be an array"))?;
        let mut out_row = Vec::with_capacity(row.len());
        for (col_idx, cell) in row.iter().enumerate() {
            let value = cell.as_u64().with_context(|| {
                format!("code[{row_idx}][{col_idx}] must be a non-negative integer")
            })?;
            out_row.push(
                u32::try_from(value)
                    .with_context(|| format!("code[{row_idx}][{col_idx}] does not fit u32"))?,
            );
        }
        out.push(out_row);
    }
    validate_rows(&out, "code rows")?;
    Ok(out)
}

fn validate_rows(rows: &[Vec<u32>], label: &str) -> Result<()> {
    if rows.is_empty() {
        bail!("{label} must contain at least one row");
    }
    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != NUM_CODEBOOKS {
            bail!(
                "{label} row {row_idx} has {} codebooks, expected {NUM_CODEBOOKS}",
                row.len()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delay_pattern::BOC_ID;
    use crate::delay_pattern::EOC_ID;

    #[test]
    fn delayed_rows_reverse_and_clamp_codec_sentinels() {
        let delayed = vec![
            vec![1, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID],
            vec![2, 11, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID],
            vec![3, 12, 21, BOC_ID, BOC_ID, BOC_ID, BOC_ID, BOC_ID],
            vec![4, 13, 22, 31, BOC_ID, BOC_ID, BOC_ID, BOC_ID],
            vec![5, 14, 23, 32, 41, BOC_ID, BOC_ID, BOC_ID],
            vec![6, 15, 24, 33, 42, 51, BOC_ID, BOC_ID],
            vec![7, 16, 25, 34, 43, 52, 61, BOC_ID],
            vec![EOC_ID, 17, 26, 35, 44, 53, 62, 71],
        ];

        let codec = codec_input_from_rows(&delayed, CodeRowsLayout::Delayed).unwrap();

        assert_eq!(codec.len(), 1);
        assert_eq!(codec[0], vec![1, 11, 21, 31, 41, 51, 61, 71]);
    }

    #[test]
    fn raw_rows_clamp_values_outside_codec_audio_vocab() {
        let raw = vec![vec![1, 1023, 1024, 1025, 7, 8, 9, 10]];

        let codec = codec_input_from_rows(&raw, CodeRowsLayout::Raw).unwrap();

        assert_eq!(codec, vec![vec![1, 1023, 0, 0, 7, 8, 9, 10]]);
    }

    #[test]
    fn json_object_uses_layout_specific_key() {
        let value = serde_json::json!({
            "raw_codes": [[1, 2, 3, 4, 5, 6, 7, 8]],
            "delayed_codes": [[9, 10, 11, 12, 13, 14, 15, 16]],
        });

        assert_eq!(
            code_rows_from_json_value(&value, CodeRowsLayout::Raw).unwrap(),
            vec![vec![1, 2, 3, 4, 5, 6, 7, 8]]
        );
        assert_eq!(
            code_rows_from_json_value(&value, CodeRowsLayout::Delayed).unwrap(),
            vec![vec![9, 10, 11, 12, 13, 14, 15, 16]]
        );
    }
}

use anyhow::Result;
use anyhow::bail;
use half::bf16;

use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
use crate::one_step_golden::HIDDEN_SIZE;
use crate::one_step_golden::NUM_CODEBOOKS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedCodebookShape {
    pub num_codebooks: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl FusedCodebookShape {
    pub const fn higgs_v3() -> Self {
        Self {
            num_codebooks: NUM_CODEBOOKS,
            vocab_size: CODEBOOK_VOCAB_SIZE,
            hidden_size: HIDDEN_SIZE,
        }
    }

    fn expected_weight_len(self) -> usize {
        self.num_codebooks * self.vocab_size * self.hidden_size
    }
}

pub fn fused_codebook_embedding_cpu(codes: &[u32], fused_embedding: &[bf16]) -> Result<Vec<f32>> {
    fused_codebook_embedding_cpu_with_shape(codes, fused_embedding, FusedCodebookShape::higgs_v3())
}

pub fn fused_codebook_embedding_cpu_with_shape(
    codes: &[u32],
    fused_embedding: &[bf16],
    shape: FusedCodebookShape,
) -> Result<Vec<f32>> {
    validate_fused_codebook_inputs(codes, fused_embedding, shape)?;
    let mut out = vec![0.0f32; shape.hidden_size];
    for (codebook, &code) in codes.iter().enumerate() {
        let row = codebook * shape.vocab_size + code as usize;
        let offset = row * shape.hidden_size;
        for (dst, weight) in out
            .iter_mut()
            .zip(&fused_embedding[offset..offset + shape.hidden_size])
        {
            *dst += weight.to_f32();
        }
    }
    Ok(out)
}

fn validate_fused_codebook_inputs(
    codes: &[u32],
    fused_embedding: &[bf16],
    shape: FusedCodebookShape,
) -> Result<()> {
    if shape.num_codebooks == 0 || shape.vocab_size == 0 || shape.hidden_size == 0 {
        bail!("fused codebook shape dimensions must be non-zero: {shape:?}");
    }
    if codes.len() != shape.num_codebooks {
        bail!(
            "code row has {} codebooks, expected {}",
            codes.len(),
            shape.num_codebooks
        );
    }
    if fused_embedding.len() != shape.expected_weight_len() {
        bail!(
            "fused codebook embedding len mismatch: expected {}, got {}",
            shape.expected_weight_len(),
            fused_embedding.len()
        );
    }
    for (codebook, &code) in codes.iter().enumerate() {
        if code as usize >= shape.vocab_size {
            bail!(
                "codebook {codebook} code id {code} is outside vocab size {}",
                shape.vocab_size
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weight(shape: FusedCodebookShape) -> Vec<bf16> {
        let mut values = Vec::with_capacity(shape.expected_weight_len());
        for codebook in 0..shape.num_codebooks {
            for code in 0..shape.vocab_size {
                for dim in 0..shape.hidden_size {
                    values.push(bf16::from_f32(
                        100.0 * codebook as f32 + 10.0 * code as f32 + dim as f32,
                    ));
                }
            }
        }
        values
    }

    #[test]
    fn fused_embedding_offsets_code_ids_by_codebook_and_sums_hidden_vectors() {
        let shape = FusedCodebookShape {
            num_codebooks: 3,
            vocab_size: 5,
            hidden_size: 4,
        };
        let out = fused_codebook_embedding_cpu_with_shape(&[1, 3, 4], &weight(shape), shape)
            .expect("fused embedding");

        assert_eq!(out, vec![380.0, 383.0, 386.0, 389.0]);
    }

    #[test]
    fn fused_embedding_accepts_boc_and_eoc_as_codec_vocab_ids() {
        let shape = FusedCodebookShape {
            num_codebooks: 2,
            vocab_size: 1026,
            hidden_size: 1,
        };
        let mut weight = vec![bf16::from_f32(0.0); shape.expected_weight_len()];
        weight[1024] = bf16::from_f32(3.0);
        weight[shape.vocab_size + 1025] = bf16::from_f32(4.0);

        let out = fused_codebook_embedding_cpu_with_shape(&[1024, 1025], &weight, shape)
            .expect("BOC/EOC are valid feedback embedding ids");

        assert_eq!(out, vec![7.0]);
    }

    #[test]
    fn fused_embedding_rejects_shape_and_vocab_mismatches() {
        let shape = FusedCodebookShape {
            num_codebooks: 2,
            vocab_size: 4,
            hidden_size: 3,
        };
        let weight = weight(shape);

        let err = fused_codebook_embedding_cpu_with_shape(&[1], &weight, shape)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 2"));

        let err = fused_codebook_embedding_cpu_with_shape(&[1, 4], &weight, shape)
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside vocab"));

        let err =
            fused_codebook_embedding_cpu_with_shape(&[1, 2], &weight[..weight.len() - 1], shape)
                .unwrap_err()
                .to_string();
        assert!(err.contains("len mismatch"));
    }
}

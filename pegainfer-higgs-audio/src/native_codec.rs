use std::collections::BTreeMap;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "runtime-qwen3")]
use anyhow::bail;
use anyhow::ensure;
use half::bf16;
use memmap2::Mmap;
#[cfg(feature = "runtime-qwen3")]
use pegainfer_kernels::tensor::DeviceContext;
use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;
use serde::Deserialize;
use serde_json::Value;

use crate::codec_input::CODEC_AUDIO_VOCAB_SIZE;
use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
use crate::one_step_golden::NUM_CODEBOOKS;
use crate::weights::HiggsWeightManifest;

pub const CODEC_TTS_PREFIX: &str = "tied.embedding.modality_embeddings.0.model.";
pub const NATIVE_CODEC_CLAIM: &str = "native_codec_vocoder_waveform_bringup";
pub const QUANTIZER_HIDDEN_SIZE: usize = 1024;
pub const ACOUSTIC_HIDDEN_SIZE: usize = 256;
pub const ACOUSTIC_DECODER_HIDDEN_SIZE: usize = 1024;
pub const ACOUSTIC_DECODER_CONV1_KERNEL: usize = 7;
pub const ACOUSTIC_DECODER_CONV1_PADDING: usize = 3;
pub const DECODER_BLOCK0_OUT_CHANNELS: usize = 512;
pub const DECODER_BLOCK0_CONVT_KERNEL: usize = 16;
pub const DECODER_BLOCK0_CONVT_STRIDE: usize = 8;
pub const DECODER_BLOCK0_CONVT_PADDING: usize = 4;
pub const DECODER_RESIDUAL_KERNEL: usize = 7;
pub const DECODER_FINAL_CONV_KERNEL: usize = 7;
pub const DECODER_FINAL_CONV_PADDING: usize = 3;

const BUNDLED_CODEC_CONFIG: &str =
    include_str!("../../tools/higgs/configs/higgs_audio_v2_tokenizer.json");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCodecConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    pub frame_rate: usize,
    pub num_quantizers: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub quantizer_hidden_size: usize,
    pub acoustic_hidden_size: usize,
    pub decoder_hidden_size: usize,
    pub upsampling_ratios: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCodecTensorSpec {
    pub full_name: String,
    pub codec_name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCodecManifestSummary {
    pub codec_tensors_in_manifest: usize,
    pub required_tensors_checked: usize,
    pub files_checked: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativePcmAudio {
    pub sample_rate: usize,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeCodecBackend {
    CudaSkeleton,
    #[cfg(feature = "runtime-qwen3")]
    Cuda {
        device_ordinal: usize,
    },
}

#[derive(Debug, Clone)]
pub struct NativeHiggsCodecDecoder {
    config: NativeCodecConfig,
    backend: NativeCodecBackend,
    rvq_codebooks: RvqCodebooks,
    quantizer_project: QuantizerProjectWeights,
    fc2: Fc2Weights,
    acoustic_decoder_conv1: Conv1dWeights,
    decoder_blocks: Vec<DecoderUpsampleBlockWeights>,
    final_snake_alpha: Vec<bf16>,
    final_conv: Conv1dWeights,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RvqCodebooks {
    data: Vec<bf16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RvqDecodeOutput {
    pub frames: usize,
    pub dim: usize,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodecStageCudaCheck {
    pub stage: &'static str,
    pub frames: usize,
    pub dim: usize,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizerProjectWeights {
    weight: Vec<bf16>,
    bias: Vec<bf16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fc2Weights {
    weight: Vec<bf16>,
    bias: Vec<bf16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Conv1dWeights {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
    weight: Vec<bf16>,
    bias: Vec<bf16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderBlock0ConvTransposeWeights {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    snake_alpha: Vec<bf16>,
    weight: Vec<bf16>,
    bias: Vec<bf16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderResidualUnitWeights {
    channels: usize,
    dilation: usize,
    snake1_alpha: Vec<bf16>,
    conv1_weight: Vec<bf16>,
    conv1_bias: Vec<bf16>,
    snake2_alpha: Vec<bf16>,
    conv2_weight: Vec<bf16>,
    conv2_bias: Vec<bf16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderUpsampleBlockWeights {
    convt: DecoderBlock0ConvTransposeWeights,
    res1: DecoderResidualUnitWeights,
    res2: DecoderResidualUnitWeights,
    res3: DecoderResidualUnitWeights,
}

#[derive(Debug, Deserialize)]
struct RawCodecConfig {
    sample_rate: usize,
    codebook_size: usize,
    codebook_dim: usize,
    acoustic_model_config: RawAcousticModelConfig,
}

#[derive(Debug, Deserialize)]
struct RawAcousticModelConfig {
    hidden_size: usize,
    decoder_hidden_size: usize,
    hop_length: usize,
    upsampling_ratios: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    byte_range: [usize; 2],
}

impl NativeCodecConfig {
    pub fn bundled_higgs_v2() -> Result<Self> {
        let raw: RawCodecConfig = serde_json::from_str(BUNDLED_CODEC_CONFIG)
            .context("parse bundled Higgs codec config")?;
        ensure!(raw.sample_rate > 0, "codec sample_rate must be positive");
        ensure!(
            raw.acoustic_model_config.hop_length > 0,
            "codec hop_length must be positive"
        );
        ensure!(
            raw.sample_rate % raw.acoustic_model_config.hop_length == 0,
            "codec sample_rate must be divisible by hop_length"
        );
        ensure!(
            raw.codebook_size == CODEBOOK_VOCAB_SIZE - 2,
            "codec codebook_size mismatch: expected {}, got {}",
            CODEBOOK_VOCAB_SIZE - 2,
            raw.codebook_size
        );
        ensure!(
            raw.codebook_dim == 64,
            "Higgs native codec skeleton expects codebook_dim=64, got {}",
            raw.codebook_dim
        );
        Ok(Self {
            sample_rate: raw.sample_rate,
            hop_length: raw.acoustic_model_config.hop_length,
            frame_rate: raw.sample_rate / raw.acoustic_model_config.hop_length,
            num_quantizers: NUM_CODEBOOKS,
            codebook_size: raw.codebook_size,
            codebook_dim: raw.codebook_dim,
            quantizer_hidden_size: QUANTIZER_HIDDEN_SIZE,
            acoustic_hidden_size: raw.acoustic_model_config.hidden_size,
            decoder_hidden_size: raw.acoustic_model_config.decoder_hidden_size,
            upsampling_ratios: raw.acoustic_model_config.upsampling_ratios,
        })
    }
}

impl NativeHiggsCodecDecoder {
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        backend: NativeCodecBackend,
    ) -> Result<Self> {
        let config = NativeCodecConfig::bundled_higgs_v2()?;
        let manifest = HiggsWeightManifest::from_model_dir(&model_dir)?;
        validate_native_codec_manifest(&model_dir, &manifest, &config)?;
        let rvq_codebooks = load_rvq_codebooks(&model_dir, &manifest, &config)?;
        let quantizer_project = load_quantizer_project(&model_dir, &manifest, &config)?;
        let fc2 = load_fc2(&model_dir, &manifest, &config)?;
        let acoustic_decoder_conv1 = load_acoustic_decoder_conv1(&model_dir, &manifest, &config)?;
        let decoder_blocks = load_decoder_upsample_blocks(&model_dir, &manifest, &config)?;
        let final_snake_alpha =
            load_final_snake_alpha(&model_dir, &manifest, final_decoder_channels(&config)?)?;
        let final_conv = load_final_conv(&model_dir, &manifest, final_decoder_channels(&config)?)?;
        Ok(Self {
            config,
            backend,
            rvq_codebooks,
            quantizer_project,
            fc2,
            acoustic_decoder_conv1,
            decoder_blocks,
            final_snake_alpha,
            final_conv,
        })
    }

    pub fn config(&self) -> &NativeCodecConfig {
        &self.config
    }

    pub fn decode_raw_codes(&self, raw_codes: &[Vec<u32>]) -> Result<NativePcmAudio> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        match self.backend {
            NativeCodecBackend::CudaSkeleton => self.decode_waveform_cpu(raw_codes),
            #[cfg(feature = "runtime-qwen3")]
            NativeCodecBackend::Cuda { .. } => self.decode_waveform_cuda(raw_codes),
        }
    }

    pub fn decode_rvq_cpu(&self, raw_codes: &[Vec<u32>]) -> Result<RvqDecodeOutput> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        self.rvq_codebooks.decode_cpu(raw_codes, &self.config)
    }

    pub fn decode_quantizer_cpu(&self, raw_codes: &[Vec<u32>]) -> Result<RvqDecodeOutput> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        self.quantizer_project
            .decode_cpu(raw_codes, &self.rvq_codebooks, &self.config)
    }

    pub fn decode_acoustic_input_cpu(&self, raw_codes: &[Vec<u32>]) -> Result<RvqDecodeOutput> {
        let quantized = self.decode_quantizer_cpu(raw_codes)?;
        self.fc2.decode_cpu(&quantized, &self.config)
    }

    pub fn decode_acoustic_decoder_conv1_cpu(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<RvqDecodeOutput> {
        let acoustic_input = self.decode_acoustic_input_cpu(raw_codes)?;
        self.acoustic_decoder_conv1.decode_cpu(&acoustic_input)
    }

    pub fn decode_decoder_block0_convt_cpu(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<RvqDecodeOutput> {
        let decoder_conv1 = self.decode_acoustic_decoder_conv1_cpu(raw_codes)?;
        self.decoder_block(0)?.convt.decode_cpu(&decoder_conv1)
    }

    pub fn decode_decoder_block0_res1_cpu(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<RvqDecodeOutput> {
        let decoder_block0_convt = self.decode_decoder_block0_convt_cpu(raw_codes)?;
        self.decoder_block(0)?
            .res1
            .decode_cpu(&decoder_block0_convt)
    }

    pub fn decode_decoder_block0_res_stack_cpu(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<RvqDecodeOutput> {
        let res1 = self.decode_decoder_block0_res1_cpu(raw_codes)?;
        let res2 = self.decoder_block(0)?.res2.decode_cpu(&res1)?;
        self.decoder_block(0)?.res3.decode_cpu(&res2)
    }

    fn decode_waveform_cpu(&self, raw_codes: &[Vec<u32>]) -> Result<NativePcmAudio> {
        let hidden = self.decode_acoustic_decoder_hidden_cpu(raw_codes)?;
        self.decode_final_audio_cpu(&hidden)
    }

    fn decode_acoustic_decoder_hidden_cpu(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<RvqDecodeOutput> {
        let mut hidden = self.decode_acoustic_decoder_conv1_cpu(raw_codes)?;
        for block in &self.decoder_blocks {
            hidden = block.decode_cpu(&hidden)?;
        }
        Ok(hidden)
    }

    fn decode_final_audio_cpu(&self, hidden: &RvqDecodeOutput) -> Result<NativePcmAudio> {
        ensure!(
            hidden.dim == self.final_snake_alpha.len(),
            "final decoder hidden dim mismatch: expected {}, got {}",
            self.final_snake_alpha.len(),
            hidden.dim
        );
        let snaked = apply_snake_cpu(hidden, &self.final_snake_alpha)?;
        let out = self.final_conv.decode_cpu(&snaked)?;
        ensure!(
            out.dim == 1,
            "final conv output dim must be 1, got {}",
            out.dim
        );
        Ok(NativePcmAudio {
            sample_rate: self.config.sample_rate,
            samples: out.values,
        })
    }

    fn decoder_block(&self, index: usize) -> Result<&DecoderUpsampleBlockWeights> {
        self.decoder_blocks
            .get(index)
            .with_context(|| format!("missing Higgs decoder block {index}"))
    }

    #[cfg(feature = "runtime-qwen3")]
    pub fn decode_rvq_cuda_with_cpu_check(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<(RvqDecodeOutput, CodecStageCudaCheck)> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        let cpu = self.decode_rvq_cpu(raw_codes)?;
        let device_ordinal = match self.backend {
            NativeCodecBackend::Cuda { device_ordinal } => device_ordinal,
            NativeCodecBackend::CudaSkeleton => {
                bail!("Higgs native RVQ CUDA decode requested with CudaSkeleton backend")
            }
        };
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let flat_codes = flatten_raw_codes(raw_codes, &self.config)?;
        let mut out = self
            .rvq_codebooks
            .decode_cuda(&ctx, &flat_codes, raw_codes.len())?;
        ctx.sync()?;
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        for (actual, expected) in out.values.iter().zip(cpu.values.iter()) {
            let diff = (actual - expected).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += diff;
        }
        let mean_abs_diff = if out.values.is_empty() {
            0.0
        } else {
            sum_abs_diff / out.values.len() as f32
        };
        out.frames = raw_codes.len();
        Ok((
            out,
            CodecStageCudaCheck {
                stage: "rvq_embedding",
                frames: raw_codes.len(),
                dim: self.config.codebook_dim,
                max_abs_diff,
                mean_abs_diff,
            },
        ))
    }

    #[cfg(feature = "runtime-qwen3")]
    pub fn decode_quantizer_cuda_with_cpu_check(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<(RvqDecodeOutput, CodecStageCudaCheck)> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        let cpu = self.decode_quantizer_cpu(raw_codes)?;
        let device_ordinal = self.cuda_device_ordinal()?;
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let flat_codes = flatten_raw_codes(raw_codes, &self.config)?;
        let out = self.quantizer_project.decode_cuda(
            &ctx,
            &flat_codes,
            raw_codes.len(),
            &self.rvq_codebooks,
        )?;
        ctx.sync()?;
        Ok((
            out.clone(),
            diff_check("quantizer_project_out", &out, &cpu)?,
        ))
    }

    #[cfg(feature = "runtime-qwen3")]
    pub fn decode_acoustic_input_cuda_with_cpu_check(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<(RvqDecodeOutput, CodecStageCudaCheck)> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        let cpu = self.decode_acoustic_input_cpu(raw_codes)?;
        let device_ordinal = self.cuda_device_ordinal()?;
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let flat_codes = flatten_raw_codes(raw_codes, &self.config)?;
        let quantized = self.quantizer_project.decode_cuda(
            &ctx,
            &flat_codes,
            raw_codes.len(),
            &self.rvq_codebooks,
        )?;
        let out = self.fc2.decode_cuda(&ctx, &quantized, raw_codes.len())?;
        ctx.sync()?;
        Ok((out.clone(), diff_check("fc2_acoustic_input", &out, &cpu)?))
    }

    #[cfg(feature = "runtime-qwen3")]
    pub fn decode_acoustic_decoder_conv1_cuda_with_cpu_check(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<(RvqDecodeOutput, CodecStageCudaCheck)> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        let cpu = self.decode_acoustic_decoder_conv1_cpu(raw_codes)?;
        let device_ordinal = self.cuda_device_ordinal()?;
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let flat_codes = flatten_raw_codes(raw_codes, &self.config)?;
        let quantized = self.quantizer_project.decode_cuda(
            &ctx,
            &flat_codes,
            raw_codes.len(),
            &self.rvq_codebooks,
        )?;
        let acoustic_input = self.fc2.decode_cuda(&ctx, &quantized, raw_codes.len())?;
        let out = self
            .acoustic_decoder_conv1
            .decode_cuda(&ctx, &acoustic_input)?;
        ctx.sync()?;
        Ok((
            out.clone(),
            diff_check("acoustic_decoder_conv1", &out, &cpu)?,
        ))
    }

    #[cfg(feature = "runtime-qwen3")]
    pub fn decode_decoder_block0_convt_cuda_with_cpu_check(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<(RvqDecodeOutput, CodecStageCudaCheck)> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        let cpu = self.decode_decoder_block0_convt_cpu(raw_codes)?;
        let device_ordinal = self.cuda_device_ordinal()?;
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let flat_codes = flatten_raw_codes(raw_codes, &self.config)?;
        let quantized = self.quantizer_project.decode_cuda(
            &ctx,
            &flat_codes,
            raw_codes.len(),
            &self.rvq_codebooks,
        )?;
        let acoustic_input = self.fc2.decode_cuda(&ctx, &quantized, raw_codes.len())?;
        let decoder_conv1 = self
            .acoustic_decoder_conv1
            .decode_cuda(&ctx, &acoustic_input)?;
        let out = self
            .decoder_block(0)?
            .convt
            .decode_cuda(&ctx, &decoder_conv1)?;
        ctx.sync()?;
        Ok((out.clone(), diff_check("decoder_block0_convt", &out, &cpu)?))
    }

    #[cfg(feature = "runtime-qwen3")]
    pub fn decode_decoder_block0_res1_cuda_with_cpu_check(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<(RvqDecodeOutput, CodecStageCudaCheck)> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        let cpu = self.decode_decoder_block0_res1_cpu(raw_codes)?;
        let device_ordinal = self.cuda_device_ordinal()?;
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let flat_codes = flatten_raw_codes(raw_codes, &self.config)?;
        let quantized = self.quantizer_project.decode_cuda(
            &ctx,
            &flat_codes,
            raw_codes.len(),
            &self.rvq_codebooks,
        )?;
        let acoustic_input = self.fc2.decode_cuda(&ctx, &quantized, raw_codes.len())?;
        let decoder_conv1 = self
            .acoustic_decoder_conv1
            .decode_cuda(&ctx, &acoustic_input)?;
        let decoder_block0_convt = self
            .decoder_block(0)?
            .convt
            .decode_cuda(&ctx, &decoder_conv1)?;
        let out = self
            .decoder_block(0)?
            .res1
            .decode_cuda(&ctx, &decoder_block0_convt)?;
        ctx.sync()?;
        Ok((out.clone(), diff_check("decoder_block0_res1", &out, &cpu)?))
    }

    #[cfg(feature = "runtime-qwen3")]
    pub fn decode_decoder_block0_res_stack_cuda_with_cpu_check(
        &self,
        raw_codes: &[Vec<u32>],
    ) -> Result<(RvqDecodeOutput, CodecStageCudaCheck)> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        let cpu = self.decode_decoder_block0_res_stack_cpu(raw_codes)?;
        let device_ordinal = self.cuda_device_ordinal()?;
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let flat_codes = flatten_raw_codes(raw_codes, &self.config)?;
        let quantized = self.quantizer_project.decode_cuda(
            &ctx,
            &flat_codes,
            raw_codes.len(),
            &self.rvq_codebooks,
        )?;
        let acoustic_input = self.fc2.decode_cuda(&ctx, &quantized, raw_codes.len())?;
        let decoder_conv1 = self
            .acoustic_decoder_conv1
            .decode_cuda(&ctx, &acoustic_input)?;
        let decoder_block0_convt = self
            .decoder_block(0)?
            .convt
            .decode_cuda(&ctx, &decoder_conv1)?;
        let res1 = self
            .decoder_block(0)?
            .res1
            .decode_cuda(&ctx, &decoder_block0_convt)?;
        let res2 = self.decoder_block(0)?.res2.decode_cuda(&ctx, &res1)?;
        let out = self.decoder_block(0)?.res3.decode_cuda(&ctx, &res2)?;
        ctx.sync()?;
        Ok((
            out.clone(),
            diff_check("decoder_block0_res_stack", &out, &cpu)?,
        ))
    }

    #[cfg(feature = "runtime-qwen3")]
    fn cuda_device_ordinal(&self) -> Result<usize> {
        match self.backend {
            NativeCodecBackend::Cuda { device_ordinal } => Ok(device_ordinal),
            NativeCodecBackend::CudaSkeleton => {
                bail!("Higgs native CUDA decode requested with CudaSkeleton backend")
            }
        }
    }

    #[cfg(feature = "runtime-qwen3")]
    fn decode_waveform_cuda(&self, raw_codes: &[Vec<u32>]) -> Result<NativePcmAudio> {
        validate_raw_codec_rows(raw_codes, &self.config)?;
        let device_ordinal = self.cuda_device_ordinal()?;
        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let flat_codes = flatten_raw_codes(raw_codes, &self.config)?;
        let quantized = self.quantizer_project.decode_cuda(
            &ctx,
            &flat_codes,
            raw_codes.len(),
            &self.rvq_codebooks,
        )?;
        let acoustic_input = self.fc2.decode_cuda(&ctx, &quantized, raw_codes.len())?;
        let mut hidden = self
            .acoustic_decoder_conv1
            .decode_cuda(&ctx, &acoustic_input)?;
        for block in &self.decoder_blocks {
            hidden = block.decode_cuda(&ctx, &hidden)?;
        }
        let audio = self.decode_final_audio_cuda(&ctx, &hidden)?;
        ctx.sync()?;
        Ok(audio)
    }

    #[cfg(feature = "runtime-qwen3")]
    fn decode_final_audio_cuda(
        &self,
        ctx: &DeviceContext,
        hidden: &RvqDecodeOutput,
    ) -> Result<NativePcmAudio> {
        ensure!(
            hidden.dim == self.final_snake_alpha.len(),
            "final decoder hidden dim mismatch: expected {}, got {}",
            self.final_snake_alpha.len(),
            hidden.dim
        );
        let input_d = ctx
            .stream
            .clone_htod(&hidden.values)
            .context("upload Higgs final decoder hidden")?;
        let alpha_d = ctx
            .stream
            .clone_htod(&self.final_snake_alpha)
            .context("upload Higgs final decoder snake alpha")?;
        let weight_d = ctx
            .stream
            .clone_htod(&self.final_conv.weight)
            .context("upload Higgs final conv weight")?;
        let bias_d = ctx
            .stream
            .clone_htod(&self.final_conv.bias)
            .context("upload Higgs final conv bias")?;
        let mut out_d = ctx
            .stream
            .alloc_zeros(hidden.frames)
            .context("allocate Higgs final PCM output")?;
        pegainfer_kernels::ops::higgs_audio_snake_conv1d_generic_into(
            ctx,
            &input_d,
            &alpha_d,
            &weight_d,
            &bias_d,
            &mut out_d,
            hidden.frames,
            hidden.dim,
            1,
            self.final_conv.kernel_size,
            self.final_conv.padding,
        )?;
        let samples = ctx
            .stream
            .clone_dtoh(&out_d)
            .context("download Higgs final PCM output")?;
        Ok(NativePcmAudio {
            sample_rate: self.config.sample_rate,
            samples,
        })
    }
}

pub fn validate_raw_codec_rows(rows: &[Vec<u32>], config: &NativeCodecConfig) -> Result<()> {
    ensure!(
        !rows.is_empty(),
        "native codec requires at least one raw codec frame"
    );
    for (row_idx, row) in rows.iter().enumerate() {
        ensure!(
            row.len() == config.num_quantizers,
            "raw codec row {row_idx} has {} quantizers, expected {}",
            row.len(),
            config.num_quantizers
        );
        for (codebook, code) in row.iter().enumerate() {
            ensure!(
                *code < CODEC_AUDIO_VOCAB_SIZE,
                "raw codec row {row_idx} codebook {codebook} has code {code}, expected < {CODEC_AUDIO_VOCAB_SIZE}"
            );
        }
    }
    Ok(())
}

impl RvqCodebooks {
    pub fn from_bf16(data: Vec<bf16>, config: &NativeCodecConfig) -> Result<Self> {
        let expected = config
            .num_quantizers
            .checked_mul(config.codebook_size)
            .and_then(|n| n.checked_mul(config.codebook_dim))
            .context("RVQ codebook size overflow")?;
        ensure!(
            data.len() == expected,
            "RVQ codebook len mismatch: expected {expected}, got {}",
            data.len()
        );
        Ok(Self { data })
    }

    pub fn decode_cpu(
        &self,
        raw_codes: &[Vec<u32>],
        config: &NativeCodecConfig,
    ) -> Result<RvqDecodeOutput> {
        validate_raw_codec_rows(raw_codes, config)?;
        let mut values = vec![0.0f32; raw_codes.len() * config.codebook_dim];
        for (frame_idx, row) in raw_codes.iter().enumerate() {
            for (quantizer, code) in row.iter().enumerate() {
                let code = usize::try_from(*code).context("RVQ code does not fit usize")?;
                let base = (quantizer * config.codebook_size + code) * config.codebook_dim;
                let out_base = frame_idx * config.codebook_dim;
                for dim in 0..config.codebook_dim {
                    values[out_base + dim] += self.data[base + dim].to_f32();
                }
            }
        }
        Ok(RvqDecodeOutput {
            frames: raw_codes.len(),
            dim: config.codebook_dim,
            values,
        })
    }

    #[cfg(feature = "runtime-qwen3")]
    fn decode_cuda(
        &self,
        ctx: &DeviceContext,
        flat_codes: &[u32],
        frames: usize,
    ) -> Result<RvqDecodeOutput> {
        let codes_d = ctx
            .stream
            .clone_htod(flat_codes)
            .context("upload Higgs RVQ codes")?;
        let codebooks_d = ctx
            .stream
            .clone_htod(&self.data)
            .context("upload Higgs RVQ codebooks")?;
        let mut out_d = ctx
            .stream
            .alloc_zeros(frames * 64)
            .context("allocate Higgs RVQ output")?;
        pegainfer_kernels::ops::higgs_audio_rvq_decode_into(
            ctx,
            &codes_d,
            &codebooks_d,
            &mut out_d,
            frames,
        )?;
        let values = ctx
            .stream
            .clone_dtoh(&out_d)
            .context("download Higgs RVQ output")?;
        Ok(RvqDecodeOutput {
            frames,
            dim: 64,
            values,
        })
    }
}

impl QuantizerProjectWeights {
    pub fn from_bf16(
        weight: Vec<bf16>,
        bias: Vec<bf16>,
        config: &NativeCodecConfig,
    ) -> Result<Self> {
        let expected_weight = config
            .num_quantizers
            .checked_mul(config.quantizer_hidden_size)
            .and_then(|n| n.checked_mul(config.codebook_dim))
            .context("quantizer project weight size overflow")?;
        let expected_bias = config
            .num_quantizers
            .checked_mul(config.quantizer_hidden_size)
            .context("quantizer project bias size overflow")?;
        ensure!(
            weight.len() == expected_weight,
            "quantizer project weight len mismatch: expected {expected_weight}, got {}",
            weight.len()
        );
        ensure!(
            bias.len() == expected_bias,
            "quantizer project bias len mismatch: expected {expected_bias}, got {}",
            bias.len()
        );
        Ok(Self { weight, bias })
    }

    pub fn decode_cpu(
        &self,
        raw_codes: &[Vec<u32>],
        codebooks: &RvqCodebooks,
        config: &NativeCodecConfig,
    ) -> Result<RvqDecodeOutput> {
        validate_raw_codec_rows(raw_codes, config)?;
        let mut values = vec![0.0f32; raw_codes.len() * config.quantizer_hidden_size];
        for (frame_idx, row) in raw_codes.iter().enumerate() {
            for hidden in 0..config.quantizer_hidden_size {
                let mut acc = 0.0f32;
                for (quantizer, code) in row.iter().enumerate() {
                    let code = usize::try_from(*code).context("RVQ code does not fit usize")?;
                    let mut q_acc =
                        self.bias[quantizer * config.quantizer_hidden_size + hidden].to_f32();
                    let codebook_base =
                        (quantizer * config.codebook_size + code) * config.codebook_dim;
                    let weight_base =
                        (quantizer * config.quantizer_hidden_size + hidden) * config.codebook_dim;
                    for dim in 0..config.codebook_dim {
                        q_acc += codebooks.data[codebook_base + dim].to_f32()
                            * self.weight[weight_base + dim].to_f32();
                    }
                    acc += q_acc;
                }
                values[frame_idx * config.quantizer_hidden_size + hidden] = acc;
            }
        }
        Ok(RvqDecodeOutput {
            frames: raw_codes.len(),
            dim: config.quantizer_hidden_size,
            values,
        })
    }

    #[cfg(feature = "runtime-qwen3")]
    fn decode_cuda(
        &self,
        ctx: &DeviceContext,
        flat_codes: &[u32],
        frames: usize,
        codebooks: &RvqCodebooks,
    ) -> Result<RvqDecodeOutput> {
        let codes_d = ctx
            .stream
            .clone_htod(flat_codes)
            .context("upload Higgs quantizer codes")?;
        let codebooks_d = ctx
            .stream
            .clone_htod(&codebooks.data)
            .context("upload Higgs quantizer codebooks")?;
        let weight_d = ctx
            .stream
            .clone_htod(&self.weight)
            .context("upload Higgs quantizer project weights")?;
        let bias_d = ctx
            .stream
            .clone_htod(&self.bias)
            .context("upload Higgs quantizer project bias")?;
        let mut out_d = ctx
            .stream
            .alloc_zeros(frames * QUANTIZER_HIDDEN_SIZE)
            .context("allocate Higgs quantizer output")?;
        pegainfer_kernels::ops::higgs_audio_quantizer_decode_into(
            ctx,
            &codes_d,
            &codebooks_d,
            &weight_d,
            &bias_d,
            &mut out_d,
            frames,
        )?;
        let values = ctx
            .stream
            .clone_dtoh(&out_d)
            .context("download Higgs quantizer output")?;
        Ok(RvqDecodeOutput {
            frames,
            dim: QUANTIZER_HIDDEN_SIZE,
            values,
        })
    }
}

impl Fc2Weights {
    pub fn from_bf16(
        weight: Vec<bf16>,
        bias: Vec<bf16>,
        config: &NativeCodecConfig,
    ) -> Result<Self> {
        let expected_weight = config
            .acoustic_hidden_size
            .checked_mul(config.quantizer_hidden_size)
            .context("fc2 weight size overflow")?;
        ensure!(
            weight.len() == expected_weight,
            "fc2 weight len mismatch: expected {expected_weight}, got {}",
            weight.len()
        );
        ensure!(
            bias.len() == config.acoustic_hidden_size,
            "fc2 bias len mismatch: expected {}, got {}",
            config.acoustic_hidden_size,
            bias.len()
        );
        Ok(Self { weight, bias })
    }

    pub fn decode_cpu(
        &self,
        quantized: &RvqDecodeOutput,
        config: &NativeCodecConfig,
    ) -> Result<RvqDecodeOutput> {
        ensure!(
            quantized.dim == config.quantizer_hidden_size,
            "fc2 input dim mismatch: expected {}, got {}",
            config.quantizer_hidden_size,
            quantized.dim
        );
        let mut values = vec![0.0f32; quantized.frames * config.acoustic_hidden_size];
        for frame in 0..quantized.frames {
            for channel in 0..config.acoustic_hidden_size {
                let mut acc = self.bias[channel].to_f32();
                let weight_base = channel * config.quantizer_hidden_size;
                let hidden_base = frame * config.quantizer_hidden_size;
                for hidden in 0..config.quantizer_hidden_size {
                    acc += quantized.values[hidden_base + hidden]
                        * self.weight[weight_base + hidden].to_f32();
                }
                values[frame * config.acoustic_hidden_size + channel] = acc;
            }
        }
        Ok(RvqDecodeOutput {
            frames: quantized.frames,
            dim: config.acoustic_hidden_size,
            values,
        })
    }

    #[cfg(feature = "runtime-qwen3")]
    fn decode_cuda(
        &self,
        ctx: &DeviceContext,
        quantized: &RvqDecodeOutput,
        frames: usize,
    ) -> Result<RvqDecodeOutput> {
        ensure!(
            quantized.dim == QUANTIZER_HIDDEN_SIZE,
            "fc2 CUDA input dim mismatch: expected {QUANTIZER_HIDDEN_SIZE}, got {}",
            quantized.dim
        );
        let hidden_d = ctx
            .stream
            .clone_htod(&quantized.values)
            .context("upload Higgs fc2 hidden")?;
        let weight_d = ctx
            .stream
            .clone_htod(&self.weight)
            .context("upload Higgs fc2 weight")?;
        let bias_d = ctx
            .stream
            .clone_htod(&self.bias)
            .context("upload Higgs fc2 bias")?;
        let mut out_d = ctx
            .stream
            .alloc_zeros(frames * ACOUSTIC_HIDDEN_SIZE)
            .context("allocate Higgs fc2 output")?;
        pegainfer_kernels::ops::higgs_audio_fc2_into(
            ctx, &hidden_d, &weight_d, &bias_d, &mut out_d, frames,
        )?;
        let values = ctx
            .stream
            .clone_dtoh(&out_d)
            .context("download Higgs fc2 output")?;
        Ok(RvqDecodeOutput {
            frames,
            dim: ACOUSTIC_HIDDEN_SIZE,
            values,
        })
    }
}

impl Conv1dWeights {
    pub fn from_bf16(
        weight: Vec<bf16>,
        bias: Vec<bf16>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
    ) -> Result<Self> {
        let expected_weight = out_channels
            .checked_mul(in_channels)
            .and_then(|n| n.checked_mul(kernel_size))
            .context("conv1d weight size overflow")?;
        ensure!(
            weight.len() == expected_weight,
            "conv1d weight len mismatch: expected {expected_weight}, got {}",
            weight.len()
        );
        ensure!(
            bias.len() == out_channels,
            "conv1d bias len mismatch: expected {out_channels}, got {}",
            bias.len()
        );
        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            padding,
            weight,
            bias,
        })
    }

    pub fn decode_cpu(&self, input: &RvqDecodeOutput) -> Result<RvqDecodeOutput> {
        ensure!(
            input.dim == self.in_channels,
            "conv1d input dim mismatch: expected {}, got {}",
            self.in_channels,
            input.dim
        );
        let mut values = vec![0.0f32; input.frames * self.out_channels];
        for frame in 0..input.frames {
            for out_channel in 0..self.out_channels {
                let mut acc = self.bias[out_channel].to_f32();
                for in_channel in 0..self.in_channels {
                    for kernel_idx in 0..self.kernel_size {
                        let padded_pos = frame + kernel_idx;
                        if padded_pos < self.padding {
                            continue;
                        }
                        let input_frame = padded_pos - self.padding;
                        if input_frame >= input.frames {
                            continue;
                        }
                        let input_value = input.values[input_frame * self.in_channels + in_channel];
                        let weight_idx = (out_channel * self.in_channels + in_channel)
                            * self.kernel_size
                            + kernel_idx;
                        acc += input_value * self.weight[weight_idx].to_f32();
                    }
                }
                values[frame * self.out_channels + out_channel] = acc;
            }
        }
        Ok(RvqDecodeOutput {
            frames: input.frames,
            dim: self.out_channels,
            values,
        })
    }

    #[cfg(feature = "runtime-qwen3")]
    fn decode_cuda(&self, ctx: &DeviceContext, input: &RvqDecodeOutput) -> Result<RvqDecodeOutput> {
        ensure!(
            input.dim == self.in_channels,
            "conv1d CUDA input dim mismatch: expected {}, got {}",
            self.in_channels,
            input.dim
        );
        let input_d = ctx
            .stream
            .clone_htod(&input.values)
            .context("upload Higgs Conv1d input")?;
        let weight_d = ctx
            .stream
            .clone_htod(&self.weight)
            .context("upload Higgs Conv1d weight")?;
        let bias_d = ctx
            .stream
            .clone_htod(&self.bias)
            .context("upload Higgs Conv1d bias")?;
        let mut out_d = ctx
            .stream
            .alloc_zeros(input.frames * self.out_channels)
            .context("allocate Higgs Conv1d output")?;
        pegainfer_kernels::ops::higgs_audio_conv1d_generic_into(
            ctx,
            &input_d,
            &weight_d,
            &bias_d,
            &mut out_d,
            input.frames,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.padding,
        )?;
        let values = ctx
            .stream
            .clone_dtoh(&out_d)
            .context("download Higgs Conv1d output")?;
        Ok(RvqDecodeOutput {
            frames: input.frames,
            dim: self.out_channels,
            values,
        })
    }
}

impl DecoderBlock0ConvTransposeWeights {
    pub fn from_bf16(
        snake_alpha: Vec<bf16>,
        weight: Vec<bf16>,
        bias: Vec<bf16>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Result<Self> {
        ensure!(
            snake_alpha.len() == in_channels,
            "decoder ConvTranspose snake alpha len mismatch: expected {in_channels}, got {}",
            snake_alpha.len()
        );
        let expected_weight = in_channels
            .checked_mul(out_channels)
            .and_then(|n| n.checked_mul(kernel_size))
            .context("decoder ConvTranspose weight size overflow")?;
        ensure!(
            weight.len() == expected_weight,
            "decoder ConvTranspose weight len mismatch: expected {expected_weight}, got {}",
            weight.len()
        );
        ensure!(
            bias.len() == out_channels,
            "decoder ConvTranspose bias len mismatch: expected {out_channels}, got {}",
            bias.len()
        );
        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            snake_alpha,
            weight,
            bias,
        })
    }

    fn output_frames(&self, frames: usize) -> Result<usize> {
        frames
            .checked_sub(1)
            .and_then(|n| n.checked_mul(self.stride))
            .and_then(|n| n.checked_add(self.kernel_size))
            .and_then(|n| n.checked_add(self.output_padding))
            .and_then(|n| n.checked_sub(2 * self.padding))
            .context("decoder ConvTranspose output frame count overflow")
    }

    pub fn decode_cpu(&self, input: &RvqDecodeOutput) -> Result<RvqDecodeOutput> {
        ensure!(
            input.dim == self.in_channels,
            "decoder ConvTranspose input dim mismatch: expected {}, got {}",
            self.in_channels,
            input.dim
        );
        let out_frames = self.output_frames(input.frames)?;
        let mut values = vec![0.0f32; out_frames * self.out_channels];
        for out_frame in 0..out_frames {
            for out_channel in 0..self.out_channels {
                let mut acc = self.bias[out_channel].to_f32();
                for in_channel in 0..self.in_channels {
                    for kernel_idx in 0..self.kernel_size {
                        let shifted = out_frame + self.padding;
                        if shifted < kernel_idx {
                            continue;
                        }
                        let numerator = shifted - kernel_idx;
                        if numerator % self.stride != 0 {
                            continue;
                        }
                        let input_frame = numerator / self.stride;
                        if input_frame >= input.frames {
                            continue;
                        }
                        let x = input.values[input_frame * self.in_channels + in_channel];
                        let alpha = self.snake_alpha[in_channel].to_f32();
                        let snake = x + (alpha * x).sin().powi(2) / (alpha + 1.0e-9);
                        let weight_idx = (in_channel * self.out_channels + out_channel)
                            * self.kernel_size
                            + kernel_idx;
                        acc += snake * self.weight[weight_idx].to_f32();
                    }
                }
                values[out_frame * self.out_channels + out_channel] = acc;
            }
        }
        Ok(RvqDecodeOutput {
            frames: out_frames,
            dim: self.out_channels,
            values,
        })
    }

    #[cfg(feature = "runtime-qwen3")]
    fn decode_cuda(&self, ctx: &DeviceContext, input: &RvqDecodeOutput) -> Result<RvqDecodeOutput> {
        ensure!(
            input.dim == self.in_channels,
            "decoder ConvTranspose CUDA input dim mismatch: expected {}, got {}",
            self.in_channels,
            input.dim
        );
        let input_d = ctx
            .stream
            .clone_htod(&input.values)
            .context("upload Higgs decoder ConvTranspose input")?;
        let alpha_d = ctx
            .stream
            .clone_htod(&self.snake_alpha)
            .context("upload Higgs decoder ConvTranspose snake alpha")?;
        let weight_d = ctx
            .stream
            .clone_htod(&self.weight)
            .context("upload Higgs decoder ConvTranspose weight")?;
        let bias_d = ctx
            .stream
            .clone_htod(&self.bias)
            .context("upload Higgs decoder ConvTranspose bias")?;
        let out_frames = self.output_frames(input.frames)?;
        let mut out_d = ctx
            .stream
            .alloc_zeros(out_frames * self.out_channels)
            .context("allocate Higgs decoder ConvTranspose output")?;
        pegainfer_kernels::ops::higgs_audio_decoder_convt_generic_into(
            ctx,
            &input_d,
            &alpha_d,
            &weight_d,
            &bias_d,
            &mut out_d,
            input.frames,
            pegainfer_kernels::ops::HiggsAudioConvTransposeSpec {
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                output_padding: self.output_padding,
            },
        )?;
        let values = ctx
            .stream
            .clone_dtoh(&out_d)
            .context("download Higgs decoder ConvTranspose output")?;
        Ok(RvqDecodeOutput {
            frames: out_frames,
            dim: self.out_channels,
            values,
        })
    }
}

impl DecoderUpsampleBlockWeights {
    fn decode_cpu(&self, input: &RvqDecodeOutput) -> Result<RvqDecodeOutput> {
        let convt = self.convt.decode_cpu(input)?;
        let res1 = self.res1.decode_cpu(&convt)?;
        let res2 = self.res2.decode_cpu(&res1)?;
        self.res3.decode_cpu(&res2)
    }

    #[cfg(feature = "runtime-qwen3")]
    fn decode_cuda(&self, ctx: &DeviceContext, input: &RvqDecodeOutput) -> Result<RvqDecodeOutput> {
        let convt = self.convt.decode_cuda(ctx, input)?;
        let res1 = self.res1.decode_cuda(ctx, &convt)?;
        let res2 = self.res2.decode_cuda(ctx, &res1)?;
        self.res3.decode_cuda(ctx, &res2)
    }
}

impl DecoderResidualUnitWeights {
    pub fn from_bf16(
        channels: usize,
        dilation: usize,
        snake1_alpha: Vec<bf16>,
        conv1_weight: Vec<bf16>,
        conv1_bias: Vec<bf16>,
        snake2_alpha: Vec<bf16>,
        conv2_weight: Vec<bf16>,
        conv2_bias: Vec<bf16>,
    ) -> Result<Self> {
        ensure!(channels > 0, "decoder residual channels must be positive");
        ensure!(dilation > 0, "decoder residual dilation must be positive");
        ensure!(
            snake1_alpha.len() == channels,
            "decoder residual snake1 alpha len mismatch: expected {channels}, got {}",
            snake1_alpha.len()
        );
        ensure!(
            snake2_alpha.len() == channels,
            "decoder residual snake2 alpha len mismatch: expected {channels}, got {}",
            snake2_alpha.len()
        );
        ensure!(
            conv1_weight.len() == channels * channels * DECODER_RESIDUAL_KERNEL,
            "decoder residual conv1 weight len mismatch: expected {}, got {}",
            channels * channels * DECODER_RESIDUAL_KERNEL,
            conv1_weight.len()
        );
        ensure!(
            conv1_bias.len() == channels,
            "decoder residual conv1 bias len mismatch: expected {channels}, got {}",
            conv1_bias.len()
        );
        ensure!(
            conv2_weight.len() == channels * channels,
            "decoder residual conv2 weight len mismatch: expected {}, got {}",
            channels * channels,
            conv2_weight.len()
        );
        ensure!(
            conv2_bias.len() == channels,
            "decoder residual conv2 bias len mismatch: expected {channels}, got {}",
            conv2_bias.len()
        );
        Ok(Self {
            channels,
            dilation,
            snake1_alpha,
            conv1_weight,
            conv1_bias,
            snake2_alpha,
            conv2_weight,
            conv2_bias,
        })
    }

    pub fn decode_cpu(&self, input: &RvqDecodeOutput) -> Result<RvqDecodeOutput> {
        ensure!(
            input.dim == self.channels,
            "decoder residual input dim mismatch: expected {}, got {}",
            self.channels,
            input.dim
        );
        let padding = ((DECODER_RESIDUAL_KERNEL - 1) * self.dilation) / 2;
        let mut conv1 = vec![0.0f32; input.frames * self.channels];
        for frame in 0..input.frames {
            for out_channel in 0..self.channels {
                let mut acc = self.conv1_bias[out_channel].to_f32();
                for in_channel in 0..self.channels {
                    for kernel_idx in 0..DECODER_RESIDUAL_KERNEL {
                        let offset = kernel_idx * self.dilation;
                        let shifted = frame + offset;
                        if shifted < padding {
                            continue;
                        }
                        let input_frame = shifted - padding;
                        if input_frame >= input.frames {
                            continue;
                        }
                        let x = input.values[input_frame * self.channels + in_channel];
                        let alpha = self.snake1_alpha[in_channel].to_f32();
                        let snake = x + (alpha * x).sin().powi(2) / (alpha + 1.0e-9);
                        let weight_idx = (out_channel * self.channels + in_channel)
                            * DECODER_RESIDUAL_KERNEL
                            + kernel_idx;
                        acc += snake * self.conv1_weight[weight_idx].to_f32();
                    }
                }
                conv1[frame * self.channels + out_channel] = acc;
            }
        }

        let mut values = vec![0.0f32; input.frames * self.channels];
        for frame in 0..input.frames {
            for out_channel in 0..self.channels {
                let mut acc = self.conv2_bias[out_channel].to_f32();
                for in_channel in 0..self.channels {
                    let x = conv1[frame * self.channels + in_channel];
                    let alpha = self.snake2_alpha[in_channel].to_f32();
                    let snake = x + (alpha * x).sin().powi(2) / (alpha + 1.0e-9);
                    let weight_idx = out_channel * self.channels + in_channel;
                    acc += snake * self.conv2_weight[weight_idx].to_f32();
                }
                values[frame * self.channels + out_channel] =
                    input.values[frame * self.channels + out_channel] + acc;
            }
        }
        Ok(RvqDecodeOutput {
            frames: input.frames,
            dim: self.channels,
            values,
        })
    }

    #[cfg(feature = "runtime-qwen3")]
    fn decode_cuda(&self, ctx: &DeviceContext, input: &RvqDecodeOutput) -> Result<RvqDecodeOutput> {
        ensure!(
            input.dim == self.channels,
            "decoder residual CUDA input dim mismatch: expected {}, got {}",
            self.channels,
            input.dim
        );
        let input_d = ctx
            .stream
            .clone_htod(&input.values)
            .context("upload Higgs decoder residual input")?;
        let snake1_d = ctx
            .stream
            .clone_htod(&self.snake1_alpha)
            .context("upload Higgs decoder residual snake1 alpha")?;
        let conv1_weight_d = ctx
            .stream
            .clone_htod(&self.conv1_weight)
            .context("upload Higgs decoder residual conv1 weight")?;
        let conv1_bias_d = ctx
            .stream
            .clone_htod(&self.conv1_bias)
            .context("upload Higgs decoder residual conv1 bias")?;
        let mut conv1_d = ctx
            .stream
            .alloc_zeros(input.frames * self.channels)
            .context("allocate Higgs decoder residual conv1 output")?;
        pegainfer_kernels::ops::higgs_audio_decoder_residual_conv1_generic_into(
            ctx,
            &input_d,
            &snake1_d,
            &conv1_weight_d,
            &conv1_bias_d,
            &mut conv1_d,
            input.frames,
            self.channels,
            self.dilation,
        )?;
        let snake2_d = ctx
            .stream
            .clone_htod(&self.snake2_alpha)
            .context("upload Higgs decoder residual snake2 alpha")?;
        let conv2_weight_d = ctx
            .stream
            .clone_htod(&self.conv2_weight)
            .context("upload Higgs decoder residual conv2 weight")?;
        let conv2_bias_d = ctx
            .stream
            .clone_htod(&self.conv2_bias)
            .context("upload Higgs decoder residual conv2 bias")?;
        let mut out_d = ctx
            .stream
            .alloc_zeros(input.frames * self.channels)
            .context("allocate Higgs decoder residual output")?;
        pegainfer_kernels::ops::higgs_audio_decoder_residual_conv2_add_generic_into(
            ctx,
            &input_d,
            &conv1_d,
            &snake2_d,
            &conv2_weight_d,
            &conv2_bias_d,
            &mut out_d,
            input.frames,
            self.channels,
        )?;
        let values = ctx
            .stream
            .clone_dtoh(&out_d)
            .context("download Higgs decoder residual output")?;
        Ok(RvqDecodeOutput {
            frames: input.frames,
            dim: self.channels,
            values,
        })
    }
}

#[cfg(feature = "runtime-qwen3")]
fn diff_check(
    stage: &'static str,
    actual: &RvqDecodeOutput,
    expected: &RvqDecodeOutput,
) -> Result<CodecStageCudaCheck> {
    ensure!(actual.frames == expected.frames, "{stage} frame mismatch");
    ensure!(actual.dim == expected.dim, "{stage} dim mismatch");
    ensure!(
        actual.values.len() == expected.values.len(),
        "{stage} value len mismatch"
    );
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f32;
    for (actual, expected) in actual.values.iter().zip(expected.values.iter()) {
        let diff = (actual - expected).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff;
    }
    let mean_abs_diff = if actual.values.is_empty() {
        0.0
    } else {
        sum_abs_diff / actual.values.len() as f32
    };
    Ok(CodecStageCudaCheck {
        stage,
        frames: actual.frames,
        dim: actual.dim,
        max_abs_diff,
        mean_abs_diff,
    })
}

#[cfg(feature = "runtime-qwen3")]
fn flatten_raw_codes(rows: &[Vec<u32>], config: &NativeCodecConfig) -> Result<Vec<u32>> {
    validate_raw_codec_rows(rows, config)?;
    Ok(rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>())
}

fn load_rvq_codebooks(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    config: &NativeCodecConfig,
) -> Result<RvqCodebooks> {
    let mut data =
        Vec::with_capacity(config.num_quantizers * config.codebook_size * config.codebook_dim);
    for quantizer in 0..config.num_quantizers {
        let name = format!("{CODEC_TTS_PREFIX}quantizer.quantizers.{quantizer}.codebook.embed");
        let file = manifest
            .weight_map
            .get(&name)
            .with_context(|| format!("manifest missing RVQ codebook tensor {name}"))?;
        let shard_path = model_dir.as_ref().join(file);
        let file = std::fs::File::open(&shard_path)
            .with_context(|| format!("open {}", shard_path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap {}", shard_path.display()))?;
        let st = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("parse {}", shard_path.display()))?;
        let tensor = st
            .tensor(&name)
            .with_context(|| format!("{} missing tensor {name}", shard_path.display()))?;
        ensure!(tensor.dtype() == Dtype::BF16, "{name} must be BF16");
        ensure!(
            tensor.shape() == [config.codebook_size, config.codebook_dim],
            "{name} shape mismatch: expected [{}, {}], got {:?}",
            config.codebook_size,
            config.codebook_dim,
            tensor.shape()
        );
        data.extend(bf16_values(tensor)?);
    }
    RvqCodebooks::from_bf16(data, config)
}

fn load_quantizer_project(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    config: &NativeCodecConfig,
) -> Result<QuantizerProjectWeights> {
    let mut weight = Vec::with_capacity(
        config.num_quantizers * config.quantizer_hidden_size * config.codebook_dim,
    );
    let mut bias = Vec::with_capacity(config.num_quantizers * config.quantizer_hidden_size);
    for quantizer in 0..config.num_quantizers {
        let weight_name =
            format!("{CODEC_TTS_PREFIX}quantizer.quantizers.{quantizer}.project_out.weight");
        let bias_name =
            format!("{CODEC_TTS_PREFIX}quantizer.quantizers.{quantizer}.project_out.bias");
        weight.extend(load_bf16_tensor(
            &model_dir,
            manifest,
            &weight_name,
            &[config.quantizer_hidden_size, config.codebook_dim],
        )?);
        bias.extend(load_bf16_tensor(
            &model_dir,
            manifest,
            &bias_name,
            &[config.quantizer_hidden_size],
        )?);
    }
    QuantizerProjectWeights::from_bf16(weight, bias, config)
}

fn load_fc2(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    config: &NativeCodecConfig,
) -> Result<Fc2Weights> {
    let weight_name = format!("{CODEC_TTS_PREFIX}fc2.weight");
    let bias_name = format!("{CODEC_TTS_PREFIX}fc2.bias");
    let weight = load_bf16_tensor(
        &model_dir,
        manifest,
        &weight_name,
        &[config.acoustic_hidden_size, config.quantizer_hidden_size],
    )?;
    let bias = load_bf16_tensor(
        &model_dir,
        manifest,
        &bias_name,
        &[config.acoustic_hidden_size],
    )?;
    Fc2Weights::from_bf16(weight, bias, config)
}

fn load_acoustic_decoder_conv1(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    config: &NativeCodecConfig,
) -> Result<Conv1dWeights> {
    let weight_name = format!("{CODEC_TTS_PREFIX}acoustic_decoder.conv1.weight");
    let bias_name = format!("{CODEC_TTS_PREFIX}acoustic_decoder.conv1.bias");
    let weight = load_bf16_tensor(
        &model_dir,
        manifest,
        &weight_name,
        &[
            config.decoder_hidden_size,
            config.acoustic_hidden_size,
            ACOUSTIC_DECODER_CONV1_KERNEL,
        ],
    )?;
    let bias = load_bf16_tensor(
        &model_dir,
        manifest,
        &bias_name,
        &[config.decoder_hidden_size],
    )?;
    Conv1dWeights::from_bf16(
        weight,
        bias,
        config.acoustic_hidden_size,
        config.decoder_hidden_size,
        ACOUSTIC_DECODER_CONV1_KERNEL,
        ACOUSTIC_DECODER_CONV1_PADDING,
    )
}

fn load_decoder_upsample_blocks(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    config: &NativeCodecConfig,
) -> Result<Vec<DecoderUpsampleBlockWeights>> {
    let mut blocks = Vec::with_capacity(config.upsampling_ratios.len());
    for (index, stride) in config.upsampling_ratios.iter().copied().enumerate() {
        let (in_channels, out_channels) = decoder_block_channels(config, index)?;
        let convt = load_decoder_block_convt(
            &model_dir,
            manifest,
            index,
            in_channels,
            out_channels,
            stride,
        )?;
        let res1 = load_decoder_residual_unit(
            &model_dir,
            manifest,
            &format!("acoustic_decoder.block.{index}.res_unit1"),
            out_channels,
            1,
        )?;
        let res2 = load_decoder_residual_unit(
            &model_dir,
            manifest,
            &format!("acoustic_decoder.block.{index}.res_unit2"),
            out_channels,
            3,
        )?;
        let res3 = load_decoder_residual_unit(
            &model_dir,
            manifest,
            &format!("acoustic_decoder.block.{index}.res_unit3"),
            out_channels,
            9,
        )?;
        blocks.push(DecoderUpsampleBlockWeights {
            convt,
            res1,
            res2,
            res3,
        });
    }
    Ok(blocks)
}

fn load_decoder_block_convt(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    index: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
) -> Result<DecoderBlock0ConvTransposeWeights> {
    let kernel_size = stride
        .checked_mul(2)
        .context("decoder ConvTranspose kernel size overflow")?;
    let padding = stride.div_ceil(2);
    let output_padding = stride % 2;
    let alpha_name = format!("{CODEC_TTS_PREFIX}acoustic_decoder.block.{index}.snake1.alpha");
    let weight_name = format!("{CODEC_TTS_PREFIX}acoustic_decoder.block.{index}.conv_t1.weight");
    let bias_name = format!("{CODEC_TTS_PREFIX}acoustic_decoder.block.{index}.conv_t1.bias");
    let snake_alpha = load_bf16_tensor(&model_dir, manifest, &alpha_name, &[1, in_channels, 1])?;
    let weight = load_bf16_tensor(
        &model_dir,
        manifest,
        &weight_name,
        &[in_channels, out_channels, kernel_size],
    )?;
    let bias = load_bf16_tensor(&model_dir, manifest, &bias_name, &[out_channels])?;
    DecoderBlock0ConvTransposeWeights::from_bf16(
        snake_alpha,
        weight,
        bias,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
    )
}

fn load_decoder_residual_unit(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    codec_prefix: &str,
    channels: usize,
    dilation: usize,
) -> Result<DecoderResidualUnitWeights> {
    let snake1_alpha = load_bf16_tensor(
        &model_dir,
        manifest,
        &format!("{CODEC_TTS_PREFIX}{codec_prefix}.snake1.alpha"),
        &[1, channels, 1],
    )?;
    let conv1_weight = load_bf16_tensor(
        &model_dir,
        manifest,
        &format!("{CODEC_TTS_PREFIX}{codec_prefix}.conv1.weight"),
        &[channels, channels, DECODER_RESIDUAL_KERNEL],
    )?;
    let conv1_bias = load_bf16_tensor(
        &model_dir,
        manifest,
        &format!("{CODEC_TTS_PREFIX}{codec_prefix}.conv1.bias"),
        &[channels],
    )?;
    let snake2_alpha = load_bf16_tensor(
        &model_dir,
        manifest,
        &format!("{CODEC_TTS_PREFIX}{codec_prefix}.snake2.alpha"),
        &[1, channels, 1],
    )?;
    let conv2_weight = load_bf16_tensor(
        &model_dir,
        manifest,
        &format!("{CODEC_TTS_PREFIX}{codec_prefix}.conv2.weight"),
        &[channels, channels, 1],
    )?;
    let conv2_bias = load_bf16_tensor(
        &model_dir,
        manifest,
        &format!("{CODEC_TTS_PREFIX}{codec_prefix}.conv2.bias"),
        &[channels],
    )?;
    DecoderResidualUnitWeights::from_bf16(
        channels,
        dilation,
        snake1_alpha,
        conv1_weight,
        conv1_bias,
        snake2_alpha,
        conv2_weight,
        conv2_bias,
    )
}

fn load_final_snake_alpha(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    channels: usize,
) -> Result<Vec<bf16>> {
    load_bf16_tensor(
        model_dir,
        manifest,
        &format!("{CODEC_TTS_PREFIX}acoustic_decoder.snake1.alpha"),
        &[1, channels, 1],
    )
}

fn load_final_conv(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    in_channels: usize,
) -> Result<Conv1dWeights> {
    let weight_name = format!("{CODEC_TTS_PREFIX}acoustic_decoder.conv2.weight");
    let bias_name = format!("{CODEC_TTS_PREFIX}acoustic_decoder.conv2.bias");
    let weight = load_bf16_tensor(
        &model_dir,
        manifest,
        &weight_name,
        &[1, in_channels, DECODER_FINAL_CONV_KERNEL],
    )?;
    let bias = load_bf16_tensor(&model_dir, manifest, &bias_name, &[1])?;
    Conv1dWeights::from_bf16(
        weight,
        bias,
        in_channels,
        1,
        DECODER_FINAL_CONV_KERNEL,
        DECODER_FINAL_CONV_PADDING,
    )
}

fn decoder_block_channels(config: &NativeCodecConfig, index: usize) -> Result<(usize, usize)> {
    ensure!(
        index < config.upsampling_ratios.len(),
        "decoder block index {index} out of range"
    );
    let in_channels = config
        .decoder_hidden_size
        .checked_div(1usize << index)
        .context("decoder block input channel division overflow")?;
    let out_channels = config
        .decoder_hidden_size
        .checked_div(1usize << (index + 1))
        .context("decoder block output channel division overflow")?;
    ensure!(
        in_channels > 0 && out_channels > 0,
        "decoder block {index} produced invalid channel shape {in_channels}->{out_channels}"
    );
    Ok((in_channels, out_channels))
}

fn final_decoder_channels(config: &NativeCodecConfig) -> Result<usize> {
    let (_, out_channels) =
        decoder_block_channels(config, config.upsampling_ratios.len().saturating_sub(1))?;
    Ok(out_channels)
}

fn apply_snake_cpu(input: &RvqDecodeOutput, alpha: &[bf16]) -> Result<RvqDecodeOutput> {
    ensure!(
        input.dim == alpha.len(),
        "snake alpha dim mismatch: expected {}, got {}",
        input.dim,
        alpha.len()
    );
    let mut values = vec![0.0f32; input.values.len()];
    for frame in 0..input.frames {
        for channel in 0..input.dim {
            let x = input.values[frame * input.dim + channel];
            let a = alpha[channel].to_f32();
            values[frame * input.dim + channel] = x + (a * x).sin().powi(2) / (a + 1.0e-9);
        }
    }
    Ok(RvqDecodeOutput {
        frames: input.frames,
        dim: input.dim,
        values,
    })
}

fn load_bf16_tensor(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<bf16>> {
    let file = manifest
        .weight_map
        .get(name)
        .with_context(|| format!("manifest missing codec tensor {name}"))?;
    let shard_path = model_dir.as_ref().join(file);
    let file = std::fs::File::open(&shard_path)
        .with_context(|| format!("open {}", shard_path.display()))?;
    let mmap =
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", shard_path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parse {}", shard_path.display()))?;
    let tensor = st
        .tensor(name)
        .with_context(|| format!("{} missing tensor {name}", shard_path.display()))?;
    ensure!(tensor.dtype() == Dtype::BF16, "{name} must be BF16");
    ensure!(
        tensor.shape() == expected_shape,
        "{name} shape mismatch: expected {:?}, got {:?}",
        expected_shape,
        tensor.shape()
    );
    bf16_values(tensor)
}

pub fn expected_native_codec_tensor_specs(
    config: &NativeCodecConfig,
) -> Vec<NativeCodecTensorSpec> {
    let required = [
        (
            "quantizer.quantizers.0.codebook.embed",
            vec![config.codebook_size, config.codebook_dim],
        ),
        (
            "quantizer.quantizers.7.codebook.embed",
            vec![config.codebook_size, config.codebook_dim],
        ),
        (
            "acoustic_decoder.conv1.weight",
            vec![
                config.decoder_hidden_size,
                config.acoustic_hidden_size,
                ACOUSTIC_DECODER_CONV1_KERNEL,
            ],
        ),
        (
            "acoustic_decoder.block.0.snake1.alpha",
            vec![1, config.decoder_hidden_size, 1],
        ),
        (
            "acoustic_decoder.block.0.conv_t1.weight",
            vec![
                config.decoder_hidden_size,
                DECODER_BLOCK0_OUT_CHANNELS,
                DECODER_BLOCK0_CONVT_KERNEL,
            ],
        ),
        (
            "acoustic_decoder.block.0.res_unit1.snake1.alpha",
            vec![1, DECODER_BLOCK0_OUT_CHANNELS, 1],
        ),
        (
            "acoustic_decoder.block.0.res_unit1.conv1.weight",
            vec![
                DECODER_BLOCK0_OUT_CHANNELS,
                DECODER_BLOCK0_OUT_CHANNELS,
                DECODER_RESIDUAL_KERNEL,
            ],
        ),
        (
            "acoustic_decoder.block.0.res_unit1.conv2.weight",
            vec![DECODER_BLOCK0_OUT_CHANNELS, DECODER_BLOCK0_OUT_CHANNELS, 1],
        ),
    ];
    required
        .into_iter()
        .map(|(codec_name, shape)| NativeCodecTensorSpec {
            full_name: format!("{CODEC_TTS_PREFIX}{codec_name}"),
            codec_name: codec_name.to_string(),
            dtype: "BF16",
            shape,
        })
        .collect()
}

pub fn validate_native_codec_manifest(
    model_dir: impl AsRef<Path>,
    manifest: &HiggsWeightManifest,
    config: &NativeCodecConfig,
) -> Result<NativeCodecManifestSummary> {
    let codec_tensors_in_manifest = manifest
        .weight_map
        .keys()
        .filter(|name| name.starts_with(CODEC_TTS_PREFIX))
        .count();
    ensure!(
        codec_tensors_in_manifest >= 527,
        "Higgs codec manifest is too sparse: found {codec_tensors_in_manifest} tensors with prefix {CODEC_TTS_PREFIX:?}"
    );

    let specs = expected_native_codec_tensor_specs(config);
    let mut by_file: BTreeMap<String, Vec<&NativeCodecTensorSpec>> = BTreeMap::new();
    for spec in &specs {
        let file = manifest
            .weight_map
            .get(&spec.full_name)
            .with_context(|| format!("manifest missing native codec tensor {}", spec.full_name))?;
        by_file.entry(file.clone()).or_default().push(spec);
    }

    for (file, specs) in &by_file {
        let header = read_safetensors_header(&model_dir.as_ref().join(file))?;
        for spec in specs {
            let actual = header
                .get(&spec.full_name)
                .with_context(|| format!("{file} missing tensor {}", spec.full_name))?;
            ensure!(
                actual.dtype == spec.dtype,
                "{} dtype mismatch: expected {}, got {}",
                spec.full_name,
                spec.dtype,
                actual.dtype
            );
            ensure!(
                actual.shape == spec.shape,
                "{} shape mismatch: expected {:?}, got {:?}",
                spec.full_name,
                spec.shape,
                actual.shape
            );
            ensure!(
                actual.byte_range[0] <= actual.byte_range[1],
                "{} has invalid data_offsets {:?}",
                spec.full_name,
                actual.byte_range
            );
        }
    }

    Ok(NativeCodecManifestSummary {
        codec_tensors_in_manifest,
        required_tensors_checked: specs.len(),
        files_checked: by_file.len(),
    })
}

pub fn write_pcm16_wav(path: impl AsRef<Path>, audio: &NativePcmAudio) -> Result<()> {
    ensure!(audio.sample_rate > 0, "sample_rate must be positive");
    let data_bytes = audio.samples.len() * 2;
    let riff_size = 36usize
        .checked_add(data_bytes)
        .context("wav size overflow")?;
    let mut bytes = Vec::with_capacity(44 + data_bytes);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(riff_size as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&(audio.sample_rate as u32).to_le_bytes());
    bytes.extend_from_slice(&((audio.sample_rate * 2) as u32).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for sample in &audio.samples {
        let clamped = if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        bytes.extend_from_slice(&((clamped * 32767.0).round() as i16).to_le_bytes());
    }
    std::fs::write(path.as_ref(), bytes)
        .with_context(|| format!("write {}", path.as_ref().display()))
}

fn read_safetensors_header(path: &Path) -> Result<HashMap<String, TensorHeader>> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .with_context(|| format!("read safetensors header length from {}", path.display()))?;
    let header_len = usize::try_from(u64::from_le_bytes(len_bytes)).with_context(|| {
        format!(
            "{} safetensors header length does not fit usize",
            path.display()
        )
    })?;
    ensure!(
        header_len < 512 * 1024 * 1024,
        "{} safetensors header is unexpectedly large: {} bytes",
        path.display(),
        header_len
    );
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .with_context(|| format!("read safetensors header from {}", path.display()))?;
    let value: Value = serde_json::from_slice(&header_bytes)
        .with_context(|| format!("parse safetensors header from {}", path.display()))?;
    let object = value
        .as_object()
        .with_context(|| format!("{} safetensors header is not a JSON object", path.display()))?;
    let mut tensors = HashMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(Value::as_str)
            .with_context(|| format!("{name} missing dtype"))?
            .to_string();
        let shape = value
            .get("shape")
            .and_then(Value::as_array)
            .with_context(|| format!("{name} missing shape"))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .context("shape dim is not u64")
                    .and_then(|dim| usize::try_from(dim).context("shape dim does not fit usize"))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = value
            .get("data_offsets")
            .and_then(Value::as_array)
            .with_context(|| format!("{name} missing data_offsets"))?;
        ensure!(offsets.len() == 2, "{name} data_offsets must have length 2");
        let start = offsets[0]
            .as_u64()
            .with_context(|| format!("{name} data_offsets[0] is not u64"))
            .and_then(|offset| usize::try_from(offset).context("offset does not fit usize"))?;
        let end = offsets[1]
            .as_u64()
            .with_context(|| format!("{name} data_offsets[1] is not u64"))
            .and_then(|offset| usize::try_from(offset).context("offset does not fit usize"))?;
        tensors.insert(
            name.clone(),
            TensorHeader {
                dtype,
                shape,
                byte_range: [start, end],
            },
        );
    }
    Ok(tensors)
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
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn bundled_codec_config_matches_higgs_v2_decode_contract() {
        let config = NativeCodecConfig::bundled_higgs_v2().unwrap();
        assert_eq!(config.sample_rate, 24_000);
        assert_eq!(config.hop_length, 960);
        assert_eq!(config.frame_rate, 25);
        assert_eq!(config.num_quantizers, 8);
        assert_eq!(config.codebook_size, 1024);
        assert_eq!(config.codebook_dim, 64);
        assert_eq!(config.quantizer_hidden_size, 1024);
        assert_eq!(config.acoustic_hidden_size, 256);
        assert_eq!(config.decoder_hidden_size, 1024);
        assert_eq!(config.upsampling_ratios, [8, 5, 4, 2, 3]);
    }

    #[test]
    fn expected_specs_pin_reference_decoder_surface() {
        let config = NativeCodecConfig::bundled_higgs_v2().unwrap();
        let specs = expected_native_codec_tensor_specs(&config);
        assert!(specs.iter().any(|spec| {
            spec.full_name
                == "tied.embedding.modality_embeddings.0.model.quantizer.quantizers.0.codebook.embed"
                && spec.shape == [1024, 64]
        }));
        assert!(specs.iter().any(|spec| {
            spec.full_name
                == "tied.embedding.modality_embeddings.0.model.acoustic_decoder.block.0.conv_t1.weight"
                && spec.shape == [1024, 512, 16]
        }));
        assert!(specs.iter().any(|spec| {
            spec.full_name
                == "tied.embedding.modality_embeddings.0.model.acoustic_decoder.block.0.snake1.alpha"
                && spec.shape == [1, 1024, 1]
        }));
        assert!(specs.iter().any(|spec| {
            spec.full_name
                == "tied.embedding.modality_embeddings.0.model.acoustic_decoder.block.0.res_unit1.conv1.weight"
                && spec.shape == [512, 512, 7]
        }));
    }

    #[test]
    fn raw_codec_rows_reject_sentinels_before_native_decode() {
        let config = NativeCodecConfig::bundled_higgs_v2().unwrap();
        let err = validate_raw_codec_rows(&[vec![0, 1, 2, 3, 4, 5, 6, 1024]], &config)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected < 1024"));
    }

    #[test]
    fn cuda_skeleton_decodes_toy_pcm_on_cpu() {
        let config = NativeCodecConfig {
            sample_rate: 24_000,
            hop_length: 1,
            frame_rate: 24_000,
            num_quantizers: 8,
            codebook_size: 1024,
            codebook_dim: 64,
            quantizer_hidden_size: 1,
            acoustic_hidden_size: 1,
            decoder_hidden_size: 2,
            upsampling_ratios: vec![1],
        };
        let rvq_codebooks = RvqCodebooks::from_bf16(
            vec![
                bf16::from_f32(0.0);
                config.num_quantizers * config.codebook_size * config.codebook_dim
            ],
            &config,
        )
        .unwrap();
        let decoder = NativeHiggsCodecDecoder {
            config,
            backend: NativeCodecBackend::CudaSkeleton,
            rvq_codebooks,
            quantizer_project: QuantizerProjectWeights {
                weight: vec![bf16::from_f32(0.0); 8 * 1024 * 64],
                bias: vec![bf16::from_f32(0.0); 8 * 1024],
            },
            fc2: Fc2Weights {
                weight: vec![bf16::from_f32(0.0); 1],
                bias: vec![bf16::from_f32(0.0); 1],
            },
            acoustic_decoder_conv1: Conv1dWeights {
                in_channels: 1,
                out_channels: 2,
                kernel_size: 1,
                padding: 0,
                weight: vec![bf16::from_f32(0.0); 2],
                bias: vec![bf16::from_f32(0.0); 2],
            },
            decoder_blocks: vec![DecoderUpsampleBlockWeights {
                convt: DecoderBlock0ConvTransposeWeights {
                    in_channels: 2,
                    out_channels: 1,
                    kernel_size: 1,
                    stride: 1,
                    padding: 0,
                    output_padding: 0,
                    snake_alpha: vec![bf16::from_f32(1.0); 2],
                    weight: vec![bf16::from_f32(0.0); 2],
                    bias: vec![bf16::from_f32(0.0); 1],
                },
                res1: DecoderResidualUnitWeights {
                    channels: 1,
                    dilation: 1,
                    snake1_alpha: vec![bf16::from_f32(1.0); 1],
                    conv1_weight: vec![bf16::from_f32(0.0); 7],
                    conv1_bias: vec![bf16::from_f32(0.0); 1],
                    snake2_alpha: vec![bf16::from_f32(1.0); 1],
                    conv2_weight: vec![bf16::from_f32(0.0); 1],
                    conv2_bias: vec![bf16::from_f32(0.0); 1],
                },
                res2: DecoderResidualUnitWeights {
                    channels: 1,
                    dilation: 3,
                    snake1_alpha: vec![bf16::from_f32(1.0); 1],
                    conv1_weight: vec![bf16::from_f32(0.0); 7],
                    conv1_bias: vec![bf16::from_f32(0.0); 1],
                    snake2_alpha: vec![bf16::from_f32(1.0); 1],
                    conv2_weight: vec![bf16::from_f32(0.0); 1],
                    conv2_bias: vec![bf16::from_f32(0.0); 1],
                },
                res3: DecoderResidualUnitWeights {
                    channels: 1,
                    dilation: 9,
                    snake1_alpha: vec![bf16::from_f32(1.0); 1],
                    conv1_weight: vec![bf16::from_f32(0.0); 7],
                    conv1_bias: vec![bf16::from_f32(0.0); 1],
                    snake2_alpha: vec![bf16::from_f32(1.0); 1],
                    conv2_weight: vec![bf16::from_f32(0.0); 1],
                    conv2_bias: vec![bf16::from_f32(0.0); 1],
                },
            }],
            final_snake_alpha: vec![bf16::from_f32(1.0); 1],
            final_conv: Conv1dWeights {
                in_channels: 1,
                out_channels: 1,
                kernel_size: 1,
                padding: 0,
                weight: vec![bf16::from_f32(0.0); 1],
                bias: vec![bf16::from_f32(0.125); 1],
            },
        };
        let audio = decoder
            .decode_raw_codes(&[vec![0, 1, 2, 3, 4, 5, 6, 7]])
            .unwrap();
        assert_eq!(audio.sample_rate, 24_000);
        assert_eq!(audio.samples, vec![0.125]);
    }

    #[test]
    fn rvq_cpu_decode_sums_quantizer_codebooks() {
        let config = NativeCodecConfig::bundled_higgs_v2().unwrap();
        let mut data = vec![
            bf16::from_f32(0.0);
            config.num_quantizers * config.codebook_size * config.codebook_dim
        ];
        for quantizer in 0..config.num_quantizers {
            for dim in 0..config.codebook_dim {
                let idx =
                    (quantizer * config.codebook_size + quantizer) * config.codebook_dim + dim;
                data[idx] = bf16::from_f32(quantizer as f32 + dim as f32 / 100.0);
            }
        }
        let expected_dim1: f32 = (0..config.num_quantizers)
            .map(|quantizer| {
                data[(quantizer * config.codebook_size + quantizer) * config.codebook_dim + 1]
                    .to_f32()
            })
            .sum();
        let codebooks = RvqCodebooks::from_bf16(data, &config).unwrap();
        let out = codebooks
            .decode_cpu(&[vec![0, 1, 2, 3, 4, 5, 6, 7]], &config)
            .unwrap();
        assert_eq!(out.frames, 1);
        assert_eq!(out.dim, 64);
        assert_eq!(out.values[0], 28.0);
        assert_eq!(out.values[1], expected_dim1);
    }

    #[test]
    fn quantizer_project_and_fc2_cpu_match_transformers_linear_layout() {
        let config = NativeCodecConfig::bundled_higgs_v2().unwrap();
        let mut codebook_data =
            vec![
                bf16::from_f32(0.0);
                config.num_quantizers * config.codebook_size * config.codebook_dim
            ];
        for quantizer in 0..config.num_quantizers {
            let base = (quantizer * config.codebook_size + quantizer) * config.codebook_dim;
            codebook_data[base] = bf16::from_f32(1.0 + quantizer as f32);
            codebook_data[base + 1] = bf16::from_f32(2.0);
        }
        let codebooks = RvqCodebooks::from_bf16(codebook_data, &config).unwrap();

        let mut project_weight =
            vec![
                bf16::from_f32(0.0);
                config.num_quantizers * config.quantizer_hidden_size * config.codebook_dim
            ];
        let mut project_bias =
            vec![bf16::from_f32(0.0); config.num_quantizers * config.quantizer_hidden_size];
        for quantizer in 0..config.num_quantizers {
            let weight_base = (quantizer * config.quantizer_hidden_size) * config.codebook_dim;
            project_weight[weight_base] = bf16::from_f32(0.5);
            project_weight[weight_base + 1] = bf16::from_f32(1.5);
            project_bias[quantizer * config.quantizer_hidden_size] = bf16::from_f32(0.25);
        }
        let project =
            QuantizerProjectWeights::from_bf16(project_weight, project_bias, &config).unwrap();
        let quantized = project
            .decode_cpu(&[vec![0, 1, 2, 3, 4, 5, 6, 7]], &codebooks, &config)
            .unwrap();
        assert_eq!(quantized.dim, 1024);
        let expected_hidden0: f32 = (0..8)
            .map(|q| 0.25 + (1.0 + q as f32) * 0.5 + 2.0 * 1.5)
            .sum();
        assert_eq!(quantized.values[0], expected_hidden0);

        let mut fc2_weight =
            vec![bf16::from_f32(0.0); config.acoustic_hidden_size * config.quantizer_hidden_size];
        fc2_weight[0] = bf16::from_f32(2.0);
        let mut fc2_bias = vec![bf16::from_f32(0.0); config.acoustic_hidden_size];
        fc2_bias[0] = bf16::from_f32(1.0);
        let fc2 = Fc2Weights::from_bf16(fc2_weight, fc2_bias, &config).unwrap();
        let acoustic = fc2.decode_cpu(&quantized, &config).unwrap();
        assert_eq!(acoustic.dim, 256);
        assert_eq!(acoustic.values[0], 1.0 + 2.0 * expected_hidden0);
    }

    #[test]
    fn conv1d_cpu_matches_pytorch_channel_first_padding_layout() {
        let conv = Conv1dWeights::from_bf16(
            vec![bf16::from_f32(1.0); 1 * 2 * 3],
            vec![bf16::from_f32(0.5)],
            2,
            1,
            3,
            1,
        )
        .unwrap();
        let input = RvqDecodeOutput {
            frames: 3,
            dim: 2,
            values: vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
        };
        let out = conv.decode_cpu(&input).unwrap();
        assert_eq!(out.frames, 3);
        assert_eq!(out.dim, 1);
        assert_eq!(out.values, vec![33.5, 66.5, 55.5]);
    }

    #[test]
    fn wav_writer_emits_pcm16_header() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("out.wav");
        write_pcm16_wav(
            &path,
            &NativePcmAudio {
                sample_rate: 24_000,
                samples: vec![0.0, 0.5, -0.5],
            },
        )
        .unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 50);
    }

    #[test]
    fn manifest_validation_rejects_sparse_codec_surface() {
        let config = NativeCodecConfig::bundled_higgs_v2().unwrap();
        let mut weight_map = HashMap::new();
        weight_map.insert(
            format!("{CODEC_TTS_PREFIX}quantizer.quantizers.0.codebook.embed"),
            "model.safetensors".to_string(),
        );
        let manifest = HiggsWeightManifest {
            total_size: None,
            weight_map,
        };
        let err = validate_native_codec_manifest(".", &manifest, &config)
            .unwrap_err()
            .to_string();
        assert!(err.contains("too sparse"));
    }

    #[test]
    fn all_expected_specs_are_unique() {
        let config = NativeCodecConfig::bundled_higgs_v2().unwrap();
        let specs = expected_native_codec_tensor_specs(&config);
        let names: BTreeSet<_> = specs.iter().map(|spec| spec.full_name.as_str()).collect();
        assert_eq!(names.len(), specs.len());
    }
}

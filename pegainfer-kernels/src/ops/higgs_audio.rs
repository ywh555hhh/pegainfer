use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

pub const HIGGS_AUDIO_RVQ_QUANTIZERS: usize = 8;
pub const HIGGS_AUDIO_RVQ_CODEBOOK_SIZE: usize = 1024;
pub const HIGGS_AUDIO_RVQ_DIM: usize = 64;
pub const HIGGS_AUDIO_QUANTIZER_HIDDEN: usize = 1024;
pub const HIGGS_AUDIO_ACOUSTIC_HIDDEN: usize = 256;
pub const HIGGS_AUDIO_DECODER_HIDDEN: usize = 1024;
pub const HIGGS_AUDIO_DECODER_CONV1_KERNEL: usize = 7;
pub const HIGGS_AUDIO_DECODER_BLOCK0_OUT: usize = 512;
pub const HIGGS_AUDIO_DECODER_BLOCK0_KERNEL: usize = 16;
pub const HIGGS_AUDIO_DECODER_BLOCK0_STRIDE: usize = 8;
pub const HIGGS_AUDIO_DECODER_RESIDUAL_KERNEL: usize = 7;

pub fn higgs_audio_rvq_decode_into(
    ctx: &DeviceContext,
    codes: &CudaSlice<u32>,
    codebooks: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
) -> Result<()> {
    ensure!(
        frames > 0,
        "Higgs-Audio RVQ decode needs at least one frame"
    );
    ensure!(
        codes.len() >= frames * HIGGS_AUDIO_RVQ_QUANTIZERS,
        "Higgs-Audio RVQ codes buffer is too small: len {} < {}",
        codes.len(),
        frames * HIGGS_AUDIO_RVQ_QUANTIZERS
    );
    ensure!(
        codebooks.len()
            >= HIGGS_AUDIO_RVQ_QUANTIZERS * HIGGS_AUDIO_RVQ_CODEBOOK_SIZE * HIGGS_AUDIO_RVQ_DIM,
        "Higgs-Audio RVQ codebook buffer is too small: len {}",
        codebooks.len()
    );
    ensure!(
        out.len() >= frames * HIGGS_AUDIO_RVQ_DIM,
        "Higgs-Audio RVQ output buffer is too small: len {} < {}",
        out.len(),
        frames * HIGGS_AUDIO_RVQ_DIM
    );
    let frames_i32 = i32::try_from(frames)
        .map_err(|_| anyhow!("Higgs-Audio RVQ frames {frames} does not fit i32"))?;

    let (codes_ptr, _codes_guard) = codes.device_ptr(&ctx.stream);
    let (codebook_ptr, _codebook_guard) = codebooks.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_rvq_decode_cuda(
            codes_ptr as *const u32,
            codebook_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio RVQ decode CUDA launch failed: {err}"))
}

pub fn higgs_audio_quantizer_decode_into(
    ctx: &DeviceContext,
    codes: &CudaSlice<u32>,
    codebooks: &CudaSlice<bf16>,
    project_weight: &CudaSlice<bf16>,
    project_bias: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
) -> Result<()> {
    ensure!(
        frames > 0,
        "Higgs-Audio quantizer decode needs at least one frame"
    );
    ensure!(
        codes.len() >= frames * HIGGS_AUDIO_RVQ_QUANTIZERS,
        "Higgs-Audio quantizer codes buffer is too small: len {} < {}",
        codes.len(),
        frames * HIGGS_AUDIO_RVQ_QUANTIZERS
    );
    ensure!(
        codebooks.len()
            >= HIGGS_AUDIO_RVQ_QUANTIZERS * HIGGS_AUDIO_RVQ_CODEBOOK_SIZE * HIGGS_AUDIO_RVQ_DIM,
        "Higgs-Audio quantizer codebook buffer is too small: len {}",
        codebooks.len()
    );
    ensure!(
        project_weight.len()
            >= HIGGS_AUDIO_RVQ_QUANTIZERS * HIGGS_AUDIO_QUANTIZER_HIDDEN * HIGGS_AUDIO_RVQ_DIM,
        "Higgs-Audio quantizer project weight buffer is too small: len {}",
        project_weight.len()
    );
    ensure!(
        project_bias.len() >= HIGGS_AUDIO_RVQ_QUANTIZERS * HIGGS_AUDIO_QUANTIZER_HIDDEN,
        "Higgs-Audio quantizer project bias buffer is too small: len {}",
        project_bias.len()
    );
    ensure!(
        out.len() >= frames * HIGGS_AUDIO_QUANTIZER_HIDDEN,
        "Higgs-Audio quantizer output buffer is too small: len {} < {}",
        out.len(),
        frames * HIGGS_AUDIO_QUANTIZER_HIDDEN
    );
    let frames_i32 = i32::try_from(frames)
        .map_err(|_| anyhow!("Higgs-Audio quantizer frames {frames} does not fit i32"))?;

    let (codes_ptr, _codes_guard) = codes.device_ptr(&ctx.stream);
    let (codebook_ptr, _codebook_guard) = codebooks.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = project_weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = project_bias.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_quantizer_decode_cuda(
            codes_ptr as *const u32,
            codebook_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio quantizer decode CUDA launch failed: {err}"))
}

pub fn higgs_audio_fc2_into(
    ctx: &DeviceContext,
    hidden: &CudaSlice<f32>,
    weight: &CudaSlice<bf16>,
    bias: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
) -> Result<()> {
    ensure!(frames > 0, "Higgs-Audio fc2 needs at least one frame");
    ensure!(
        hidden.len() >= frames * HIGGS_AUDIO_QUANTIZER_HIDDEN,
        "Higgs-Audio fc2 hidden buffer is too small: len {} < {}",
        hidden.len(),
        frames * HIGGS_AUDIO_QUANTIZER_HIDDEN
    );
    ensure!(
        weight.len() >= HIGGS_AUDIO_ACOUSTIC_HIDDEN * HIGGS_AUDIO_QUANTIZER_HIDDEN,
        "Higgs-Audio fc2 weight buffer is too small: len {}",
        weight.len()
    );
    ensure!(
        bias.len() >= HIGGS_AUDIO_ACOUSTIC_HIDDEN,
        "Higgs-Audio fc2 bias buffer is too small: len {}",
        bias.len()
    );
    ensure!(
        out.len() >= frames * HIGGS_AUDIO_ACOUSTIC_HIDDEN,
        "Higgs-Audio fc2 output buffer is too small: len {} < {}",
        out.len(),
        frames * HIGGS_AUDIO_ACOUSTIC_HIDDEN
    );
    let frames_i32 = i32::try_from(frames)
        .map_err(|_| anyhow!("Higgs-Audio fc2 frames {frames} does not fit i32"))?;

    let (hidden_ptr, _hidden_guard) = hidden.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_fc2_cuda(
            hidden_ptr as *const f32,
            weight_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio fc2 CUDA launch failed: {err}"))
}

pub fn higgs_audio_acoustic_decoder_conv1_into(
    ctx: &DeviceContext,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<bf16>,
    bias: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
) -> Result<()> {
    ensure!(
        frames > 0,
        "Higgs-Audio acoustic decoder conv1 needs at least one frame"
    );
    ensure!(
        input.len() >= frames * HIGGS_AUDIO_ACOUSTIC_HIDDEN,
        "Higgs-Audio acoustic decoder conv1 input buffer is too small: len {} < {}",
        input.len(),
        frames * HIGGS_AUDIO_ACOUSTIC_HIDDEN
    );
    ensure!(
        weight.len()
            >= HIGGS_AUDIO_DECODER_HIDDEN
                * HIGGS_AUDIO_ACOUSTIC_HIDDEN
                * HIGGS_AUDIO_DECODER_CONV1_KERNEL,
        "Higgs-Audio acoustic decoder conv1 weight buffer is too small: len {}",
        weight.len()
    );
    ensure!(
        bias.len() >= HIGGS_AUDIO_DECODER_HIDDEN,
        "Higgs-Audio acoustic decoder conv1 bias buffer is too small: len {}",
        bias.len()
    );
    ensure!(
        out.len() >= frames * HIGGS_AUDIO_DECODER_HIDDEN,
        "Higgs-Audio acoustic decoder conv1 output buffer is too small: len {} < {}",
        out.len(),
        frames * HIGGS_AUDIO_DECODER_HIDDEN
    );
    let frames_i32 = i32::try_from(frames).map_err(|_| {
        anyhow!("Higgs-Audio acoustic decoder conv1 frames {frames} does not fit i32")
    })?;

    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_acoustic_decoder_conv1_cuda(
            input_ptr as *const f32,
            weight_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio acoustic decoder conv1 CUDA launch failed: {err}"))
}

pub fn higgs_audio_decoder_block0_convt_into(
    ctx: &DeviceContext,
    input: &CudaSlice<f32>,
    snake_alpha: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
) -> Result<()> {
    ensure!(
        frames > 0,
        "Higgs-Audio decoder block0 conv_t1 needs at least one frame"
    );
    ensure!(
        input.len() >= frames * HIGGS_AUDIO_DECODER_HIDDEN,
        "Higgs-Audio decoder block0 conv_t1 input buffer is too small: len {} < {}",
        input.len(),
        frames * HIGGS_AUDIO_DECODER_HIDDEN
    );
    ensure!(
        snake_alpha.len() >= HIGGS_AUDIO_DECODER_HIDDEN,
        "Higgs-Audio decoder block0 snake alpha buffer is too small: len {}",
        snake_alpha.len()
    );
    ensure!(
        weight.len()
            >= HIGGS_AUDIO_DECODER_HIDDEN
                * HIGGS_AUDIO_DECODER_BLOCK0_OUT
                * HIGGS_AUDIO_DECODER_BLOCK0_KERNEL,
        "Higgs-Audio decoder block0 conv_t1 weight buffer is too small: len {}",
        weight.len()
    );
    ensure!(
        out.len() >= frames * HIGGS_AUDIO_DECODER_BLOCK0_STRIDE * HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder block0 conv_t1 output buffer is too small: len {} < {}",
        out.len(),
        frames * HIGGS_AUDIO_DECODER_BLOCK0_STRIDE * HIGGS_AUDIO_DECODER_BLOCK0_OUT
    );
    let frames_i32 = i32::try_from(frames).map_err(|_| {
        anyhow!("Higgs-Audio decoder block0 conv_t1 frames {frames} does not fit i32")
    })?;

    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (alpha_ptr, _alpha_guard) = snake_alpha.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_decoder_block0_convt_cuda(
            input_ptr as *const f32,
            alpha_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio decoder block0 conv_t1 CUDA launch failed: {err}"))
}

pub fn higgs_audio_decoder_residual_conv1_into(
    ctx: &DeviceContext,
    input: &CudaSlice<f32>,
    snake_alpha: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
    dilation: usize,
) -> Result<()> {
    ensure!(
        frames > 0,
        "Higgs-Audio decoder residual conv1 needs at least one frame"
    );
    ensure!(
        dilation > 0,
        "Higgs-Audio decoder residual conv1 dilation must be positive"
    );
    ensure!(
        input.len() >= frames * HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder residual conv1 input buffer is too small: len {} < {}",
        input.len(),
        frames * HIGGS_AUDIO_DECODER_BLOCK0_OUT
    );
    ensure!(
        snake_alpha.len() >= HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder residual snake alpha buffer is too small: len {}",
        snake_alpha.len()
    );
    ensure!(
        weight.len()
            >= HIGGS_AUDIO_DECODER_BLOCK0_OUT
                * HIGGS_AUDIO_DECODER_BLOCK0_OUT
                * HIGGS_AUDIO_DECODER_RESIDUAL_KERNEL,
        "Higgs-Audio decoder residual conv1 weight buffer is too small: len {}",
        weight.len()
    );
    ensure!(
        bias.len() >= HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder residual conv1 bias buffer is too small: len {}",
        bias.len()
    );
    ensure!(
        out.len() >= frames * HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder residual conv1 output buffer is too small: len {} < {}",
        out.len(),
        frames * HIGGS_AUDIO_DECODER_BLOCK0_OUT
    );
    let frames_i32 = i32::try_from(frames).map_err(|_| {
        anyhow!("Higgs-Audio decoder residual conv1 frames {frames} does not fit i32")
    })?;
    let dilation_i32 = i32::try_from(dilation).map_err(|_| {
        anyhow!("Higgs-Audio decoder residual conv1 dilation {dilation} does not fit i32")
    })?;

    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (alpha_ptr, _alpha_guard) = snake_alpha.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_decoder_residual_conv1_cuda(
            input_ptr as *const f32,
            alpha_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            dilation_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio decoder residual conv1 CUDA launch failed: {err}"))
}

pub fn higgs_audio_decoder_residual_conv2_add_into(
    ctx: &DeviceContext,
    residual: &CudaSlice<f32>,
    conv1: &CudaSlice<f32>,
    snake_alpha: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
) -> Result<()> {
    ensure!(
        frames > 0,
        "Higgs-Audio decoder residual conv2 add needs at least one frame"
    );
    ensure!(
        residual.len() >= frames * HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder residual buffer is too small: len {} < {}",
        residual.len(),
        frames * HIGGS_AUDIO_DECODER_BLOCK0_OUT
    );
    ensure!(
        conv1.len() >= frames * HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder residual conv1 buffer is too small: len {} < {}",
        conv1.len(),
        frames * HIGGS_AUDIO_DECODER_BLOCK0_OUT
    );
    ensure!(
        snake_alpha.len() >= HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder residual snake2 alpha buffer is too small: len {}",
        snake_alpha.len()
    );
    ensure!(
        weight.len() >= HIGGS_AUDIO_DECODER_BLOCK0_OUT * HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder residual conv2 weight buffer is too small: len {}",
        weight.len()
    );
    ensure!(
        bias.len() >= HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder residual conv2 bias buffer is too small: len {}",
        bias.len()
    );
    ensure!(
        out.len() >= frames * HIGGS_AUDIO_DECODER_BLOCK0_OUT,
        "Higgs-Audio decoder residual output buffer is too small: len {} < {}",
        out.len(),
        frames * HIGGS_AUDIO_DECODER_BLOCK0_OUT
    );
    let frames_i32 = i32::try_from(frames).map_err(|_| {
        anyhow!("Higgs-Audio decoder residual conv2 frames {frames} does not fit i32")
    })?;

    let (residual_ptr, _residual_guard) = residual.device_ptr(&ctx.stream);
    let (conv1_ptr, _conv1_guard) = conv1.device_ptr(&ctx.stream);
    let (alpha_ptr, _alpha_guard) = snake_alpha.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_decoder_residual_conv2_add_cuda(
            residual_ptr as *const f32,
            conv1_ptr as *const f32,
            alpha_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio decoder residual conv2 add CUDA launch failed: {err}"))
}

#[derive(Debug, Clone, Copy)]
pub struct HiggsAudioConvTransposeSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
}

impl HiggsAudioConvTransposeSpec {
    pub fn out_frames(self, frames: usize) -> Result<usize> {
        ensure!(
            frames > 0,
            "Higgs-Audio ConvTranspose needs at least one frame"
        );
        frames
            .checked_sub(1)
            .and_then(|n| n.checked_mul(self.stride))
            .and_then(|n| n.checked_add(self.kernel_size))
            .and_then(|n| n.checked_add(self.output_padding))
            .and_then(|n| n.checked_sub(2 * self.padding))
            .ok_or_else(|| anyhow!("Higgs-Audio ConvTranspose output frame count overflow"))
    }
}

pub fn higgs_audio_decoder_convt_generic_into(
    ctx: &DeviceContext,
    input: &CudaSlice<f32>,
    snake_alpha: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
    spec: HiggsAudioConvTransposeSpec,
) -> Result<()> {
    ensure!(
        spec.in_channels > 0,
        "Higgs-Audio ConvTranspose in_channels must be positive"
    );
    ensure!(
        spec.out_channels > 0,
        "Higgs-Audio ConvTranspose out_channels must be positive"
    );
    ensure!(
        spec.kernel_size > 0,
        "Higgs-Audio ConvTranspose kernel_size must be positive"
    );
    ensure!(
        spec.stride > 0,
        "Higgs-Audio ConvTranspose stride must be positive"
    );
    let out_frames = spec.out_frames(frames)?;
    ensure!(
        input.len() >= frames * spec.in_channels,
        "Higgs-Audio ConvTranspose input buffer is too small: len {} < {}",
        input.len(),
        frames * spec.in_channels
    );
    ensure!(
        snake_alpha.len() >= spec.in_channels,
        "Higgs-Audio ConvTranspose snake alpha buffer is too small: len {} < {}",
        snake_alpha.len(),
        spec.in_channels
    );
    ensure!(
        weight.len() >= spec.in_channels * spec.out_channels * spec.kernel_size,
        "Higgs-Audio ConvTranspose weight buffer is too small: len {} < {}",
        weight.len(),
        spec.in_channels * spec.out_channels * spec.kernel_size
    );
    ensure!(
        bias.len() >= spec.out_channels,
        "Higgs-Audio ConvTranspose bias buffer is too small: len {} < {}",
        bias.len(),
        spec.out_channels
    );
    ensure!(
        out.len() >= out_frames * spec.out_channels,
        "Higgs-Audio ConvTranspose output buffer is too small: len {} < {}",
        out.len(),
        out_frames * spec.out_channels
    );
    let frames_i32 =
        i32::try_from(frames).map_err(|_| anyhow!("frames {frames} does not fit i32"))?;
    let in_channels_i32 = i32::try_from(spec.in_channels)
        .map_err(|_| anyhow!("in_channels {} does not fit i32", spec.in_channels))?;
    let out_channels_i32 = i32::try_from(spec.out_channels)
        .map_err(|_| anyhow!("out_channels {} does not fit i32", spec.out_channels))?;
    let kernel_i32 = i32::try_from(spec.kernel_size)
        .map_err(|_| anyhow!("kernel_size {} does not fit i32", spec.kernel_size))?;
    let stride_i32 = i32::try_from(spec.stride)
        .map_err(|_| anyhow!("stride {} does not fit i32", spec.stride))?;
    let padding_i32 = i32::try_from(spec.padding)
        .map_err(|_| anyhow!("padding {} does not fit i32", spec.padding))?;
    let output_padding_i32 = i32::try_from(spec.output_padding)
        .map_err(|_| anyhow!("output_padding {} does not fit i32", spec.output_padding))?;

    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (alpha_ptr, _alpha_guard) = snake_alpha.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_decoder_convt_generic_cuda(
            input_ptr as *const f32,
            alpha_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            in_channels_i32,
            out_channels_i32,
            kernel_i32,
            stride_i32,
            padding_i32,
            output_padding_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio generic ConvTranspose CUDA launch failed: {err}"))
}

pub fn higgs_audio_decoder_residual_conv1_generic_into(
    ctx: &DeviceContext,
    input: &CudaSlice<f32>,
    snake_alpha: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
    channels: usize,
    dilation: usize,
) -> Result<()> {
    ensure!(
        channels > 0,
        "Higgs-Audio residual channels must be positive"
    );
    ensure!(
        dilation > 0,
        "Higgs-Audio residual dilation must be positive"
    );
    ensure!(
        input.len() >= frames * channels,
        "Higgs-Audio residual input too small"
    );
    ensure!(
        snake_alpha.len() >= channels,
        "Higgs-Audio residual snake alpha too small"
    );
    ensure!(
        weight.len() >= channels * channels * HIGGS_AUDIO_DECODER_RESIDUAL_KERNEL,
        "Higgs-Audio residual conv1 weight too small"
    );
    ensure!(
        bias.len() >= channels,
        "Higgs-Audio residual conv1 bias too small"
    );
    ensure!(
        out.len() >= frames * channels,
        "Higgs-Audio residual conv1 output too small"
    );
    let frames_i32 =
        i32::try_from(frames).map_err(|_| anyhow!("frames {frames} does not fit i32"))?;
    let channels_i32 =
        i32::try_from(channels).map_err(|_| anyhow!("channels {channels} does not fit i32"))?;
    let dilation_i32 =
        i32::try_from(dilation).map_err(|_| anyhow!("dilation {dilation} does not fit i32"))?;
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (alpha_ptr, _alpha_guard) = snake_alpha.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_decoder_residual_conv1_generic_cuda(
            input_ptr as *const f32,
            alpha_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            channels_i32,
            dilation_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio generic residual conv1 CUDA launch failed: {err}"))
}

pub fn higgs_audio_decoder_residual_conv2_add_generic_into(
    ctx: &DeviceContext,
    residual: &CudaSlice<f32>,
    conv1: &CudaSlice<f32>,
    snake_alpha: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
    channels: usize,
) -> Result<()> {
    ensure!(
        channels > 0,
        "Higgs-Audio residual channels must be positive"
    );
    ensure!(
        residual.len() >= frames * channels,
        "Higgs-Audio residual buffer too small"
    );
    ensure!(
        conv1.len() >= frames * channels,
        "Higgs-Audio residual conv1 buffer too small"
    );
    ensure!(
        snake_alpha.len() >= channels,
        "Higgs-Audio residual snake2 alpha too small"
    );
    ensure!(
        weight.len() >= channels * channels,
        "Higgs-Audio residual conv2 weight too small"
    );
    ensure!(
        bias.len() >= channels,
        "Higgs-Audio residual conv2 bias too small"
    );
    ensure!(
        out.len() >= frames * channels,
        "Higgs-Audio residual output too small"
    );
    let frames_i32 =
        i32::try_from(frames).map_err(|_| anyhow!("frames {frames} does not fit i32"))?;
    let channels_i32 =
        i32::try_from(channels).map_err(|_| anyhow!("channels {channels} does not fit i32"))?;
    let (residual_ptr, _residual_guard) = residual.device_ptr(&ctx.stream);
    let (conv1_ptr, _conv1_guard) = conv1.device_ptr(&ctx.stream);
    let (alpha_ptr, _alpha_guard) = snake_alpha.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_decoder_residual_conv2_add_generic_cuda(
            residual_ptr as *const f32,
            conv1_ptr as *const f32,
            alpha_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            channels_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio generic residual conv2 add CUDA launch failed: {err}"))
}

pub fn higgs_audio_conv1d_generic_into(
    ctx: &DeviceContext,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<bf16>,
    bias: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
) -> Result<()> {
    ensure!(
        input.len() >= frames * in_channels,
        "Higgs-Audio Conv1d input too small"
    );
    ensure!(
        weight.len() >= out_channels * in_channels * kernel_size,
        "Higgs-Audio Conv1d weight too small"
    );
    ensure!(
        bias.len() >= out_channels,
        "Higgs-Audio Conv1d bias too small"
    );
    ensure!(
        out.len() >= frames * out_channels,
        "Higgs-Audio Conv1d output too small"
    );
    let frames_i32 =
        i32::try_from(frames).map_err(|_| anyhow!("frames {frames} does not fit i32"))?;
    let in_i32 = i32::try_from(in_channels)
        .map_err(|_| anyhow!("in_channels {in_channels} does not fit i32"))?;
    let out_i32 = i32::try_from(out_channels)
        .map_err(|_| anyhow!("out_channels {out_channels} does not fit i32"))?;
    let kernel_i32 = i32::try_from(kernel_size)
        .map_err(|_| anyhow!("kernel_size {kernel_size} does not fit i32"))?;
    let padding_i32 =
        i32::try_from(padding).map_err(|_| anyhow!("padding {padding} does not fit i32"))?;
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_conv1d_generic_cuda(
            input_ptr as *const f32,
            weight_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            in_i32,
            out_i32,
            kernel_i32,
            padding_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio generic Conv1d CUDA launch failed: {err}"))
}

pub fn higgs_audio_snake_conv1d_generic_into(
    ctx: &DeviceContext,
    input: &CudaSlice<f32>,
    snake_alpha: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    bias: &CudaSlice<bf16>,
    out: &mut CudaSlice<f32>,
    frames: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
) -> Result<()> {
    ensure!(
        input.len() >= frames * in_channels,
        "Higgs-Audio Snake+Conv1d input too small"
    );
    ensure!(
        snake_alpha.len() >= in_channels,
        "Higgs-Audio Snake+Conv1d alpha too small"
    );
    ensure!(
        weight.len() >= out_channels * in_channels * kernel_size,
        "Higgs-Audio Snake+Conv1d weight too small"
    );
    ensure!(
        bias.len() >= out_channels,
        "Higgs-Audio Snake+Conv1d bias too small"
    );
    ensure!(
        out.len() >= frames * out_channels,
        "Higgs-Audio Snake+Conv1d output too small"
    );
    let frames_i32 =
        i32::try_from(frames).map_err(|_| anyhow!("frames {frames} does not fit i32"))?;
    let in_i32 = i32::try_from(in_channels)
        .map_err(|_| anyhow!("in_channels {in_channels} does not fit i32"))?;
    let out_i32 = i32::try_from(out_channels)
        .map_err(|_| anyhow!("out_channels {out_channels} does not fit i32"))?;
    let kernel_i32 = i32::try_from(kernel_size)
        .map_err(|_| anyhow!("kernel_size {kernel_size} does not fit i32"))?;
    let padding_i32 =
        i32::try_from(padding).map_err(|_| anyhow!("padding {padding} does not fit i32"))?;
    let (input_ptr, _input_guard) = input.device_ptr(&ctx.stream);
    let (alpha_ptr, _alpha_guard) = snake_alpha.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::higgs_audio_snake_conv1d_generic_cuda(
            input_ptr as *const f32,
            alpha_ptr as *const ffi::Half,
            weight_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            frames_i32,
            in_i32,
            out_i32,
            kernel_i32,
            padding_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("Higgs-Audio generic Snake+Conv1d CUDA launch failed: {err}"))
}

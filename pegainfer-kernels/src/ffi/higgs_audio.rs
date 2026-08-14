use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;

use super::Half;

unsafe extern "C" {
    pub fn higgs_audio_rvq_decode_cuda(
        codes: *const u32,
        codebooks: *const Half,
        out: *mut f32,
        frames: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_quantizer_decode_cuda(
        codes: *const u32,
        codebooks: *const Half,
        project_weight: *const Half,
        project_bias: *const Half,
        out: *mut f32,
        frames: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_fc2_cuda(
        hidden: *const f32,
        weight: *const Half,
        bias: *const Half,
        out: *mut f32,
        frames: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_acoustic_decoder_conv1_cuda(
        input: *const f32,
        weight: *const Half,
        bias: *const Half,
        out: *mut f32,
        frames: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_decoder_block0_convt_cuda(
        input: *const f32,
        snake_alpha: *const Half,
        weight: *const Half,
        out: *mut f32,
        frames: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_decoder_residual_conv1_cuda(
        input: *const f32,
        snake_alpha: *const Half,
        weight: *const Half,
        bias: *const Half,
        out: *mut f32,
        frames: i32,
        dilation: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_decoder_residual_conv2_add_cuda(
        residual: *const f32,
        conv1: *const f32,
        snake_alpha: *const Half,
        weight: *const Half,
        bias: *const Half,
        out: *mut f32,
        frames: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_decoder_convt_generic_cuda(
        input: *const f32,
        snake_alpha: *const Half,
        weight: *const Half,
        bias: *const Half,
        out: *mut f32,
        frames: i32,
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        output_padding: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_decoder_residual_conv1_generic_cuda(
        input: *const f32,
        snake_alpha: *const Half,
        weight: *const Half,
        bias: *const Half,
        out: *mut f32,
        frames: i32,
        channels: i32,
        dilation: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_decoder_residual_conv2_add_generic_cuda(
        residual: *const f32,
        conv1: *const f32,
        snake_alpha: *const Half,
        weight: *const Half,
        bias: *const Half,
        out: *mut f32,
        frames: i32,
        channels: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_conv1d_generic_cuda(
        input: *const f32,
        weight: *const Half,
        bias: *const Half,
        out: *mut f32,
        frames: i32,
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        padding: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn higgs_audio_snake_conv1d_generic_cuda(
        input: *const f32,
        snake_alpha: *const Half,
        weight: *const Half,
        bias: *const Half,
        out: *mut f32,
        frames: i32,
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        padding: i32,
        stream: CUstream,
    ) -> CUresult;
}

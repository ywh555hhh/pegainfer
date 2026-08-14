#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQuantizers = 8;
constexpr int kCodebookSize = 1024;
constexpr int kDim = 64;
constexpr int kHidden = 1024;
constexpr int kAcousticHidden = 256;
constexpr int kDecoderHidden = 1024;
constexpr int kDecoderConv1Kernel = 7;
constexpr int kDecoderConv1Padding = 3;
constexpr int kDecoderBlock0Out = 512;
constexpr int kDecoderBlock0Kernel = 16;
constexpr int kDecoderBlock0Stride = 8;
constexpr int kDecoderBlock0Padding = 4;
constexpr int kDecoderResidualKernel = 7;

__global__ void higgs_audio_rvq_decode_kernel(
    const uint32_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ codebooks,
    float* __restrict__ out,
    int frames) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = frames * kDim;
  if (idx >= total) {
    return;
  }
  const int frame = idx / kDim;
  const int dim = idx - frame * kDim;
  float acc = 0.0f;
  for (int q = 0; q < kQuantizers; ++q) {
    const uint32_t code = codes[static_cast<size_t>(frame) * kQuantizers + q];
    if (code >= kCodebookSize) {
      out[idx] = nanf("");
      return;
    }
    const size_t offset =
        (static_cast<size_t>(q) * kCodebookSize + code) * kDim + dim;
    acc += __bfloat162float(codebooks[offset]);
  }
  out[idx] = acc;
}

__global__ void higgs_audio_quantizer_decode_kernel(
    const uint32_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ codebooks,
    const __nv_bfloat16* __restrict__ project_weight,
    const __nv_bfloat16* __restrict__ project_bias,
    float* __restrict__ out,
    int frames) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = frames * kHidden;
  if (idx >= total) {
    return;
  }
  const int frame = idx / kHidden;
  const int hidden = idx - frame * kHidden;
  float acc = 0.0f;
  for (int q = 0; q < kQuantizers; ++q) {
    const uint32_t code = codes[static_cast<size_t>(frame) * kQuantizers + q];
    if (code >= kCodebookSize) {
      out[idx] = nanf("");
      return;
    }
    float q_acc = __bfloat162float(project_bias[static_cast<size_t>(q) * kHidden + hidden]);
    const size_t codebook_base =
        (static_cast<size_t>(q) * kCodebookSize + code) * kDim;
    const size_t weight_base =
        (static_cast<size_t>(q) * kHidden + hidden) * kDim;
    for (int dim = 0; dim < kDim; ++dim) {
      q_acc += __bfloat162float(codebooks[codebook_base + dim]) *
               __bfloat162float(project_weight[weight_base + dim]);
    }
    acc += q_acc;
  }
  out[idx] = acc;
}

__global__ void higgs_audio_fc2_kernel(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ out,
    int frames) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = frames * kAcousticHidden;
  if (idx >= total) {
    return;
  }
  const int frame = idx / kAcousticHidden;
  const int channel = idx - frame * kAcousticHidden;
  float acc = __bfloat162float(bias[channel]);
  const float* row_hidden = hidden + static_cast<size_t>(frame) * kHidden;
  const __nv_bfloat16* row_weight = weight + static_cast<size_t>(channel) * kHidden;
  for (int hidden_idx = 0; hidden_idx < kHidden; ++hidden_idx) {
    acc += row_hidden[hidden_idx] * __bfloat162float(row_weight[hidden_idx]);
  }
  out[idx] = acc;
}

__global__ void higgs_audio_acoustic_decoder_conv1_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ out,
    int frames) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = frames * kDecoderHidden;
  if (idx >= total) {
    return;
  }
  const int frame = idx / kDecoderHidden;
  const int out_channel = idx - frame * kDecoderHidden;
  float acc = __bfloat162float(bias[out_channel]);
  for (int in_channel = 0; in_channel < kAcousticHidden; ++in_channel) {
    for (int kernel_idx = 0; kernel_idx < kDecoderConv1Kernel; ++kernel_idx) {
      const int input_frame = frame + kernel_idx - kDecoderConv1Padding;
      if (input_frame < 0 || input_frame >= frames) {
        continue;
      }
      const float input_value =
          input[static_cast<size_t>(input_frame) * kAcousticHidden + in_channel];
      const size_t weight_idx =
          (static_cast<size_t>(out_channel) * kAcousticHidden + in_channel) *
              kDecoderConv1Kernel +
          kernel_idx;
      acc += input_value * __bfloat162float(weight[weight_idx]);
    }
  }
  out[idx] = acc;
}

__global__ void higgs_audio_decoder_block0_convt_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ snake_alpha,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ out,
    int frames) {
  const int out_frames = frames * kDecoderBlock0Stride;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = out_frames * kDecoderBlock0Out;
  if (idx >= total) {
    return;
  }
  const int out_frame = idx / kDecoderBlock0Out;
  const int out_channel = idx - out_frame * kDecoderBlock0Out;
  float acc = 0.0f;
  for (int in_channel = 0; in_channel < kDecoderHidden; ++in_channel) {
    for (int kernel_idx = 0; kernel_idx < kDecoderBlock0Kernel; ++kernel_idx) {
      const int numerator = out_frame + kDecoderBlock0Padding - kernel_idx;
      if (numerator < 0 || numerator % kDecoderBlock0Stride != 0) {
        continue;
      }
      const int input_frame = numerator / kDecoderBlock0Stride;
      if (input_frame >= frames) {
        continue;
      }
      const float x =
          input[static_cast<size_t>(input_frame) * kDecoderHidden + in_channel];
      const float alpha = __bfloat162float(snake_alpha[in_channel]);
      const float s = __sinf(alpha * x);
      const float snake = x + (s * s) / (alpha + 1.0e-9f);
      const size_t weight_idx =
          (static_cast<size_t>(in_channel) * kDecoderBlock0Out + out_channel) *
              kDecoderBlock0Kernel +
          kernel_idx;
      acc += snake * __bfloat162float(weight[weight_idx]);
    }
  }
  out[idx] = acc;
}

__global__ void higgs_audio_decoder_residual_conv1_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ snake_alpha,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ out,
    int frames,
    int dilation) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = frames * kDecoderBlock0Out;
  if (idx >= total) {
    return;
  }
  const int frame = idx / kDecoderBlock0Out;
  const int out_channel = idx - frame * kDecoderBlock0Out;
  const int padding = ((kDecoderResidualKernel - 1) * dilation) / 2;
  float acc = __bfloat162float(bias[out_channel]);
  for (int in_channel = 0; in_channel < kDecoderBlock0Out; ++in_channel) {
    for (int kernel_idx = 0; kernel_idx < kDecoderResidualKernel; ++kernel_idx) {
      const int input_frame = frame + kernel_idx * dilation - padding;
      if (input_frame < 0 || input_frame >= frames) {
        continue;
      }
      const float x =
          input[static_cast<size_t>(input_frame) * kDecoderBlock0Out + in_channel];
      const float alpha = __bfloat162float(snake_alpha[in_channel]);
      const float s = __sinf(alpha * x);
      const float snake = x + (s * s) / (alpha + 1.0e-9f);
      const size_t weight_idx =
          (static_cast<size_t>(out_channel) * kDecoderBlock0Out + in_channel) *
              kDecoderResidualKernel +
          kernel_idx;
      acc += snake * __bfloat162float(weight[weight_idx]);
    }
  }
  out[idx] = acc;
}

__global__ void higgs_audio_decoder_residual_conv2_add_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ conv1,
    const __nv_bfloat16* __restrict__ snake_alpha,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ out,
    int frames) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = frames * kDecoderBlock0Out;
  if (idx >= total) {
    return;
  }
  const int frame = idx / kDecoderBlock0Out;
  const int out_channel = idx - frame * kDecoderBlock0Out;
  float acc = __bfloat162float(bias[out_channel]);
  for (int in_channel = 0; in_channel < kDecoderBlock0Out; ++in_channel) {
    const float x = conv1[static_cast<size_t>(frame) * kDecoderBlock0Out + in_channel];
    const float alpha = __bfloat162float(snake_alpha[in_channel]);
    const float s = __sinf(alpha * x);
    const float snake = x + (s * s) / (alpha + 1.0e-9f);
    const size_t weight_idx =
        static_cast<size_t>(out_channel) * kDecoderBlock0Out + in_channel;
    acc += snake * __bfloat162float(weight[weight_idx]);
  }
  out[idx] = residual[idx] + acc;
}

__global__ void higgs_audio_decoder_convt_generic_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ snake_alpha,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ out,
    int frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding) {
  const int out_frames = (frames - 1) * stride - 2 * padding + kernel_size + output_padding;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = out_frames * out_channels;
  if (idx >= total) {
    return;
  }
  const int out_frame = idx / out_channels;
  const int out_channel = idx - out_frame * out_channels;
  float acc = __bfloat162float(bias[out_channel]);
  for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
    for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
      const int numerator = out_frame + padding - kernel_idx;
      if (numerator < 0 || numerator % stride != 0) {
        continue;
      }
      const int input_frame = numerator / stride;
      if (input_frame >= frames) {
        continue;
      }
      const float x = input[static_cast<size_t>(input_frame) * in_channels + in_channel];
      const float alpha = __bfloat162float(snake_alpha[in_channel]);
      const float s = __sinf(alpha * x);
      const float snake = x + (s * s) / (alpha + 1.0e-9f);
      const size_t weight_idx =
          (static_cast<size_t>(in_channel) * out_channels + out_channel) * kernel_size +
          kernel_idx;
      acc += snake * __bfloat162float(weight[weight_idx]);
    }
  }
  out[idx] = acc;
}

__global__ void higgs_audio_decoder_residual_conv1_generic_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ snake_alpha,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ out,
    int frames,
    int channels,
    int dilation) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = frames * channels;
  if (idx >= total) {
    return;
  }
  const int frame = idx / channels;
  const int out_channel = idx - frame * channels;
  const int padding = ((kDecoderResidualKernel - 1) * dilation) / 2;
  float acc = __bfloat162float(bias[out_channel]);
  for (int in_channel = 0; in_channel < channels; ++in_channel) {
    for (int kernel_idx = 0; kernel_idx < kDecoderResidualKernel; ++kernel_idx) {
      const int input_frame = frame + kernel_idx * dilation - padding;
      if (input_frame < 0 || input_frame >= frames) {
        continue;
      }
      const float x = input[static_cast<size_t>(input_frame) * channels + in_channel];
      const float alpha = __bfloat162float(snake_alpha[in_channel]);
      const float s = __sinf(alpha * x);
      const float snake = x + (s * s) / (alpha + 1.0e-9f);
      const size_t weight_idx =
          (static_cast<size_t>(out_channel) * channels + in_channel) * kDecoderResidualKernel +
          kernel_idx;
      acc += snake * __bfloat162float(weight[weight_idx]);
    }
  }
  out[idx] = acc;
}

__global__ void higgs_audio_decoder_residual_conv2_add_generic_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ conv1,
    const __nv_bfloat16* __restrict__ snake_alpha,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ out,
    int frames,
    int channels) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = frames * channels;
  if (idx >= total) {
    return;
  }
  const int frame = idx / channels;
  const int out_channel = idx - frame * channels;
  float acc = __bfloat162float(bias[out_channel]);
  for (int in_channel = 0; in_channel < channels; ++in_channel) {
    const float x = conv1[static_cast<size_t>(frame) * channels + in_channel];
    const float alpha = __bfloat162float(snake_alpha[in_channel]);
    const float s = __sinf(alpha * x);
    const float snake = x + (s * s) / (alpha + 1.0e-9f);
    const size_t weight_idx = static_cast<size_t>(out_channel) * channels + in_channel;
    acc += snake * __bfloat162float(weight[weight_idx]);
  }
  out[idx] = residual[idx] + acc;
}

__global__ void higgs_audio_conv1d_generic_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ out,
    int frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int padding) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = frames * out_channels;
  if (idx >= total) {
    return;
  }
  const int frame = idx / out_channels;
  const int out_channel = idx - frame * out_channels;
  float acc = __bfloat162float(bias[out_channel]);
  for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
    for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
      const int input_frame = frame + kernel_idx - padding;
      if (input_frame < 0 || input_frame >= frames) {
        continue;
      }
      const float input_value = input[static_cast<size_t>(input_frame) * in_channels + in_channel];
      const size_t weight_idx =
          (static_cast<size_t>(out_channel) * in_channels + in_channel) * kernel_size +
          kernel_idx;
      acc += input_value * __bfloat162float(weight[weight_idx]);
    }
  }
  out[idx] = acc;
}

__global__ void higgs_audio_snake_conv1d_generic_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ snake_alpha,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float* __restrict__ out,
    int frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int padding) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = frames * out_channels;
  if (idx >= total) {
    return;
  }
  const int frame = idx / out_channels;
  const int out_channel = idx - frame * out_channels;
  float acc = __bfloat162float(bias[out_channel]);
  for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
    for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
      const int input_frame = frame + kernel_idx - padding;
      if (input_frame < 0 || input_frame >= frames) {
        continue;
      }
      const float x = input[static_cast<size_t>(input_frame) * in_channels + in_channel];
      const float alpha = __bfloat162float(snake_alpha[in_channel]);
      const float s = __sinf(alpha * x);
      const float snake = x + (s * s) / (alpha + 1.0e-9f);
      const size_t weight_idx =
          (static_cast<size_t>(out_channel) * in_channels + in_channel) * kernel_size +
          kernel_idx;
      acc += snake * __bfloat162float(weight[weight_idx]);
    }
  }
  out[idx] = acc;
}

}  // namespace

extern "C" cudaError_t higgs_audio_rvq_decode_cuda(
    const uint32_t* codes,
    const __nv_bfloat16* codebooks,
    float* out,
    int frames,
    cudaStream_t stream) {
  if (codes == nullptr || codebooks == nullptr || out == nullptr || frames <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * kDim;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_rvq_decode_kernel<<<grid, block, 0, stream>>>(
      codes, codebooks, out, frames);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_quantizer_decode_cuda(
    const uint32_t* codes,
    const __nv_bfloat16* codebooks,
    const __nv_bfloat16* project_weight,
    const __nv_bfloat16* project_bias,
    float* out,
    int frames,
    cudaStream_t stream) {
  if (codes == nullptr || codebooks == nullptr || project_weight == nullptr ||
      project_bias == nullptr || out == nullptr || frames <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * kHidden;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_quantizer_decode_kernel<<<grid, block, 0, stream>>>(
      codes, codebooks, project_weight, project_bias, out, frames);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_fc2_cuda(
    const float* hidden,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float* out,
    int frames,
    cudaStream_t stream) {
  if (hidden == nullptr || weight == nullptr || bias == nullptr || out == nullptr ||
      frames <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * kAcousticHidden;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_fc2_kernel<<<grid, block, 0, stream>>>(
      hidden, weight, bias, out, frames);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_acoustic_decoder_conv1_cuda(
    const float* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float* out,
    int frames,
    cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr || out == nullptr ||
      frames <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * kDecoderHidden;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_acoustic_decoder_conv1_kernel<<<grid, block, 0, stream>>>(
      input, weight, bias, out, frames);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_decoder_block0_convt_cuda(
    const float* input,
    const __nv_bfloat16* snake_alpha,
    const __nv_bfloat16* weight,
    float* out,
    int frames,
    cudaStream_t stream) {
  if (input == nullptr || snake_alpha == nullptr || weight == nullptr ||
      out == nullptr || frames <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * kDecoderBlock0Stride * kDecoderBlock0Out;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_decoder_block0_convt_kernel<<<grid, block, 0, stream>>>(
      input, snake_alpha, weight, out, frames);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_decoder_residual_conv1_cuda(
    const float* input,
    const __nv_bfloat16* snake_alpha,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float* out,
    int frames,
    int dilation,
    cudaStream_t stream) {
  if (input == nullptr || snake_alpha == nullptr || weight == nullptr ||
      bias == nullptr || out == nullptr || frames <= 0 || dilation <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * kDecoderBlock0Out;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_decoder_residual_conv1_kernel<<<grid, block, 0, stream>>>(
      input, snake_alpha, weight, bias, out, frames, dilation);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_decoder_residual_conv2_add_cuda(
    const float* residual,
    const float* conv1,
    const __nv_bfloat16* snake_alpha,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float* out,
    int frames,
    cudaStream_t stream) {
  if (residual == nullptr || conv1 == nullptr || snake_alpha == nullptr ||
      weight == nullptr || bias == nullptr || out == nullptr || frames <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * kDecoderBlock0Out;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_decoder_residual_conv2_add_kernel<<<grid, block, 0, stream>>>(
      residual, conv1, snake_alpha, weight, bias, out, frames);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_decoder_convt_generic_cuda(
    const float* input,
    const __nv_bfloat16* snake_alpha,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float* out,
    int frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream) {
  if (input == nullptr || snake_alpha == nullptr || weight == nullptr ||
      bias == nullptr || out == nullptr || frames <= 0 || in_channels <= 0 ||
      out_channels <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0 ||
      output_padding < 0) {
    return cudaErrorInvalidValue;
  }
  const int out_frames = (frames - 1) * stride - 2 * padding + kernel_size + output_padding;
  if (out_frames <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = out_frames * out_channels;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_decoder_convt_generic_kernel<<<grid, block, 0, stream>>>(
      input, snake_alpha, weight, bias, out, frames, in_channels, out_channels,
      kernel_size, stride, padding, output_padding);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_decoder_residual_conv1_generic_cuda(
    const float* input,
    const __nv_bfloat16* snake_alpha,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float* out,
    int frames,
    int channels,
    int dilation,
    cudaStream_t stream) {
  if (input == nullptr || snake_alpha == nullptr || weight == nullptr ||
      bias == nullptr || out == nullptr || frames <= 0 || channels <= 0 ||
      dilation <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * channels;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_decoder_residual_conv1_generic_kernel<<<grid, block, 0, stream>>>(
      input, snake_alpha, weight, bias, out, frames, channels, dilation);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_decoder_residual_conv2_add_generic_cuda(
    const float* residual,
    const float* conv1,
    const __nv_bfloat16* snake_alpha,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float* out,
    int frames,
    int channels,
    cudaStream_t stream) {
  if (residual == nullptr || conv1 == nullptr || snake_alpha == nullptr ||
      weight == nullptr || bias == nullptr || out == nullptr || frames <= 0 ||
      channels <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * channels;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_decoder_residual_conv2_add_generic_kernel<<<grid, block, 0, stream>>>(
      residual, conv1, snake_alpha, weight, bias, out, frames, channels);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_conv1d_generic_cuda(
    const float* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float* out,
    int frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int padding,
    cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr || out == nullptr ||
      frames <= 0 || in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 ||
      padding < 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * out_channels;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_conv1d_generic_kernel<<<grid, block, 0, stream>>>(
      input, weight, bias, out, frames, in_channels, out_channels, kernel_size, padding);
  return cudaGetLastError();
}

extern "C" cudaError_t higgs_audio_snake_conv1d_generic_cuda(
    const float* input,
    const __nv_bfloat16* snake_alpha,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float* out,
    int frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int padding,
    cudaStream_t stream) {
  if (input == nullptr || snake_alpha == nullptr || weight == nullptr ||
      bias == nullptr || out == nullptr || frames <= 0 || in_channels <= 0 ||
      out_channels <= 0 || kernel_size <= 0 || padding < 0) {
    return cudaErrorInvalidValue;
  }
  const int total = frames * out_channels;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  higgs_audio_snake_conv1d_generic_kernel<<<grid, block, 0, stream>>>(
      input, snake_alpha, weight, bias, out, frames, in_channels, out_channels,
      kernel_size, padding);
  return cudaGetLastError();
}

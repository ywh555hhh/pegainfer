# Higgs-Audio Native Codec/Vocoder

TL;DR: The Higgs-Audio native codec/vocoder bring-up now produces a real WAV on
the RTX 4090 through model-local Rust + CUDA: native retained audio-code
generation emits raw codec rows, CUDA RVQ/quantizer/fc2/acoustic decoder stages
run through all five DAC upsampling blocks and final Conv, and the smoke writes
a 24 kHz mono PCM artifact without a Python sidecar. Strict parity,
performance, CUDA graph capture, and production serving are still not claimed.

Last touched: 2026-08

## Boundary

The production runtime boundary remains pure Rust + CUDA. Python is allowed for
reference/golden creation only; it is not part of the native E2E claim.

The SGLang-Omni reference wraps Transformers'
`HiggsAudioV2TokenizerModel`. The PegaInfer port should own the equivalent
model-local Rust/CUDA implementation under `pegainfer-higgs-audio`, not a shared
kernel or core abstraction.

Current PegaInfer claim:

```text
native_codec_vocoder_waveform_bringup
```

This means:

- native retained-KV code generation runs;
- raw codec rows are produced;
- codec config and checkpoint tensor contract are validated natively;
- all 8 RVQ codebooks are loaded from the Higgs checkpoint;
- CUDA RVQ embedding decode emits `[frames, 64]` as a diagnostic probe;
- CUDA quantizer decode applies each quantizer `project_out(64 -> 1024)` and
  sums the projected outputs, matching Transformers' `quantizer.decode`;
- CUDA `fc2(1024 -> 256)` emits the acoustic decoder input `[frames, 256]`;
- CUDA acoustic decoder `conv1(256 -> 1024, kernel=7, padding=3)` is implemented;
- CUDA decoder block0 `Snake + ConvTranspose1d(1024 -> 512, stride=8, kernel=16, padding=4)` is implemented;
- CUDA decoder block0 residual stack `res_unit1/2/3` is implemented with dilations `1/3/9`;
- CUDA decoder blocks 1 through 4 run the same model-local
  ConvTranspose/residual stack pattern;
- CUDA final `Snake + Conv1d(32 -> 1, kernel=7, padding=3)` emits f32 PCM;
- the smoke writes a PCM16 WAV artifact natively;
- implemented diagnostic stages through block0 are checked against a Rust CPU
  reference;
- strict trace parity, audio quality parity, optimized memory traffic, CUDA
  graph capture, and production serving are not claimed yet.

## Reference Facts

Source of truth: `sglang_omni/models/higgs_tts/audio_codec.py` and
`vocoder_scheduler.py` in `sgl-project/sglang-omni`.

Important facts captured in `native_codec.rs`:

- codec weights are bundled in the Higgs TTS checkpoint under
  `tied.embedding.modality_embeddings.0.model.`;
- config is Higgs Audio V2 tokenizer;
- sample rate is `24000`;
- hop length is `960`;
- frame rate is `25`;
- runtime audio-code quantizers are `8`;
- codec audio vocab is `1024`, while the AR codebook vocab is `1026` including
  BOC/EOC sentinels;
- checkpoint tensors observed on the 4090 host are BF16.

Pinned tensor contract examples:

```text
quantizer.quantizers.0.codebook.embed: [1024, 64]
quantizer.quantizers.0.project_out.weight: [1024, 64]
quantizer.quantizers.0.project_out.bias: [1024]
quantizer.quantizers.7.codebook.embed: [1024, 64]
fc2.weight: [256, 1024]
fc2.bias: [256]
acoustic_decoder.conv1.weight: [1024, 256, 7]
acoustic_decoder.conv1.bias: [1024]
acoustic_decoder.block.0.snake1.alpha: [1, 1024, 1]
acoustic_decoder.block.0.conv_t1.weight: [1024, 512, 16]
acoustic_decoder.block.0.conv_t1.bias: [512]
acoustic_decoder.block.0.res_unit1.conv1.weight: [512, 512, 7]
acoustic_decoder.block.0.res_unit1.conv2.weight: [512, 512, 1]
acoustic_decoder.block.4.conv_t1.weight: [64, 32, 6]
acoustic_decoder.snake1.alpha: [1, 32, 1]
acoustic_decoder.conv2.weight: [1, 32, 7]
acoustic_decoder.conv2.bias: [1]
```

Important correction: `[frames, 64]` is only the codebook embedding-space
diagnostic. The actual Transformers decode path is:

```text
raw codes [T, 8]
  -> per-quantizer embedding [T, 64]
  -> per-quantizer project_out [T, 1024]
  -> sum quantizers [T, 1024]
  -> fc2 [T, 256]
  -> acoustic_decoder Conv1d / ConvTranspose / residual stack
  -> PCM waveform
```

## 4090 Native WAV Evidence

Host:

```text
RTX 4090, sm_89
driver 595.71.05
CUDA 13.0
Rust nightly 2026-08-13
```

Command:

```bash
cargo run --release -p pegainfer-higgs-audio \
  --features runtime-qwen3 \
  --bin higgs_native_e2e_smoke -- \
  --model-dir /root/autodl-tmp/models/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561 \
  --golden /root/autodl-tmp/results/higgs-one-step-sglang-omni-golden-autodl.safetensors \
  --steps 48 \
  --trace-out /data/results/pegainfer/higgs-audio/native-e2e/higgs-native-e2e-trace-native-codec-vocoder-20260814.json \
  --report-out /data/results/pegainfer/higgs-audio/native-e2e/higgs-native-e2e-report-native-codec-vocoder-20260814.txt \
  --wav-out /data/results/pegainfer/higgs-audio/native-e2e/higgs-native-e2e-audio-20260814.wav
```

Result:

```text
status=ok
claim=native_codec_vocoder_waveform_bringup
runtime_boundary=Rust+CUDA native code generation plus native codec/vocoder compute; no Python sidecar
model_dir=/root/autodl-tmp/models/higgs-tts-3-4b-7556c17e05201fccd9c8cc120bc216dcc7b5d561
golden=/root/autodl-tmp/results/higgs-one-step-sglang-omni-golden-autodl.safetensors
prompt_tokens=10
requested_steps=48
completed_steps=48
trace_steps=49
raw_codec_rows=42
rvq_embedding_backend=cuda
rvq_embedding_frames=42
rvq_embedding_dim=64
rvq_embedding_values=2688
rvq_embedding_cpu_cuda_max_abs_diff=0.00000000e0
rvq_embedding_cpu_cuda_mean_abs_diff=0.00000000e0
quantizer_backend=cuda
quantizer_frames=42
quantizer_dim=1024
quantizer_values=43008
quantizer_cpu_cuda_max_abs_diff=0.00000000e0
quantizer_cpu_cuda_mean_abs_diff=0.00000000e0
acoustic_input_backend=cuda
acoustic_input_frames=42
acoustic_input_dim=256
acoustic_input_values=10752
acoustic_input_cpu_cuda_max_abs_diff=1.52587891e-5
acoustic_input_cpu_cuda_mean_abs_diff=1.22067320e-6
decoder_conv1_backend=cuda
decoder_conv1_frames=42
decoder_conv1_dim=1024
decoder_conv1_values=43008
decoder_conv1_cpu_cuda_max_abs_diff=3.43322754e-5
decoder_conv1_cpu_cuda_mean_abs_diff=1.53073290e-6
decoder_block0_convt_backend=cuda
decoder_block0_convt_frames=336
decoder_block0_convt_dim=512
decoder_block0_convt_values=172032
decoder_block0_convt_cpu_cuda_max_abs_diff=1.90734863e-5
decoder_block0_convt_cpu_cuda_mean_abs_diff=1.08438314e-6
decoder_block0_res1_backend=cuda
decoder_block0_res1_frames=336
decoder_block0_res1_dim=512
decoder_block0_res1_values=172032
decoder_block0_res1_cpu_cuda_max_abs_diff=7.05718994e-5
decoder_block0_res1_cpu_cuda_mean_abs_diff=3.67000416e-6
decoder_block0_res_stack_backend=cuda
decoder_block0_res_stack_frames=336
decoder_block0_res_stack_dim=512
decoder_block0_res_stack_values=172032
decoder_block0_res_stack_cpu_cuda_max_abs_diff=1.58691406e-3
decoder_block0_res_stack_cpu_cuda_mean_abs_diff=1.48696818e-5
native_audio_samples=40320
native_wav_out=/data/results/pegainfer/higgs-audio/native-e2e/higgs-native-e2e-audio-20260814.wav
codec_sample_rate=24000
codec_frame_rate=25
codec_hop_length=960
trace_out=/data/results/pegainfer/higgs-audio/native-e2e/higgs-native-e2e-trace-native-codec-vocoder-20260814.json
strict_parity=not_claimed
native_codec_decoder_error=none samples=40320 sample_rate=24000
```

Artifact metadata:

```text
wav_channels=1
sample_rate=24000
frames=40320
sample_width=2
duration=1.680000
report_sha256=fbbdf593d15560afb2fd4ec1d87578a317c650f3acefd69324e2a1299d755059
trace_sha256=9a495dbd88ce1dad0b81ddfa21d615a30aeb92e11b15490c244cc97bf4465df4
wav_sha256=a3097331b248ea07894d5ade2edaa7d273d501c8e636646ca8b46205606f9a25
```

Local artifact copy:

```text
/Users/yiweihan/Documents/muxi/higgs-audio-migration/4090-results-native-codec-vocoder-20260814/
```

## Implementation Ladder

1. Contract skeleton: config, weight prefix, key tensor shapes, raw-code
   validation, fail-closed native decoder. Done.
2. Weight loader: load all decoder-side tensors into model-local native
   structures, still no compute claim.
3. RVQ embedding probe: implement quantizer embedding lookup and additive
   codebook sum. Done, but not the acoustic decoder input.
4. Quantizer decode: apply every quantizer `project_out` and sum the `[T, 1024]`
   projected outputs. Done.
5. FC2 acoustic input: apply `fc2` to produce `[T, 256]` for the DAC decoder.
   Done.
6. Decoder block0: acoustic decoder `conv1`, block0 `Snake + ConvTranspose1d`,
   and block0 residual units are implemented and 4090-validated. Done.
7. Remaining decoder blocks: repeat the ConvTranspose + residual pattern for
   block1 through block4. Done for bring-up.
8. Full waveform path: acoustic decoder emits f32 PCM, `write_pcm16_wav` writes
   the artifact. Done for bring-up.
9. Performance path: capture/decode fixed frame-count graphs after correctness
   is meaningful.

Do not upgrade the claim beyond the matching ladder rung. This is native WAV
bring-up evidence, not strict parity or production-ready serving evidence. A
nonempty wav from a Python sidecar is still not native E2E evidence.

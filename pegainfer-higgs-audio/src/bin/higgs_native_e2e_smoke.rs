use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use pegainfer_higgs_audio::codec_input::CodeRowsLayout;
use pegainfer_higgs_audio::codec_input::codec_input_from_rows;
use pegainfer_higgs_audio::decode_trace::write_decode_trace_json;
use pegainfer_higgs_audio::native_codec::NATIVE_CODEC_CLAIM;
use pegainfer_higgs_audio::native_codec::NativeCodecBackend;
use pegainfer_higgs_audio::native_codec::NativeHiggsCodecDecoder;
use pegainfer_higgs_audio::native_codec::write_pcm16_wav;
use pegainfer_higgs_audio::one_step_actual::load_fused_audio_head_bf16;
use pegainfer_higgs_audio::one_step_actual::load_prompt_from_golden;
use pegainfer_higgs_audio::runtime_bridge::AudioHeadBackend;
use pegainfer_higgs_audio::runtime_bridge::HiggsAudioRuntime;
use pegainfer_higgs_audio::runtime_bridge::HiggsPromptSession;
use pegainfer_higgs_audio::runtime_bridge::HiggsRuntimeSource;
use pegainfer_higgs_audio::runtime_source::Qwen3RuntimeSourcePath;
use pegainfer_higgs_audio::runtime_source::select_qwen3_runtime_source;

#[derive(Parser)]
#[command(about = "Run the pure Rust/CUDA Higgs native E2E skeleton gate")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    golden: PathBuf,
    #[arg(long, conflicts_with = "qwen3_config_dir")]
    qwen3_body_dir: Option<PathBuf>,
    #[arg(long, conflicts_with = "qwen3_body_dir")]
    qwen3_config_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    session_id: u64,
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
    #[arg(long, default_value_t = 48)]
    steps: usize,
    #[arg(long)]
    trace_out: PathBuf,
    #[arg(long)]
    report_out: PathBuf,
    #[arg(long)]
    wav_out: Option<PathBuf>,
    #[arg(long)]
    allow_unimplemented_codec: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.steps > 0, "--steps must be greater than zero");
    let source_path = select_qwen3_runtime_source(
        args.qwen3_body_dir.as_deref(),
        args.qwen3_config_dir.as_deref(),
        &args.trace_out,
    )?;
    let source = match &source_path {
        Qwen3RuntimeSourcePath::BodyView(qwen3_body_dir) => {
            HiggsRuntimeSource::Qwen3BodyView { qwen3_body_dir }
        }
        Qwen3RuntimeSourcePath::ConfigAlias(qwen3_config_dir) => {
            HiggsRuntimeSource::Qwen3ConfigAlias { qwen3_config_dir }
        }
        Qwen3RuntimeSourcePath::AutoConfigAlias(qwen3_config_dir) => {
            HiggsRuntimeSource::AutoConfigAlias { qwen3_config_dir }
        }
    };

    let prompt = load_prompt_from_golden(&args.golden)?;
    let prompt_ids = prompt.prompt_ids()?;
    let fused_embedding = load_fused_audio_head_bf16(&args.model_dir)?;
    let decoder = NativeHiggsCodecDecoder::from_model_dir(
        &args.model_dir,
        NativeCodecBackend::Cuda {
            device_ordinal: args.device_ordinal,
        },
    )?;
    let mut runtime = HiggsAudioRuntime::from_model_dir(
        &args.model_dir,
        source,
        AudioHeadBackend::CudaBf16,
        args.device_ordinal,
    )?;

    let session_handle = HiggsPromptSession::new(args.session_id);
    let mut seed = runtime.start_audio_codegen_session_with_output_budget(
        session_handle,
        &prompt_ids,
        args.steps + 1,
    )?;
    let mut completed = 0usize;
    for _ in 0..args.steps {
        if seed.codegen.generation_done() {
            break;
        }
        runtime
            .continue_audio_codegen_step_from_feedback(
                seed.session,
                &mut seed.codegen,
                &fused_embedding,
            )?
            .context("Higgs codegen did not expose a feedback row")?;
        completed += 1;
    }

    write_decode_trace_json(&args.trace_out, seed.codegen.trace())?;
    let raw_codes = seed.codegen.raw_codes(true)?;
    let codec_rows = codec_input_from_rows(&raw_codes, CodeRowsLayout::Raw)?;
    let (rvq_out, rvq_check) = decoder.decode_rvq_cuda_with_cpu_check(&codec_rows)?;
    let (quantizer_out, quantizer_check) =
        decoder.decode_quantizer_cuda_with_cpu_check(&codec_rows)?;
    let (acoustic_input, acoustic_check) =
        decoder.decode_acoustic_input_cuda_with_cpu_check(&codec_rows)?;
    let (decoder_conv1, decoder_conv1_check) =
        decoder.decode_acoustic_decoder_conv1_cuda_with_cpu_check(&codec_rows)?;
    let (decoder_block0_convt, decoder_block0_convt_check) =
        decoder.decode_decoder_block0_convt_cuda_with_cpu_check(&codec_rows)?;
    let (decoder_block0_res1, decoder_block0_res1_check) =
        decoder.decode_decoder_block0_res1_cuda_with_cpu_check(&codec_rows)?;
    let (decoder_block0_res_stack, decoder_block0_res_stack_check) =
        decoder.decode_decoder_block0_res_stack_cuda_with_cpu_check(&codec_rows)?;
    let decode_result = decoder.decode_raw_codes(&codec_rows);
    runtime.drop_prompt_session(seed.session)?;

    let (status, decoder_error, audio_samples, wav_out) = match decode_result {
        Ok(audio) => {
            let wav_out = if let Some(path) = &args.wav_out {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create {}", parent.display()))?;
                }
                write_pcm16_wav(path, &audio)?;
                path.display().to_string()
            } else {
                "not_requested".to_string()
            };
            (
                "ok".to_string(),
                format!(
                    "none samples={} sample_rate={}",
                    audio.samples.len(),
                    audio.sample_rate
                ),
                audio.samples.len(),
                wav_out,
            )
        }
        Err(err) if args.allow_unimplemented_codec => (
            "decoder_block0_res_stack_ok_waveform_unimplemented".to_string(),
            err.to_string(),
            0,
            "not_written".to_string(),
        ),
        Err(err) => return Err(err),
    };

    let report = format!(
        "status={status}\nclaim={NATIVE_CODEC_CLAIM}\nruntime_boundary=Rust+CUDA native code generation plus native codec/vocoder compute; no Python sidecar\nmodel_dir={}\ngolden={}\nprompt_tokens={}\nrequested_steps={}\ncompleted_steps={completed}\ntrace_steps={}\nraw_codec_rows={}\nrvq_embedding_backend=cuda\nrvq_embedding_frames={}\nrvq_embedding_dim={}\nrvq_embedding_values={}\nrvq_embedding_cpu_cuda_max_abs_diff={:.8e}\nrvq_embedding_cpu_cuda_mean_abs_diff={:.8e}\nquantizer_backend=cuda\nquantizer_frames={}\nquantizer_dim={}\nquantizer_values={}\nquantizer_cpu_cuda_max_abs_diff={:.8e}\nquantizer_cpu_cuda_mean_abs_diff={:.8e}\nacoustic_input_backend=cuda\nacoustic_input_frames={}\nacoustic_input_dim={}\nacoustic_input_values={}\nacoustic_input_cpu_cuda_max_abs_diff={:.8e}\nacoustic_input_cpu_cuda_mean_abs_diff={:.8e}\ndecoder_conv1_backend=cuda\ndecoder_conv1_frames={}\ndecoder_conv1_dim={}\ndecoder_conv1_values={}\ndecoder_conv1_cpu_cuda_max_abs_diff={:.8e}\ndecoder_conv1_cpu_cuda_mean_abs_diff={:.8e}\ndecoder_block0_convt_backend=cuda\ndecoder_block0_convt_frames={}\ndecoder_block0_convt_dim={}\ndecoder_block0_convt_values={}\ndecoder_block0_convt_cpu_cuda_max_abs_diff={:.8e}\ndecoder_block0_convt_cpu_cuda_mean_abs_diff={:.8e}\ndecoder_block0_res1_backend=cuda\ndecoder_block0_res1_frames={}\ndecoder_block0_res1_dim={}\ndecoder_block0_res1_values={}\ndecoder_block0_res1_cpu_cuda_max_abs_diff={:.8e}\ndecoder_block0_res1_cpu_cuda_mean_abs_diff={:.8e}\ndecoder_block0_res_stack_backend=cuda\ndecoder_block0_res_stack_frames={}\ndecoder_block0_res_stack_dim={}\ndecoder_block0_res_stack_values={}\ndecoder_block0_res_stack_cpu_cuda_max_abs_diff={:.8e}\ndecoder_block0_res_stack_cpu_cuda_mean_abs_diff={:.8e}\nnative_audio_samples={audio_samples}\nnative_wav_out={wav_out}\ncodec_sample_rate={}\ncodec_frame_rate={}\ncodec_hop_length={}\ntrace_out={}\nstrict_parity=not_claimed\nnative_codec_decoder_error={decoder_error}\n",
        args.model_dir.display(),
        args.golden.display(),
        prompt_ids.len(),
        args.steps,
        seed.codegen.steps(),
        codec_rows.len(),
        rvq_out.frames,
        rvq_out.dim,
        rvq_out.values.len(),
        rvq_check.max_abs_diff,
        rvq_check.mean_abs_diff,
        quantizer_out.frames,
        quantizer_out.dim,
        quantizer_out.values.len(),
        quantizer_check.max_abs_diff,
        quantizer_check.mean_abs_diff,
        acoustic_input.frames,
        acoustic_input.dim,
        acoustic_input.values.len(),
        acoustic_check.max_abs_diff,
        acoustic_check.mean_abs_diff,
        decoder_conv1.frames,
        decoder_conv1.dim,
        decoder_conv1.values.len(),
        decoder_conv1_check.max_abs_diff,
        decoder_conv1_check.mean_abs_diff,
        decoder_block0_convt.frames,
        decoder_block0_convt.dim,
        decoder_block0_convt.values.len(),
        decoder_block0_convt_check.max_abs_diff,
        decoder_block0_convt_check.mean_abs_diff,
        decoder_block0_res1.frames,
        decoder_block0_res1.dim,
        decoder_block0_res1.values.len(),
        decoder_block0_res1_check.max_abs_diff,
        decoder_block0_res1_check.mean_abs_diff,
        decoder_block0_res_stack.frames,
        decoder_block0_res_stack.dim,
        decoder_block0_res_stack.values.len(),
        decoder_block0_res_stack_check.max_abs_diff,
        decoder_block0_res_stack_check.mean_abs_diff,
        decoder.config().sample_rate,
        decoder.config().frame_rate,
        decoder.config().hop_length,
        args.trace_out.display(),
    );
    if let Some(parent) = args.report_out.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(&args.report_out, report)
        .with_context(|| format!("write {}", args.report_out.display()))?;

    println!("higgs native e2e smoke: {status}");
    println!("  claim: {NATIVE_CODEC_CLAIM}");
    println!("  completed_steps: {completed}");
    println!("  raw_codec_rows: {}", codec_rows.len());
    println!(
        "  rvq embedding: frames={} dim={} max_abs_diff={:.8e}",
        rvq_check.frames, rvq_check.dim, rvq_check.max_abs_diff
    );
    println!(
        "  quantizer: frames={} dim={} max_abs_diff={:.8e}",
        quantizer_check.frames, quantizer_check.dim, quantizer_check.max_abs_diff
    );
    println!(
        "  acoustic input: frames={} dim={} max_abs_diff={:.8e}",
        acoustic_check.frames, acoustic_check.dim, acoustic_check.max_abs_diff
    );
    println!(
        "  decoder conv1: frames={} dim={} max_abs_diff={:.8e}",
        decoder_conv1_check.frames, decoder_conv1_check.dim, decoder_conv1_check.max_abs_diff
    );
    println!(
        "  decoder block0 conv_t1: frames={} dim={} max_abs_diff={:.8e}",
        decoder_block0_convt_check.frames,
        decoder_block0_convt_check.dim,
        decoder_block0_convt_check.max_abs_diff
    );
    println!(
        "  decoder block0 res_unit1: frames={} dim={} max_abs_diff={:.8e}",
        decoder_block0_res1_check.frames,
        decoder_block0_res1_check.dim,
        decoder_block0_res1_check.max_abs_diff
    );
    println!(
        "  decoder block0 residual stack: frames={} dim={} max_abs_diff={:.8e}",
        decoder_block0_res_stack_check.frames,
        decoder_block0_res_stack_check.dim,
        decoder_block0_res_stack_check.max_abs_diff
    );
    println!("  report_out: {}", args.report_out.display());
    println!("  wav_out: {wav_out}");
    Ok(())
}

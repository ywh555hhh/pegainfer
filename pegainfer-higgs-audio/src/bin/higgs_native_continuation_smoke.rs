use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use pegainfer_higgs_audio::codec_input::CodeRowsLayout;
use pegainfer_higgs_audio::codec_input::codec_input_from_rows;
use pegainfer_higgs_audio::codec_input::write_codec_input_json;
use pegainfer_higgs_audio::decode_trace::write_decode_trace_json;
use pegainfer_higgs_audio::one_step_actual::load_fused_audio_head_bf16;
use pegainfer_higgs_audio::one_step_actual::load_prompt_from_golden;
use pegainfer_higgs_audio::runtime_bridge::AudioHeadBackend as RuntimeAudioHeadBackend;
use pegainfer_higgs_audio::runtime_bridge::HiggsAudioRuntime;
use pegainfer_higgs_audio::runtime_bridge::HiggsPromptSession;
use pegainfer_higgs_audio::runtime_bridge::HiggsRuntimeSource;
use pegainfer_higgs_audio::runtime_source::Qwen3RuntimeSourcePath;
use pegainfer_higgs_audio::runtime_source::select_qwen3_runtime_source;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum AudioHeadBackend {
    CudaBf16,
    CpuFp32,
}

#[derive(Parser)]
#[command(about = "Run Higgs native retained continuation steps and write trace artifacts")]
struct Args {
    /// Original Higgs checkpoint directory containing the fused audio head and codebook embedding.
    #[arg(long)]
    model_dir: PathBuf,
    /// Optional fallback Qwen3-compatible body view produced by higgs_materialize_qwen3_body.
    #[arg(long, conflicts_with = "qwen3_config_dir")]
    qwen3_body_dir: Option<PathBuf>,
    /// Optional Qwen3 config-only view; by default a small view is written next to --trace-out.
    #[arg(long, conflicts_with = "qwen3_body_dir")]
    qwen3_config_dir: Option<PathBuf>,
    /// Golden safetensors fixture; prompt tensors are copied from this file.
    #[arg(long)]
    golden: PathBuf,
    /// Higgs-owned session id used for the retained prompt KV.
    #[arg(long, default_value_t = 1)]
    session_id: u64,
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
    /// Number of retained audio-code continuation steps to execute.
    #[arg(long, default_value_t = 8)]
    steps: usize,
    /// Audio head execution backend used for prompt seed and continuation logits.
    #[arg(long, value_enum, default_value_t = AudioHeadBackend::CudaBf16)]
    audio_head_backend: AudioHeadBackend,
    /// Output native code-generation trace JSON path.
    #[arg(long)]
    trace_out: PathBuf,
    /// Output raw codec rows JSON path.
    #[arg(long)]
    codec_input_out: PathBuf,
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
    let session_handle = HiggsPromptSession::new(args.session_id);
    let mut runtime = HiggsAudioRuntime::from_model_dir(
        &args.model_dir,
        source,
        args.audio_head_backend.into(),
        args.device_ordinal,
    )?;

    // Qwen3's KV ledger counts the final prefill token as the first generated
    // token. Reserve that bookkeeping slot plus the requested continuation
    // steps so the diagnostic embedding-fed decode path can advance.
    let max_output_tokens = args.steps + 1;
    let mut seed = runtime.start_audio_codegen_session_with_output_budget(
        session_handle,
        &prompt_ids,
        max_output_tokens,
    )?;
    let mut continuations = Vec::with_capacity(args.steps);
    for _ in 0..args.steps {
        if seed.codegen.generation_done() {
            anyhow::bail!(
                "Higgs codegen finished before requested {} continuation steps; completed {}",
                args.steps,
                continuations.len()
            );
        }
        let continuation = runtime
            .continue_audio_codegen_step_from_feedback(
                seed.session,
                &mut seed.codegen,
                &fused_embedding,
            )?
            .ok_or_else(|| anyhow::anyhow!("Higgs codegen did not expose a feedback row"))?;
        continuations.push(continuation);
    }

    write_decode_trace_json(&args.trace_out, seed.codegen.trace())?;
    let raw_codes = seed.codegen.raw_codes(true)?;
    let codec_rows = codec_input_from_rows(&raw_codes, CodeRowsLayout::Raw)?;
    write_codec_input_json(&args.codec_input_out, &codec_rows)?;
    runtime.drop_prompt_session(seed.session)?;

    println!("higgs native continuation smoke: ok");
    println!("  model_dir: {}", args.model_dir.display());
    println!("  golden: {}", args.golden.display());
    println!("  prompt_tokens: {}", prompt_ids.len());
    println!("  session_id: {}", seed.session.id());
    println!("  continuation_steps: {}", continuations.len());
    println!(
        "  last_continuation_step: {}",
        continuations.last().expect("steps > 0").step
    );
    println!("  trace_steps: {}", seed.codegen.steps());
    println!("  raw_codec_rows: {}", codec_rows.len());
    println!("  trace_out: {}", args.trace_out.display());
    println!("  codec_input_out: {}", args.codec_input_out.display());
    Ok(())
}

impl From<AudioHeadBackend> for RuntimeAudioHeadBackend {
    fn from(value: AudioHeadBackend) -> Self {
        match value {
            AudioHeadBackend::CudaBf16 => Self::CudaBf16,
            AudioHeadBackend::CpuFp32 => Self::CpuFp32,
        }
    }
}

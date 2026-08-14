use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use pegainfer_higgs_audio::audio_codegen::AudioCodeGenerationSession;
use pegainfer_higgs_audio::audio_codegen::AudioCodegenSessionId;
use pegainfer_higgs_audio::codec_input::CodeRowsLayout;
use pegainfer_higgs_audio::codec_input::codec_input_from_rows;
use pegainfer_higgs_audio::codec_input::write_codec_input_json;
use pegainfer_higgs_audio::decode_trace::AudioLogitsSummary;
use pegainfer_higgs_audio::decode_trace::TRACE_SCHEMA;
use pegainfer_higgs_audio::decode_trace::write_decode_trace_json;
use pegainfer_higgs_audio::one_step_actual::load_fused_audio_head_bf16;
use pegainfer_higgs_audio::one_step_actual::load_prompt_from_golden;
use pegainfer_higgs_audio::one_step_golden::NUM_CODEBOOKS;
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
#[command(
    about = "Run Higgs native retained continuation while forcing sampled rows from a reference trace"
)]
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
    /// Reference trace whose sampled rows define the forced common prefix.
    #[arg(long)]
    reference_trace: PathBuf,
    /// Higgs-owned session id used for the retained prompt KV.
    #[arg(long, default_value_t = 1)]
    session_id: u64,
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
    /// Number of retained audio-code continuation steps to execute after the prompt seed.
    #[arg(long, default_value_t = 8)]
    steps: usize,
    /// Audio head execution backend used for prompt seed logits.
    #[arg(long, value_enum, default_value_t = AudioHeadBackend::CudaBf16)]
    audio_head_backend: AudioHeadBackend,
    /// Output forced-prefix trace JSON path.
    #[arg(long)]
    trace_out: PathBuf,
    /// Output raw codec rows JSON path.
    #[arg(long)]
    codec_input_out: PathBuf,
}

struct ReferenceTraceRows {
    prompt_tokens: usize,
    sampled_rows: Vec<Vec<u32>>,
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

    let reference = load_reference_trace_rows(&args.reference_trace, args.steps + 1)?;
    let prompt = load_prompt_from_golden(&args.golden)?;
    let prompt_ids = prompt.prompt_ids()?;
    anyhow::ensure!(
        reference.prompt_tokens == prompt_ids.len(),
        "reference prompt_tokens {} must match golden prompt length {}",
        reference.prompt_tokens,
        prompt_ids.len()
    );

    let fused_embedding = load_fused_audio_head_bf16(&args.model_dir)?;
    let session_handle = HiggsPromptSession::new(args.session_id);
    let mut runtime = HiggsAudioRuntime::from_model_dir(
        &args.model_dir,
        source,
        args.audio_head_backend.into(),
        args.device_ordinal,
    )?;

    let max_output_tokens = args.steps + 1;
    let prefill = runtime.prefill_prompt_session_with_output_budget(
        session_handle,
        &prompt_ids,
        max_output_tokens,
    )?;
    let mut codegen = AudioCodeGenerationSession::new_with_id(
        AudioCodegenSessionId::new(prefill.session.id()),
        prefill.prompt_tokens,
    );
    codegen.push_sampled_codes(
        reference.sampled_rows[0].clone(),
        AudioLogitsSummary::from_prediction(&prefill.audio),
    )?;

    let mut continuations = Vec::with_capacity(args.steps);
    for step in 1..=args.steps {
        if codegen.generation_done() {
            anyhow::bail!(
                "Higgs forced-prefix codegen finished before requested {} continuation steps; completed {}",
                args.steps,
                continuations.len()
            );
        }
        let continuation = runtime
            .continue_audio_codegen_step_from_feedback_with_sampled_codes(
                prefill.session,
                &mut codegen,
                &fused_embedding,
                reference.sampled_rows[step].clone(),
            )?
            .ok_or_else(|| {
                anyhow::anyhow!("Higgs forced-prefix codegen did not expose a feedback row")
            })?;
        continuations.push(continuation);
    }

    write_decode_trace_json(&args.trace_out, codegen.trace())?;
    let raw_codes = codegen.raw_codes(true)?;
    let codec_rows = codec_input_from_rows(&raw_codes, CodeRowsLayout::Raw)?;
    write_codec_input_json(&args.codec_input_out, &codec_rows)?;
    runtime.drop_prompt_session(prefill.session)?;

    println!("higgs native forced-prefix smoke: ok");
    println!("  model_dir: {}", args.model_dir.display());
    println!("  golden: {}", args.golden.display());
    println!("  reference_trace: {}", args.reference_trace.display());
    println!("  prompt_tokens: {}", prompt_ids.len());
    println!("  session_id: {}", prefill.session.id());
    println!("  continuation_steps: {}", continuations.len());
    println!("  trace_steps: {}", codegen.steps());
    println!("  raw_codec_rows: {}", codec_rows.len());
    println!("  trace_out: {}", args.trace_out.display());
    println!("  codec_input_out: {}", args.codec_input_out.display());
    Ok(())
}

fn load_reference_trace_rows(path: &Path, required_steps: usize) -> Result<ReferenceTraceRows> {
    let value: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))?;
    let schema = value
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .context("reference trace missing schema")?;
    anyhow::ensure!(
        schema == TRACE_SCHEMA,
        "reference trace schema {schema:?} must be {TRACE_SCHEMA:?}"
    );
    let prompt_tokens = value
        .get("prompt_tokens")
        .and_then(serde_json::Value::as_u64)
        .context("reference trace missing prompt_tokens")? as usize;
    let steps = value
        .get("steps")
        .and_then(serde_json::Value::as_array)
        .context("reference trace missing steps")?;
    anyhow::ensure!(
        steps.len() >= required_steps,
        "reference trace has {} steps, need at least {required_steps}",
        steps.len()
    );

    let mut sampled_rows = Vec::with_capacity(required_steps);
    for (step_idx, step) in steps.iter().take(required_steps).enumerate() {
        let sampled = step
            .get("sampled_codes")
            .and_then(serde_json::Value::as_array)
            .with_context(|| format!("reference trace step {step_idx} missing sampled_codes"))?;
        anyhow::ensure!(
            sampled.len() == NUM_CODEBOOKS,
            "reference trace step {step_idx} sampled width {} must be {NUM_CODEBOOKS}",
            sampled.len()
        );
        let row = sampled
            .iter()
            .enumerate()
            .map(|(codebook, value)| {
                let id = value.as_u64().with_context(|| {
                    format!("reference trace step {step_idx} codebook {codebook} is not u32")
                })?;
                u32::try_from(id).with_context(|| {
                    format!(
                        "reference trace step {step_idx} codebook {codebook} id {id} is too large"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        sampled_rows.push(row);
    }
    Ok(ReferenceTraceRows {
        prompt_tokens,
        sampled_rows,
    })
}

impl From<AudioHeadBackend> for RuntimeAudioHeadBackend {
    fn from(value: AudioHeadBackend) -> Self {
        match value {
            AudioHeadBackend::CudaBf16 => Self::CudaBf16,
            AudioHeadBackend::CpuFp32 => Self::CpuFp32,
        }
    }
}

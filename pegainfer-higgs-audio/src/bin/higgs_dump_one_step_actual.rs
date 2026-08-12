use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use pegainfer_higgs_audio::runtime_bridge::{
    AudioHeadBackend as RuntimeAudioHeadBackend, HiggsOneStepRuntime, HiggsRuntimeSource,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum AudioHeadBackend {
    /// Match the Python golden generator's CUDA BF16 F.linear contract.
    CudaBf16,
    /// Legacy diagnostic fallback: compute the audio head as a CPU FP32 dot.
    CpuFp32,
}

#[derive(Parser)]
#[command(about = "Dump a Higgs Audio one-step actual safetensors file from PegaInfer runtime")]
struct Args {
    /// Original Higgs checkpoint directory containing the fused audio head.
    #[arg(long)]
    model_dir: PathBuf,
    /// Qwen3-compatible body view produced by higgs_materialize_qwen3_body.
    #[arg(
        long,
        conflicts_with = "qwen3_config_dir",
        required_unless_present = "qwen3_config_dir"
    )]
    qwen3_body_dir: Option<PathBuf>,
    /// Qwen3 config-only view; weights are read directly from --model-dir through aliases.
    #[arg(
        long,
        conflicts_with = "qwen3_body_dir",
        required_unless_present = "qwen3_body_dir"
    )]
    qwen3_config_dir: Option<PathBuf>,
    /// Golden safetensors fixture; prompt tensors are copied from this file.
    #[arg(long)]
    golden: PathBuf,
    /// Output actual safetensors path.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
    /// Audio head execution backend used for the actual logits dump.
    #[arg(long, value_enum, default_value_t = AudioHeadBackend::CudaBf16)]
    audio_head_backend: AudioHeadBackend,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let source = match (&args.qwen3_body_dir, &args.qwen3_config_dir) {
        (Some(qwen3_body_dir), None) => HiggsRuntimeSource::Qwen3BodyView { qwen3_body_dir },
        (None, Some(qwen3_config_dir)) => HiggsRuntimeSource::Qwen3ConfigAlias { qwen3_config_dir },
        _ => unreachable!("clap enforces exactly one Qwen3 runtime source"),
    };
    let mut runtime = HiggsOneStepRuntime::from_model_dir(
        &args.model_dir,
        source,
        args.audio_head_backend.into(),
        args.device_ordinal,
    )?;
    let summary = runtime.dump_one_step_actual(&args.golden, &args.out)?;

    println!("higgs one-step actual dump: ok");
    println!("  out: {}", summary.output_path.display());
    println!("  audio_head_backend: {:?}", args.audio_head_backend);
    println!("  prompt_tokens: {}", summary.prompt_tokens);
    println!("  hidden_values: {}", summary.hidden_values);
    println!("  audio_logits: {}", summary.audio_logits);
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

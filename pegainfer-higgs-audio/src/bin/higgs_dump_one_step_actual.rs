use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use pegainfer_higgs_audio::one_step_actual::{
    load_fused_audio_head_bf16, load_prompt_from_golden, write_one_step_actual,
    write_one_step_actual_with_gpu_audio_head,
};
use pegainfer_qwen3_4b::runtime::Qwen3Executor;

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
    #[arg(long)]
    qwen3_body_dir: PathBuf,
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
    let prompt = load_prompt_from_golden(&args.golden)?;
    let prompt_ids = prompt.prompt_ids()?;
    let qwen3_body_dir = args
        .qwen3_body_dir
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("qwen3 body dir must be valid UTF-8"))?;
    let mut executor = Qwen3Executor::from_runtime(qwen3_body_dir, false, &[args.device_ordinal])?;
    let hidden = executor
        .prefill_last_hidden_bf16(prompt_ids.clone())?
        .hidden_bf16;
    let audio_head = load_fused_audio_head_bf16(&args.model_dir)?;
    let summary = match args.audio_head_backend {
        AudioHeadBackend::CudaBf16 => write_one_step_actual_with_gpu_audio_head(
            &args.out,
            &prompt,
            &hidden,
            &audio_head,
            args.device_ordinal,
        )?,
        AudioHeadBackend::CpuFp32 => {
            write_one_step_actual(&args.out, &prompt, &hidden, &audio_head)?
        }
    };

    println!("higgs one-step actual dump: ok");
    println!("  out: {}", summary.output_path.display());
    println!("  audio_head_backend: {:?}", args.audio_head_backend);
    println!("  prompt_tokens: {}", summary.prompt_tokens);
    println!("  hidden_values: {}", summary.hidden_values);
    println!("  audio_logits: {}", summary.audio_logits);
    Ok(())
}

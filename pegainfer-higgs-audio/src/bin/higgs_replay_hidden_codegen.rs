use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use half::bf16;
use pegainfer_higgs_audio::audio_codegen::AudioCodeGenerationSession;
use pegainfer_higgs_audio::audio_codegen::AudioCodegenSessionId;
use pegainfer_higgs_audio::codec_input::CodeRowsLayout;
use pegainfer_higgs_audio::codec_input::codec_input_from_rows;
use pegainfer_higgs_audio::codec_input::write_codec_input_json;
use pegainfer_higgs_audio::decode_trace::write_decode_trace_json;
use pegainfer_higgs_audio::one_step_actual::load_fused_audio_head_bf16;
use pegainfer_higgs_audio::one_step_golden::HIDDEN_SIZE;
use pegainfer_higgs_audio::one_step_golden::NUM_CODEBOOKS;
use safetensors::Dtype;
use safetensors::SafeTensors;

#[derive(Parser)]
#[command(
    about = "Replay backend-returned Higgs final hidden rows through the audio head into codegen artifacts"
)]
struct Args {
    /// Original Higgs checkpoint directory containing the fused audio head.
    #[arg(long)]
    model_dir: PathBuf,
    /// Safetensors file with tensor final_hidden.bf16 shaped [steps, 2560].
    #[arg(long)]
    hidden_safetensors: PathBuf,
    /// Initial one-step sampled code row from prompt prefill, comma-separated 8 ids.
    #[arg(long)]
    seed_sampled_codes: String,
    /// Prompt token count recorded in the trace artifact.
    #[arg(long)]
    prompt_tokens: usize,
    /// Higgs-owned session id recorded in the code-generation session.
    #[arg(long, default_value_t = 1)]
    session_id: u64,
    /// Output native code-generation trace JSON path.
    #[arg(long)]
    trace_out: PathBuf,
    /// Output raw codec rows JSON path.
    #[arg(long)]
    codec_input_out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let audio_head = load_fused_audio_head_bf16(&args.model_dir)?;
    let seed = parse_code_row(&args.seed_sampled_codes)?;
    let hidden_rows = load_final_hidden_rows(&args.hidden_safetensors)?;

    let mut session = AudioCodeGenerationSession::new_with_id(
        AudioCodegenSessionId::new(args.session_id),
        args.prompt_tokens,
    );
    let seed_argmax = seed.iter().map(|&code| i64::from(code)).collect();
    session.push_sampled_codes(
        seed,
        pegainfer_higgs_audio::decode_trace::AudioLogitsSummary {
            argmax: seed_argmax,
            top_ids: None,
            logits: None,
            top64_min_overlap_with_previous: None,
            logits_l2_norm: 0.0,
        },
    )?;
    for hidden in &hidden_rows {
        if session.generation_done() {
            break;
        }
        session.push_final_normed_hidden_cpu(hidden, &audio_head)?;
    }

    write_decode_trace_json(&args.trace_out, session.trace())?;
    let raw_codes = session.raw_codes(true)?;
    let codec_rows = codec_input_from_rows(&raw_codes, CodeRowsLayout::Raw)?;
    write_codec_input_json(&args.codec_input_out, &codec_rows)?;

    println!("higgs replay hidden codegen: ok");
    println!("  model_dir: {}", args.model_dir.display());
    println!(
        "  hidden_safetensors: {}",
        args.hidden_safetensors.display()
    );
    println!("  prompt_tokens: {}", args.prompt_tokens);
    println!("  session_id: {}", args.session_id);
    println!("  hidden_rows: {}", hidden_rows.len());
    println!("  trace_steps: {}", session.steps());
    println!("  raw_codec_rows: {}", codec_rows.len());
    println!("  trace_out: {}", args.trace_out.display());
    println!("  codec_input_out: {}", args.codec_input_out.display());
    Ok(())
}

fn parse_code_row(value: &str) -> Result<Vec<u32>> {
    let row = value
        .split(',')
        .map(|cell| {
            cell.trim()
                .parse::<u32>()
                .with_context(|| format!("parse code id {cell:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if row.len() != NUM_CODEBOOKS {
        bail!(
            "seed sampled code row has {} codebooks, expected {NUM_CODEBOOKS}",
            row.len()
        );
    }
    Ok(row)
}

fn load_final_hidden_rows(path: &Path) -> Result<Vec<Vec<bf16>>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st =
        SafeTensors::deserialize(&bytes).with_context(|| format!("parse {}", path.display()))?;
    let tensor = st
        .tensor("final_hidden.bf16")
        .context("hidden safetensors missing tensor final_hidden.bf16")?;
    if tensor.dtype() != Dtype::BF16 {
        bail!("final_hidden.bf16 must be BF16, got {:?}", tensor.dtype());
    }
    let shape = tensor.shape();
    if shape.len() != 2 || shape[1] != HIDDEN_SIZE {
        bail!(
            "final_hidden.bf16 shape mismatch: expected [steps, {HIDDEN_SIZE}], got {:?}",
            shape
        );
    }
    let rows = shape[0];
    if rows == 0 {
        bail!("final_hidden.bf16 must contain at least one row");
    }
    Ok(tensor
        .data()
        .chunks_exact(HIDDEN_SIZE * 2)
        .take(rows)
        .map(|row| {
            row.chunks_exact(2)
                .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
                .collect()
        })
        .collect())
}

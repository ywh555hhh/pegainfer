use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use pegainfer_higgs_audio::codec_input::CodeRowsLayout;
use pegainfer_higgs_audio::codec_input::code_rows_from_json_value;
use pegainfer_higgs_audio::codec_input::codec_input_from_rows;
use pegainfer_higgs_audio::codec_input::write_codec_input_json;
use pegainfer_higgs_audio::decode_trace::trace_sampled_code_rows;
use pegainfer_higgs_audio::decode_trace::write_decode_trace_json;

#[derive(Parser)]
#[command(
    about = "Prepare Higgs Audio native code-generation trace and codec artifacts from sampled code rows"
)]
struct Args {
    /// JSON array or object containing sampled codebook rows, shape [steps, 8].
    #[arg(long)]
    sampled_codes_json: PathBuf,
    /// Prompt token count recorded in the trace artifact.
    #[arg(long)]
    prompt_tokens: usize,
    /// Output native code-generation trace JSON path.
    #[arg(long)]
    trace_out: PathBuf,
    /// Output raw codec rows JSON path.
    #[arg(long)]
    codec_input_out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sampled_rows = load_sampled_code_rows_json(&args.sampled_codes_json)?;
    let trace = trace_sampled_code_rows(args.prompt_tokens, &sampled_rows)?;
    let raw_rows = trace
        .steps
        .last()
        .map(|step| step.raw_codes.as_slice())
        .context("trace has no steps; sampled rows must contain at least one row")?;
    if raw_rows.is_empty() {
        bail!(
            "trace has no raw codec rows yet; provide enough sampled steps to pass the Higgs delay-pattern warmup"
        );
    }
    write_decode_trace_json(&args.trace_out, &trace)?;
    let codec_rows = codec_input_from_rows(raw_rows, CodeRowsLayout::Raw)?;
    write_codec_input_json(&args.codec_input_out, &codec_rows)?;

    println!("higgs prepare codegen artifacts: ok");
    println!(
        "  sampled_codes_json: {}",
        args.sampled_codes_json.display()
    );
    println!("  prompt_tokens: {}", args.prompt_tokens);
    println!("  steps: {}", trace.steps.len());
    println!("  raw_codec_rows: {}", codec_rows.len());
    println!("  trace_out: {}", args.trace_out.display());
    println!("  codec_input_out: {}", args.codec_input_out.display());
    Ok(())
}

fn load_sampled_code_rows_json(path: &PathBuf) -> Result<Vec<Vec<u32>>> {
    let value: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))?;
    let sampled = match &value {
        serde_json::Value::Object(map) => map
            .get("sampled_codes")
            .context("sampled code JSON object missing key \"sampled_codes\"")?,
        _ => &value,
    };
    code_rows_from_json_value(sampled, CodeRowsLayout::Raw)
}

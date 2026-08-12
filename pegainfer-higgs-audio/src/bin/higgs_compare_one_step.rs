use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use pegainfer_higgs_audio::compare::{
    OneStepTolerances, compare_one_step_files, ensure_comparison_passed,
};

#[derive(Parser)]
#[command(about = "Compare a Higgs Audio one-step actual safetensors dump against the golden")]
struct Args {
    #[arg(long)]
    golden: PathBuf,
    #[arg(long)]
    actual: PathBuf,
    #[arg(long, default_value_t = OneStepTolerances::default().hidden_abs_tol)]
    hidden_abs_tol: f32,
    #[arg(long, default_value_t = OneStepTolerances::default().hidden_mean_abs_tol)]
    hidden_mean_abs_tol: f32,
    #[arg(long, default_value_t = OneStepTolerances::default().logits_abs_tol)]
    logits_abs_tol: f32,
    #[arg(long, default_value_t = OneStepTolerances::default().logits_mean_abs_tol)]
    logits_mean_abs_tol: f32,
    #[arg(long, default_value_t = OneStepTolerances::default().top_logprobs_abs_tol)]
    top_logprobs_abs_tol: f32,
    #[arg(long, default_value_t = OneStepTolerances::default().top_logprobs_mean_abs_tol)]
    top_logprobs_mean_abs_tol: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tolerances = OneStepTolerances {
        hidden_abs_tol: args.hidden_abs_tol,
        hidden_mean_abs_tol: args.hidden_mean_abs_tol,
        logits_abs_tol: args.logits_abs_tol,
        logits_mean_abs_tol: args.logits_mean_abs_tol,
        top_logprobs_abs_tol: args.top_logprobs_abs_tol,
        top_logprobs_mean_abs_tol: args.top_logprobs_mean_abs_tol,
    };
    let comparison = compare_one_step_files(&args.golden, &args.actual, tolerances)?;
    println!("higgs one-step comparison:");
    for tensor in &comparison.tensors {
        println!(
            "  {:32} pass={} elems={} exact_mismatch={} max_abs={:.6} mean_abs={:.6} rmse={:.6} p99_abs={:.6} abs_tol={:.6} mean_tol={:.6}",
            tensor.name,
            tensor.passed,
            tensor.elements,
            tensor.exact_mismatches,
            tensor.max_abs,
            tensor.mean_abs,
            tensor.rmse,
            tensor.p99_abs,
            tensor.abs_tol,
            tensor.mean_abs_tol
        );
    }
    ensure_comparison_passed(&comparison)?;
    println!("higgs one-step comparison: ok");
    Ok(())
}

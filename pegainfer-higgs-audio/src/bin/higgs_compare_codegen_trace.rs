use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use pegainfer_higgs_audio::trace_compare::CodegenTraceTolerances;
use pegainfer_higgs_audio::trace_compare::compare_codegen_trace_files;
use pegainfer_higgs_audio::trace_compare::ensure_codegen_trace_comparison_passed;

#[derive(Parser)]
#[command(about = "Compare Higgs Audio native code-generation trace against a reference trace")]
struct Args {
    /// Official/HF incremental past_key_values reference trace JSON.
    #[arg(long)]
    reference: PathBuf,
    /// Native PegaInfer retained-KV code-generation trace JSON.
    #[arg(long)]
    actual: PathBuf,
    #[arg(long, default_value_t = CodegenTraceTolerances::default().logits_cosine_min)]
    logits_cosine_min: f32,
    #[arg(long, default_value_t = CodegenTraceTolerances::default().logits_max_abs_tol)]
    logits_max_abs_tol: f32,
    #[arg(long, default_value_t = CodegenTraceTolerances::default().logits_mean_abs_tol)]
    logits_mean_abs_tol: f32,
    #[arg(long, default_value_t = CodegenTraceTolerances::default().logits_p99_abs_tol)]
    logits_p99_abs_tol: f32,
    #[arg(long, default_value_t = CodegenTraceTolerances::default().argmax_regret_tol)]
    argmax_regret_tol: f32,
    #[arg(long, default_value_t = CodegenTraceTolerances::default().topk_min_overlap)]
    topk_min_overlap: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tolerances = CodegenTraceTolerances {
        logits_cosine_min: args.logits_cosine_min,
        logits_max_abs_tol: args.logits_max_abs_tol,
        logits_mean_abs_tol: args.logits_mean_abs_tol,
        logits_p99_abs_tol: args.logits_p99_abs_tol,
        argmax_regret_tol: args.argmax_regret_tol,
        topk_min_overlap: args.topk_min_overlap,
    };
    let comparison = compare_codegen_trace_files(&args.reference, &args.actual, tolerances)?;

    println!("higgs codegen trace comparison:");
    println!(
        "  prompt_tokens_match={} steps_compared={} first_divergent_step={:?}",
        comparison.prompt_tokens_match, comparison.steps_compared, comparison.first_divergent_step
    );
    println!(
        "  sampled_rows_exact={} raw_rows_exact={} generation_done_exact={}",
        comparison.sampled_rows_exact, comparison.raw_rows_exact, comparison.generation_done_exact
    );
    println!(
        "  argmax_agreement={:.6} argmax_matches={}/{}",
        comparison.argmax_agreement, comparison.argmax_matches, comparison.argmax_positions
    );
    println!(
        "  full_logits_available={} logits_cosine={:?} logits_cosine_min={:.6}",
        comparison.full_logits_available,
        comparison.logits_cosine,
        comparison.tolerances.logits_cosine_min
    );
    println!(
        "  logits_max_abs={:?} max_tol={:.6} logits_mean_abs={:?} mean_tol={:.6} logits_p99_abs={:?} p99_tol={:.6}",
        comparison.logits_max_abs,
        comparison.tolerances.logits_max_abs_tol,
        comparison.logits_mean_abs,
        comparison.tolerances.logits_mean_abs_tol,
        comparison.logits_p99_abs,
        comparison.tolerances.logits_p99_abs_tol
    );
    println!(
        "  max_argmax_regret={:?} regret_tol={:.6}",
        comparison.max_argmax_regret, comparison.tolerances.argmax_regret_tol
    );
    println!(
        "  topk_available={} topk_min_overlap={:?} topk_mean_overlap={:?} topk_min_overlap_tol={}",
        comparison.topk_available,
        comparison.topk_min_overlap,
        comparison.topk_mean_overlap,
        comparison.tolerances.topk_min_overlap
    );
    ensure_codegen_trace_comparison_passed(&comparison)?;
    println!("higgs codegen trace comparison: ok");
    Ok(())
}

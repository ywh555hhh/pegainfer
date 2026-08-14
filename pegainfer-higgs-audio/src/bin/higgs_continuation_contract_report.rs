use anyhow::Result;
use pegainfer_higgs_audio::continuation_contract::ContinuationInputKind;
use pegainfer_higgs_audio::continuation_contract::ContinuationOutputKind;
use pegainfer_higgs_audio::continuation_contract::higgs_v3_continuation_contract;

fn main() -> Result<()> {
    let contract = higgs_v3_continuation_contract();
    contract.validate()?;

    println!("higgs continuation contract: ok");
    println!("  input: {}", input_kind(contract.input));
    println!("  output: {}", output_kind(contract.output));
    println!("  input_hidden_size: {}", contract.input_hidden_size);
    println!("  output_hidden_size: {}", contract.output_hidden_size);
    println!("  codebooks: {}", contract.codebooks);
    println!("  retained_kv: {}", contract.retained_kv);
    println!("  full_prompt_rebuild: {}", contract.full_prompt_rebuild);
    println!("  next_gpu_gate: feedback embedding -> retained-KV body -> final normed hidden");
    Ok(())
}

fn input_kind(kind: ContinuationInputKind) -> &'static str {
    match kind {
        ContinuationInputKind::FeedbackEmbedding => "feedback_embedding",
    }
}

fn output_kind(kind: ContinuationOutputKind) -> &'static str {
    match kind {
        ContinuationOutputKind::FinalNormedHidden => "final_normed_hidden",
    }
}

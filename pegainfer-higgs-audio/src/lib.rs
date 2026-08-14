//! Higgs Audio model-line scaffolding.
//!
//! This crate starts with the artifact and golden-contract boundary for the
//! zero-shot, one-step `[8, 1026]` audio-logits gate. Runtime execution is added
//! behind this boundary in later slices so stale fixture assumptions cannot leak
//! into the model implementation.

pub mod audio_codegen;
pub mod codebook_embedding;
pub mod codec_input;
pub mod compare;
pub mod config;
pub mod continuation_contract;
pub mod decode_session;
pub mod decode_trace;
pub mod delay_pattern;
pub mod kernel_plan;
pub mod launch_preflight;
pub mod layer_dump;
pub mod load_plan;
pub mod materialize_qwen3;
#[cfg(feature = "server-line")]
pub mod model_line;
pub mod native_codec;
pub mod one_step_actual;
pub mod one_step_golden;
#[cfg(feature = "runtime-qwen3")]
pub mod runtime_bridge;
pub mod runtime_source;
pub mod trace_compare;
pub mod weights;

pub use kernel_plan::kernel_plan;

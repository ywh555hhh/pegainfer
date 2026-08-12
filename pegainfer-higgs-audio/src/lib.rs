//! Higgs Audio model-line scaffolding.
//!
//! This crate starts with the artifact and golden-contract boundary for the
//! zero-shot, one-step `[8, 1026]` audio-logits gate. Runtime execution is added
//! behind this boundary in later slices so stale fixture assumptions cannot leak
//! into the model implementation.

pub mod compare;
pub mod config;
pub mod one_step_golden;
pub mod weights;

//! Configuration for the Gemma model adapter.

use litert_lm_rust::Backend;

/// Tuning options for the Gemma engine.
///
/// All fields default to sensible CPU-friendly values; `Backend` is re-exported
/// from `litert-lm-rust` (this is the one crate where LiteRT types may
/// appear — nothing outside `noema-gemma` ever sees them).
#[derive(Debug, Clone, PartialEq)]
pub struct GemmaOptions {
    /// The main execution backend. Defaults to CPU.
    pub backend: Backend,
    /// Optional vision backend for image input.
    pub vision_backend: Option<Backend>,
    /// Optional audio backend for audio input.
    pub audio_backend: Option<Backend>,
    /// Number of CPU threads.
    pub num_threads: i32,
    /// Maximum context (prefill + decode) tokens.
    pub max_num_tokens: Option<i32>,
    /// Default maximum output tokens per turn.
    pub max_output_tokens: i32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-k value used alongside top-p sampling.
    pub top_k: i32,
    /// Top-p (nucleus) sampling probability.
    pub top_p: f32,
    /// The conversation's system prompt, set once when the conversation is
    /// created. (Per-request system overrides arrive with context assembly.)
    pub system_prompt: Option<String>,
    /// Whether to enable native benchmarking so per-turn token usage can be
    /// reported.
    pub benchmark: bool,
    /// A stable identifier reported through the model trait.
    pub id: String,
}

impl Default for GemmaOptions {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            // The LiteRT engine only loads the vision/audio executors when
            // these are set — without them an image turn fails with "Vision
            // executor should not be null". CPU is the only backend this
            // project uses, so default them on.
            vision_backend: Some(Backend::Cpu),
            audio_backend: Some(Backend::Cpu),
            num_threads: 4,
            max_num_tokens: None,
            max_output_tokens: 512,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            system_prompt: None,
            benchmark: true,
            id: "gemma-4".into(),
        }
    }
}

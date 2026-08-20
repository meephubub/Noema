//! The Gemma 4 model adapter.
//!
//! [`GemmaModel`] implements Noema's model trait over LiteRT-LM. Each turn is
//! streamed token-by-token through [`ModelResponse::Stream`], with
//! cancellation and per-turn usage metadata.
//!
//! # Conversation state
//!
//! The `Model` trait is session-agnostic (requests carry messages but no
//! session id), and LiteRT-LM's *streaming* conversation path does not commit
//! the assistant turn into the native conversation — so relying on native
//! state would lose multi-turn memory (verified empirically). `GemmaModel`
//! instead keeps the conversation history in Rust and seeds a fresh native
//! conversation from it each turn (the engine's `messages` preface), which
//! preserves memory and makes each turn self-contained. Multiple sessions
//! sharing one `GemmaModel` would still interleave in this single history;
//! session-keyed context assembly (the context milestone) will key history
//! per session.

use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use litert_lm_rust::{
    Backend, Conversation, ConversationConfig, Engine, Message, SamplerParams, SendOptions,
    SessionConfig, StreamEvent,
};
use noema_core::{Model, ModelChunk, ModelRequest, ModelResponse, NoemaError, Result, Usage};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::mapping::{map_litert, map_message};
use crate::options::GemmaOptions;

/// The default Gemma model file name inside the workspace `models/` directory.
pub const DEFAULT_MODEL_FILE: &str = "gemma-4-E2B-it.litertlm";

/// The `Model` adapter for Gemma 4, running through `litert-lm-rust`.
pub struct GemmaModel {
    id: String,
    options: GemmaOptions,
    /// The conversation history (user and assistant turns), kept in Rust so
    /// multi-turn memory survives the streaming path. Replayed as the native
    /// conversation preface each turn.
    history: Arc<Mutex<Vec<Message>>>,
    /// Serialises whole turns so concurrent `generate` calls never interleave
    /// on the history. The guard is moved into the draining task and released
    /// only after the turn has been committed to history.
    turn_lock: Arc<tokio::sync::Mutex<()>>,
    /// The usage of the most recently completed turn, when benchmarking is
    /// enabled.
    last_usage: Arc<Mutex<Option<Usage>>>,
    /// The LiteRT engine. Declared **after** the other fields so it is
    /// dropped last (fields drop in declaration order); native conversations
    /// must be released before the engine is destroyed.
    engine: Arc<Engine>,
}

impl fmt::Debug for GemmaModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GemmaModel")
            .field("id", &self.id)
            .field("options", &self.options)
            .field(
                "history_turns",
                &self.history.lock().map(|h| h.len()).unwrap_or(0),
            )
            .field("last_usage", &self.last_usage.lock().unwrap())
            .finish_non_exhaustive()
    }
}

/// Newtype making the native stream receiver movable to a blocking thread.
struct SendReceiver(litert_lm_rust::StreamEventReceiver);

// SAFETY: `StreamEventReceiver` is `!Send` only because of its `_inputs:
// Option<OwnedInputs>` field, which conversation streams never populate
// (only the raw session API sets it). The other fields — a `Receiver` and a
// callback guard — are `Send`. Holding this wrapper is therefore sound for
// the conversation streams `GemmaModel` uses. (The `iter` method below
// takes `self` so the whole wrapper — not its `!Send` inner field — is
// captured when the receiver is moved into the drain thread.)
unsafe impl Send for SendReceiver {}

impl SendReceiver {
    /// Drains the native stream, forwarding text deltas into `tx` until the
    /// stream ends, the consumer drops, or the token is cancelled. Returns
    /// the full generated text and whether it was cancelled.
    fn drain(
        self,
        conversation: &mut Conversation<'static>,
        cancel: &CancellationToken,
        tx: &tokio::sync::mpsc::Sender<std::result::Result<ModelChunk, NoemaError>>,
    ) -> (String, bool) {
        let mut full = String::new();
        let mut cancelled = false;

        for event in self.0.iter() {
            if cancel.is_cancelled() {
                conversation.cancel_process();
                let _ = tx.blocking_send(Err(NoemaError::Model(
                    "gemma generation cancelled".into(),
                )));
                cancelled = true;
                break;
            }
            match event {
                StreamEvent::StartFailed(code) => {
                    let _ = tx.blocking_send(Err(NoemaError::Model(format!(
                        "gemma stream failed to start: {code}"
                    ))));
                    break;
                }
                StreamEvent::Chunk(chunk) => {
                    if let Some(err) = chunk.error {
                        let _ = tx.blocking_send(Err(NoemaError::Model(format!(
                            "gemma stream error: {err}"
                        ))));
                        break;
                    }
                    if let Some(text) = chunk.text {
                        // Streamed chunks carry a serialized message
                        // (`{"role":"assistant","content":[...]}`), not
                        // raw text; extract the text delta.
                        if let Some(delta) = Message::from_json_str(&text)
                            .ok()
                            .and_then(|message| message.text())
                        {
                            if delta.is_empty() {
                                continue;
                            }
                            full.push_str(&delta);
                            if tx.blocking_send(Ok(ModelChunk::new(delta.clone()))).is_err() {
                                // The consumer is gone; stop generating.
                                conversation.cancel_process();
                                break;
                            }
                        }
                    }
                    if chunk.is_final {
                        break;
                    }
                }
            }
        }

        (full, cancelled)
    }
}

impl GemmaModel {
    /// Starts a builder for a Gemma model at the given `.litertlm` path.
    pub fn builder(model_path: impl Into<PathBuf>) -> GemmaModelBuilder {
        GemmaModelBuilder {
            model_path: model_path.into(),
            options: GemmaOptions::default(),
        }
    }

    /// Builds a Gemma model with default options, resolving the model file
    /// from `NOEMA_GEMMA_MODEL` or the workspace `models/` directory.
    pub fn from_default() -> Result<Self> {
        let path = default_model_path().ok_or_else(|| {
            NoemaError::Model(format!(
                "no Gemma model found; set NOEMA_GEMMA_MODEL or place {DEFAULT_MODEL_FILE} in models/"
            ))
        })?;
        Self::builder(path).build()
    }

    /// The token usage of the most recently completed turn, if benchmarking
    /// was enabled. (Streamed responses do not carry usage through the trait,
    /// so this is the way to read it.)
    pub fn last_usage(&self) -> Option<Usage> {
        *self.last_usage.lock().expect("usage lock poisoned")
    }

    /// Clears the conversation history, rewinding context while keeping the
    /// loaded weights.
    pub async fn reset_conversation(&self) -> Result<()> {
        let _turn = self.turn_lock.clone().lock_owned().await;
        self.history.lock().expect("history poisoned").clear();
        *self.last_usage.lock().expect("usage lock poisoned") = None;
        Ok(())
    }

    /// Builds the LiteRT message and per-turn options for a request.
    fn message_for(&self, request: &ModelRequest) -> Result<(Message, SendOptions)> {
        let message = request.messages.last().ok_or_else(|| {
            NoemaError::Model("Gemma request carried no messages".into())
        })?;
        let native = map_message(message)?;
        let optional_args = request.options.max_tokens.map(|max| {
            litert_lm_rust::ConversationOptionalArgs {
                max_output_tokens: Some(max.min(i32::MAX as u32) as i32),
                ..Default::default()
            }
        });
        let options = SendOptions {
            optional_args,
            ..Default::default()
        };
        Ok((native, options))
    }

    /// Estimates this turn's input tokens by tokenizing the system prompt,
    /// the history, and the current message. (The native conversation's
    /// `token_count` is 0 before the first send, so it cannot be used on a
    /// freshly seeded conversation.)
    fn input_tokens(&self, history: &[Message], current: &Message) -> u64 {
        let mut tokens = 0u64;
        if let Some(system) = &self.options.system_prompt {
            if let Ok(t) = self.engine.tokenize(system) {
                tokens += t.len() as u64;
            }
        }
        for message in history.iter().chain(std::iter::once(current)) {
            if let Some(text) = message.text() {
                if let Ok(t) = self.engine.tokenize(&text) {
                    tokens += t.len() as u64;
                }
            }
        }
        tokens
    }

    /// Creates a fresh native conversation seeded with the given history as
    /// the engine's `messages` preface, so the streamed turn sees prior
    /// context without relying on native conversation state.
    fn seeded_conversation(&self, history: &[Message]) -> Result<Conversation<'static>> {
        let config = ConversationConfig {
            session: SessionConfig {
                max_output_tokens: Some(self.options.max_output_tokens),
                // The CPU accelerator implements the top-p and greedy
                // samplers; top-k sampling is not supported by this backend.
                sampler: Some(
                    SamplerParams::top_p(self.options.top_p)
                        .with_top_k(self.options.top_k)
                        .with_temperature(self.options.temperature),
                ),
                ..Default::default()
            },
            system_message: self.options.system_prompt.as_ref().map(|system| {
                serde_json::to_value(Message::system(system)).expect("system message serializes")
            }),
            messages: (!history.is_empty())
                .then(|| serde_json::to_value(history).expect("history serializes")),
            ..Default::default()
        };
        let conversation = self.engine.create_conversation(config).map_err(map_litert)?;
        // SAFETY: `Conversation<'a>` carries only a phantom lifetime marker
        // (`EngineLifetime` in litert-lm-rust); it holds no actual reference
        // to the engine. The underlying native handle stays valid as long as
        // the engine outlives it, which `GemmaModel` guarantees by owning
        // both (`engine` is declared last, so it is dropped last).
        Ok(unsafe { std::mem::transmute::<Conversation<'_>, Conversation<'static>>(conversation) })
    }
}

#[async_trait]
impl Model for GemmaModel {
    fn id(&self) -> &str {
        &self.id
    }

    async fn generate(
        &self,
        request: ModelRequest,
        cancel: CancellationToken,
    ) -> Result<ModelResponse> {
        let (message, send_options) = self.message_for(&request)?;

        // Serialise turns and hold the guard until the turn is committed:
        // it is moved into the draining task below, so a later `generate`
        // never starts while a previous turn is still streaming or writing
        // history.
        let turn_guard = self.turn_lock.clone().lock_owned().await;

        let history = self.history.lock().expect("history poisoned").clone();
        let mut conversation = self.seeded_conversation(&history)?;
        let input_tokens = self.input_tokens(&history, &message);

        let receiver = SendReceiver(
            conversation
                .send_message_stream_with(message.clone(), send_options)
                .map_err(map_litert)?,
        );

        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<ModelChunk, NoemaError>>(16);

        let engine = Arc::clone(&self.engine);
        let history_slot = Arc::clone(&self.history);
        let usage_slot = Arc::clone(&self.last_usage);

        // The native receiver is a blocking std channel; drain it on a
        // dedicated thread and forward chunks into the async stream.
        std::thread::spawn(move || {
            let (full, cancelled) = receiver.drain(&mut conversation, &cancel, &tx);

            if !cancelled && !full.is_empty() {
                // Commit the turn: user message + assistant reply.
                if let Ok(mut history) = history_slot.lock() {
                    history.push(message);
                    history.push(Message::model(full.trim_end()));
                    let output_tokens = engine
                        .tokenize(&full)
                        .ok()
                        .map(|tokens| tokens.len() as u64)
                        .unwrap_or(0);
                    let usage = Usage {
                        input_tokens,
                        output_tokens,
                    };
                    if let Ok(mut slot) = usage_slot.lock() {
                        *slot = Some(usage);
                    }
                }
            }

            // Dropping `turn_guard` here (after history is written) lets the
            // next turn start.
            drop(turn_guard);
        });

        let stream = ReceiverStream::new(rx);
        Ok(ModelResponse::Stream(Box::pin(stream)))
    }
}

/// Resolves the Gemma model file from `NOEMA_GEMMA_MODEL` or the workspace
/// `models/` directory.
pub fn default_model_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("NOEMA_GEMMA_MODEL") {
        return Some(PathBuf::from(path));
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo = manifest.parent()?.parent()?;
    let candidate = repo.join("models").join(DEFAULT_MODEL_FILE);
    candidate.exists().then_some(candidate)
}

/// Builder for a [`GemmaModel`].
#[derive(Debug, Clone)]
pub struct GemmaModelBuilder {
    model_path: PathBuf,
    options: GemmaOptions,
}

impl GemmaModelBuilder {
    /// Sets the main execution backend (default: CPU).
    pub fn backend(mut self, backend: Backend) -> Self {
        self.options.backend = backend;
        self
    }

    /// Sets the vision backend for image input.
    pub fn vision_backend(mut self, backend: Backend) -> Self {
        self.options.vision_backend = Some(backend);
        self
    }

    /// Sets the audio backend for audio input.
    pub fn audio_backend(mut self, backend: Backend) -> Self {
        self.options.audio_backend = Some(backend);
        self
    }

    /// Sets the number of CPU threads.
    pub fn num_threads(mut self, threads: i32) -> Self {
        self.options.num_threads = threads;
        self
    }

    /// Sets the maximum context tokens.
    pub fn max_num_tokens(mut self, tokens: i32) -> Self {
        self.options.max_num_tokens = Some(tokens);
        self
    }

    /// Sets the default maximum output tokens per turn.
    pub fn max_output_tokens(mut self, tokens: i32) -> Self {
        self.options.max_output_tokens = tokens;
        self
    }

    /// Sets the sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.options.temperature = temperature;
        self
    }

    /// Sets the conversation's system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.options.system_prompt = Some(prompt.into());
        self
    }

    /// Enables or disables native benchmarking (usage reporting).
    pub fn benchmark(mut self, enabled: bool) -> Self {
        self.options.benchmark = enabled;
        self
    }

    /// Overrides the model id reported through the model trait.
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.options.id = id.into();
        self
    }

    /// Loads the model weights and builds the adapter.
    ///
    /// This blocks while the weights load; call it from a background task in
    /// async contexts.
    pub fn build(self) -> Result<GemmaModel> {
        let mut builder = Engine::builder(&self.model_path)
            .backend(self.options.backend.clone())
            .num_threads(self.options.num_threads);
        if let Some(vision) = &self.options.vision_backend {
            builder = builder.vision_backend(vision.clone());
        }
        if let Some(audio) = &self.options.audio_backend {
            builder = builder.audio_backend(audio.clone());
        }
        if let Some(max) = self.options.max_num_tokens {
            builder = builder.max_num_tokens(max);
        }
        if self.options.benchmark {
            builder = builder.enable_benchmark();
        }
        let engine = Arc::new(builder.build().map_err(map_litert)?);

        tracing::info!(
            model = %self.options.id,
            path = %self.model_path.display(),
            backend = %self.options.backend.as_str(),
            "gemma engine loaded"
        );

        Ok(GemmaModel {
            id: self.options.id.clone(),
            options: self.options,
            history: Arc::new(Mutex::new(Vec::new())),
            turn_lock: Arc::new(tokio::sync::Mutex::new(())),
            last_usage: Arc::new(Mutex::new(None)),
            engine,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_path_resolution() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo = manifest.parent().and_then(|p| p.parent()).unwrap();
        let expected = repo.join("models").join(DEFAULT_MODEL_FILE);
        if expected.exists() {
            let resolved = default_model_path().expect("model file present");
            assert_eq!(resolved, expected);
        }
    }

    #[test]
    fn options_survive_builder_round_trip() {
        let options = GemmaOptions {
            backend: Backend::Cpu,
            vision_backend: None,
            audio_backend: None,
            num_threads: 2,
            max_num_tokens: Some(2048),
            max_output_tokens: 128,
            temperature: 0.2,
            top_k: 10,
            top_p: 0.95,
            system_prompt: Some("be brief".into()),
            benchmark: true,
            id: "test-gemma".into(),
        };
        assert_eq!(options.max_output_tokens, 128);
        assert_eq!(options.system_prompt.as_deref(), Some("be brief"));
    }
}

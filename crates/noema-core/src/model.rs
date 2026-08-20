//! Model abstractions: the interface every model backend implements.
//!
//! Noema is model-agnostic. Gemma 4, Needle 2, cloud providers, and future
//! models all speak through the traits in this module; nothing else in the
//! codebase depends on a specific inference backend.
//!
//! The abstraction supports: text, images, audio, streaming, tool-related
//! messages, system prompts, model escalation, cancellation, usage metadata,
//! and errors.

use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;
use tokio_util::sync::CancellationToken;

use crate::error::{NoemaError, Result};

/// A boxed, sendable stream.
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

/// Who produced a [`Message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// System instructions.
    System,
    /// The user.
    User,
    /// The assistant (model).
    Assistant,
    /// A message produced by a tool.
    Tool,
}

/// A single ordered part of a message's content.
///
/// Messages carry an ordered list of parts so mixed requests — text plus an
/// image, audio plus text, and so on — are first-class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContentPart {
    /// Plain text.
    Text(String),
    /// An image.
    Image(ImageData),
    /// An audio clip.
    Audio(AudioData),
}

impl ContentPart {
    /// A text content part.
    pub fn text(content: impl Into<String>) -> Self {
        ContentPart::Text(content.into())
    }

    /// An image content part.
    pub fn image(bytes: Vec<u8>, mime: impl Into<String>) -> Self {
        ContentPart::Image(ImageData {
            bytes,
            mime: mime.into(),
        })
    }

    /// An audio content part.
    pub fn audio(bytes: Vec<u8>, mime: impl Into<String>) -> Self {
        ContentPart::Audio(AudioData {
            bytes,
            mime: mime.into(),
        })
    }
}

/// Binary image content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageData {
    /// The raw image bytes.
    pub bytes: Vec<u8>,
    /// The MIME type, e.g. `image/png`.
    pub mime: String,
}

/// Binary audio content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioData {
    /// The raw audio bytes.
    pub bytes: Vec<u8>,
    /// The MIME type, e.g. `audio/wav`.
    pub mime: String,
}

/// A message with ordered, potentially multimodal content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// The speaker.
    pub role: Role,
    /// The content parts, in order.
    pub content: Vec<ContentPart>,
}

impl Message {
    /// A message with the given content parts.
    pub fn new(role: Role, content: Vec<ContentPart>) -> Self {
        Self { role, content }
    }

    /// A plain-text message.
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text(content.into())],
        }
    }
}

/// Token usage metadata reported by a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Input (prompt) tokens.
    pub input_tokens: u64,
    /// Output (completion) tokens.
    pub output_tokens: u64,
}

impl Usage {
    /// Total tokens consumed.
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Sampling and generation options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelOptions {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum number of output tokens.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling probability.
    pub top_p: Option<f32>,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: None,
            top_p: None,
        }
    }
}

/// A request to a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    /// System instructions, if any.
    pub system: Option<String>,
    /// The conversation, in order.
    pub messages: Vec<Message>,
    /// Generation options.
    pub options: ModelOptions,
}

impl ModelRequest {
    /// A request from the given conversation.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            system: None,
            messages,
            options: ModelOptions::default(),
        }
    }

    /// Sets the system instructions.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the generation options.
    pub fn with_options(mut self, options: ModelOptions) -> Self {
        self.options = options;
        self
    }
}

/// A structured request to escalate to a larger model.
///
/// Models signal escalation with structured metadata rather than a
/// natural-language statement; Noema then decides whether and how to
/// escalate, subject to policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EscalationRequest {
    /// Why escalation is needed.
    pub reason: String,
    /// The context the larger model should see.
    pub context: Vec<Message>,
}

impl EscalationRequest {
    /// An escalation request with the given reason and context.
    pub fn new(reason: impl Into<String>, context: Vec<Message>) -> Self {
        Self {
            reason: reason.into(),
            context,
        }
    }
}

/// A chunk of a streaming response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelChunk {
    /// The text delta produced so far.
    pub delta: String,
}

impl ModelChunk {
    /// A chunk carrying the given text delta.
    pub fn new(delta: impl Into<String>) -> Self {
        Self { delta: delta.into() }
    }
}

/// A model's response.
//
// `Debug` is implemented by hand because a boxed stream is not `Debug`.
pub enum ModelResponse {
    /// A complete, non-streaming text response.
    Text {
        /// The full generated text.
        content: String,
        /// Usage metadata, when the backend reports it.
        usage: Option<Usage>,
    },
    /// A stream of response chunks.
    Stream(BoxStream<std::result::Result<ModelChunk, NoemaError>>),
    /// The model requests escalation to a larger model.
    Escalate(EscalationRequest),
}

impl fmt::Debug for ModelResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelResponse::Text { content, usage } => f
                .debug_struct("ModelResponse::Text")
                .field("content", content)
                .field("usage", usage)
                .finish(),
            ModelResponse::Stream(_) => f.write_str("ModelResponse::Stream(..)"),
            ModelResponse::Escalate(request) => f
                .debug_struct("ModelResponse::Escalate")
                .field("reason", &request.reason)
                .finish(),
        }
    }
}

/// A local model backend (Gemma, Needle, or a future local model).
///
/// Implementations must honour [`CancellationToken`]: when it is cancelled,
/// generation should stop promptly and return
/// [`NoemaError::Model`](crate::error::NoemaError::Model).
#[async_trait]
pub trait Model: fmt::Debug + Send + Sync + 'static {
    /// A stable identifier for this model instance.
    fn id(&self) -> &str;

    /// Generates a response for the given request.
    async fn generate(
        &self,
        request: ModelRequest,
        cancel: CancellationToken,
    ) -> Result<ModelResponse>;
}

/// A cloud escalation provider (Gemini, OpenAI, or a future provider).
///
/// Providers are abstract: the core agent never hard-codes one.
#[async_trait]
pub trait ModelProvider: fmt::Debug + Send + Sync + 'static {
    /// A stable identifier for this provider, e.g. `gemini` or `openai`.
    fn id(&self) -> &str;

    /// Completes the given request on the larger cloud model.
    async fn complete(
        &self,
        request: ModelRequest,
        cancel: CancellationToken,
    ) -> Result<ModelResponse>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_message_builds() {
        let message = Message::text(Role::User, "hello");
        assert_eq!(message.role, Role::User);
        assert_eq!(
            message.content,
            vec![ContentPart::Text("hello".into())]
        );
    }

    #[test]
    fn multimodal_message_preserves_order() {
        let message = Message::new(
            Role::User,
            vec![
                ContentPart::text("explain this"),
                ContentPart::image(vec![1, 2, 3], "image/png"),
                ContentPart::audio(vec![4, 5], "audio/wav"),
            ],
        );
        assert_eq!(message.content.len(), 3);
        assert!(matches!(message.content[1], ContentPart::Image(_)));
        assert!(matches!(message.content[2], ContentPart::Audio(_)));
    }

    #[test]
    fn request_builders_work() {
        let request = ModelRequest::new(vec![Message::text(Role::User, "hi")])
            .with_system("be brief")
            .with_options(ModelOptions {
                temperature: Some(0.0),
                ..Default::default()
            });
        assert_eq!(request.system.as_deref(), Some("be brief"));
        assert_eq!(request.options.temperature, Some(0.0));
        assert_eq!(request.messages.len(), 1);
    }

    #[test]
    fn usage_totals() {
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
        };
        assert_eq!(usage.total(), 15);
    }

    #[test]
    fn escalation_request_carries_context() {
        let request = EscalationRequest::new(
            "needs bigger reasoning",
            vec![Message::text(Role::User, "hard problem")],
        );
        assert_eq!(request.context.len(), 1);
    }
}

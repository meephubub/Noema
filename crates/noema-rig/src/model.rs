//! A Rig [`CompletionModel`] adapter over Noema's [`Model`] trait.
//!
//! [`NoemaCompletionModel`] lets any Noema model — local Gemma, local Needle,
//! or a future cloud provider — be driven by rig agents and the rig request
//! vocabulary, without rig ever seeing the backend's types.
//!
//! # Conversation state
//!
//! Noema's local models own their conversation state (a `GemmaModel` holds
//! one native LiteRT conversation, for example), so by default the adapter
//! forwards only the latest user message of a rig chat history and lets the
//! model keep the context. Stateless models can opt into receiving the full
//! history via [`NoemaCompletionModel::send_full_history`].

use std::sync::Arc;

use noema_core::{Model, ModelOptions, ModelRequest, ModelResponse, NoemaError};
use rig_core::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    FinishReason, Usage as RigUsage,
};
use rig_core::message::Text;
use rig_core::streaming::{
    RawStreamingChoice, StreamFinal, StreamingCompletionResponse, StreamingResult,
};
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::message::{last_user_message, rig_to_noema};

/// A rig completion model backed by any Noema [`Model`].
///
/// Cloning shares the underlying model (an `Arc`), never duplicating it.
#[derive(Debug)]
pub struct NoemaCompletionModel<M: ?Sized> {
    model: Arc<M>,
    provider: String,
    /// When set, the full rig chat history is forwarded to the model each
    /// turn (for stateless backends). Defaults to `false`: local models own
    /// their conversation state, so only the latest user message is sent.
    send_full_history: bool,
}

impl<M: ?Sized> Clone for NoemaCompletionModel<M> {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            provider: self.provider.clone(),
            send_full_history: self.send_full_history,
        }
    }
}

impl<M: Model + ?Sized> NoemaCompletionModel<M> {
    /// Wraps the given model. `provider` names the model in rig responses
    /// (the model's own id is used when not overridden).
    pub fn new(model: Arc<M>) -> Self {
        Self {
            model,
            provider: "noema".to_string(),
            send_full_history: false,
        }
    }

    /// Overrides the provider name reported in rig responses.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    /// Sets whether the full chat history is forwarded each turn.
    ///
    /// Defaults to `false`, matching Noema's stateful local models. Enable
    /// for stateless backends that rebuild their own context from the
    /// history.
    pub fn send_full_history(mut self, enabled: bool) -> Self {
        self.send_full_history = enabled;
        self
    }

    /// Builds the Noema request for a rig completion request.
    fn request_to_noema(
        &self,
        request: &CompletionRequest,
    ) -> Result<ModelRequest, CompletionError> {
        request.validate_message_content()?;
        let messages = if self.send_full_history {
            request
                .chat_history
                .iter()
                .map(rig_to_noema)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let latest = last_user_message(&request.chat_history).ok_or_else(|| {
                CompletionError::RequestError(
                    "rig request carried no user message".to_owned().into(),
                )
            })?;
            vec![rig_to_noema(latest)?]
        };
        let options = ModelOptions {
            temperature: request.temperature.map(|value| value as f32),
            max_tokens: request.max_tokens.map(|value| value.min(u32::MAX as u64) as u32),
            top_p: None,
        };
        Ok(ModelRequest::new(messages).with_options(options))
    }

    /// Runs the model and reduces any response shape to text + usage.
    async fn generate_text(
        &self,
        request: ModelRequest,
    ) -> Result<(String, Option<noema_core::Usage>), CompletionError> {
        let response = self
            .model
            .generate(request, CancellationToken::new())
            .await
            .map_err(map_noema_error)?;
        reduce_response(response).await
    }
}

impl<M: Model + ?Sized> CompletionModel for NoemaCompletionModel<M> {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let noema_request = self.request_to_noema(&request)?;
        let (text, usage) = self.generate_text(noema_request).await?;
        let choice = if text.is_empty() {
            Vec::new()
        } else {
            vec![AssistantContent::Text(Text::new(text))]
        };
        Ok(CompletionResponse::new(choice, usage_to_rig(usage), &self.provider)
            .with_finish_reason(FinishReason::Stop))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let noema_request = self.request_to_noema(&request)?;
        let (text, usage) = self.generate_text(noema_request).await?;
        let provider = self.provider.clone();

        let mut items: Vec<Result<RawStreamingChoice<StreamFinal>, CompletionError>> = Vec::new();
        if !text.is_empty() {
            items.push(Ok(RawStreamingChoice::Message(text)));
        }
        items.push(Ok(RawStreamingChoice::FinalResponse(
            StreamFinal::new(provider.clone(), usage_to_rig(usage)).with_finish_reason(
                FinishReason::Stop,
            ),
        )));

        let inner: StreamingResult = Box::pin(tokio_stream::iter(items));
        Ok(StreamingCompletionResponse::stream(provider, inner))
    }
}

/// Reduces a Noema model response to text and usage for rig.
async fn reduce_response(
    response: ModelResponse,
) -> Result<(String, Option<noema_core::Usage>), CompletionError> {
    match response {
        ModelResponse::Text { content, usage } => Ok((content, usage)),
        ModelResponse::Stream(stream) => {
            let mut stream = Box::pin(stream);
            let mut content = String::new();
            while let Some(chunk) = stream.next().await {
                content.push_str(&chunk.map_err(|error| {
                    CompletionError::ProviderError(error.to_string())
                })?.delta);
            }
            Ok((content, None))
        }
        ModelResponse::Escalate(request) => Err(CompletionError::ProviderError(format!(
            "model requested escalation: {}",
            request.reason
        ))),
    }
}

/// Converts Noema usage into rig's usage record (zero-valued when the model
/// reported none).
fn usage_to_rig(usage: Option<noema_core::Usage>) -> RigUsage {
    match usage {
        Some(usage) => RigUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total(),
            ..Default::default()
        },
        None => RigUsage::default(),
    }
}

/// Maps a Noema error onto rig's completion error vocabulary.
pub(crate) fn map_noema_error(error: NoemaError) -> CompletionError {
    CompletionError::ProviderError(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use noema_core::ModelRequest;
    use rig_core::completion::Message as RigMessage;

    #[derive(Debug)]
    struct EchoModel {
        seen: std::sync::Mutex<Vec<Vec<String>>>,
    }

    impl EchoModel {
        fn new() -> Self {
            Self {
                seen: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Model for EchoModel {
        fn id(&self) -> &str {
            "echo"
        }

        async fn generate(
            &self,
            request: ModelRequest,
            _cancel: CancellationToken,
        ) -> Result<ModelResponse, NoemaError> {
            let texts = request
                .messages
                .iter()
                .map(|message| match message.content.first() {
                    Some(noema_core::ContentPart::Text(text)) => text.clone(),
                    _ => String::new(),
                })
                .collect();
            self.seen.lock().unwrap().push(texts);
            Ok(ModelResponse::Text {
                content: "pong".into(),
                usage: Some(noema_core::Usage {
                    input_tokens: 3,
                    output_tokens: 4,
                }),
            })
        }
    }

    #[tokio::test]
    async fn completion_returns_text_and_usage() {
        let model = NoemaCompletionModel::new(Arc::new(EchoModel::new()));
        let request = CompletionRequestBuilderAdapter::build(model.clone(), "ping");
        let response = model.completion(request).await.expect("completion");

        assert_eq!(response.choice.len(), 1);
        match &response.choice[0] {
            AssistantContent::Text(text) => assert_eq!(text.text, "pong"),
            other => panic!("expected text, got {other:?}"),
        }
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.output_tokens, 4);
        assert_eq!(response.usage.total_tokens, 7);
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn stream_yields_chunk_then_final() {
        let model = NoemaCompletionModel::new(Arc::new(EchoModel::new()));
        let request = CompletionRequestBuilderAdapter::build(model.clone(), "ping");
        let mut stream = model.stream(request).await.expect("stream");

        let mut texts = Vec::new();
        let mut usage = None;
        while let Some(item) = tokio_stream::StreamExt::next(&mut stream).await {
            match item.expect("item") {
                rig_core::streaming::StreamedAssistantContent::Text(text) => {
                    texts.push(text.text);
                }
                rig_core::streaming::StreamedAssistantContent::Final(final_) => {
                    usage = Some(final_.usage);
                }
                other => panic!("unexpected stream item: {other:?}"),
            }
        }
        assert_eq!(texts, vec!["pong"]);
        let usage = usage.expect("final record");
        assert_eq!(usage.input_tokens, 3);
        assert_eq!(usage.output_tokens, 4);
    }

    #[tokio::test]
    async fn default_forwards_only_the_latest_user_message() {
        let echo = Arc::new(EchoModel::new());
        let model = NoemaCompletionModel::new(Arc::clone(&echo));
        // History like rig agents build up across turns.
        let request = CompletionRequestBuilderAdapter::with_history(
            model.clone(),
            vec![
                RigMessage::user("old prompt"),
                RigMessage::assistant("reply"),
                RigMessage::user("latest prompt"),
            ],
        );
        model.completion(request).await.expect("completion");

        let seen = echo.seen.lock().unwrap();
        let last = seen.last().expect("one turn");
        assert_eq!(last.len(), 1);
        assert_eq!(last[0], "latest prompt");
    }

    #[tokio::test]
    async fn full_history_mode_forwards_every_message() {
        let echo = Arc::new(EchoModel::new());
        let model = NoemaCompletionModel::new(Arc::clone(&echo)).send_full_history(true);
        let request = CompletionRequestBuilderAdapter::with_history(
            model.clone(),
            vec![
                RigMessage::system("be brief"),
                RigMessage::user("first"),
                RigMessage::assistant("reply"),
            ],
        );
        model.completion(request).await.expect("completion");

        let seen = echo.seen.lock().unwrap();
        let last = seen.last().expect("one turn");
        assert_eq!(last.len(), 3);
        assert_eq!(last[1], "first");
    }

    /// Builds rig requests without requiring `Self: Clone` gymnastics in the
    /// tests above.
    struct CompletionRequestBuilderAdapter;

    impl CompletionRequestBuilderAdapter {
        fn build<M: Model + ?Sized>(
            model: NoemaCompletionModel<M>,
            prompt: &str,
        ) -> CompletionRequest {
            let builder =
                rig_core::completion::CompletionRequestBuilder::new(model.clone(), prompt.to_string());
            builder.build()
        }

        fn with_history<M: Model + ?Sized>(
            model: NoemaCompletionModel<M>,
            history: Vec<RigMessage>,
        ) -> CompletionRequest {
            let mut builder = rig_core::completion::CompletionRequestBuilder::new(
                model.clone(),
                history.last().unwrap().clone(),
            );
            for message in &history[..history.len() - 1] {
                builder = builder.message(message.clone());
            }
            builder.build()
        }
    }
}

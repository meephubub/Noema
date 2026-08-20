//! A deterministic rig completion provider for tests, examples, and
//! development without a real model.
//!
//! Every completion returns the configured canned response, optionally with
//! usage metadata. Streaming delivers the response as a single text chunk
//! followed by the terminal usage record.

use rig_core::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    FinishReason, Usage,
};
use rig_core::message::Text;
use rig_core::streaming::{
    RawStreamingChoice, StreamFinal, StreamingCompletionResponse, StreamingResult,
};

/// A rig completion provider that always answers with a canned response.
#[derive(Debug, Clone)]
pub struct StubProvider {
    model: String,
    response: String,
    usage: Option<noema_core::Usage>,
}

impl StubProvider {
    /// A stub provider for the given model name that always answers with
    /// `response`.
    pub fn new(model: impl Into<String>, response: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            response: response.into(),
            usage: None,
        }
    }

    /// Attaches token usage metadata to every completion.
    pub fn with_usage(mut self, usage: noema_core::Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    fn usage(&self) -> Usage {
        match self.usage {
            Some(usage) => Usage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: usage.total(),
                ..Default::default()
            },
            None => Usage::default(),
        }
    }
}

impl CompletionModel for StubProvider {
    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let choice = if self.response.is_empty() {
            Vec::new()
        } else {
            vec![AssistantContent::Text(Text::new(self.response.clone()))]
        };
        Ok(CompletionResponse::new(choice, self.usage(), &self.model)
            .with_finish_reason(FinishReason::Stop))
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let mut items: Vec<Result<RawStreamingChoice<StreamFinal>, CompletionError>> = Vec::new();
        if !self.response.is_empty() {
            items.push(Ok(RawStreamingChoice::Message(self.response.clone())));
        }
        items.push(Ok(RawStreamingChoice::FinalResponse(
            StreamFinal::new(self.model.clone(), self.usage())
                .with_finish_reason(FinishReason::Stop),
        )));
        let inner: StreamingResult = Box::pin(tokio_stream::iter(items));
        Ok(StreamingCompletionResponse::stream(self.model.clone(), inner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::CompletionRequestBuilder;
    use rig_core::streaming::StreamedAssistantContent;
    use tokio_stream::StreamExt;

    #[tokio::test]
    async fn stub_completion_returns_canned_text() {
        let stub = StubProvider::new("stub", "forty-two");
        let request = CompletionRequestBuilder::new(stub.clone(), "what is 6 * 7?")
            .build();
        let response = stub.completion(request).await.expect("completion");

        match &response.choice[0] {
            AssistantContent::Text(text) => assert_eq!(text.text, "forty-two"),
            other => panic!("expected text, got {other:?}"),
        }
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn stub_stream_delivers_chunk_then_final() {
        let stub = StubProvider::new("stub", "hello").with_usage(noema_core::Usage {
            input_tokens: 5,
            output_tokens: 2,
        });
        let request = CompletionRequestBuilder::new(stub.clone(), "hi").build();
        let mut stream = stub.stream(request).await.expect("stream");

        let mut texts = Vec::new();
        let mut usage = None;
        while let Some(item) = stream.next().await {
            match item.expect("item") {
                StreamedAssistantContent::Text(text) => texts.push(text.text),
                StreamedAssistantContent::Final(final_) => usage = Some(final_.usage),
                other => panic!("unexpected item: {other:?}"),
            }
        }
        assert_eq!(texts, vec!["hello"]);
        let usage = usage.expect("final record");
        assert_eq!(usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn stub_completion_request_builder_works() {
        let stub = StubProvider::new("stub", "pong");
        let request = CompletionRequestBuilder::new(stub.clone(), "ping").build();
        let response = stub.completion(request).await.expect("completion");
        match &response.choice[0] {
            AssistantContent::Text(text) => assert_eq!(text.text, "pong"),
            other => panic!("expected text, got {other:?}"),
        }
    }
}

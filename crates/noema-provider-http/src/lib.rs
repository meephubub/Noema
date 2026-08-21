//! OpenAI-compatible HTTP model provider for Noema cloud escalation.
//!
//! [`OpenAICompatibleProvider`] implements
//! [`noema_core::ModelProvider`] over the widely-supported OpenAI
//! chat-completions protocol, so a single implementation reaches Gemini
//! (via its OpenAI-compatible endpoint), OpenAI, LocalAI, Ollama, vLLM, and
//! any other server that speaks that protocol. It is configured with three
//! things and nothing else:
//!
//! * the **model name** (e.g. `gemini-2.5-pro` or `gpt-4o`),
//! * the **base URL** (e.g. `https://generativelanguage.googleapis.com/v1beta/openai`),
//! * the **API key** (`None` is fine for local endpoints that need no auth).
//!
//! Streaming is optional ([`OpenAICompatibleProvider::with_streaming`]);
//! both paths honour the session's cancellation token.
//!
//! # Example
//!
//! ```
//! use noema_core::ModelProvider;
//! use noema_provider_http::OpenAICompatibleProvider;
//!
//! let provider = OpenAICompatibleProvider::new(
//!     "openai",                       // provider id (the policy's preferred_provider)
//!     "gpt-4o",                       // model name
//!     "https://api.openai.com/v1",    // base URL
//!     Some("sk-...".into()),          // API key
//! );
//! assert_eq!(provider.id(), "openai");
//! assert_eq!(provider.model(), "gpt-4o");
//! assert_eq!(provider.api_key(), Some("sk-..."));
//! ```

use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use noema_core::{
    ContentPart, ModelChunk, ModelProvider, ModelRequest, ModelResponse, NoemaError, Result, Role,
    Usage,
};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// A [`ModelProvider`] speaking the OpenAI-compatible chat-completions
/// protocol.
#[derive(Debug, Clone)]
pub struct OpenAICompatibleProvider {
    id: String,
    model: String,
    base_url: String,
    api_key: Option<String>,
    streaming: bool,
    timeout: Option<Duration>,
    client: reqwest::Client,
}

impl OpenAICompatibleProvider {
    /// A provider for the given id, model name, base URL, and API key.
    ///
    /// The id is what the escalation policy's `preferred_provider` refers
    /// to. `api_key` may be `None` for endpoints that need no
    /// authentication.
    pub fn new(
        id: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            model: model.into(),
            base_url: base_url.into(),
            api_key,
            streaming: false,
            timeout: None,
            client: reqwest::Client::new(),
        }
    }

    /// Streams the completion over the event bus when `true`.
    ///
    /// Defaults to `false` (a single `ModelResponse::Text`). Streaming uses
    /// the protocol's server-sent-events mode and yields one
    /// [`ModelChunk`] per content delta.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Caps the whole completion (request and, when streaming, the full
    /// stream) at the given duration. Applies in addition to any policy
    /// latency limit enforced by the session.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// The model name served by this provider.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The provider's base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The API key, when one is configured.
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl ModelProvider for OpenAICompatibleProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(
        &self,
        request: ModelRequest,
        cancel: CancellationToken,
    ) -> Result<ModelResponse> {
        let body = build_chat_body(&self.model, &request, self.streaming);
        if self.streaming {
            self.stream_chat(body, cancel)
        } else {
            self.chat_once(body, cancel).await
        }
    }
}

impl OpenAICompatibleProvider {
    /// Non-streaming completion: one HTTP request, one text response.
    async fn chat_once(&self, body: ChatBody, cancel: CancellationToken) -> Result<ModelResponse> {
        let mut builder = self.client.post(self.endpoint()).json(&body);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        if let Some(timeout) = self.timeout {
            builder = builder.timeout(timeout);
        }

        let response = tokio::select! {
            result = builder.send() => result,
            _ = cancel.cancelled() => {
                return Err(NoemaError::Model(format!(
                    "cloud completion cancelled (provider '{}')",
                    self.id
                )));
            }
        }
        .map_err(|error| {
            NoemaError::Model(format!("provider '{}' request failed: {error}", self.id))
        })?;

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| NoemaError::Model(format!("provider '{}' read failed: {error}", self.id)))?;
        if !status.is_success() {
            let detail = String::from_utf8_lossy(&bytes);
            return Err(NoemaError::Model(format!(
                "provider '{}' returned {status}: {detail}",
                self.id
            )));
        }

        let parsed: ChatResponse = serde_json::from_slice(&bytes).map_err(|error| {
            NoemaError::Model(format!(
                "provider '{}' returned invalid JSON: {error}",
                self.id
            ))
        })?;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .unwrap_or_default();
        let usage = parsed.usage.map(|usage| Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        });
        Ok(ModelResponse::Text { content, usage })
    }

    /// Streaming completion via the protocol's server-sent-events mode.
    ///
    /// Yields one [`ModelChunk`] per content delta. The returned stream is
    /// cancellable mid-flight and carries no borrow of `self`.
    fn stream_chat(&self, body: ChatBody, cancel: CancellationToken) -> Result<ModelResponse> {
        let client = self.client.clone();
        let url = self.endpoint();
        let id = self.id.clone();
        let api_key = self.api_key.clone();
        let timeout = self.timeout;

        let stream = async_stream::try_stream! {
            let mut builder = client.post(url).json(&body);
            if let Some(key) = &api_key {
                builder = builder.bearer_auth(key);
            }
            if let Some(timeout) = timeout {
                builder = builder.timeout(timeout);
            }

            let response = tokio::select! {
                result = builder.send() => result.map_err(|error| {
                    NoemaError::Model(format!("provider '{id}' request failed: {error}"))
                }),
                _ = cancel.cancelled() => Err(NoemaError::Model(format!(
                    "cloud completion cancelled (provider '{id}')"
                ))),
            }?;

            let status = response.status();
            if !status.is_success() {
                let bytes = response.bytes().await.map_err(|error| {
                    NoemaError::Model(format!("provider '{id}' read failed: {error}"))
                })?;
                let detail = String::from_utf8_lossy(&bytes);
                Err(NoemaError::Model(format!(
                    "provider '{id}' returned {status}: {detail}"
                )))?;
                unreachable!();
            }

            let mut stream = response.bytes_stream();
            let mut buffer: Vec<u8> = Vec::new();
            loop {
                let bytes = tokio::select! {
                    _ = cancel.cancelled() => Err(NoemaError::Model(format!(
                        "cloud completion cancelled (provider '{id}')"
                    ))),
                    chunk = stream.next() => match chunk {
                        None => break,
                        Some(Ok(bytes)) => Ok(bytes),
                        Some(Err(error)) => Err(NoemaError::Model(format!(
                            "provider '{id}' stream failed: {error}"
                        ))),
                    },
                }?;
                buffer.extend_from_slice(&bytes);
                // Emit every complete line; keep the trailing partial line
                // for the next chunk.
                while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buffer.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line);
                    let line = line.trim();
                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }
                    let payload = match line.strip_prefix("data:") {
                        Some(payload) => payload.trim(),
                        None => continue,
                    };
                    if payload == "[DONE]" {
                        break;
                    }
                    if let Ok(chunk) = serde_json::from_str::<StreamChunk>(payload) {
                        if let Some(delta) = chunk
                            .choices
                            .into_iter()
                            .next()
                            .and_then(|choice| choice.delta.content)
                        {
                            if !delta.is_empty() {
                                yield ModelChunk::new(delta);
                            }
                        }
                    }
                }
            }
        };

        Ok(ModelResponse::Stream(Box::pin(stream)))
    }
}

/// The chat-completions request body.
#[derive(Debug, Serialize)]
struct ChatBody {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

/// Requests usage metadata in the final streaming chunk.
#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// One chat message.
#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// The non-streaming chat-completions response.
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// One server-sent event of a streaming response.
#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

/// Builds the request body for a [`ModelRequest`].
fn build_chat_body(model: &str, request: &ModelRequest, streaming: bool) -> ChatBody {
    let mut messages = Vec::with_capacity(request.messages.len() + 1);
    if let Some(system) = &request.system {
        messages.push(ChatMessage {
            role: "system".into(),
            content: system.clone(),
        });
    }
    for message in &request.messages {
        messages.push(ChatMessage {
            role: role_str(message.role).into(),
            content: parts_text(&message.content),
        });
    }
    ChatBody {
        model: model.into(),
        messages,
        stream: streaming,
        temperature: request.options.temperature,
        max_tokens: request.options.max_tokens,
        top_p: request.options.top_p,
        stream_options: if streaming {
            Some(StreamOptions { include_usage: true })
        } else {
            None
        },
    }
}

/// The OpenAI wire role for a Noema role.
fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Flattens multimodal content to text for the wire; non-text parts become
/// placeholders.
fn parts_text(parts: &[ContentPart]) -> String {
    let mut text = String::new();
    for part in parts {
        match part {
            ContentPart::Text(text_part) => text.push_str(text_part),
            ContentPart::Image(_) => text.push_str("[image content]"),
            ContentPart::Audio(_) => text.push_str("[audio content]"),
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use noema_core::{Message, ModelOptions};
    use std::sync::{Arc, Mutex};

    /// Serves a single canned HTTP response and captures the request line /
    /// headers / body so tests can assert on what was sent.
    async fn spawn_server(
        response: Vec<u8>,
    ) -> (tokio::task::JoinHandle<()>, String, Arc<Mutex<Option<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let handle = tokio::spawn({
            let captured = Arc::clone(&captured);
            async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            loop {
                let n = tokio::io::AsyncReadExt::read(&mut socket, &mut tmp)
                    .await
                    .expect("read");
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                if let Some(header_end) = find_header_end(&buf) {
                    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                    let cl = content_length(&head);
                    match cl {
                        Some(len) if buf.len() >= header_end + len => break,
                        None => break,
                        _ => {}
                    }
                }
            }
                *captured.lock().unwrap() = Some(String::from_utf8_lossy(&buf).to_string());
                let _ = tokio::io::AsyncWriteExt::write_all(&mut socket, &response).await;
                let _ = tokio::io::AsyncWriteExt::shutdown(&mut socket).await;
            }
        });
        (handle, format!("http://{addr}"), captured)
    }

    fn find_header_end(buf: &[u8]) -> Option<usize> {
        buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
    }

    fn content_length(head: &str) -> Option<usize> {
        head.lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    value.trim().parse().ok()
                } else {
                    None
                }
            })
    }

    fn provider(base_url: &str, streaming: bool) -> OpenAICompatibleProvider {
        OpenAICompatibleProvider::new("openai", "test-model", base_url, Some("test-key".into()))
            .with_streaming(streaming)
    }

    fn request() -> ModelRequest {
        ModelRequest::new(vec![Message::text(Role::User, "hello")])
            .with_system("be brief")
            .with_options(ModelOptions {
                temperature: Some(0.5),
                ..Default::default()
            })
    }

    #[tokio::test]
    async fn non_streaming_completion_parses_text_and_usage() {
        let body = serde_json::json!({
            "choices": [{ "message": { "content": "hello world" } }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 },
        });
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.to_string().len(),
            body
        )
        .into_bytes();
        let (handle, base_url, captured) = spawn_server(response).await;
        let provider = provider(&base_url, false);

        match provider.complete(request(), CancellationToken::new()).await {
            Ok(ModelResponse::Text { content, usage }) => {
                assert_eq!(content, "hello world");
                let usage = usage.expect("usage parsed");
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 5);
            }
            other => panic!("expected text response, got {other:?}"),
        }
        handle.await.expect("server finished");

        let request_text = captured.lock().unwrap().clone().expect("request captured");
        assert!(
            request_text.contains("Bearer test-key"),
            "auth header missing; captured request:\n{request_text}"
        );
        assert!(request_text.contains("\"model\":\"test-model\""));
        assert!(request_text.contains("\"stream\":false"));
        assert!(request_text.contains("\"temperature\":0.5"));
        assert!(request_text.contains("\"role\":\"system\""));
    }

    #[tokio::test]
    async fn streaming_completion_yields_chunks() {
        let events = [
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo \"}}]}",
            "data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
            "data: [DONE]",
        ]
        .join("\n\n");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
            events.len(),
            events
        )
        .into_bytes();
        let (handle, base_url, _captured) = spawn_server(response).await;
        let provider = provider(&base_url, true);

        let mut response = provider.complete(request(), CancellationToken::new()).await.expect("complete");
        let ModelResponse::Stream(stream) = &mut response else {
            panic!("expected a stream");
        };
        let mut deltas = Vec::new();
        while let Some(chunk) = stream.next().await {
            deltas.push(chunk.expect("chunk").delta);
        }
        assert_eq!(deltas, vec!["Hel", "lo ", "world"]);
        handle.await.expect("server finished");
    }

    #[tokio::test]
    async fn error_status_surfaces_the_provider_message() {
        let body = r#"{"error":{"message":"invalid api key"}}"#;
        let response = format!(
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        )
        .into_bytes();
        let (handle, base_url, _captured) = spawn_server(response).await;
        let provider = provider(&base_url, false);

        let error = provider
            .complete(request(), CancellationToken::new())
            .await
            .expect_err("401 fails");
        assert!(matches!(error, NoemaError::Model(_)));
        assert!(error.to_string().contains("401"), "status in error: {error}");
        handle.await.expect("server finished");
    }

    #[tokio::test]
    async fn cancellation_aborts_the_request() {
        // A server that never answers keeps the request pending until the
        // cancellation token fires.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            // Accept and then just hold the connection open without
            // responding.
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut tmp = [0u8; 4096];
            while let Ok(n) = tokio::io::AsyncReadExt::read(&mut socket, &mut tmp).await {
                if n == 0 {
                    break;
                }
            }
        });
        let provider = OpenAICompatibleProvider::new(
            "openai",
            "test-model",
            format!("http://{addr}"),
            None,
        );
        let cancel = CancellationToken::new();
        let handle = tokio::spawn({
            let provider = provider.clone();
            let cancel = cancel.clone();
            async move { provider.complete(request(), cancel).await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("completion finished")
            .expect("task ok");
        let error = result.expect_err("cancelled");
        assert!(matches!(error, NoemaError::Model(_)));
        assert!(error.to_string().contains("cancelled"));
        server.abort();
    }

    #[test]
    fn body_builds_system_and_messages_in_order() {
        let request = request();
        let body = build_chat_body("m", &request, true);
        assert_eq!(body.messages[0].role, "system");
        assert_eq!(body.messages[1].role, "user");
        assert_eq!(body.messages[1].content, "hello");
        assert!(body.stream);
        assert!(body.stream_options.as_ref().unwrap().include_usage);
    }

    #[test]
    fn multimodal_parts_become_placeholders() {
        let parts = vec![
            ContentPart::text("see "),
            ContentPart::image(vec![1, 2, 3], "image/png"),
            ContentPart::audio(vec![4, 5], "audio/wav"),
        ];
        assert_eq!(parts_text(&parts), "see [image content][audio content]");
    }
}

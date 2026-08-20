//! Basic Noema example.
//!
//! Starts a Noema runtime with a mock model, creates a session, sends a
//! message, and prints the events streamed out of the runtime.
//!
//! Run with: `cargo run -p noema-example-basic`

use std::time::Duration;

use noema_api::prelude::*;
use tokio_util::sync::CancellationToken;

/// A stand-in for a real model backend (Gemma, Needle, cloud).
///
/// Real models implement the same [`Model`] trait and are registered the
/// same way: `Noema::builder().with_model(model)`.
#[derive(Debug)]
struct MockModel;

#[async_trait::async_trait]
impl Model for MockModel {
    fn id(&self) -> &str {
        "mock"
    }

    async fn generate(
        &self,
        request: ModelRequest,
        cancel: CancellationToken,
    ) -> Result<ModelResponse> {
        let prompt = request
            .messages
            .first()
            .and_then(|m| match m.content.first() {
                Some(ContentPart::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();

        // Simulate a streaming model: a short delay, then one chunk per word.
        let words: Vec<String> = prompt.split_whitespace().map(str::to_string).collect();
        let chunks: Vec<Result<ModelChunk>> = if words.is_empty() {
            vec![Ok(ModelChunk::new("Hello from the mock model!"))]
        } else {
            words
                .iter()
                .map(|w| Ok(ModelChunk::new(format!("{w} "))))
                .collect()
        };

        // Honours cancellation, as every real model must.
        if cancel.is_cancelled() {
            return Err(NoemaError::Model("generation cancelled".into()));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;

        Ok(ModelResponse::Stream(Box::pin(tokio_stream::iter(chunks))))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging(LogLevel::Info)?;

    let noema = Noema::builder()
        .with_model(MockModel)
        .build()
        .await?;
    println!(
        "noema runtime started (model = {:?}, offline_mode = {})",
        noema.model().map(|m| m.id()),
        noema.config().offline_mode
    );

    let session = noema.create_session().await?;
    println!("created session {}", session.id());

    let mut events = session.events();

    let response = session
        .send(Message::text(Role::User, "hello from the basic example"))
        .await?;

    match response {
        ModelResponse::Text { content, usage } => {
            println!("assistant: {content}");
            if let Some(usage) = usage {
                println!("usage: {usage:?}");
            }
        }
        other => println!("model responded: {other:?}"),
    }

    session.close().await?;
    println!("closed session {}", session.id());

    println!("--- session events ---");
    while let Some(event) = events.next().await {
        println!("{event:?}");
    }

    println!("done");
    Ok(())
}

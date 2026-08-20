//! Runs a local Gemma 4 conversation through the Noema runtime and through
//! Rig's completion interface.
//!
//! Usage:
//!
//! ```sh
//! cargo run -p gemma-example
//! ```
//!
//! The model file is resolved from `NOEMA_GEMMA_MODEL` or
//! `models/gemma-4-E2B-it.litertlm`. The LiteRT-LM DLLs are staged next to
//! the executable automatically at build time.

use noema_core::{
    init_logging, ContentPart, LogLevel, Message, Model, ModelResponse, Noema, Role, SendOutcome,
};
use noema_gemma::GemmaModel;
use noema_rig::NoemaCompletionModel;
use rig_core::completion::{AssistantContent, CompletionModel, CompletionRequestBuilder};
use rig_core::streaming::StreamedAssistantContent;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging(LogLevel::Info)?;

    let gemma = std::sync::Arc::new(GemmaModel::from_default()?);
    println!("gemma engine loaded: {}\n", gemma.id());

    println!("--- through the Noema runtime (streamed) ---");
    let noema = Noema::builder()
        .with_model(std::sync::Arc::clone(&gemma))
        .build()
        .await?;
    let session = noema.create_session().await?;

    for prompt in [
        "My name is Zorp and my favorite color is teal.",
        "What is my name?",
        "What is my favorite color?",
    ] {
        println!(">>> {prompt}");
        let outcome = session.send(Message::text(Role::User, prompt)).await?;
        match outcome {
            SendOutcome::Routed(action) => println!("<<< routed to {}\n", action.id),
            SendOutcome::Model(ModelResponse::Text { content, .. }) => {
                println!("<<< {content}");
                let usage = gemma.last_usage();
                if let Some(usage) = usage {
                    println!(
                        "    ({} in / {} out tokens)\n",
                        usage.input_tokens, usage.output_tokens
                    );
                } else {
                    println!();
                }
            }
            SendOutcome::Model(ModelResponse::Escalate(request)) => {
                println!("<<< escalated: {}\n", request.reason)
            }
            SendOutcome::Model(other) => println!("<<< (streamed) {other:?}\n"),
        }
    }
    session.close().await?;

    println!("--- multimodal turns ---");
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/noema-gemma/tests/fixtures");
    let image = std::fs::read(fixtures.join("red.png"))?;
    let audio = std::fs::read(fixtures.join("tone.wav"))?;

    let session = noema.create_session().await?;
    println!(">>> [image] What color is this image? Answer in one word.");
    let outcome = session
        .send(Message::new(
            Role::User,
            vec![
                ContentPart::text("What color is this image? Answer in one word."),
                ContentPart::image(image, "image/png"),
            ],
        ))
        .await?;
    if let SendOutcome::Model(ModelResponse::Text { content, .. }) = outcome {
        println!("<<< {content}\n");
    }

    println!(">>> [audio] What did you hear in this audio?");
    let outcome = session
        .send(Message::new(
            Role::User,
            vec![
                ContentPart::text("What did you hear in this audio?"),
                ContentPart::audio(audio, "audio/wav"),
            ],
        ))
        .await?;
    if let SendOutcome::Model(ModelResponse::Text { content, .. }) = outcome {
        println!("<<< {content}\n");
    }
    session.close().await?;

    println!("--- through Rig's completion interface ---");
    let gemma = GemmaModel::from_default()?;
    let model = NoemaCompletionModel::new(std::sync::Arc::new(gemma)).with_provider("gemma-4");

    // A plain completion request.
    let request = CompletionRequestBuilder::new(model.clone(), "Say hello in one short sentence.")
        .build();
    let response = model.completion(request).await?;
    for item in response.choice {
        if let AssistantContent::Text(text) = item {
            println!("<<< {}", text.text);
        }
    }
    println!(
        "    ({} in / {} out tokens)\n",
        response.usage.input_tokens, response.usage.output_tokens
    );

    // A streamed completion request.
    println!(">>> stream: count from one to three.");
    let request = CompletionRequestBuilder::new(model.clone(), "Count from one to three.").build();
    let mut stream = model.stream(request).await?;
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Text(text) = item? {
            print!("{}", text.text);
        }
    }
    println!();

    Ok(())
}

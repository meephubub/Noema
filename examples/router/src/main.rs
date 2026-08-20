//! The initial text router end-to-end example.
//!
//! Every plain-text request goes through Needle 2 first:
//!
//! ```text
//! "open my flashcards"          → routed → open_flashcards (Gemma never runs)
//! "what is the capital of france?" → escalated → Gemma 4 answers
//! ```
//!
//! Usage:
//!
//! ```sh
//! cargo run -p router-example
//! ```
//!
//! Needs the Needle engine (`prebuilt/needle/`) and the Gemma model
//! (`models/` or `NOEMA_GEMMA_MODEL`).

use noema_core::{init_logging, LogLevel, Message, ModelResponse, Noema, Role, SendOutcome};
use noema_events::Event;
use noema_gemma::GemmaModel;
use noema_router::NeedleRouter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging(LogLevel::Info)?;

    let gemma = GemmaModel::from_default()?;
    let router = NeedleRouter::from_default()?;

    let noema = Noema::builder()
        .with_model(gemma)
        .with_router(router)
        .build()
        .await?;
    println!(
        "noema started (model = {}, router = {})\n",
        noema.model().map(|m| m.id().to_string()).unwrap_or_default(),
        noema.router().map(|r| r.id().to_string()).unwrap_or_default(),
    );

    let session = noema.create_session().await?;
    let mut events = session.events();

    for prompt in [
        "Open my flashcards",
        "Show me my notes",
        "What is the capital of France?",
        "Go to settings",
    ] {
        println!(">>> {prompt}");
        let outcome = session.send(Message::text(Role::User, prompt)).await?;
        match outcome {
            SendOutcome::Routed(action) => {
                println!(
                    "<<< routed → {} (confidence {:?})\n",
                    action.id, action.confidence
                );
            }
            SendOutcome::Model(response) => {
                let text = match response {
                    ModelResponse::Text { content, .. } => content,
                    other => format!("{other:?}"),
                };
                println!("<<< escalated → Gemma: {text}\n");
            }
        }
    }
    session.close().await?;

    println!("--- routing events from the session ---");
    while let Some(event) = events.next().await {
        match event {
            Event::RoutingStarted { .. }
            | Event::RoutingCompleted { .. }
            | Event::RoutingEscalated { .. } => println!("{event:?}"),
            _ => {}
        }
        if matches!(event, Event::SessionCompleted { .. }) {
            break;
        }
    }

    Ok(())
}

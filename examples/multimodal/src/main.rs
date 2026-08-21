//! Phase 12: the multimodal agent, end to end.
//!
//! Audio/image → Gemma 4 → reasoning → tools → response. This example walks
//! the plan's multimodal paths on the real engine:
//!
//! 1. **Mixed text/image** — a question plus an image (red.png).
//! 2. **Mixed text/audio** — a question plus an audio clip (tone.wav).
//! 3. **Multimodal → tool workflow** — an image turn whose reasoning leads
//!    to a tool call: the agent loop detects the semantic tool request,
//!    Needle formats it, and the filesearch tool executes — all from a
//!    single `session.send`.
//!
//! The E2B checkpoint has a working vision channel but no audio channel:
//! the audio turn is accepted and flows through the same path, and the
//! model declines gracefully. A future audio-capable checkpoint answers
//! directly with no code changes. Likewise the small checkpoint is not
//! reliably agentic, so the tool step is best-effort: if Gemma names the
//! tool the loop runs it, otherwise Gemma answers directly.
//!
//! Usage:
//!
//! ```sh
//! cargo run -p multimodal-example
//! ```
//!
//! Needs `models/gemma-4-E2B-it.litertlm` (or `NOEMA_GEMMA_MODEL`) and the
//! Needle engine (`prebuilt/needle/`).

use noema_core::{
    init_logging, ContentPart, LogLevel, Message, ModelResponse, Noema, Role, SendOutcome,
};
use noema_events::Event;
use noema_filesearch::{Filesearch, TOOL_NAME};
use noema_gemma::GemmaModel;
use noema_router::NeedleToolFormatter;
use noema_tools::ToolRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging(LogLevel::Info)?;

    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/noema-gemma/tests/fixtures");
    let image = std::fs::read(fixtures.join("red.png"))?;
    let audio = std::fs::read(fixtures.join("tone.wav"))?;

    // Register the filesearch tool and its logical Needle agent, so the
    // multimodal turn below can drive a real tool call.
    let mut registry = ToolRegistry::new();
    registry.register(Filesearch::default())?;
    let schema = registry.get(TOOL_NAME).expect(TOOL_NAME).schema();

    let noema = Noema::builder()
        .with_model(GemmaModel::from_default()?)
        .with_tools(registry.clone())
        .with_tool_formatter_for(TOOL_NAME, NeedleToolFormatter::from_tool(&schema, None)?)
        .build()
        .await?;
    let session = noema.create_session().await?;

    // 1. Mixed text/image.
    println!("=== 1. mixed text/image ===\n");
    println!(">>> What color is this image? Answer in one word.");
    let outcome = session
        .send(Message::new(
            Role::User,
            vec![
                ContentPart::text("What color is this image? Answer in one word."),
                ContentPart::image(image.clone(), "image/png"),
            ],
        ))
        .await?;
    if let SendOutcome::Model(ModelResponse::Text { content, .. }) = outcome {
        println!("<<< {content}\n");
    }

    // 2. Mixed text/audio. The current checkpoint has no audio channel, so
    //    the model declines gracefully — the turn still flows end-to-end.
    println!("=== 2. mixed text/audio ===\n");
    println!(">>> What did you hear in this audio clip?");
    let outcome = session
        .send(Message::new(
            Role::User,
            vec![
                ContentPart::text("What did you hear in this audio clip?"),
                ContentPart::audio(audio, "audio/wav"),
            ],
        ))
        .await?;
    if let SendOutcome::Model(ModelResponse::Text { content, .. }) = outcome {
        println!("<<< {content}\n");
    }

    // 3. Multimodal → tool workflow: the image turn's reasoning leads to a
    //    filesearch call inside one session.send.
    println!("=== 3. image turn → tool workflow ===\n");
    let study_dir = std::env::current_dir().expect("cwd").join(".noema-multimodal-demo");
    std::fs::create_dir_all(&study_dir)?;
    let image_note = study_dir.join("image-notes.txt");
    if !image_note.exists() {
        std::fs::write(&image_note, "the red square image note")?;
    }

    println!(">>> [image attached] find the image-notes.txt file in this folder");
    let outcome = session
        .send(Message::new(
            Role::User,
            vec![
                ContentPart::text("Find the image-notes.txt file in this folder and tell me its path."),
                ContentPart::image(image, "image/png"),
            ],
        ))
        .await?;

    let mut tool_steps = 0usize;
    let mut events = session.events();
    println!("--- event stream ---");
    while let Ok(Some(event)) =
        tokio::time::timeout(std::time::Duration::from_millis(250), events.next()).await
    {
        match event {
            Event::ModelStarted { model, .. } => println!("model: {model} started"),
            Event::ModelDelta { delta, .. } => print!("{delta}"),
            Event::ModelCompleted { .. } => println!(),
            Event::ToolRequested { .. } => {
                tool_steps += 1;
                println!("tool: semantic request detected");
            }
            Event::ToolFormatted { .. } => println!("tool: formatted into a structured call"),
            Event::ToolStarted { .. } => println!("tool: executing…"),
            Event::ToolCompleted { .. } => println!("tool: completed"),
            Event::ToolFailed { .. } => println!("tool: FAILED"),
            Event::Error { error, .. } => println!("error: {error}"),
            _ => {}
        }
    }

    match outcome {
        SendOutcome::Model(ModelResponse::Text { content, .. }) => {
            println!("\n<<< final answer: {content}");
        }
        SendOutcome::Model(other) => println!("\n<<< {other:?}"),
        SendOutcome::Routed(action) => println!("\n<<< routed to {}", action.id),
    }
    if tool_steps > 0 {
        println!("\nThe image turn drove {tool_steps} tool step(s) before answering.");
    } else {
        println!("\nThe small checkpoint answered directly (best-effort tool step, as documented).");
    }

    session.close().await?;
    std::fs::remove_dir_all(&study_dir).ok();
    Ok(())
}

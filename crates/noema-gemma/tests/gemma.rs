//! Real-inference integration tests for the Gemma 4 adapter.
//!
//! These tests load the actual model from `models/` (or `NOEMA_GEMMA_MODEL`)
//! and exercise LiteRT-LM through the vendored binding and through the
//! `GemmaModel` adapter itself. They are `#[ignore]`d by default because they
//! need the 2.5 GB model file and take tens of seconds; run them with:
//!
//! ```text
//! cargo test -p noema-gemma --test gemma -- --ignored --nocapture
//! ```

use litert_lm_rust::{
    ConversationConfig, Engine, Message, SamplerParams, SessionConfig, StreamEvent,
};
use noema_core::{ContentPart, Model, ModelRequest, ModelResponse, Role};
use noema_gemma::{default_model_path, GemmaModel};
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

/// A CPU-friendly conversation config with the top-p sampler this backend
/// supports.
fn config(system: Option<&str>) -> ConversationConfig {
    ConversationConfig {
        session: SessionConfig {
            max_output_tokens: Some(96),
            sampler: Some(SamplerParams::top_p(0.9).with_top_k(40).with_temperature(0.6)),
            ..Default::default()
        },
        system_message: system.map(|s| {
            serde_json::to_value(Message::system(s)).expect("system message serializes")
        }),
        ..Default::default()
    }
}

fn engine() -> Engine {
    let path = default_model_path().expect("set NOEMA_GEMMA_MODEL or put the model in models/");
    Engine::builder(path)
        .num_threads(4)
        .vision_backend(litert_lm_rust::Backend::Cpu)
        .audio_backend(litert_lm_rust::Backend::Cpu)
        .build()
        .expect("engine loads")
}

/// Extracts the plain-text delta from a streamed chunk. LiteRT-LM streams
/// serialized messages (`{"role":"assistant","content":[{"type":"text",...}]}`),
/// not raw text.
fn extract_delta(chunk_text: &str) -> Option<String> {
    Message::from_json_str(chunk_text)
        .ok()
        .and_then(|message| message.text())
}

/// Drains an engine-level stream receiver into plain text.
fn drain_receiver(receiver: litert_lm_rust::StreamEventReceiver) -> String {
    let mut text = String::new();
    for event in receiver.iter() {
        match event {
            StreamEvent::StartFailed(code) => panic!("stream failed to start: {code}"),
            StreamEvent::Chunk(chunk) => {
                if let Some(err) = chunk.error {
                    panic!("stream error: {err}");
                }
                if let Some(t) = chunk.text {
                    if let Some(delta) = extract_delta(&t) {
                        text.push_str(&delta);
                    }
                }
                if chunk.is_final {
                    break;
                }
            }
        }
    }
    text
}

/// Drains a trait-level stream into plain text, returning any error item.
/// (`Pin<Box<dyn Stream>>` is `Unpin`, so `next()` works directly.)
async fn drain_model_stream(
    mut stream: std::pin::Pin<
        Box<dyn tokio_stream::Stream<
            Item = std::result::Result<noema_core::ModelChunk, noema_core::NoemaError>,
        > + Send>,
    >,
) -> (String, Option<noema_core::NoemaError>) {
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) => text.push_str(&chunk.delta),
            Err(error) => return (text, Some(error)),
        }
    }
    (text, None)
}

/// Fixture bytes for multimodal turns: an 8x8 solid-red PNG and a 440 Hz
/// tone WAV, generated at `tests/fixtures/`.
fn fixtures() -> (Vec<u8>, Vec<u8>) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    (
        std::fs::read(dir.join("red.png")).expect("red.png fixture"),
        std::fs::read(dir.join("tone.wav")).expect("tone.wav fixture"),
    )
}

#[test]
#[ignore = "needs the Gemma model file"]
fn engine_accepts_image_and_audio_turns() {
    let engine = engine();
    let mut conversation = engine
        .create_conversation(config(Some("Be very brief.")))
        .expect("conversation");
    let (image, audio) = fixtures();

    let image_message = Message::user_parts([
        litert_lm_rust::ContentPart::text("What color is this image?"),
        litert_lm_rust::ContentPart::image_bytes(&image),
    ])
    .expect("image message");
    match conversation.send_message(image_message) {
        Ok(reply) => {
            let text = reply.text().unwrap_or_default();
            eprintln!("image turn reply: {text:?}");
            assert!(!text.trim().is_empty());
        }
        Err(error) => {
            eprintln!("image turn FAILED: {error}");
            panic!("model does not accept image input: {error}");
        }
    }

    let audio_message = Message::user_parts([
        litert_lm_rust::ContentPart::text("What did you hear in this audio?"),
        litert_lm_rust::ContentPart::audio_bytes(&audio),
    ])
    .expect("audio message");
    match conversation.send_message(audio_message) {
        Ok(reply) => {
            let text = reply.text().unwrap_or_default();
            eprintln!("audio turn reply: {text:?}");
            assert!(!text.trim().is_empty());
        }
        Err(error) => {
            eprintln!("audio turn FAILED: {error}");
            panic!("model does not accept audio input: {error}");
        }
    }
}

#[test]
#[ignore = "needs the Gemma model file"]
fn engine_stream_delivers_plain_text_deltas() {
    let engine = engine();
    let mut conversation = engine
        .create_conversation(config(Some("Be very brief.")))
        .expect("conversation");

    let receiver = conversation
        .send_message_stream(Message::user("Count from one to three."))
        .expect("stream starts");

    let text = drain_receiver(receiver);
    assert!(!text.trim().is_empty(), "expected generated text, got {text:?}");
    eprintln!("streamed text: {text:?}");
}

#[test]
#[ignore = "needs the Gemma model file"]
fn engine_replay_seeded_history_preserves_multi_turn_memory() {
    let engine = engine();

    // Turn 1: learn a fact with the blocking path (records the assistant turn).
    let mut first = engine
        .create_conversation(config(Some("Be very brief.")))
        .expect("conversation");
    let reply = first
        .send_message(Message::user("Remember: my name is Zorp."))
        .expect("turn 1");
    let first_reply = reply.text().expect("reply text");
    assert!(!first_reply.trim().is_empty());
    eprintln!("turn 1 reply: {first_reply:?}");

    // Turn 2: seed a fresh conversation with the recorded history (preface
    // messages) and stream the follow-up.
    let history = serde_json::json!([
        Message::user("Remember: my name is Zorp."),
        Message::model(&first_reply),
    ]);
    let mut second = engine
        .create_conversation(ConversationConfig {
            messages: Some(history),
            ..config(Some("Be very brief."))
        })
        .expect("seeded conversation");

    let receiver = second
        .send_message_stream(Message::user("What is my name?"))
        .expect("stream starts");
    let reply_text = drain_receiver(receiver);
    eprintln!("turn 2 reply: {reply_text:?}");
    assert!(
        reply_text.to_lowercase().contains("zorp"),
        "seeded conversation should remember the name, got: {reply_text:?}"
    );
}

#[tokio::test]
#[ignore = "needs the Gemma model file"]
async fn gemma_model_streams_and_reports_usage() {
    let model = GemmaModel::from_default().expect("model loads");
    let response = model
        .generate(
            ModelRequest::new(vec![noema_core::Message::text(Role::User, "Say hello.")]),
            CancellationToken::new(),
        )
        .await
        .expect("generate");

    match response {
        ModelResponse::Stream(stream) => {
            let (text, error) = drain_model_stream(stream).await;
            assert!(error.is_none(), "stream errored: {error:?}");
            assert!(!text.trim().is_empty(), "expected generated text, got {text:?}");
            let usage = model.last_usage().expect("usage reported");
            assert!(usage.output_tokens > 0, "expected output tokens, got {usage:?}");
            eprintln!("reply: {text:?} usage: {usage:?}");
        }
        other => panic!("expected a stream, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "needs the Gemma model file"]
async fn gemma_model_keeps_multi_turn_memory() {
    let model = GemmaModel::from_default().expect("model loads");

    // The adapter is request-driven: each request carries the conversation,
    // so the caller accumulates it (the session does this in real use).
    let mut history: Vec<noema_core::Message> = Vec::new();

    async fn turn(
        model: &GemmaModel,
        history: &mut Vec<noema_core::Message>,
        text: &str,
    ) -> String {
        history.push(noema_core::Message::text(Role::User, text));
        let response = model
            .generate(
                ModelRequest::new(history.clone()),
                CancellationToken::new(),
            )
            .await
            .expect("generate");
        let reply = match response {
            ModelResponse::Stream(stream) => {
                let (text, error) = drain_model_stream(stream).await;
                assert!(error.is_none(), "stream errored: {error:?}");
                text
            }
            other => panic!("expected a stream, got {other:?}"),
        };
        history.push(noema_core::Message::text(Role::Assistant, &reply));
        reply
    }

    turn(&model, &mut history, "Remember: my name is Zorp.").await;
    let reply = turn(&model, &mut history, "What is my name?").await;
    eprintln!("follow-up reply: {reply:?}");
    assert!(
        reply.to_lowercase().contains("zorp"),
        "model should remember the name across streamed turns, got: {reply:?}"
    );
}

#[tokio::test]
#[ignore = "needs the Gemma model file"]
async fn gemma_model_understands_images() {
    let model = GemmaModel::from_default().expect("model loads");
    let (image, _) = fixtures();

    let response = model
        .generate(
            ModelRequest::new(vec![noema_core::Message::new(
                Role::User,
                vec![
                    ContentPart::text("What color is this image? Answer in one word."),
                    ContentPart::image(image, "image/png"),
                ],
            )]),
            CancellationToken::new(),
        )
        .await
        .expect("generate");

    match response {
        ModelResponse::Stream(stream) => {
            let (text, error) = drain_model_stream(stream).await;
            assert!(error.is_none(), "stream errored: {error:?}");
            assert!(
                text.to_lowercase().contains("red"),
                "expected the model to see a red image, got: {text:?}"
            );
            eprintln!("image reply: {text:?}");
        }
        other => panic!("expected a stream, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "needs the Gemma model file"]
async fn gemma_model_accepts_audio_turns() {
    let model = GemmaModel::from_default().expect("model loads");
    let (_, audio) = fixtures();

    let response = model
        .generate(
            ModelRequest::new(vec![noema_core::Message::new(
                Role::User,
                vec![
                    ContentPart::text("What did you hear in this audio?"),
                    ContentPart::audio(audio, "audio/wav"),
                ],
            )]),
            CancellationToken::new(),
        )
        .await
        .expect("generate");

    match response {
        ModelResponse::Stream(stream) => {
            // The audio path must not error — the current E2B checkpoint has
            // no audio channel, so the model declines gracefully; a future
            // audio-capable checkpoint should answer here.
            let (text, error) = drain_model_stream(stream).await;
            assert!(error.is_none(), "stream errored: {error:?}");
            assert!(!text.trim().is_empty(), "expected a reply, got empty");
            eprintln!("audio reply: {text:?}");
        }
        other => panic!("expected a stream, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "needs the Gemma model file"]
async fn gemma_model_cancellation_stops_generation() {
    let model = GemmaModel::from_default().expect("model loads");
    let cancel = CancellationToken::new();
    cancel.cancel();

    let response = model
        .generate(
            ModelRequest::new(vec![noema_core::Message::text(
                Role::User,
                "Write a very long essay about the history of paper.",
            )]),
            cancel,
        )
        .await
        .expect("generate");

    match response {
        ModelResponse::Stream(stream) => {
            let (_, error) = drain_model_stream(stream).await;
            assert!(
                error.is_some(),
                "a pre-cancelled generation should surface a model error"
            );
        }
        other => panic!("expected a stream, got {other:?}"),
    }
}

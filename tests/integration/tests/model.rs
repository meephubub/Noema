//! Integration tests for the model abstraction, exercised through the
//! public `noema-api` surface with mock models.

use std::time::Duration;

use noema_api::prelude::*;
use tokio_util::sync::CancellationToken;

/// Echoes the text content of the last user message back.
#[derive(Debug)]
struct EchoModel;

#[async_trait::async_trait]
impl Model for EchoModel {
    fn id(&self) -> &str {
        "echo"
    }

    async fn generate(
        &self,
        request: ModelRequest,
        _cancel: CancellationToken,
    ) -> Result<ModelResponse> {
        let content = request
            .messages
            .iter()
            .rev()
            .find_map(|m| match m.content.first() {
                Some(ContentPart::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();
        Ok(ModelResponse::Text { content, usage: None })
    }
}

/// Streams a fixed sequence of chunks.
#[derive(Debug)]
struct StreamingModel;

#[async_trait::async_trait]
impl Model for StreamingModel {
    fn id(&self) -> &str {
        "streaming"
    }

    async fn generate(
        &self,
        _request: ModelRequest,
        _cancel: CancellationToken,
    ) -> Result<ModelResponse> {
        let stream = tokio_stream::iter(vec![
            Ok(ModelChunk::new("The ")),
            Ok(ModelChunk::new("quick ")),
            Ok(ModelChunk::new("fox")),
        ]);
        Ok(ModelResponse::Stream(Box::pin(stream)))
    }
}

/// Reports usage metadata.
#[derive(Debug)]
struct UsageModel;

#[async_trait::async_trait]
impl Model for UsageModel {
    fn id(&self) -> &str {
        "usage"
    }

    async fn generate(
        &self,
        _request: ModelRequest,
        _cancel: CancellationToken,
    ) -> Result<ModelResponse> {
        Ok(ModelResponse::Text {
            content: "done".into(),
            usage: Some(Usage {
                input_tokens: 12,
                output_tokens: 3,
            }),
        })
    }
}

/// Never finishes unless cancelled.
#[derive(Debug)]
struct HangingModel;

#[async_trait::async_trait]
impl Model for HangingModel {
    fn id(&self) -> &str {
        "hanging"
    }

    async fn generate(
        &self,
        _request: ModelRequest,
        cancel: CancellationToken,
    ) -> Result<ModelResponse> {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(60)) => {
                Ok(ModelResponse::Text { content: "done".into(), usage: None })
            }
            _ = cancel.cancelled() => {
                Err(NoemaError::Model("cancelled".into()))
            }
        }
    }
}

/// Requests escalation to a larger model.
#[derive(Debug)]
struct EscalatingModel;

#[async_trait::async_trait]
impl Model for EscalatingModel {
    fn id(&self) -> &str {
        "escalator"
    }

    async fn generate(
        &self,
        _request: ModelRequest,
        _cancel: CancellationToken,
    ) -> Result<ModelResponse> {
        Ok(ModelResponse::Escalate(EscalationRequest::new(
            "requires substantially larger reasoning capacity",
            vec![Message::text(Role::User, "hard problem")],
        )))
    }
}

async fn runtime_with(model: impl Model) -> Noema {
    Noema::builder().with_model(model).build().await.expect("build runtime")
}

#[tokio::test]
async fn session_sends_message_to_model() {
    let noema = runtime_with(EchoModel).await;
    let session = noema.create_session().await.expect("create session");

    let response = session
        .send(Message::text(Role::User, "hello from agora"))
        .await
        .expect("send")
        .into_model()
        .expect("model outcome");

    match response {
        ModelResponse::Text { content, .. } => assert_eq!(content, "hello from agora"),
        _ => panic!("expected text response"),
    }
}

#[tokio::test]
async fn send_without_model_fails_cleanly() {
    let noema = Noema::builder().build().await.expect("build runtime");
    let session = noema.create_session().await.expect("create session");
    let result = session.send(Message::text(Role::User, "hi")).await;
    assert!(matches!(result, Err(NoemaError::Model(_))));
}

#[tokio::test]
async fn streaming_is_drained_and_emitted_as_deltas() {
    let noema = runtime_with(StreamingModel).await;
    let session = noema.create_session().await.expect("create session");
    let mut events = session.events();

    let response = session
        .send(Message::text(Role::User, "go"))
        .await
        .expect("send")
        .into_model()
        .expect("model outcome");

    match response {
        ModelResponse::Text { content, .. } => assert_eq!(content, "The quick fox"),
        _ => panic!("expected drained stream"),
    }

    let mut deltas = Vec::new();
    while let Some(event) = events.next().await {
        match event {
            Event::ModelDelta { delta, .. } => deltas.push(delta),
            Event::ModelCompleted { .. } => break,
            _ => {}
        }
    }
    assert_eq!(deltas, vec!["The ", "quick ", "fox"]);
}

#[tokio::test]
async fn usage_metadata_is_preserved() {
    let noema = runtime_with(UsageModel).await;
    let session = noema.create_session().await.expect("create session");

    let response = session
        .send(Message::text(Role::User, "go"))
        .await
        .expect("send")
        .into_model()
        .expect("model outcome");

    match response {
        ModelResponse::Text { usage, .. } => {
            let usage = usage.expect("usage reported");
            assert_eq!(usage.input_tokens, 12);
            assert_eq!(usage.output_tokens, 3);
            assert_eq!(usage.total(), 15);
        }
        _ => panic!("expected text response"),
    }
}

#[tokio::test]
async fn multimodal_messages_reach_the_model() {
    #[derive(Debug)]
    struct MultimodalModel;

    #[async_trait::async_trait]
    impl Model for MultimodalModel {
        fn id(&self) -> &str {
            "multimodal"
        }

        async fn generate(
            &self,
            request: ModelRequest,
            _cancel: CancellationToken,
        ) -> Result<ModelResponse> {
            let message = request.messages.first().expect("one message");
            let parts = message.content.len();
            let has_image = matches!(
                message.content.get(1),
                Some(ContentPart::Image(_))
            );
            let has_audio = matches!(
                message.content.get(2),
                Some(ContentPart::Audio(_))
            );
            Ok(ModelResponse::Text {
                content: format!("parts={parts} image={has_image} audio={has_audio}"),
                usage: None,
            })
        }
    }

    let noema = runtime_with(MultimodalModel).await;
    let session = noema.create_session().await.expect("create session");

    let message = Message::new(
        Role::User,
        vec![
            ContentPart::text("explain this"),
            ContentPart::image(vec![1, 2, 3], "image/png"),
            ContentPart::audio(vec![4, 5], "audio/wav"),
        ],
    );

    let response = session
        .send(message)
        .await
        .expect("send")
        .into_model()
        .expect("model outcome");
    match response {
        ModelResponse::Text { content, .. } => {
            assert_eq!(content, "parts=3 image=true audio=true");
        }
        _ => panic!("expected text response"),
    }
}

#[tokio::test]
async fn cancel_stops_generation() {
    let noema = runtime_with(HangingModel).await;
    let session = noema.create_session().await.expect("create session");

    let handle = tokio::spawn({
        let session = session.clone();
        async move { session.send(Message::text(Role::User, "go")).await }
    });

    tokio::time::sleep(Duration::from_millis(50)).await;
    session.cancel().await.expect("cancel");

    let result = tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("send finished")
        .expect("task ok");
    assert!(matches!(result, Err(NoemaError::Model(_))));
}

#[tokio::test]
async fn escalation_request_is_surfaced() {
    let noema = runtime_with(EscalatingModel).await;
    let session = noema.create_session().await.expect("create session");

    let response = session
        .send(Message::text(Role::User, "very hard question"))
        .await
        .expect("send")
        .into_model()
        .expect("model outcome");

    match response {
        ModelResponse::Escalate(request) => {
            assert!(request.reason.contains("reasoning capacity"));
            assert_eq!(request.context.len(), 1);
        }
        _ => panic!("expected escalation"),
    }
}

/// A router that handles "open my flashcards" and escalates everything else.
#[derive(Debug)]
struct FakeRouter;

#[async_trait::async_trait]
impl Router for FakeRouter {
    fn id(&self) -> &str {
        "fake-router"
    }

    async fn route(
        &self,
        text: &str,
        _cancel: CancellationToken,
    ) -> Result<Route> {
        if text == "open my flashcards" {
            Ok(Route::Action(RoutedAction::new("open_flashcards")))
        } else {
            Ok(Route::Escalate {
                reason: "not an app action".into(),
            })
        }
    }
}

#[tokio::test]
async fn router_handles_simple_actions_without_the_model() {
    let noema = Noema::builder()
        .with_model(EchoModel)
        .with_router(FakeRouter)
        .build()
        .await
        .expect("build runtime");
    let session = noema.create_session().await.expect("create session");
    let mut events = session.events();

    let outcome = session
        .send(Message::text(Role::User, "open my flashcards"))
        .await
        .expect("send");
    let action = outcome.into_routed().expect("routed outcome");
    assert_eq!(action.id, "open_flashcards");

    let mut saw_started = false;
    let mut saw_completed = false;
    while let Some(event) = events.next().await {
        match event {
            Event::RoutingStarted { .. } => saw_started = true,
            Event::RoutingCompleted { .. } => {
                saw_completed = true;
                break;
            }
            _ => {}
        }
    }
    assert!(saw_started && saw_completed);
}

#[tokio::test]
async fn router_escalation_flows_into_the_model() {
    let noema = Noema::builder()
        .with_model(EchoModel)
        .with_router(FakeRouter)
        .build()
        .await
        .expect("build runtime");
    let session = noema.create_session().await.expect("create session");
    let mut events = session.events();

    let outcome = session
        .send(Message::text(Role::User, "what is the capital of france"))
        .await
        .expect("send")
        .into_model()
        .expect("model outcome");
    match outcome {
        ModelResponse::Text { content, .. } => {
            assert_eq!(content, "what is the capital of france")
        }
        _ => panic!("expected echo"),
    }

    let mut saw_escalated = false;
    let mut saw_model = false;
    while let Some(event) = events.next().await {
        match event {
            Event::RoutingEscalated { .. } => saw_escalated = true,
            Event::ModelStarted { .. } => {
                saw_model = true;
                break;
            }
            _ => {}
        }
    }
    assert!(saw_escalated && saw_model);
}

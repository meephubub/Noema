//! Ephemeral session state.

use std::sync::Arc;
use std::time::SystemTime;

use noema_events::{Event, EventBus, EventStream, SessionId};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::error::{NoemaError, Result};
use crate::model::{ContentPart, Message, Model, ModelRequest, ModelResponse, Role};
use crate::router::{Route, Router, SendOutcome};

/// The lifecycle state of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// The session is open and accepting work.
    Active,
    /// The session has been closed and must not accept further work.
    Closed,
}

/// An ephemeral Noema session.
///
/// A session owns no persistent state; anything worth remembering belongs in
/// Mnemo. Sessions hold the current conversation and pending work only for
/// the lifetime of the session.
#[derive(Debug, Clone)]
pub struct Session {
    id: SessionId,
    created_at: SystemTime,
    state: Arc<Mutex<SessionState>>,
    events: EventBus,
    model: Option<Arc<dyn Model>>,
    router: Option<Arc<dyn Router>>,
    current_op: Arc<Mutex<Option<CancellationToken>>>,
}

impl Session {
    pub(crate) fn new(
        id: SessionId,
        events: EventBus,
        state: Arc<Mutex<SessionState>>,
        model: Option<Arc<dyn Model>>,
        router: Option<Arc<dyn Router>>,
    ) -> Self {
        Self {
            id,
            created_at: SystemTime::now(),
            state,
            events,
            model,
            router,
            current_op: Arc::new(Mutex::new(None)),
        }
    }

    /// The router registered on this session, if any.
    pub fn router(&self) -> Option<&Arc<dyn Router>> {
        self.router.as_ref()
    }

    /// The session's unique id.
    pub fn id(&self) -> &SessionId {
        &self.id
    }

    /// When the session was created.
    pub fn created_at(&self) -> SystemTime {
        self.created_at
    }

    /// The current session state.
    pub async fn state(&self) -> SessionState {
        *self.state.lock().await
    }

    /// Subscribes to the events belonging to this session.
    pub fn events(&self) -> EventStream {
        self.events.subscribe(self.id.clone())
    }

    /// Sends a message and returns what happened.
    ///
    /// Plain-text user requests are first offered to the session's router
    /// (Needle 2 in practice). When the router handles the request, the
    /// returned outcome is [`SendOutcome::Routed`] and the reasoning model is
    /// never invoked; [`Event::RoutingStarted`] and [`Event::RoutingCompleted`]
    /// are published. Otherwise the request escalates to the model
    /// ([`Event::RoutingEscalated`]) and the outcome is
    /// [`SendOutcome::Model`].
    ///
    /// Model turns are streamed through the event bus:
    /// [`Event::UserMessageReceived`], [`Event::ModelStarted`],
    /// [`Event::ModelDelta`] (per chunk), and [`Event::ModelCompleted`] are
    /// published as the model works. Streaming responses are drained into a
    /// complete [`ModelResponse::Text`] before returning.
    ///
    /// Requires a model to have been registered on the runtime via
    /// [`NoemaBuilder::with_model`](crate::NoemaBuilder::with_model).
    pub async fn send(&self, message: Message) -> Result<SendOutcome> {
        {
            let state = self.state.lock().await;
            if *state == SessionState::Closed {
                return Err(NoemaError::Session(format!(
                    "session {} is closed",
                    self.id
                )));
            }
        }

        let model = self.model.clone().ok_or_else(|| {
            NoemaError::Model(
                "no model registered with this runtime; use Noema::builder().with_model(..)"
                    .to_string(),
            )
        })?;

        self.events.publish(Event::UserMessageReceived {
            session_id: self.id.clone(),
        });

        let token = CancellationToken::new();
        *self.current_op.lock().await = Some(token.clone());

        // Initial text routing: plain-text user requests are offered to the
        // router first; handled requests never reach the reasoning model.
        if let Some(outcome) = self.route_message(&message, &token).await {
            *self.current_op.lock().await = None;
            return Ok(outcome);
        }

        let request = ModelRequest::new(vec![message]);
        self.events.publish(Event::ModelStarted {
            session_id: self.id.clone(),
            model: model.id().to_string(),
        });

        let result = model.generate(request, token).await;

        // The operation is over (success, error, or cancellation).
        *self.current_op.lock().await = None;

        let response = result?;
        let response = self.finish_response(response).await?;
        Ok(SendOutcome::Model(response))
    }

    /// Routes a plain-text user message through the registered router.
    ///
    /// Returns `Some(SendOutcome::Routed(..))` when the router handled the
    /// request, and `None` when it should escalate to the model (including
    /// non-text or non-user messages, no router registered, and router
    /// failures — which are surfaced as events but still escalate so the
    /// user gets an answer).
    async fn route_message(
        &self,
        message: &Message,
        token: &CancellationToken,
    ) -> Option<SendOutcome> {
        let router = self.router.as_ref()?;
        let text = match plain_user_text(message) {
            Some(text) => text,
            // Multimodal and non-user turns skip routing.
            None => return None,
        };

        self.events.publish(Event::RoutingStarted {
            session_id: self.id.clone(),
        });
        match router.route(&text, token.clone()).await {
            Ok(Route::Action(action)) => {
                self.events.publish(Event::RoutingCompleted {
                    session_id: self.id.clone(),
                });
                Some(SendOutcome::Routed(action))
            }
            Ok(Route::Escalate { .. }) => {
                self.events.publish(Event::RoutingEscalated {
                    session_id: self.id.clone(),
                });
                None
            }
            Err(error) => {
                tracing::warn!(router = %router.id(), error = %error, "router failed; escalating");
                self.events.publish(Event::RoutingEscalated {
                    session_id: self.id.clone(),
                });
                self.events.publish(Event::Error {
                    session_id: self.id.clone(),
                    error: error.to_string(),
                });
                None
            }
        }
    }

    /// Cancels the in-flight operation, if any.
    ///
    /// The model observes the cancellation and stops generating; the
    /// in-flight [`send`](Self::send) returns a model error.
    pub async fn cancel(&self) -> Result<()> {
        let current = self.current_op.lock().await;
        if let Some(token) = current.as_ref() {
            token.cancel();
        }
        Ok(())
    }

    /// Closes the session and emits [`Event::SessionCompleted`].
    ///
    /// Closing an already-closed session is an error.
    pub async fn close(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        if *state == SessionState::Closed {
            return Err(NoemaError::Session(format!(
                "session {} is already closed",
                self.id
            )));
        }
        *state = SessionState::Closed;
        self.events
            .publish(Event::SessionCompleted { session_id: self.id.clone() });
        Ok(())
    }

    async fn finish_response(&self, response: ModelResponse) -> Result<ModelResponse> {
        match response {
            ModelResponse::Text { content, usage } => {
                if !content.is_empty() {
                    self.events.publish(Event::ModelDelta {
                        session_id: self.id.clone(),
                        delta: content.clone(),
                    });
                }
                self.events
                    .publish(Event::ModelCompleted { session_id: self.id.clone() });
                Ok(ModelResponse::Text { content, usage })
            }
            ModelResponse::Stream(stream) => {
                let mut stream = Box::pin(stream);
                let mut content = String::new();
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(chunk) => {
                            if !chunk.delta.is_empty() {
                                content.push_str(&chunk.delta);
                                self.events.publish(Event::ModelDelta {
                                    session_id: self.id.clone(),
                                    delta: chunk.delta,
                                });
                            }
                        }
                        Err(error) => {
                            self.events.publish(Event::Error {
                                session_id: self.id.clone(),
                                error: error.to_string(),
                            });
                            return Err(error);
                        }
                    }
                }
                self.events
                    .publish(Event::ModelCompleted { session_id: self.id.clone() });
                Ok(ModelResponse::Text { content, usage: None })
            }
            // Escalation is surfaced to the caller; policy handling arrives
            // with the escalation milestone.
            other @ ModelResponse::Escalate(_) => Ok(other),
        }
    }
}

/// The message text when a message is a plain-text user turn, `None`
/// otherwise (multimodal messages and non-user roles skip routing).
fn plain_user_text(message: &Message) -> Option<String> {
    if message.role != Role::User {
        return None;
    }
    match message.content.as_slice() {
        [ContentPart::Text(text)] => Some(text.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ContentPart, ModelChunk, ModelRequest, ModelResponse, Role};
    use crate::router::RoutedAction;

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
            let text = request
                .messages
                .iter()
                .filter_map(|m| match m.content.first() {
                    Some(ContentPart::Text(t)) => Some(t.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            Ok(ModelResponse::Text {
                content: text,
                usage: None,
            })
        }
    }

    fn test_session(model: Option<Arc<dyn Model>>) -> Session {
        Session::new(
            SessionId::generate(),
            EventBus::default(),
            Arc::new(Mutex::new(SessionState::Active)),
            model,
            None,
        )
    }

    fn test_session_with_router(model: Option<Arc<dyn Model>>, router: Option<Arc<dyn Router>>) -> Session {
        Session::new(
            SessionId::generate(),
            EventBus::default(),
            Arc::new(Mutex::new(SessionState::Active)),
            model,
            router,
        )
    }

    #[tokio::test]
    async fn session_starts_active_and_closes_once() {
        let session = test_session(None);
        assert_eq!(session.state().await, SessionState::Active);
        assert!(session.close().await.is_ok());
        assert_eq!(session.state().await, SessionState::Closed);
        assert!(matches!(
            session.close().await,
            Err(NoemaError::Session(_))
        ));
    }

    #[tokio::test]
    async fn closing_emits_session_completed() {
        let bus = EventBus::new(16);
        let id = SessionId::generate();
        let session = Session::new(
            id.clone(),
            bus.clone(),
            Arc::new(Mutex::new(SessionState::Active)),
            None,
            None,
        );
        let mut events = bus.subscribe(id.clone());

        session.close().await.expect("close");
        assert_eq!(
            events.next().await,
            Some(Event::SessionCompleted { session_id: id })
        );
    }

    #[tokio::test]
    async fn send_requires_a_registered_model() {
        let session = test_session(None);
        let result = session.send(Message::text(Role::User, "hi")).await;
        assert!(matches!(result, Err(NoemaError::Model(_))));
    }

    #[tokio::test]
    async fn send_returns_echo_and_emits_events() {
        let session = test_session(Some(Arc::new(EchoModel)));
        let mut events = session.events();

        let response = session
            .send(Message::text(Role::User, "hello model"))
            .await
            .expect("send")
            .into_model()
            .expect("model outcome");

        match response {
            ModelResponse::Text { content, .. } => assert_eq!(content, "hello model"),
            _ => panic!("expected text response"),
        }

        let first = events.next().await.expect("event");
        assert!(matches!(first, Event::UserMessageReceived { .. }));
        let second = events.next().await.expect("event");
        assert!(matches!(second, Event::ModelStarted { model, .. } if model == "echo"));
        let third = events.next().await.expect("event");
        assert!(matches!(third, Event::ModelDelta { delta, .. } if delta == "hello model"));
        let fourth = events.next().await.expect("event");
        assert!(matches!(fourth, Event::ModelCompleted { .. }));
    }

    #[tokio::test]
    async fn send_rejects_when_closed() {
        let session = test_session(Some(Arc::new(EchoModel)));
        session.close().await.expect("close");
        let result = session.send(Message::text(Role::User, "hi")).await;
        assert!(matches!(result, Err(NoemaError::Session(_))));
    }

    #[tokio::test]
    async fn streaming_response_is_drained_and_streamed_as_events() {
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
                    Ok(ModelChunk::new("Hel")),
                    Ok(ModelChunk::new("lo ")),
                    Ok(ModelChunk::new("world")),
                ]);
                Ok(ModelResponse::Stream(Box::pin(stream)))
            }
        }

        let session = test_session(Some(Arc::new(StreamingModel)));
        let mut events = session.events();

        let response = session
            .send(Message::text(Role::User, "stream"))
            .await
            .expect("send")
            .into_model()
            .expect("model outcome");

        match response {
            ModelResponse::Text { content, .. } => assert_eq!(content, "Hello world"),
            _ => panic!("expected drained text response"),
        }

        let mut deltas = Vec::new();
        while let Some(event) = events.next().await {
            match event {
                Event::ModelDelta { delta, .. } => deltas.push(delta),
                // The bus stays open while the session holds it, so the
                // terminal event for a send is the end of this turn.
                Event::ModelCompleted { .. } => break,
                _ => {}
            }
        }
        assert_eq!(deltas, vec!["Hel", "lo ", "world"]);
    }

    #[tokio::test]
    async fn escalation_response_is_surfaced() {
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
                Ok(ModelResponse::Escalate(crate::model::EscalationRequest::new(
                    "too hard",
                    vec![],
                )))
            }
        }

        let session = test_session(Some(Arc::new(EscalatingModel)));
        let response = session
            .send(Message::text(Role::User, "hard"))
            .await
            .expect("send")
            .into_model()
            .expect("model outcome");
        assert!(matches!(response, ModelResponse::Escalate(_)));
    }

    #[tokio::test]
    async fn cancel_stops_an_in_flight_generation() {
        #[derive(Debug)]
        struct SlowModel;

        #[async_trait::async_trait]
        impl Model for SlowModel {
            fn id(&self) -> &str {
                "slow"
            }

            async fn generate(
                &self,
                _request: ModelRequest,
                cancel: CancellationToken,
            ) -> Result<ModelResponse> {
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(30)) => {
                        Ok(ModelResponse::Text { content: "finally".into(), usage: None })
                    }
                    _ = cancel.cancelled() => {
                        Err(NoemaError::Model("generation cancelled".into()))
                    }
                }
            }
        }

        let session = test_session(Some(Arc::new(SlowModel)));

        let handle = tokio::spawn({
            let session = session.clone();
            async move { session.send(Message::text(Role::User, "go")).await }
        });

        // Give the generation a moment to start, then cancel it.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        session.cancel().await.expect("cancel");

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            handle,
        )
        .await
        .expect("send finished")
        .expect("task ok");

        assert!(matches!(result, Err(NoemaError::Model(_))));
    }

    /// A router whose behaviour is chosen per test.
    #[derive(Debug)]
    struct FakeRouter {
        handle: bool,
        fail: bool,
        calls: std::sync::Mutex<Vec<String>>,
    }

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
            self.calls.lock().unwrap().push(text.to_string());
            if self.fail {
                return Err(NoemaError::Router("boom".into()));
            }
            if self.handle {
                Ok(Route::Action(RoutedAction::new("open_flashcards")))
            } else {
                Ok(Route::Escalate {
                    reason: "not an action".into(),
                })
            }
        }
    }

    #[tokio::test]
    async fn router_handles_plain_text_without_the_model() {
        let router = Arc::new(FakeRouter {
            handle: true,
            fail: false,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let session = test_session_with_router(Some(Arc::new(EchoModel)), Some(router));
        let mut events = session.events();

        let outcome = session
            .send(Message::text(Role::User, "open my flashcards"))
            .await
            .expect("send");
        assert!(outcome.is_routed());
        let action = outcome.into_routed().expect("routed outcome");
        assert_eq!(action.id, "open_flashcards");

        // Routing events, no model events at all.
        assert!(matches!(
            events.next().await,
            Some(Event::UserMessageReceived { .. })
        ));
        assert!(matches!(events.next().await, Some(Event::RoutingStarted { .. })));
        assert!(matches!(
            events.next().await,
            Some(Event::RoutingCompleted { .. })
        ));
    }

    #[tokio::test]
    async fn router_escalation_reaches_the_model() {
        let router = Arc::new(FakeRouter {
            handle: false,
            fail: false,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let session = test_session_with_router(Some(Arc::new(EchoModel)), Some(router));
        let mut events = session.events();

        let outcome = session
            .send(Message::text(Role::User, "what is the capital of france"))
            .await
            .expect("send");
        let response = outcome.into_model().expect("model outcome");
        match response {
            ModelResponse::Text { content, .. } => {
                assert_eq!(content, "what is the capital of france")
            }
            _ => panic!("expected text response"),
        }

        assert!(matches!(
            events.next().await,
            Some(Event::UserMessageReceived { .. })
        ));
        assert!(matches!(events.next().await, Some(Event::RoutingStarted { .. })));
        assert!(matches!(
            events.next().await,
            Some(Event::RoutingEscalated { .. })
        ));
        assert!(matches!(
            events.next().await,
            Some(Event::ModelStarted { .. })
        ));
    }

    #[tokio::test]
    async fn multimodal_messages_skip_routing() {
        let router = Arc::new(FakeRouter {
            handle: true,
            fail: false,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let session = test_session_with_router(Some(Arc::new(EchoModel)), Some(router));

        let message = Message::new(
            Role::User,
            vec![
                ContentPart::text("describe this"),
                ContentPart::image(vec![1, 2, 3], "image/png"),
            ],
        );
        let outcome = session.send(message).await.expect("send");
        assert!(outcome.into_model().is_some(), "multimodal skips the router");
    }

    #[tokio::test]
    async fn router_failure_escalates_with_error_event() {
        let router = Arc::new(FakeRouter {
            handle: false,
            fail: true,
            calls: std::sync::Mutex::new(Vec::new()),
        });
        let session = test_session_with_router(Some(Arc::new(EchoModel)), Some(router));
        let mut events = session.events();

        let outcome = session
            .send(Message::text(Role::User, "open my flashcards"))
            .await
            .expect("send");
        assert!(outcome.into_model().is_some(), "router failure escalates");

        assert!(matches!(
            events.next().await,
            Some(Event::UserMessageReceived { .. })
        ));
        assert!(matches!(events.next().await, Some(Event::RoutingStarted { .. })));
        assert!(matches!(
            events.next().await,
            Some(Event::RoutingEscalated { .. })
        ));
        assert!(matches!(events.next().await, Some(Event::Error { .. })));
    }
}

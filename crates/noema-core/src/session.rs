//! Ephemeral session state.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use noema_approval::{ApprovalDecision, ApprovalId, ApprovalPolicy, ApprovalRequest, ApprovalStore};
use noema_events::{Event, EventBus, EventStream, SessionId};
use noema_tools::{RiskLevel, ToolCall, ToolRegistry, ToolResult, ToolSchema};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::error::{NoemaError, Result};
use crate::escalation::{EscalationDecision, EscalationPolicy};
use crate::model::{ContentPart, EscalationRequest, Message, Model, ModelRequest, ModelResponse, Role};
use crate::router::{Route, Router, SendOutcome};
use crate::tooling::ToolFormatter;

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
    tools: Option<Arc<ToolRegistry>>,
    tool_formatter: Option<Arc<dyn ToolFormatter>>,
    tool_formatters: HashMap<String, Arc<dyn ToolFormatter>>,
    approval_policy: Arc<ApprovalPolicy>,
    approvals: Arc<ApprovalStore>,
    escalation: Arc<EscalationPolicy>,
    current_op: Arc<Mutex<Option<CancellationToken>>>,
}

impl Session {
    pub(crate) fn new(
        id: SessionId,
        events: EventBus,
        state: Arc<Mutex<SessionState>>,
        model: Option<Arc<dyn Model>>,
        router: Option<Arc<dyn Router>>,
        tools: Option<Arc<ToolRegistry>>,
        tool_formatter: Option<Arc<dyn ToolFormatter>>,
        tool_formatters: HashMap<String, Arc<dyn ToolFormatter>>,
        approval_policy: Arc<ApprovalPolicy>,
        approvals: Arc<ApprovalStore>,
        escalation: Arc<EscalationPolicy>,
    ) -> Self {
        Self {
            id,
            created_at: SystemTime::now(),
            state,
            events,
            model,
            router,
            tools,
            tool_formatter,
            tool_formatters,
            approval_policy,
            approvals,
            escalation,
            current_op: Arc::new(Mutex::new(None)),
        }
    }

    /// The router registered on this session, if any.
    pub fn router(&self) -> Option<&Arc<dyn Router>> {
        self.router.as_ref()
    }

    /// The tool registry available to this session, if any.
    pub fn tools(&self) -> Option<&Arc<ToolRegistry>> {
        self.tools.as_ref()
    }

    /// The default tool formatter available to this session, if any.
    pub fn tool_formatter(&self) -> Option<&Arc<dyn ToolFormatter>> {
        self.tool_formatter.as_ref()
    }

    /// The per-tool formatters available to this session.
    pub fn tool_formatters(&self) -> &HashMap<String, Arc<dyn ToolFormatter>> {
        &self.tool_formatters
    }

    /// The approval policy gating risky tool calls on this session.
    pub fn approval_policy(&self) -> &ApprovalPolicy {
        &self.approval_policy
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

    /// Formats a semantic tool request into a structured call.
    ///
    /// Resolves the formatter for `schema.name`: a per-tool formatter
    /// registered via
    /// [`NoemaBuilder::with_tool_formatter_for`](crate::NoemaBuilder::with_tool_formatter_for)
    /// wins; otherwise the default formatter (the tool-specific Needle
    /// agent) is used. The returned call is validated before it can be
    /// executed, but formatting itself does not execute anything.
    pub async fn format_tool(
        &self,
        schema: ToolSchema,
        request: &str,
    ) -> Result<ToolCall> {
        let formatter = self
            .tool_formatters
            .get(&schema.name)
            .or(self.tool_formatter.as_ref())
            .ok_or_else(|| {
                NoemaError::Tool(
                    "no tool formatter registered; use \
                     Noema::builder().with_tool_formatter(..) or \
                     .with_tool_formatter_for(..)"
                        .into(),
                )
            })?;
        let cancel = CancellationToken::new();
        let call = formatter.format(schema, request, cancel).await?;
        if let Some(tools) = &self.tools {
            tools.validate_call(&call)?;
        }
        Ok(call)
    }

    /// Executes a structured tool call and returns the result.
    ///
    /// The call is validated against the registered tool's schema before
    /// anything runs. When the tool's risk requires approval (see
    /// [`ApprovalPolicy`]), execution pauses: a [`Event::ToolApprovalRequired`]
    /// event is published and the call waits for [`Session::approve_tool`] /
    /// [`Session::reject_tool`] (or the policy timeout). Rejected or expired
    /// calls never execute.
    ///
    /// Execution is streamed through the event bus:
    /// [`Event::ToolStarted`] before the tool runs and [`Event::ToolCompleted`]
    /// / [`Event::ToolFailed`] when it finishes.
    ///
    /// Requires the tool to be registered on the runtime via
    /// [`NoemaBuilder::with_tool`](crate::NoemaBuilder::with_tool).
    pub async fn execute_tool(&self, call: ToolCall) -> Result<ToolResult> {
        let tools = self.tools.as_ref().ok_or_else(|| {
            NoemaError::Tool("no tools registered with this runtime".into())
        })?;
        tools.validate_call(&call)?;
        let tool = tools.get(&call.tool).expect("validated call's tool exists");
        let risk = tool.metadata().risk;

        // Risk gate: at/above the approval threshold (or Critical), pause
        // for a human decision before anything runs.
        if self.approval_policy.requires_approval(risk) {
            self.request_approval(&call, risk).await?;
        }

        self.events.publish(Event::ToolStarted {
            session_id: self.id.clone(),
        });
        let result = tools.execute(call).await;
        match result {
            Ok(result) => {
                self.events.publish(Event::ToolCompleted {
                    session_id: self.id.clone(),
                });
                Ok(result)
            }
            Err(error) => {
                self.events.publish(Event::ToolFailed {
                    session_id: self.id.clone(),
                });
                Err(NoemaError::Tool(error.to_string()))
            }
        }
    }

    /// Creates a pending approval for a risky call and waits for the human's
    /// decision (bounded by the policy timeout).
    async fn request_approval(&self, call: &ToolCall, risk: RiskLevel) -> Result<()> {
        let tools = self.tools.as_ref().expect("caller checked tools");
        let tool = tools.get(&call.tool).expect("validated call's tool exists");
        let metadata = tool.metadata();
        let timeout = self.approval_policy.timeout;

        let request = ApprovalRequest::new(
            self.id.to_string(),
            metadata.name,
            metadata.description,
            call.arguments.clone(),
            risk.to_string(),
            timeout,
        );
        let id = request.id.clone();
        let mut handle = self.approvals.create(request);

        self.events.publish(Event::ToolApprovalRequired {
            session_id: self.id.clone(),
        });

        let decision = handle.await_decision(timeout).await.map_err(|error| {
            // Expired requests are removed from the store so the frontend
            // cannot approve them later.
            let _ = self.approvals.expire(&id.to_string());
            NoemaError::Approval(error.to_string())
        })?;

        match decision {
            ApprovalDecision::Approved => {
                self.events.publish(Event::ToolApproved {
                    session_id: self.id.clone(),
                });
                Ok(())
            }
            ApprovalDecision::Rejected => {
                self.events.publish(Event::ToolRejected {
                    session_id: self.id.clone(),
                });
                Err(NoemaError::Approval(format!(
                    "tool call '{}' rejected by the user",
                    call.tool
                )))
            }
        }
    }

    /// Approves a pending tool approval, releasing its call for execution.
    ///
    /// The approval id comes from the [`Event::ToolApprovalRequired`]
    /// payload (see [`Session::pending_approvals`]). Approving an unknown or
    /// already-decided request is an error.
    pub fn approve_tool(&self, id: ApprovalId) -> Result<()> {
        self.resolve_approval(id, ApprovalDecision::Approved)
    }

    /// Rejects a pending tool approval, cancelling its call.
    ///
    /// See [`Session::approve_tool`] for the id source.
    pub fn reject_tool(&self, id: ApprovalId) -> Result<()> {
        self.resolve_approval(id, ApprovalDecision::Rejected)
    }

    fn resolve_approval(&self, id: ApprovalId, decision: ApprovalDecision) -> Result<()> {
        let id_string = id.as_str().to_string();
        let _ = self.approvals.decide(&id_string, decision)?;
        let event = match decision {
            ApprovalDecision::Approved => Event::ToolApproved {
                session_id: self.id.clone(),
            },
            ApprovalDecision::Rejected => Event::ToolRejected {
                session_id: self.id.clone(),
            },
        };
        self.events.publish(event);
        Ok(())
    }

    /// The approvals currently waiting on a human decision.
    pub fn pending_approvals(&self) -> Vec<ApprovalRequest> {
        self.approvals.pending()
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
        let escalated = match self.route_message(&message, &token).await {
            RouteOutcome::Routed(outcome) => {
                *self.current_op.lock().await = None;
                return Ok(outcome);
            }
            RouteOutcome::Escalated(request) => match self.start_escalation(&request).await {
                Ok(()) => true,
                Err(error) => {
                    *self.current_op.lock().await = None;
                    return Err(error);
                }
            },
            RouteOutcome::Proceed => false,
        };

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
        if escalated {
            self.events.publish(Event::EscalationCompleted {
                session_id: self.id.clone(),
            });
        }
        Ok(SendOutcome::Model(response))
    }

    /// Applies the escalation policy to a router escalation and, when the
    /// escalation proceeds, publishes [`Event::EscalationStarted`].
    ///
    /// The default policy escalates to the local reasoning model; a denied
    /// escalation is an error. Cloud decisions need a registered provider
    /// and are not wired yet (the cloud milestone).
    async fn start_escalation(&self, request: &EscalationRequest) -> Result<()> {
        match self.escalation.decide(request) {
            EscalationDecision::Local => {
                self.events.publish(Event::EscalationStarted {
                    session_id: self.id.clone(),
                });
                Ok(())
            }
            EscalationDecision::Cloud => Err(NoemaError::Escalation(
                "cloud escalation requires a registered provider (cloud milestone)".into(),
            )),
            EscalationDecision::Denied => {
                let message = format!("escalation denied by policy: {}", request.reason);
                self.events.publish(Event::Error {
                    session_id: self.id.clone(),
                    error: message.clone(),
                });
                Err(NoemaError::Escalation(message))
            }
        }
    }

    /// Routes a plain-text user message through the registered router.
    ///
    /// Multimodal and non-user turns skip routing entirely. Router failures
    /// are surfaced as events but still escalate, so the user gets an
    /// answer.
    async fn route_message(
        &self,
        message: &Message,
        token: &CancellationToken,
    ) -> RouteOutcome {
        let router = match self.router.as_ref() {
            Some(router) => router,
            None => return RouteOutcome::Proceed,
        };
        let text = match plain_user_text(message) {
            Some(text) => text,
            // Multimodal and non-user turns skip routing.
            None => return RouteOutcome::Proceed,
        };

        self.events.publish(Event::RoutingStarted {
            session_id: self.id.clone(),
        });
        match router.route(&text, token.clone()).await {
            Ok(Route::Action(action)) => {
                self.events.publish(Event::RoutingCompleted {
                    session_id: self.id.clone(),
                });
                RouteOutcome::Routed(SendOutcome::Routed(action))
            }
            Ok(Route::Escalate { reason }) => {
                self.events.publish(Event::RoutingEscalated {
                    session_id: self.id.clone(),
                });
                RouteOutcome::Escalated(EscalationRequest::new(reason, vec![message.clone()]))
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
                // A failing router still escalates: the request becomes an
                // escalation whose reason carries the failure.
                RouteOutcome::Escalated(EscalationRequest::new(
                    format!("router failed: {error}"),
                    vec![message.clone()],
                ))
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

/// The outcome of offering a message to the router.
enum RouteOutcome {
    /// The router handled the request; return the outcome directly.
    Routed(SendOutcome),
    /// The router escalated; run the escalation policy, then the model.
    Escalated(EscalationRequest),
    /// No routing applies (multimodal turn or no router); go straight to the
    /// model.
    Proceed,
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
        test_session_with_router(model, None)
    }

    fn test_session_with_router(model: Option<Arc<dyn Model>>, router: Option<Arc<dyn Router>>) -> Session {
        test_session_full(model, router, Arc::new(EscalationPolicy::default()))
    }

    fn test_session_full(
        model: Option<Arc<dyn Model>>,
        router: Option<Arc<dyn Router>>,
        escalation: Arc<EscalationPolicy>,
    ) -> Session {
        Session::new(
            SessionId::generate(),
            EventBus::default(),
            Arc::new(Mutex::new(SessionState::Active)),
            model,
            router,
            None,
            None,
            HashMap::new(),
            Arc::new(ApprovalPolicy::default()),
            Arc::new(ApprovalStore::new()),
            escalation,
        )
    }

    /// A session with tools and a formatter, for tool tests.
    fn test_session_with_tools(
        tools: ToolRegistry,
        formatter: Option<Arc<dyn ToolFormatter>>,
    ) -> Session {
        Session::new(
            SessionId::generate(),
            EventBus::default(),
            Arc::new(Mutex::new(SessionState::Active)),
            None,
            None,
            Some(Arc::new(tools)),
            formatter,
            HashMap::new(),
            Arc::new(ApprovalPolicy::default()),
            Arc::new(ApprovalStore::new()),
            Arc::new(EscalationPolicy::default()),
        )
    }

    /// A session with tools, a formatter, and an explicit approval policy.
    fn test_session_with_approval(
        tools: ToolRegistry,
        policy: ApprovalPolicy,
    ) -> Session {
        Session::new(
            SessionId::generate(),
            EventBus::default(),
            Arc::new(Mutex::new(SessionState::Active)),
            None,
            None,
            Some(Arc::new(tools)),
            None,
            HashMap::new(),
            Arc::new(policy),
            Arc::new(ApprovalStore::new()),
            Arc::new(EscalationPolicy::default()),
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
        let id = SessionId::generate();        let session = Session::new(
            id.clone(),
            bus.clone(),
            Arc::new(Mutex::new(SessionState::Active)),
            None,
            None,
            None,
            None,
            HashMap::new(),
            Arc::new(ApprovalPolicy::default()),
            Arc::new(ApprovalStore::new()),
            Arc::new(EscalationPolicy::default()),
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
            Some(Event::EscalationStarted { .. })
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

    /// A router that always escalates, for escalation-flow tests.
    #[derive(Debug)]
    struct EscalatingRouter;

    #[async_trait::async_trait]
    impl Router for EscalatingRouter {
        fn id(&self) -> &str {
            "escalating-router"
        }

        async fn route(
            &self,
            _text: &str,
            _cancel: CancellationToken,
        ) -> Result<Route> {
            Ok(Route::Escalate {
                reason: "low confidence (0.53 < 0.6)".into(),
            })
        }
    }

    #[tokio::test]
    async fn router_escalation_runs_the_policy_and_emits_events() {
        let session = test_session_with_router(Some(Arc::new(EchoModel)), Some(Arc::new(EscalatingRouter)));
        let mut events = session.events();

        let outcome = session
            .send(Message::text(Role::User, "go to settings"))
            .await
            .expect("send")
            .into_model()
            .expect("model outcome");
        match outcome {
            ModelResponse::Text { content, .. } => assert_eq!(content, "go to settings"),
            _ => panic!("expected echo"),
        }

        let mut seen = Vec::new();
        while let Some(event) = events.next().await {
            match event {
                Event::RoutingEscalated { .. }
                | Event::EscalationStarted { .. }
                | Event::ModelStarted { .. }
                | Event::ModelCompleted { .. } => seen.push(event),
                // The escalation completes only after the model finishes; it
                // is the last event of this flow, so stop draining here.
                Event::EscalationCompleted { .. } => {
                    seen.push(event);
                    break;
                }
                Event::UserMessageReceived { .. } | Event::RoutingStarted { .. } | Event::ModelDelta { .. } => {}
                _ => break,
            }
        }
        let kinds: Vec<&str> = seen
            .iter()
            .map(|e| match e {
                Event::RoutingEscalated { .. } => "routing_escalated",
                Event::EscalationStarted { .. } => "escalation_started",
                Event::ModelStarted { .. } => "model_started",
                Event::ModelCompleted { .. } => "model_completed",
                Event::EscalationCompleted { .. } => "escalation_completed",
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(
            kinds,
            vec![
                "routing_escalated",
                "escalation_started",
                "model_started",
                "model_completed",
                "escalation_completed",
            ]
        );
    }

    #[tokio::test]
    async fn denied_escalation_policy_errors() {
        let policy = Arc::new(EscalationPolicy {
            allow_local: false,
            allow_cloud: false,
            ..EscalationPolicy::default()
        });
        let session = test_session_full(
            Some(Arc::new(EchoModel)),
            Some(Arc::new(EscalatingRouter)),
            policy,
        );

        let result = session
            .send(Message::text(Role::User, "go to settings"))
            .await;
        assert!(
            matches!(result, Err(NoemaError::Escalation(_))),
            "denied escalation should error, got {result:?}"
        );
    }

    #[tokio::test]
    async fn cloud_escalation_requires_a_provider() {
        let policy = Arc::new(EscalationPolicy {
            allow_local: false,
            allow_cloud: true,
            ..EscalationPolicy::default()
        });
        let session = test_session_full(
            Some(Arc::new(EchoModel)),
            Some(Arc::new(EscalatingRouter)),
            policy,
        );

        let result = session
            .send(Message::text(Role::User, "go to settings"))
            .await;
        assert!(matches!(result, Err(NoemaError::Escalation(_))));
    }

    /// A fake tool for session tool tests.
    #[derive(Debug)]
    struct EchoTool;

    #[async_trait::async_trait]
    impl noema_tools::NoemaTool for EchoTool {
        fn metadata(&self) -> noema_tools::ToolMetadata {
            noema_tools::ToolMetadata {
                name: "echo".into(),
                crate_name: "noema-test".into(),
                description: "Echoes its message argument".into(),
                risk: noema_tools::RiskLevel::None,
            }
        }

        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "echo".into(),
                description: "Echoes its message argument".into(),
                parameters: serde_json::json!({ "type": "object", "properties": {} }),
            }
        }

        async fn execute(&self, call: ToolCall) -> noema_tools::Result<ToolResult> {
            Ok(ToolResult::ok(format!("ran {}", call.tool)))
        }
    }

    /// A recording tool with a configurable risk, for approval tests.
    #[derive(Debug)]
    struct RecordingTool {
        name: &'static str,
        risk: RiskLevel,
        runs: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl noema_tools::NoemaTool for RecordingTool {
        fn metadata(&self) -> noema_tools::ToolMetadata {
            noema_tools::ToolMetadata {
                name: self.name.into(),
                crate_name: "noema-test".into(),
                description: format!("{} tool", self.name),
                risk: self.risk,
            }
        }

        fn schema(&self) -> ToolSchema {
            ToolSchema::new(self.name, format!("{} tool", self.name))
        }

        async fn execute(&self, _call: ToolCall) -> noema_tools::Result<ToolResult> {
            self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ToolResult::ok(format!("{} executed", self.name)))
        }
    }

    fn recording_registry(name: &'static str, risk: RiskLevel) -> (ToolRegistry, Arc<std::sync::atomic::AtomicUsize>) {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry
            .register(RecordingTool { name, risk, runs: Arc::clone(&runs) })
            .expect("register");
        (registry, runs)
    }

    /// A formatter that always returns a fixed call.
    #[derive(Debug)]
    struct FixedFormatter(ToolCall);

    #[async_trait::async_trait]
    impl ToolFormatter for FixedFormatter {
        fn id(&self) -> &str {
            "fixed"
        }

        async fn format(
            &self,
            _schema: ToolSchema,
            _request: &str,
            _cancel: CancellationToken,
        ) -> Result<ToolCall> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn execute_tool_runs_and_emits_events() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool).expect("register");
        let bus = EventBus::new(16);
        let id = SessionId::generate();
        let session = Session::new(
            id.clone(),
            bus.clone(),
            Arc::new(Mutex::new(SessionState::Active)),
            None,
            None,
            Some(Arc::new(registry)),
            None,
            HashMap::new(),
            Arc::new(ApprovalPolicy::default()),
            Arc::new(ApprovalStore::new()),
            Arc::new(EscalationPolicy::default()),
        );
        let mut events = bus.subscribe(id.clone());

        let result = session
            .execute_tool(ToolCall::new("echo"))
            .await
            .expect("tool runs");
        assert!(result.success);
        assert_eq!(result.text, "ran echo");

        assert!(matches!(events.next().await, Some(Event::ToolStarted { .. })));
        assert!(matches!(events.next().await, Some(Event::ToolCompleted { .. })));
    }

    #[tokio::test]
    async fn execute_tool_requires_registration_and_validation() {
        let session = test_session_with_tools(ToolRegistry::new(), None);
        let err = session
            .execute_tool(ToolCall::new("echo"))
            .await
            .expect_err("unknown tool");
        assert!(matches!(err, NoemaError::Tool(_)));

        let session = test_session(None);
        let err = session
            .execute_tool(ToolCall::new("echo"))
            .await
            .expect_err("no tools registered");
        assert!(matches!(err, NoemaError::Tool(_)));
    }

    #[tokio::test]
    async fn format_tool_uses_the_registered_formatter() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool).expect("register");
        let session = test_session_with_tools(
            registry,
            Some(Arc::new(FixedFormatter(ToolCall::new("echo")))),
        );

        let schema = ToolSchema::new("echo", "Echoes its message argument");
        let call = session
            .format_tool(schema, "say hello")
            .await
            .expect("format");
        assert_eq!(call.tool, "echo");
    }

    #[tokio::test]
    async fn format_tool_requires_a_formatter() {
        let session = test_session(None);
        let schema = ToolSchema::new("echo", "Echoes its message argument");
        let err = session
            .format_tool(schema, "say hello")
            .await
            .expect_err("no formatter");
        assert!(matches!(err, NoemaError::Tool(_)));
    }

    #[tokio::test]
    async fn low_risk_calls_skip_approval() {
        let (registry, runs) = recording_registry("echo", RiskLevel::Low);
        let session = test_session_with_tools(registry, None);

        let result = session
            .execute_tool(ToolCall::new("echo"))
            .await
            .expect("low risk executes directly");
        assert!(result.success);
        assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(session.pending_approvals().is_empty());
    }

    #[tokio::test]
    async fn risky_call_pauses_for_approval_then_executes() {
        let (registry, runs) = recording_registry("delete", RiskLevel::Critical);
        let session = test_session_with_tools(registry, None);
        let mut events = session.events();

        let handle = tokio::spawn({
            let session = session.clone();
            async move { session.execute_tool(ToolCall::new("delete")).await }
        });

        // The call must not run before approval.
        tokio::task::yield_now().await;
        assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(session.pending_approvals().len(), 1);
        assert!(matches!(
            events.next().await,
            Some(Event::ToolApprovalRequired { .. })
        ));

        let pending = session.pending_approvals();
        let id = pending[0].id.clone();
        assert_eq!(pending[0].risk, "critical");
        session.approve_tool(id).expect("approve");

        let result = handle.await.expect("task").expect("execute");
        assert!(result.success);
        assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(session.pending_approvals().is_empty());
    }

    #[tokio::test]
    async fn rejected_call_never_executes() {
        let (registry, runs) = recording_registry("delete", RiskLevel::Critical);
        let session = test_session_with_tools(registry, None);
        let mut events = session.events();

        let handle = tokio::spawn({
            let session = session.clone();
            async move { session.execute_tool(ToolCall::new("delete")).await }
        });

        tokio::task::yield_now().await;
        assert!(matches!(
            events.next().await,
            Some(Event::ToolApprovalRequired { .. })
        ));
        let pending = session.pending_approvals();
        let id = pending[0].id.clone();
        session.reject_tool(id).expect("reject");

        let err = handle.await.expect("task").expect_err("rejected");
        assert!(matches!(err, NoemaError::Approval(_)));
        assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 0, "never executed");
        assert!(matches!(
            events.next().await,
            Some(Event::ToolRejected { .. })
        ));
    }

    #[tokio::test]
    async fn approval_expires_when_undecided() {
        let policy = ApprovalPolicy {
            require_approval_above: Some(RiskLevel::High),
            timeout: Some(std::time::Duration::from_millis(30)),
        };
        let (registry, runs) = recording_registry("delete", RiskLevel::Critical);
        let session = test_session_with_approval(registry, policy);
        let mut events = session.events();

        let handle = tokio::spawn({
            let session = session.clone();
            async move { session.execute_tool(ToolCall::new("delete")).await }
        });

        let err = handle.await.expect("task").expect_err("expired");
        assert!(matches!(err, NoemaError::Approval(_)));
        assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(session.pending_approvals().is_empty());
        assert!(matches!(
            events.next().await,
            Some(Event::ToolApprovalRequired { .. })
        ));
    }

    #[tokio::test]
    async fn approving_unknown_id_fails() {
        let (registry, _) = recording_registry("delete", RiskLevel::Critical);
        let session = test_session_with_tools(registry, None);
        let err = session
            .approve_tool(ApprovalId::generate())
            .expect_err("unknown");
        assert!(matches!(err, NoemaError::Approval(_)));
    }
}

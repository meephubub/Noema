//! The Noema runtime: construction, configuration, and session lifecycle.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use noema_approval::{ApprovalPolicy, ApprovalStore};
use noema_events::{Event, EventBus, EventStream, SessionId};
use noema_tools::{NoemaTool, ToolRegistry};
use tokio::sync::Mutex as AsyncMutex;

use crate::config::{LogLevel, NoemaConfig};
use crate::error::{NoemaError, Result};
use crate::escalation::EscalationPolicy;
use crate::model::Model;
use crate::router::Router;
use crate::session::{Session, SessionState};
use crate::tooling::ToolFormatter;

/// A Noema runtime.
///
/// This is the entry point for the Agora frontend. It owns the configuration
/// and the event bus, manages the lifecycle of ephemeral sessions, and holds
/// the models the sessions talk to.
///
/// ```
/// # use noema_core::{Noema, Result};
/// # async fn example() -> Result<()> {
/// let noema = Noema::builder().build().await?;
/// let session = noema.create_session().await?;
/// noema.close_session(session.id()).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Noema {
    config: NoemaConfig,
    events: EventBus,
    sessions: Mutex<HashMap<SessionId, Arc<AsyncMutex<SessionState>>>>,
    model: Option<Arc<dyn Model>>,
    router: Option<Arc<dyn Router>>,
    tools: Option<Arc<ToolRegistry>>,
    tool_formatter: Option<Arc<dyn ToolFormatter>>,
    tool_formatters: HashMap<String, Arc<dyn ToolFormatter>>,
    approval_policy: Arc<ApprovalPolicy>,
    approvals: Arc<ApprovalStore>,
    escalation: Arc<EscalationPolicy>,
}

impl Noema {
    /// Returns a builder for constructing a [`Noema`] runtime.
    pub fn builder() -> NoemaBuilder {
        NoemaBuilder::default()
    }

    /// The configuration this runtime was built with.
    pub fn config(&self) -> &NoemaConfig {
        &self.config
    }

    /// The model registered on this runtime, if any.
    pub fn model(&self) -> Option<&Arc<dyn Model>> {
        self.model.as_ref()
    }

    /// The router registered on this runtime, if any.
    pub fn router(&self) -> Option<&Arc<dyn Router>> {
        self.router.as_ref()
    }

    /// The tool registry registered on this runtime, if any.
    pub fn tools(&self) -> Option<&Arc<ToolRegistry>> {
        self.tools.as_ref()
    }

    /// The tool formatter registered on this runtime, if any.
    pub fn tool_formatter(&self) -> Option<&Arc<dyn ToolFormatter>> {
        self.tool_formatter.as_ref()
    }

    /// The per-tool formatters registered on this runtime, if any.
    pub fn tool_formatters(&self) -> &HashMap<String, Arc<dyn ToolFormatter>> {
        &self.tool_formatters
    }

    /// The approval policy in effect on this runtime.
    pub fn approval_policy(&self) -> &ApprovalPolicy {
        &self.approval_policy
    }

    /// The escalation policy in effect on this runtime.
    pub fn escalation(&self) -> &EscalationPolicy {
        &self.escalation
    }

    /// Creates a new ephemeral session and emits [`Event::SessionStarted`].
    ///
    /// Returns a [`Result`] to match the final public API shape; the current
    /// error paths will fill in as tool and memory wiring lands.
    pub async fn create_session(&self) -> Result<Session> {
        let id = SessionId::generate();
        let state = Arc::new(AsyncMutex::new(SessionState::Active));
        self.sessions
            .lock()
            .expect("session registry poisoned")
            .insert(id.clone(), Arc::clone(&state));
        self.events
            .publish(Event::SessionStarted { session_id: id.clone() });
        Ok(Session::new(
            id,
            self.events.clone(),
            state,
            self.model.clone(),
            self.router.clone(),
            self.tools.clone(),
            self.tool_formatter.clone(),
            self.tool_formatters.clone(),
            Arc::clone(&self.approval_policy),
            Arc::clone(&self.approvals),
            Arc::clone(&self.escalation),
        ))
    }

    /// Closes the session with the given id and emits
    /// [`Event::SessionCompleted`].
    ///
    /// Closing an unknown session is an error.
    pub async fn close_session(&self, session_id: &SessionId) -> Result<()> {
        // Clone the shared state out of the registry so the std-lock guard is
        // not held across an await point, then update the shared state.
        let state = {
            let sessions = self.sessions.lock().expect("session registry poisoned");
            sessions.get(session_id).cloned().ok_or_else(|| {
                NoemaError::Session(format!("no such session: {session_id}"))
            })?
        };
        let mut state = state.lock().await;
        if *state == SessionState::Closed {
            return Err(NoemaError::Session(format!(
                "session {session_id} is already closed"
            )));
        }
        *state = SessionState::Closed;
        drop(state);
        self.sessions
            .lock()
            .expect("session registry poisoned")
            .remove(session_id);
        self.events
            .publish(Event::SessionCompleted { session_id: session_id.clone() });
        Ok(())
    }

    /// Subscribes to every event emitted by the runtime.
    pub fn subscribe_all(&self) -> EventStream {
        self.events.subscribe_all()
    }

    /// Subscribes to the events of a single session.
    pub fn subscribe(&self, session_id: SessionId) -> EventStream {
        self.events.subscribe(session_id)
    }
}

/// Builds the approval policy from the runtime configuration.
///
/// A disabled approval flow (`NoemaConfig::approval.enabled == false`)
/// never pauses execution; otherwise the risk threshold comes from
/// `NoemaConfig::risk` and the timeout from `NoemaConfig::approval`.
fn approval_policy_from_config(config: &NoemaConfig) -> ApprovalPolicy {
    if !config.approval.enabled {
        return ApprovalPolicy {
            require_approval_above: None,
            timeout: None,
        };
    }
    ApprovalPolicy {
        require_approval_above: config.risk.require_approval_above,
        timeout: config
            .approval
            .timeout_seconds
            .map(std::time::Duration::from_secs),
    }
}

/// Builder for a [`Noema`] runtime.
#[derive(Debug, Default)]
pub struct NoemaBuilder {
    config: NoemaConfig,
    model: Option<Arc<dyn Model>>,
    router: Option<Arc<dyn Router>>,
    tools: Option<ToolRegistry>,
    tool_formatter: Option<Arc<dyn ToolFormatter>>,
    tool_formatters: HashMap<String, Arc<dyn ToolFormatter>>,
    approval_policy: Option<ApprovalPolicy>,
    escalation: Option<EscalationPolicy>,
}

impl NoemaBuilder {
    /// Overrides the runtime configuration.
    pub fn with_config(mut self, config: NoemaConfig) -> Self {
        self.config = config;
        self
    }

    /// Convenience: sets the global logging level.
    pub fn with_logging(mut self, level: LogLevel) -> Self {
        self.config.logging.level = level;
        self
    }

    /// Registers the model sessions will talk to.
    ///
    /// Gemma, Needle, and cloud adapters all implement
    /// [`Model`](crate::model::Model) and are registered here.
    pub fn with_model<M: Model>(mut self, model: M) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    /// Registers the initial text router.
    ///
    /// When set, plain-text user requests are routed through it first;
    /// handled requests never reach the model (see
    /// [`Router`](crate::router::Router)).
    pub fn with_router<R: Router>(mut self, router: R) -> Self {
        self.router = Some(Arc::new(router));
        self
    }

    /// Registers a pre-built tool registry with the runtime.
    ///
    /// Tools are available to every session; Gemma sees only their
    /// lightweight summaries (see [`ToolRegistry::gemma_tool_section`]) while
    /// the tool-specific Needle agents bind to the full schemas.
    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Registers a single tool, merging it into the runtime's registry.
    ///
    /// Replaces any registry set by a previous `with_tools`/`with_tool`
    /// call; call this repeatedly to register several tools.
    pub fn with_tool<T: NoemaTool + 'static>(mut self, tool: T) -> Self {
        let mut registry = self.tools.take().unwrap_or_default();
        registry
            .register(tool)
            .expect("registering a tool with a duplicate name");
        self.tools = Some(registry);
        self
    }

    /// Registers the tool formatter (the tool-specific Needle agent role).
    ///
    /// When set, sessions can turn semantic tool requests into structured
    /// calls; see [`ToolFormatter`](crate::ToolFormatter). This is the
    /// default formatter used for any tool without a dedicated one.
    pub fn with_tool_formatter<F: ToolFormatter + 'static>(mut self, formatter: F) -> Self {
        self.tool_formatter = Some(Arc::new(formatter));
        self
    }

    /// Registers a formatter for one specific tool.
    ///
    /// One physical Needle model, many logical agents: each tool can have its
    /// own formatter bound to its schema and instructions. A formatter
    /// registered here is preferred over the default one when formatting
    /// that tool.
    pub fn with_tool_formatter_for<F: ToolFormatter + 'static>(
        mut self,
        tool: impl Into<String>,
        formatter: F,
    ) -> Self {
        self.tool_formatters.insert(tool.into(), Arc::new(formatter));
        self
    }

    /// Overrides the approval policy gating risky tool calls.
    ///
    /// By default the policy is built from the runtime configuration:
    /// [`NoemaConfig::risk`] supplies the approval threshold and
    /// [`NoemaConfig::approval`] supplies the timeout and the enabled flag
    /// (a disabled approval flow never pauses execution).
    pub fn with_approval_policy(mut self, policy: ApprovalPolicy) -> Self {
        self.approval_policy = Some(policy);
        self
    }

    /// Overrides the escalation policy.
    ///
    /// By default the policy is built from the runtime configuration (see
    /// [`EscalationPolicy::from_config`]): local escalation is allowed and
    /// cloud escalation follows [`NoemaConfig::cloud`], with
    /// [`NoemaConfig::offline_mode`] always winning.
    pub fn with_escalation_policy(mut self, policy: EscalationPolicy) -> Self {
        self.escalation = Some(policy);
        self
    }

    /// Builds the runtime.
    ///
    /// Async to match the final public API shape; later milestones load
    /// models and register tools here.
    pub async fn build(self) -> Result<Noema> {
        if self.config.logging.level != LogLevel::Off {
            tracing::info!(level = ?self.config.logging.level, "building noema runtime");
        }
        let escalation = Arc::new(self.escalation.unwrap_or_else(|| {
            EscalationPolicy::from_config(&self.config)
        }));
        let approval_policy = Arc::new(self.approval_policy.unwrap_or_else(|| {
            approval_policy_from_config(&self.config)
        }));
        Ok(Noema {
            events: EventBus::new(self.config.streaming.event_capacity),
            config: self.config,
            sessions: Mutex::new(HashMap::new()),
            model: self.model,
            router: self.router,
            tools: self.tools.map(Arc::new),
            tool_formatter: self.tool_formatter,
            tool_formatters: self.tool_formatters,
            approval_policy: Arc::clone(&approval_policy),
            approvals: Arc::new(ApprovalStore::new()),
            escalation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Message, ModelRequest, ModelResponse, Role};

    #[derive(Debug)]
    struct TestModel;

    #[async_trait::async_trait]
    impl Model for TestModel {
        fn id(&self) -> &str {
            "test"
        }

        async fn generate(
            &self,
            _request: ModelRequest,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> Result<ModelResponse> {
            Ok(ModelResponse::Text {
                content: "pong".into(),
                usage: None,
            })
        }
    }

    #[tokio::test]
    async fn build_and_create_session() {
        let noema = Noema::builder().build().await.expect("build");
        let session = noema.create_session().await.expect("create");
        assert_eq!(session.state().await, SessionState::Active);
        noema.close_session(session.id()).await.expect("close");
    }

    #[tokio::test]
    async fn sessions_have_unique_ids() {
        let noema = Noema::builder().build().await.expect("build");
        let a = noema.create_session().await.expect("create a");
        let b = noema.create_session().await.expect("create b");
        assert_ne!(a.id(), b.id());
    }

    #[tokio::test]
    async fn close_unknown_session_fails() {
        let noema = Noema::builder().build().await.expect("build");
        let unknown = SessionId::generate();
        assert!(matches!(
            noema.close_session(&unknown).await,
            Err(NoemaError::Session(_))
        ));
    }

    #[tokio::test]
    async fn session_lifecycle_events_are_streamed() {
        let noema = Noema::builder().build().await.expect("build");
        // Subscribe before creating so the full lifecycle is visible.
        let mut events = noema.subscribe_all();

        let session = noema.create_session().await.expect("create");
        noema.close_session(session.id()).await.expect("close");

        let first = events.next().await.expect("first event");
        assert_eq!(
            first,
            Event::SessionStarted {
                session_id: session.id().clone()
            }
        );
        let second = events.next().await.expect("second event");
        assert_eq!(
            second,
            Event::SessionCompleted {
                session_id: session.id().clone()
            }
        );
    }

    #[tokio::test]
    async fn sessions_talk_to_the_registered_model() {
        let noema = Noema::builder()
            .with_model(TestModel)
            .build()
            .await
            .expect("build");
        let session = noema.create_session().await.expect("create");

        let response = session
            .send(Message::text(Role::User, "ping"))
            .await
            .expect("send")
            .into_model()
            .expect("model outcome");

        match response {
            ModelResponse::Text { content, .. } => assert_eq!(content, "pong"),
            _ => panic!("expected text response"),
        }
    }

    #[tokio::test]
    async fn builder_overrides_config() {
        let noema = Noema::builder()
            .with_logging(LogLevel::Debug)
            .build()
            .await
            .expect("build");
        assert_eq!(noema.config().logging.level, LogLevel::Debug);
    }

    #[tokio::test]
    async fn tools_register_via_the_builder() {
        let noema = Noema::builder()
            .with_tool(TestTool)
            .build()
            .await
            .expect("build");
        let tools = noema.tools().expect("registry registered");
        assert_eq!(tools.names(), vec!["test"]);
        assert!(tools.get("test").is_some());
    }

    /// A no-op tool for builder tests.
    #[derive(Debug)]
    struct TestTool;

    #[async_trait::async_trait]
    impl noema_tools::NoemaTool for TestTool {
        fn metadata(&self) -> noema_tools::ToolMetadata {
            noema_tools::ToolMetadata {
                name: "test".into(),
                crate_name: "noema-test".into(),
                description: "A no-op tool".into(),
                risk: noema_tools::RiskLevel::None,
            }
        }

        fn schema(&self) -> noema_tools::ToolSchema {
            noema_tools::ToolSchema::new("test", "A no-op tool")
        }

        async fn execute(&self, _call: noema_tools::ToolCall) -> noema_tools::Result<noema_tools::ToolResult> {
            Ok(noema_tools::ToolResult::ok("no-op"))
        }
    }
}

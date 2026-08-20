//! The Noema runtime: construction, configuration, and session lifecycle.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use noema_events::{Event, EventBus, EventStream, SessionId};
use tokio::sync::Mutex as AsyncMutex;

use crate::config::{LogLevel, NoemaConfig};
use crate::error::{NoemaError, Result};
use crate::model::Model;
use crate::router::Router;
use crate::session::{Session, SessionState};

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

/// Builder for a [`Noema`] runtime.
#[derive(Debug, Default)]
pub struct NoemaBuilder {
    config: NoemaConfig,
    model: Option<Arc<dyn Model>>,
    router: Option<Arc<dyn Router>>,
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

    /// Builds the runtime.
    ///
    /// Async to match the final public API shape; later milestones load
    /// models and register tools here.
    pub async fn build(self) -> Result<Noema> {
        if self.config.logging.level != LogLevel::Off {
            tracing::info!(level = ?self.config.logging.level, "building noema runtime");
        }
        Ok(Noema {
            events: EventBus::new(self.config.streaming.event_capacity),
            config: self.config,
            sessions: Mutex::new(HashMap::new()),
            model: self.model,
            router: self.router,
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
}

//! The Needle-backed initial text router.
//!
//! [`NeedleRouter`] implements Noema's [`Router`] trait with a Needle 2
//! engine. The registry's actions are exposed to Needle as tools; a request
//! that maps to one of them becomes a [`Route::Action`], and anything else
//! (the engine refuses with an empty call list) escalates.

use std::sync::Arc;

use async_trait::async_trait;
use noema_core::{NoemaError, Result, Route, RoutedAction, Router};
use noema_needle::{DylibEngine, EngineSettings, NeedleEngine};
use tokio_util::sync::CancellationToken;

use crate::action::ActionRegistry;

/// The default confidence below which a routed action escalates instead of
/// acting. Needle's confidence head is calibrated; the model card instructs:
/// act at or above your threshold, escalate below it.
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.6;

/// A [`Router`] backed by a Needle 2 engine.
///
/// Holds the engine (shared, so the router can be cloned cheaply) and the
/// action registry the engine was built with. A call below
/// [`DEFAULT_MIN_CONFIDENCE`] escalates rather than acting, so the router is
/// conservative: it only handles requests it is confident about.
#[derive(Debug)]
pub struct NeedleRouter<E: NeedleEngine> {
    engine: Arc<E>,
    registry: ActionRegistry,
    min_confidence: f32,
}

impl<E: NeedleEngine> NeedleRouter<E> {
    /// A router over the given engine and action registry.
    ///
    /// The engine must have been created with this registry's tool schema —
    /// the router maps the engine's calls back onto the registry.
    pub fn new(engine: Arc<E>, registry: ActionRegistry) -> Self {
        Self {
            engine,
            registry,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
        }
    }

    /// Sets the confidence threshold: calls below it escalate to the model.
    pub fn with_min_confidence(mut self, min: f32) -> Self {
        self.min_confidence = min;
        self
    }
}

impl NeedleRouter<DylibEngine> {
    /// A router over the default action registry and the default engine
    /// discovery path (see [`noema_needle::default_lib_path`]).
    ///
    /// The router's engine is bound to the registry's tools, so it cannot be
    /// shared with another logical Needle agent — create a dedicated engine
    /// per agent, as the plan specifies.
    pub fn from_default() -> Result<Self> {
        let registry = ActionRegistry::builtin();
        let settings = EngineSettings::new(registry.tools_json()).with_system(format!(
            "You are the Noema application router.\n\
             date: {}\n\
             locale: en",
            today()
        ));
        let engine = DylibEngine::from_default(settings).map_err(|error| {
            NoemaError::Router(format!("failed to load the Needle engine: {error}"))
        })?;
        Ok(Self::new(Arc::new(engine), registry))
    }
}

/// Today's date in `YYYY-MM-DD` form (the engine treats system turns as
/// facts, so the router feeds it the date like any other environment fact).
fn today() -> String {
    let now = std::time::SystemTime::now();
    let seconds = now
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = seconds / 86_400;
    // Civil-from-days algorithm (Howard Hinnant).
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[async_trait]
impl<E: NeedleEngine + 'static> Router for NeedleRouter<E> {
    fn id(&self) -> &str {
        "needle-router"
    }

    async fn route(&self, text: &str, _cancel: CancellationToken) -> Result<Route> {
        let text = text.to_string();
        let engine = Arc::clone(&self.engine);
        // Needle's C calls are blocking and the engine keeps one process-
        // global conversation; routing is stateless, so reset before each
        // request (verified: without the reset, context from earlier turns
        // degrades later routes). Keep both off the async executor.
        let response = tokio::task::spawn_blocking(move || -> noema_needle::Result<noema_needle::NeedleResponse> {
            engine.reset()?;
            engine.complete(&text, 256)
        })
        .await
        .map_err(|join| NoemaError::Router(format!("router task failed: {join}")))?
        .map_err(|error| NoemaError::Router(error.to_string()))?;

        match response.calls().first() {
            // Act only on registered actions the router is confident about;
            // anything else escalates (refusals, unregistered tools, and
            // low-confidence calls — the model card's escalation rule).
            Some(call)
                if self.registry.get(&call.name).is_some()
                    && response.confidence.unwrap_or(1.0) >= self.min_confidence =>
            {
                tracing::debug!(
                    action = %call.name,
                    confidence = ?response.confidence,
                    "routed to action"
                );
                Ok(Route::Action(RoutedAction {
                    id: call.name.clone(),
                    arguments: call.arguments.clone(),
                    confidence: response.confidence,
                }))
            }
            Some(call) if self.registry.get(&call.name).is_some() => Ok(Route::Escalate {
                reason: format!(
                    "low confidence ({:?} < {}): uncertain about {}",
                    response.confidence, self.min_confidence, call.name
                ),
            }),
            // The engine refused (no call) or produced a call for an
            // unregistered tool: this request needs the reasoning model.
            _ => Ok(Route::Escalate {
                reason: "no registered action matches the request".into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noema_core::Route;
    use noema_needle::{FunctionCall, NeedleResponse};
    use serde_json::json;

    /// A scripted engine for router tests.
    #[derive(Debug)]
    struct FakeEngine {
        response: std::sync::Mutex<NeedleResponse>,
        prompts: std::sync::Mutex<Vec<String>>,
    }

    impl FakeEngine {
        fn new(response: NeedleResponse) -> Self {
            Self {
                response: std::sync::Mutex::new(response),
                prompts: std::sync::Mutex::new(Vec::new()),        
            }
        }
    }

    impl NeedleEngine for FakeEngine {
        fn id(&self) -> &str {
            "fake"
        }

        fn complete(&self, input: &str, _max_new_tokens: u32) -> noema_needle::Result<NeedleResponse> {
            self.prompts.lock().unwrap().push(input.to_string());
            Ok(self.response.lock().unwrap().clone())
        }

        fn reset(&self) -> noema_needle::Result<()> {
            Ok(())
        }
    }

    fn call_response(name: &str, args: serde_json::Value, confidence: Option<f32>) -> NeedleResponse {
        NeedleResponse {
            response_type: "call".into(),
            function_calls: vec![FunctionCall {
                name: name.into(),
                arguments: args,
            }],
            confidence,
            ..Default::default()
        }
    }

    fn refusal() -> NeedleResponse {
        NeedleResponse {
            response_type: "call".into(),
            function_calls: vec![],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn registered_action_becomes_routed_action() {
        let engine = Arc::new(FakeEngine::new(call_response(
            "open_flashcards",
            json!({}),
            Some(0.9),
        )));
        let router = NeedleRouter::new(engine, ActionRegistry::default());

        let route = router
            .route("open my flashcards", CancellationToken::new())
            .await
            .expect("route");
        match route {
            Route::Action(action) => {
                assert_eq!(action.id, "open_flashcards");
                assert_eq!(action.confidence, Some(0.9));
            }
            other => panic!("expected action, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn low_confidence_call_escalates() {
        let engine = Arc::new(FakeEngine::new(call_response(
            "go_to_settings",
            json!({}),
            Some(0.53),
        )));
        let router = NeedleRouter::new(engine, ActionRegistry::default());

        let route = router
            .route("go to settings", CancellationToken::new())
            .await
            .expect("route");
        assert!(
            matches!(route, Route::Escalate { .. }),
            "sub-threshold calls must escalate, got {route:?}"
        );

        // Raising the threshold is stricter; lowering it acts.
        let engine = Arc::new(FakeEngine::new(call_response(
            "go_to_settings",
            json!({}),
            Some(0.53),
        )));
        let router = NeedleRouter::new(engine, ActionRegistry::default())
            .with_min_confidence(0.4);
        let route = router
            .route("go to settings", CancellationToken::new())
            .await
            .expect("route");
        assert!(matches!(route, Route::Action(_)));
    }

    #[tokio::test]
    async fn refusal_escalates() {
        let engine = Arc::new(FakeEngine::new(refusal()));
        let router = NeedleRouter::new(engine, ActionRegistry::default());

        let route = router
            .route("what is the capital of france", CancellationToken::new())
            .await
            .expect("route");
        assert!(matches!(route, Route::Escalate { .. }));
    }

    #[tokio::test]
    async fn unregistered_tool_name_escalates() {
        let engine = Arc::new(FakeEngine::new(call_response(
            "delete_all_files",
            json!({}),
            None,
        )));
        let router = NeedleRouter::new(engine, ActionRegistry::default());

        let route = router
            .route("delete everything", CancellationToken::new())
            .await
            .expect("route");
        assert!(matches!(route, Route::Escalate { .. }));
    }

    #[tokio::test]
    async fn engine_failure_is_a_router_error() {
        let engine = Arc::new(FakeEngine::new(NeedleResponse {
            response_type: "call".into(),
            function_calls: vec![],
            error: Some("boom".into()),
            ..Default::default()
        }));
        let router = NeedleRouter::new(engine, ActionRegistry::default());
        let route = router
            .route("open my flashcards", CancellationToken::new())
            .await
            .expect("route");
        // An errored-but-parseable envelope is treated as a refusal.
        assert!(matches!(route, Route::Escalate { .. }));
    }
}

//! Initial text routing.
//!
//! Plain-text user requests are routed before any reasoning model runs: a
//! lightweight [`Router`] (Needle 2 in practice) decides whether the request
//! is a simple, deterministic application action — "open my flashcards",
//! "go to settings" — or whether it needs the full model. Handled requests
//! never invoke the reasoning model; everything else escalates to it.

use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::model::ModelResponse;

/// A simple application action produced by the initial text router.
///
/// The action id names an entry in the application's action registry (for
/// example `open_flashcards`); the frontend maps it onto the actual UI
/// behaviour. Arguments carry any evidence the model extracted (currently
/// always an empty object for the parameterless router actions).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutedAction {
    /// The registry id of the action, e.g. `open_flashcards`.
    pub id: String,
    /// Structured arguments for the action.
    pub arguments: serde_json::Value,
    /// The router's confidence in the mapping, when the backend reports one.
    pub confidence: Option<f32>,
}

impl RoutedAction {
    /// A routed action with the given id and no arguments.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            arguments: serde_json::Value::Object(Default::default()),
            confidence: None,
        }
    }
}

/// The outcome of routing a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Route {
    /// The request maps to a simple application action.
    Action(RoutedAction),
    /// The request needs the reasoning model.
    Escalate {
        /// Why the router could not handle it.
        reason: String,
    },
}

/// The initial text router.
///
/// Implementations must be cheap: routing happens on every plain-text user
/// request before the reasoning model runs, so it should take milliseconds,
/// not seconds.
#[async_trait]
pub trait Router: fmt::Debug + Send + Sync + 'static {
    /// A stable identifier for this router instance.
    fn id(&self) -> &str;

    /// Routes a plain-text request to a simple action, or escalates it.
    ///
    /// Implementations should honour [`CancellationToken`]: when cancelled,
    /// they should stop promptly and return an error.
    async fn route(&self, text: &str, cancel: CancellationToken) -> Result<Route>;
}

/// What a [`Session::send`](crate::Session::send) call produced.
///
/// A message can either be handled by the initial text router — the request
/// never reaches the reasoning model — or it falls through to the model and
/// produces a normal [`ModelResponse`].
#[derive(Debug)]
pub enum SendOutcome {
    /// The request was routed to a simple application action.
    Routed(RoutedAction),
    /// The request escalated to the reasoning model, which responded.
    Model(ModelResponse),
}

impl SendOutcome {
    /// The model response, when this outcome was produced by the model.
    pub fn into_model(self) -> Option<ModelResponse> {
        match self {
            SendOutcome::Routed(_) => None,
            SendOutcome::Model(response) => Some(response),
        }
    }

    /// The routed action, when the router handled the request.
    pub fn into_routed(self) -> Option<RoutedAction> {
        match self {
            SendOutcome::Routed(action) => Some(action),
            SendOutcome::Model(_) => None,
        }
    }

    /// Whether the request was handled by the router without the model.
    pub fn is_routed(&self) -> bool {
        matches!(self, SendOutcome::Routed(_))
    }
}

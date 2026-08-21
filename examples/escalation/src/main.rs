//! Cloud escalation, configured from plain fields.
//!
//! This example shows the whole escalation surface working together:
//!
//! 1. The cloud provider is described by three plain fields — **model
//!    name**, **base URL**, and **API key** — read from `NoemaConfig` /
//!    environment variables, and turned into an
//!    [`OpenAICompatibleProvider`] (the bundled general HTTP provider).
//! 2. The escalation policy allows cloud escalation and prefers that
//!    provider.
//! 3. A request the router cannot handle escalates; the policy sends it to
//!    the provider, and the session streams the same model events a local
//!    turn would.
//!
//! No API key ships with this repo, so by default the provider points at an
//! unroutable local address and the run fails *gracefully* — you can see
//! the exact error path. Point `NOEMA_CLOUD_BASE_URL` / `NOEMA_CLOUD_MODEL`
//! / `NOEMA_CLOUD_API_KEY` at a real OpenAI-compatible endpoint to run it
//! for real.
//!
//! Usage:
//!
//! ```sh
//! cargo run -p escalation-example
//! ```

use async_trait::async_trait;
use noema_core::{
    init_logging, LogLevel, Message, Model, ModelRequest, ModelResponse, Noema, NoemaConfig, Role,
    Route, Router,
};
use noema_events::Event;
use noema_provider_http::OpenAICompatibleProvider;
use tokio_util::sync::CancellationToken;

/// A router that always escalates, so the demo reaches the cloud path
/// without needing a real Needle engine.
#[derive(Debug)]
struct AlwaysEscalatingRouter;

#[async_trait]
impl Router for AlwaysEscalatingRouter {
    fn id(&self) -> &str {
        "always-escalating"
    }

    async fn route(&self, _text: &str, _cancel: CancellationToken) -> noema_core::Result<Route> {
        Ok(Route::Escalate {
            reason: "demo request outside routing capabilities (0.21 < 0.6)".into(),
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging(LogLevel::Info)?;

    // 1. The provider is configured with model name, base URL, and API key.
    //    All three come from the environment so the same binary works
    //    against any OpenAI-compatible endpoint without recompiling.
    let model = std::env::var("NOEMA_CLOUD_MODEL").unwrap_or_else(|_| "gemini-2.5-pro".into());
    let base_url =
        std::env::var("NOEMA_CLOUD_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:9999/v1".into());
    let api_key = std::env::var("NOEMA_CLOUD_API_KEY").ok();

    let mut config = NoemaConfig::default();
    config.cloud.enabled = true;
    config.cloud.preferred_provider = Some("openai-compatible".into());
    config.cloud.model = Some(model.clone());
    config.cloud.base_url = Some(base_url.clone());
    config.cloud.api_key = api_key.clone();
    // Half a second is plenty for a local demo; a real endpoint will want
    // more. The session enforces this as the escalation latency limit.
    config.cloud.maximum_latency_ms = Some(500);

    let provider = OpenAICompatibleProvider::new(
        "openai-compatible",
        model.clone(),
        base_url.clone(),
        api_key,
    );

    // The escalation *policy* is the enforcement point: configuration only
    // enables the capability. This policy routes escalations to the cloud
    // provider (never back to the local model) with a latency limit.
    let policy = noema_core::EscalationPolicy {
        allow_local: false,
        allow_cloud: true,
        preferred_provider: Some("openai-compatible".into()),
        maximum_latency: Some(std::time::Duration::from_millis(500)),
        ..noema_core::EscalationPolicy::default()
    };

    let noema = Noema::builder()
        .with_config(config)
        .with_router(AlwaysEscalatingRouter)
        .with_provider(provider)
        .with_escalation_policy(policy)
        .with_model(EchoModel)
        .build()
        .await?;

    println!("=== Escalation configuration ===");
    println!("  provider id : openai-compatible");
    println!("  model       : {model}");
    println!("  base URL    : {base_url}");
    println!(
        "  api key     : {}",
        if base_url.contains("127.0.0.1") || base_url.contains("localhost") {
            "(none needed for a local endpoint)"
        } else {
            "(set via NOEMA_CLOUD_API_KEY)"
        }
    );
    println!("  policy      : cloud allowed, local disabled, latency limit 500ms");
    println!();

    let session = noema.create_session().await?;
    let mut events = session.events();
    println!("sending: \"what is the capital of France?\"");
    println!();

    // The router escalates; the policy routes the escalation to the cloud
    // provider. Without a reachable endpoint this fails gracefully with a
    // clear escalation error.
    let outcome = session
        .send(Message::text(Role::User, "what is the capital of France?"))
        .await;

    // Drain the events of this turn so the streamed model events are
    // visible, then stop (the bus stays open while the session lives).
    let mut seen = 0;
    while let Ok(Some(event)) =
        tokio::time::timeout(std::time::Duration::from_millis(400), events.next()).await
    {
        match event {
            Event::RoutingStarted { .. } => println!("  [event] routing started"),
            Event::RoutingEscalated { .. } => println!("  [event] routing escalated"),
            Event::EscalationStarted { .. } => println!("  [event] escalation started"),
            Event::ModelStarted { model, .. } => println!("  [event] model started: {model}"),
            Event::ModelDelta { delta, .. } => println!("  [event] model delta: {delta}"),
            Event::ModelCompleted { .. } => println!("  [event] model completed"),
            Event::EscalationCompleted { .. } => println!("  [event] escalation completed"),
            Event::Error { error, .. } => println!("  [event] error: {error}"),
            _ => {}
        }
        seen += 1;
        if seen > 32 {
            break;
        }
    }

    println!();
    match outcome {
        Ok(outcome) => println!("send succeeded (unexpected for an unroutable endpoint): {outcome:?}"),
        Err(error) => {
            println!("send failed gracefully with an escalation error:");
            println!("  {error}");
        }
    }
    println!();
    println!("To run against a real provider:");
    println!("  set NOEMA_CLOUD_MODEL, NOEMA_CLOUD_BASE_URL, NOEMA_CLOUD_API_KEY");
    println!("  and raise config.cloud.maximum_latency_ms (e.g. 30_000).");

    session.close().await?;
    Ok(())
}

/// The local reasoning model the escalation comes *from*. A stand-in for
/// the real Gemma adapter: it answers without ever mentioning a tool.
#[derive(Debug)]
struct EchoModel;

#[async_trait]
impl Model for EchoModel {
    fn id(&self) -> &str {
        "echo"
    }

    async fn generate(
        &self,
        _request: ModelRequest,
        _cancel: CancellationToken,
    ) -> noema_core::Result<ModelResponse> {
        Ok(ModelResponse::Text {
            content: "(local model would normally answer here)".into(),
            usage: None,
        })
    }
}

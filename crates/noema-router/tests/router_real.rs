//! Real-engine routing tests (ignored by default; need `prebuilt/needle/`).
//!
//! ```text
//! cargo test -p noema-router --test router_real -- --ignored --nocapture
//! ```

use noema_core::{Route, Router};
use noema_router::NeedleRouter;
use tokio_util::sync::CancellationToken;

#[tokio::test]
#[ignore = "needs the Needle engine"]
async fn default_registry_routes_the_six_actions() {
    let router = NeedleRouter::from_default().expect("router loads");

    let cases = [
        ("Open my flashcards", "open_flashcards"),
        ("Show me my notes", "show_notes"),
        ("Open my PDFs", "open_pdfs"),
        ("Go to settings", "go_to_settings"),
        ("Start a revision session", "start_revision"),
        ("Open the last document", "open_last_document"),
    ];

    // Routing is stateless, so a sequence (not just single calls) must work.
    for (prompt, expected) in cases {
        let route = router
            .route(prompt, CancellationToken::new())
            .await
            .expect("route");
        match route {
            Route::Action(action) => {
                assert_eq!(
                    action.id, expected,
                    "{prompt:?} should route to {expected}, got {}",
                    action.id
                );
            }
            Route::Escalate { reason } => {
                panic!("{prompt:?} should route to {expected}, but escalated: {reason}");
            }
        }
        eprintln!("{prompt:?} -> {expected}");
    }
}

#[tokio::test]
#[ignore = "needs the Needle engine"]
async fn questions_escalate() {
    let router = NeedleRouter::from_default().expect("router loads");
    for prompt in [
        "What is the capital of France?",
        "Explain quantum mechanics",
        "Write a poem about the ocean",
    ] {
        let route = router
            .route(prompt, CancellationToken::new())
            .await
            .expect("route");
        assert!(
            matches!(route, Route::Escalate { .. }),
            "{prompt:?} should escalate, got {route:?}"
        );
    }
}

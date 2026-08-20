//! Integration tests for the session lifecycle, exercised through the
//! public `noema-api` surface exactly as the Agora frontend would use it.

use noema_api::prelude::*;

#[tokio::test]
async fn frontend_can_build_runtime_and_manage_sessions() {
    let noema = Noema::builder().build().await.expect("build runtime");

    let session = noema.create_session().await.expect("create session");
    assert_eq!(session.state().await, SessionState::Active);

    noema.close_session(session.id()).await.expect("close session");
    assert_eq!(session.state().await, SessionState::Closed);
}

#[tokio::test]
async fn frontend_can_close_session_directly() {
    let noema = Noema::builder().build().await.expect("build runtime");
    let session = noema.create_session().await.expect("create session");

    session.close().await.expect("close session");
    assert_eq!(session.state().await, SessionState::Closed);

    // The state is shared: closing again through either path fails.
    assert!(session.close().await.is_err());
    assert!(noema.close_session(session.id()).await.is_err());
}

#[tokio::test]
async fn session_events_stream_lifecycle() {
    let noema = Noema::builder().build().await.expect("build runtime");
    let mut events = noema.subscribe_all();

    let session = noema.create_session().await.expect("create session");
    noema.close_session(session.id()).await.expect("close session");

    let first = events.next().await.expect("session started");
    assert!(matches!(first, Event::SessionStarted { .. }));

    let second = events.next().await.expect("session completed");
    assert!(matches!(second, Event::SessionCompleted { .. }));
}

#[tokio::test]
async fn session_filtered_stream_only_sees_its_session() {
    let noema = Noema::builder().build().await.expect("build runtime");
    let a = noema.create_session().await.expect("create a");
    let b = noema.create_session().await.expect("create b");

    let mut b_events = noema.subscribe(b.id().clone());

    // Close the other session first: it must not appear on b's stream.
    noema.close_session(a.id()).await.expect("close a");
    noema.close_session(b.id()).await.expect("close b");

    let event = b_events.next().await.expect("event for b");
    assert!(matches!(event, Event::SessionCompleted { .. }));
    assert_eq!(event.session_id(), Some(b.id()));
}

#[tokio::test]
async fn multiple_sessions_coexist() {
    let noema = Noema::builder().build().await.expect("build runtime");
    let sessions = vec![
        noema.create_session().await.expect("create 1"),
        noema.create_session().await.expect("create 2"),
        noema.create_session().await.expect("create 3"),
    ];

    let ids: Vec<_> = sessions.iter().map(|s| s.id().clone()).collect();
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), 3);

    for session in sessions {
        noema.close_session(session.id()).await.expect("close");
    }
}

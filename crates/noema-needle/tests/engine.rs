//! Integration tests that run the real Needle 2 engine.
//!
//! These tests exercise the official engine end to end: tool-call
//! generation, the refusal contract for unsupported input, multi-turn
//! conversation, `reset`, and the CLI fallback backend.
//!
//! The engine library is found through the normal discovery path
//! (`NEEDLE_LIB_PATH`, `prebuilt/needle/<platform>/`, or the
//! `~/.cache/cactus-needle/` cache). When no engine is present the tests
//! print a note and pass, so `cargo test` stays green on machines without
//! the artifacts.

use std::path::PathBuf;
use std::sync::Mutex;

use noema_needle::{CliEngine, DylibEngine, EngineSettings, NeedleEngine};

/// The C engine keeps one process-global session; serialise the real-engine
/// tests so conversations do not interleave.
static ENGINE_TESTS: Mutex<()> = Mutex::new(());

const TOOLS: &str = r#"[
    {
        "name": "set_lights",
        "description": "Turn a room's lights on or off and set brightness",
        "parameters": {
            "type": "object",
            "properties": {
                "room": { "type": "string", "description": "which room to control" },
                "on": { "type": "boolean" },
                "brightness": { "type": "integer", "minimum": 0, "maximum": 100 }
            },
            "required": ["room", "on"]
        }
    },
    {
        "name": "play_music",
        "description": "Play music matching a mood, genre, or artist",
        "parameters": {
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"]
        }
    }
]"#;

fn engine_lib() -> Option<PathBuf> {
    noema_needle::default_lib_path()
}

fn dylib_engine() -> Option<DylibEngine> {
    let path = engine_lib()?;
    Some(
        DylibEngine::load(path, EngineSettings::new(TOOLS))
            .expect("engine library should load"),
    )
}

#[test]
fn tool_call_is_structured_and_schema_conformant() {
    let _guard = ENGINE_TESTS.lock().unwrap();
    let Some(engine) = dylib_engine() else {
        eprintln!("skipped: needle engine library not found");
        return;
    };

    let response = engine.complete("dim the living room to 30", 256).expect("complete");

    assert!(response.is_call(), "expected a call, got {:?}", response.response_type);
    assert_eq!(response.calls().len(), 1);
    let call = &response.calls()[0];
    assert_eq!(call.name, "set_lights");

    let args = call.arguments_map().expect("arguments should be an object");
    assert_eq!(args["room"], "living room");
    assert_eq!(args["brightness"], 30);
    assert_eq!(args["on"], true);

    // The confidence head should report a calibrated score.
    let confidence = response.confidence.expect("base model reports confidence");
    assert!((0.0..=1.0).contains(&confidence), "confidence out of range: {confidence}");
}

#[test]
fn unsupported_input_is_refused_with_empty_calls() {
    let _guard = ENGINE_TESTS.lock().unwrap();
    let Some(engine) = dylib_engine() else {
        eprintln!("skipped: needle engine library not found");
        return;
    };

    let response = engine
        .complete("what is the capital of france?", 256)
        .expect("complete");

    // The documented refusal contract: no declared tool can serve this, so
    // the model answers with the empty call list.
    assert!(
        response.calls().is_empty(),
        "expected a refusal with no calls: {response:?}"
    );
}

#[test]
fn multi_turn_conversation_continues_after_result() {
    let _guard = ENGINE_TESTS.lock().unwrap();
    let Some(engine) = dylib_engine() else {
        eprintln!("skipped: needle engine library not found");
        return;
    };

    let first = engine.complete("dim the living room to 30", 256).expect("complete");
    assert!(first.is_call());
    let call = first.calls()[0].clone();

    // Execute the call and feed the result back as the next turn.
    let result = serde_json::json!({ "ok": true, "name": call.name, "arguments": call.arguments });
    let second = engine.complete(&result.to_string(), 256).expect("complete");

    // A final step answers from the results with type "respond" and no calls.
    assert!(
        second.is_respond(),
        "expected the loop to finish with respond, got {:?}",
        second.response_type
    );
    assert!(second.calls().is_empty());
}

#[test]
fn reset_rewinds_the_conversation() {
    let _guard = ENGINE_TESTS.lock().unwrap();
    let Some(engine) = dylib_engine() else {
        eprintln!("skipped: needle engine library not found");
        return;
    };

    let first = engine.complete("dim the living room to 30", 256).expect("complete");
    assert!(first.is_call());

    engine.reset().expect("reset");

    // After rewinding, the conversation is gone and unsupported input is
    // refused again.
    let response = engine
        .complete("what is the capital of france?", 256)
        .expect("complete");
    assert!(
        response.calls().is_empty(),
        "expected a refusal after reset: {response:?}"
    );
}

#[test]
fn explicit_path_load_and_single_turn() {
    let _guard = ENGINE_TESTS.lock().unwrap();
    let Some(path) = engine_lib() else {
        eprintln!("skipped: needle engine library not found");
        return;
    };

    let engine = DylibEngine::load(path, EngineSettings::new(TOOLS)).expect("load");
    let response = engine.complete("play something relaxing", 256).expect("complete");
    assert!(
        !response.calls().is_empty(),
        "expected a music tool call: {response:?}"
    );
    assert_eq!(response.calls()[0].name, "play_music");
}

#[test]
fn cli_backend_runs_a_tool_call() {
    let _guard = ENGINE_TESTS.lock().unwrap();
    let Some(binary) = noema_needle::default_cli_path() else {
        eprintln!("skipped: needle CLI binary not found");
        return;
    };

    let engine = CliEngine::new(binary, EngineSettings::new(TOOLS)).expect("create");
    let response = engine.complete("dim the living room to 30", 256).expect("complete");

    assert!(response.is_call(), "expected a call: {response:?}");
    assert_eq!(response.calls().len(), 1);
    assert_eq!(response.calls()[0].name, "set_lights");
    let args = response.calls()[0].arguments_map().expect("object");
    assert_eq!(args["room"], "living room");
}

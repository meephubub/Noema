//! Runs the Needle 2 tool-calling model end to end.
//!
//! Usage:
//!
//! ```sh
//! cargo run -p needle-example
//! ```
//!
//! The engine library is found via `NEEDLE_LIB_PATH`, the repository's
//! `prebuilt/needle/<platform>/` directory, or the shared
//! `~/.cache/cactus-needle/` cache.

use noema_needle::{DylibEngine, EngineSettings, NeedleEngine};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = DylibEngine::from_default(EngineSettings::new(TOOLS))?;
    println!("engine bound: {}\n", engine.id());

    let first = engine.complete("dim the living room to 30", 256)?;
    print_envelope("turn 1", &first)?;

    // Feed the (simulated) tool result back; the conversation continues.
    if let Some(call) = first.calls().first() {
        let result = serde_json::json!({
            "ok": true,
            "name": call.name,
            "arguments": call.arguments,
        });
        let follow_up = engine.complete(&result.to_string(), 256)?;
        print_envelope("turn 2 (result fed back)", &follow_up)?;
    }

    // Rewind the conversation, then check the refusal behaviour for
    // unsupported input.
    engine.reset()?;
    let off_topic = engine.complete("what is the capital of france?", 256)?;
    print_envelope("off-topic (after reset)", &off_topic)?;

    if off_topic.is_refusal() {
        println!("→ correctly refused as unsupported input");
    } else {
        println!("→ note: engine did not return the empty-call refusal");
    }

    Ok(())
}

fn print_envelope(
    label: &str,
    response: &noema_needle::NeedleResponse,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- {label} ---");
    println!(
        "type={} confidence={:?} calls={}",
        response.response_type,
        response.confidence,
        response.calls().len()
    );
    for call in response.calls() {
        println!("  call: {}", serde_json::to_string_pretty(call)?);
    }
    if let Some(reasoning) = &response.reasoning {
        println!("  reasoning: {reasoning}");
    }
    println!();
    Ok(())
}

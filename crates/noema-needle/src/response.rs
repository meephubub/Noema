//! The JSON envelope the Needle 2 engine returns from every turn.
//!
//! The engine is a tool-calling model: text goes in, a structured tool call
//! comes out. Every turn returns one object with the shape documented by
//! Cactus Compute (see the model card and `doc/apis.md` in
//! `cactus-compute/needle`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A single structured tool call produced by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The declared tool name.
    pub name: String,
    /// The arguments, grammar-guaranteed to match the tool's schema.
    ///
    /// Optional arguments with no evidence in the input are omitted by the
    /// model, so keys may be absent.
    pub arguments: Value,
}

impl FunctionCall {
    /// The arguments as a JSON object, when the model emitted an object.
    pub fn arguments_map(&self) -> Option<&serde_json::Map<String, Value>> {
        self.arguments.as_object()
    }
}

/// The `validation` section of the envelope, when present.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ValidationInfo {
    /// Argument paths the model could not ground in the input.
    #[serde(default)]
    pub ungrounded: Vec<Value>,
    /// Whether the call was produced under a negated instruction.
    #[serde(default)]
    pub negation: Option<bool>,
}

/// The complete envelope returned by `needle_complete`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NeedleResponse {
    /// `"call"` when the model wants tool calls (an empty `function_calls`
    /// list is the refusal for unsupported input), `"respond"` when the
    /// turn is finished and the answer is carried by the tool results.
    #[serde(rename = "type", default)]
    pub response_type: String,

    /// Whether the turn succeeded.
    #[serde(default)]
    pub success: Option<bool>,

    /// A human-readable error, when the engine reported one.
    #[serde(default)]
    pub error: Option<String>,

    /// A structured error code, when the engine reported one.
    #[serde(default)]
    pub error_code: Option<Value>,

    /// A short reason, when the engine reported one.
    #[serde(default)]
    pub reason: Option<String>,

    /// The tool calls the model wants executed.
    #[serde(default)]
    pub function_calls: Vec<FunctionCall>,

    /// The model's short derivation of each argument from its source span.
    #[serde(default)]
    pub reasoning: Option<String>,

    /// The calibrated confidence score in `[0, 1]`.
    ///
    /// `None` for tuned (`.cact`) weights, whose confidence head is not
    /// calibrated.
    #[serde(default)]
    pub confidence: Option<f32>,

    /// Prompt-processing throughput, in tokens per second.
    #[serde(default)]
    pub prefill_tps: Option<f32>,

    /// Decode throughput, in tokens per second.
    #[serde(default)]
    pub decode_tps: Option<f32>,

    /// Peak session memory, in megabytes.
    #[serde(default)]
    pub peak_ram_mb: Option<f32>,

    /// Grounding validation, when the engine reports it.
    #[serde(default)]
    pub validation: Option<ValidationInfo>,
}

impl NeedleResponse {
    /// Whether the model wants tool calls executed (`type == "call"`).
    pub fn is_call(&self) -> bool {
        self.response_type == "call"
    }

    /// Whether the turn is finished (`type == "respond"`).
    pub fn is_respond(&self) -> bool {
        self.response_type == "respond"
    }

    /// The tool calls, if any.
    pub fn calls(&self) -> &[FunctionCall] {
        &self.function_calls
    }

    /// Whether this is the documented refusal for unsupported input: a call
    /// turn with no calls at all.
    pub fn is_refusal(&self) -> bool {
        self.is_call() && self.function_calls.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENVELOPE: &str = r#"{
        "type": "call",
        "success": true,
        "error": null,
        "error_code": null,
        "reason": null,
        "function_calls": [
            {
                "name": "set_lights",
                "arguments": { "room": "living room", "on": true, "brightness": 30 }
            }
        ],
        "reasoning": "'living room' -> room; 'dim' -> on true, brightness 30",
        "confidence": 0.94,
        "prefill_tps": 4300.0,
        "decode_tps": 850.0,
        "peak_ram_mb": 28.5,
        "validation": { "ungrounded": [], "negation": false }
    }"#;

    #[test]
    fn parses_the_documented_envelope() {
        let response: NeedleResponse = serde_json::from_str(ENVELOPE).expect("parse");
        assert!(response.is_call());
        assert!(!response.is_refusal());
        assert_eq!(response.confidence, Some(0.94));
        let call = &response.function_calls[0];
        assert_eq!(call.name, "set_lights");
        let args = call.arguments_map().expect("object");
        assert_eq!(args["room"], "living room");
        assert_eq!(args["brightness"], 30);
    }

    #[test]
    fn missing_fields_default() {
        let response: NeedleResponse =
            serde_json::from_str(r#"{ "type": "call", "function_calls": [] }"#)
                .expect("parse");
        assert!(response.is_refusal());
        assert_eq!(response.confidence, None);
        assert_eq!(response.success, None);
        assert!(response.validation.is_none());
    }

    #[test]
    fn round_trips() {
        let response: NeedleResponse = serde_json::from_str(ENVELOPE).expect("parse");
        let encoded = serde_json::to_string(&response).expect("encode");
        let decoded: NeedleResponse = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(response, decoded);
    }
}

//! The tool-specific Needle formatter.
//!
//! One physical Needle 2 model, many logical agents: each tool gets its own
//! [`NeedleToolFormatter`], bound to that tool's complete schema and any
//! tool-provided instructions. The reasoning model issues a *semantic*
//! request ("find the file abc.exe"); this formatter turns it into a
//! validated, structured [`ToolCall`] the registry can execute.
//!
//! ```text
//! Gemma semantic request
//!     ↓
//! NeedleToolFormatter (Needle 2, bound to one tool's schema)
//!     ↓
//! ToolCall { tool, arguments }
//!     ↓
//! registry.validate + execute
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use noema_core::{NoemaError, Result, ToolFormatter};
use noema_needle::{DylibEngine, EngineSettings, NeedleEngine};
use noema_tools::{ToolCall, ToolSchema};
use tokio_util::sync::CancellationToken;

/// The default confidence below which a formatted call is refused.
///
/// Deliberately much lower than the text router's threshold: the router
/// decides *whether to act at all*, while the formatter's job is producing
/// the structured call, which is schema-validated before execution anyway.
/// The engine's calibrated confidence head is routing-tuned, so absolute
/// values for tool-formatting tasks run far lower (a perfect evident call
/// can score ~0.2 while refusals sit near 0.0). 0.15 accepts every
/// evidence-backed call and still refuses the genuinely uncertain ones
/// (e.g. 0.06 after a confused reply).
pub const DEFAULT_FORMATTER_MIN_CONFIDENCE: f32 = 0.15;

/// The system prompt prefix every tool formatter uses. The tool's own
/// instructions (when provided) are appended; the schema is always bound to
/// the engine itself.
const FORMATTER_PROMPT: &str = "\
You are the Noema tool formatter.
Convert the requested operation into the tool call schema bound to this \
session. Produce exactly one call. Use only values evidenced in the request; \
omit optional fields with no evidence. If the request cannot be served by \
the bound tool, refuse with no call.";

/// A tool-specific logical Needle agent.
///
/// The engine must have been created with the tool's schema bound (see
/// [`NeedleToolFormatter::from_tool`]). A call below the confidence
/// threshold, a call for a different tool, or a refusal is an error so the
/// caller can decide how to respond.
#[derive(Debug)]
pub struct NeedleToolFormatter<E: NeedleEngine> {
    engine: Arc<E>,
    tool_name: String,
    min_confidence: f32,
}

impl<E: NeedleEngine> NeedleToolFormatter<E> {
    /// A formatter over the given engine, which must be bound to `tool_name`'s
    /// schema.
    pub fn new(engine: Arc<E>, tool_name: impl Into<String>) -> Self {
        Self {
            engine,
            tool_name: tool_name.into(),
            min_confidence: DEFAULT_FORMATTER_MIN_CONFIDENCE,
        }
    }

    /// Sets the confidence threshold: calls below it are refused.
    pub fn with_min_confidence(mut self, min: f32) -> Self {
        self.min_confidence = min;
        self
    }
}

impl NeedleToolFormatter<DylibEngine> {
    /// Creates a logical Needle agent for one tool.
    ///
    /// Builds a dedicated engine bound to the tool's complete schema and a
    /// system turn combining the formatter prompt with the tool's own
    /// instructions (if any). The engine is process-global, so each logical
    /// agent gets its own [`DylibEngine`]; only one stays bound at a time,
    /// and every request resets first, so sharing the physical model is
    /// safe.
    pub fn from_tool(schema: &ToolSchema, instructions: Option<&str>) -> Result<Self> {
        let mut system = String::from(FORMATTER_PROMPT);
        if let Some(instructions) = instructions {
            system.push_str("\n\n");
            system.push_str(instructions);
        }
        // The engine binds a JSON *array* of tools; a single tool's schema
        // is wrapped here (the router's registry emits the same shape).
        let tools_json = format!("[{}]", schema.needle_json());
        let settings = EngineSettings::new(tools_json).with_system(system);
        let engine = DylibEngine::from_default(settings).map_err(|error| {
            NoemaError::Tool(format!(
                "failed to load the Needle engine for tool '{}': {error}",
                schema.name
            ))
        })?;
        Ok(Self::new(Arc::new(engine), schema.name.clone()))
    }
}

#[async_trait]
impl<E: NeedleEngine + 'static> ToolFormatter for NeedleToolFormatter<E> {
    fn id(&self) -> &str {
        "needle-tool-formatter"
    }

    async fn format(
        &self,
        schema: ToolSchema,
        request: &str,
        _cancel: CancellationToken,
    ) -> Result<ToolCall> {
        let request_owned = request.to_string();
        let engine = Arc::clone(&self.engine);
        // Needle's C calls are blocking and keep one process-global
        // conversation; formatting is stateless, so reset before each
        // request (same rule as the text router). Keep both off the async
        // executor.
        let request_for_engine = request_owned.clone();
        let response =
            tokio::task::spawn_blocking(move || -> noema_needle::Result<noema_needle::NeedleResponse> {
                engine.reset()?;
                engine.complete(&request_for_engine, 256)
            })
            .await
            .map_err(|join| NoemaError::Tool(format!("formatter task failed: {join}")))?
            .map_err(|error| NoemaError::Tool(error.to_string()))?;

        let confidence = response.confidence.unwrap_or(1.0);
        let call = response.calls().first().cloned().ok_or_else(|| {
            NoemaError::Tool(format!(
                "the tool formatter refused '{request_owned}' for tool '{}'",
                self.tool_name
            ))
        })?;

        if call.name != self.tool_name {
            return Err(NoemaError::Tool(format!(
                "the tool formatter produced a call for '{}', expected '{}'",
                call.name, self.tool_name
            )));
        }
        if confidence < self.min_confidence {
            return Err(NoemaError::Tool(format!(
                "low confidence ({confidence:?} < {}): uncertain call for '{}'",
                self.min_confidence, self.tool_name
            )));
        }

        let call = ToolCall::with_arguments(call.name.clone(), call.arguments.clone());
        schema.validate_arguments(&call.arguments)?;
        tracing::debug!(
            tool = %self.tool_name,
            confidence = ?response.confidence,
            "formatted tool call"
        );
        Ok(call)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noema_needle::{FunctionCall, NeedleResponse};
    use serde_json::json;

    /// A scripted engine for formatter tests.
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

        fn complete(
            &self,
            input: &str,
            _max_new_tokens: u32,
        ) -> noema_needle::Result<NeedleResponse> {
            self.prompts.lock().unwrap().push(input.to_string());
            Ok(self.response.lock().unwrap().clone())
        }

        fn reset(&self) -> noema_needle::Result<()> {
            Ok(())
        }
    }

    fn search_schema() -> ToolSchema {
        ToolSchema {
            name: "search_files".into(),
            description: "Search for files on the local system".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
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
    async fn semantic_request_becomes_a_validated_call() {
        let engine = Arc::new(FakeEngine::new(call_response(
            "search_files",
            json!({ "query": "abc.exe" }),
            Some(0.92),
        )));
        let formatter = NeedleToolFormatter::new(engine, "search_files");

        let call = formatter
            .format(search_schema(), "find the file abc.exe", CancellationToken::new())
            .await
            .expect("format");
        assert_eq!(call.tool, "search_files");
        assert_eq!(call.arguments["query"], "abc.exe");
    }

    #[tokio::test]
    async fn refusal_is_an_error() {
        let engine = Arc::new(FakeEngine::new(refusal()));
        let formatter = NeedleToolFormatter::new(engine, "search_files");
        let err = formatter
            .format(search_schema(), "explain quantum physics", CancellationToken::new())
            .await
            .expect_err("refusal");
        assert!(err.to_string().contains("refused"));
    }

    #[tokio::test]
    async fn wrong_tool_name_is_an_error() {
        let engine = Arc::new(FakeEngine::new(call_response(
            "delete_all_files",
            json!({}),
            Some(0.99),
        )));
        let formatter = NeedleToolFormatter::new(engine, "search_files");
        let err = formatter
            .format(search_schema(), "delete everything", CancellationToken::new())
            .await
            .expect_err("wrong tool");
        assert!(err.to_string().contains("delete_all_files"));
    }

    #[tokio::test]
    async fn low_confidence_is_an_error() {
        // A strict threshold refuses uncertain calls.
        let engine = Arc::new(FakeEngine::new(call_response(
            "search_files",
            json!({ "query": "abc" }),
            Some(0.42),
        )));
        let formatter =
            NeedleToolFormatter::new(engine, "search_files").with_min_confidence(0.6);
        let err = formatter
            .format(search_schema(), "find abc", CancellationToken::new())
            .await
            .expect_err("low confidence");
        assert!(err.to_string().contains("low confidence"));

        // Raising the threshold is stricter; the default formats.
        let engine = Arc::new(FakeEngine::new(call_response(
            "search_files",
            json!({ "query": "abc" }),
            Some(0.42),
        )));
        let formatter = NeedleToolFormatter::new(engine, "search_files");
        formatter
            .format(search_schema(), "find abc", CancellationToken::new())
            .await
            .expect("within default threshold");
    }

    #[tokio::test]
    async fn missing_required_arguments_fail_validation() {
        let engine = Arc::new(FakeEngine::new(call_response(
            "search_files",
            json!({}),
            Some(0.9),
        )));
        let formatter = NeedleToolFormatter::new(engine, "search_files");
        let err = formatter
            .format(search_schema(), "find something", CancellationToken::new())
            .await
            .expect_err("missing required query");
        assert!(err.to_string().contains("query"));
    }
}

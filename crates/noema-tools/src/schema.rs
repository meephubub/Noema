//! Tool schemas.
//!
//! A [`ToolSchema`] describes one tool's callable surface. It is the
//! *Needle-facing* form: the tool-specific Needle agent receives the full
//! schema (and any tool-provided instructions) and produces a structured
//! call. Gemma, by contrast, only ever sees the lightweight
//! [`ToolSummary`](crate::ToolSummary) — never this schema — to keep the
//! reasoning model's context small.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{Result, ToolError};

/// The schema of a single tool call.
///
/// `parameters` is a JSON Schema object (type `object` with `properties` and
/// optionally `required`). The schema serializes to the engine's tool format
/// with the `name` key first: the Needle engine's grammar-driven schema
/// parser is key-order sensitive, so `name` must precede `description`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    /// The tool name, e.g. `search_files`. Must be unique within a registry.
    pub name: String,
    /// A short human description of what the tool does.
    pub description: String,
    /// The JSON Schema object describing the tool's parameters.
    #[serde(default = "object_parameters")]
    pub parameters: Value,
}

fn object_parameters() -> Value {
    json!({ "type": "object", "properties": {} })
}

impl ToolSchema {
    /// Builds a schema with an empty parameter object.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: object_parameters(),
        }
    }

    /// The parameter names the schema marks as required.
    pub fn required(&self) -> Vec<String> {
        self.parameters
            .get("required")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The declared property names, in insertion order.
    pub fn properties(&self) -> Vec<String> {
        self.parameters
            .get("properties")
            .and_then(Value::as_object)
            .map(|props| props.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// The engine-form schema JSON: `name` first, then `description`, then
    /// `parameters` (see the type docs for why the order matters).
    pub fn needle_json(&self) -> String {
        let value = json!({
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        });
        let json = serde_json::to_string(&value).expect("tool schema serializes");
        debug_assert!(json.starts_with("{\"name\":"));
        json
    }

    /// Lightweight schema validation: arguments must be an object, and every
    /// required parameter must be present. Full JSON-Schema conformance is a
    /// later hardening milestone; this guards the model/engine boundary.
    pub fn validate_arguments(&self, arguments: &Value) -> Result<()> {
        let object = arguments.as_object().ok_or_else(|| {
            ToolError::InvalidCall(format!(
                "arguments for '{}' must be a JSON object",
                self.name
            ))
        })?;
        for required in self.required() {
            if !object.contains_key(&required) {
                return Err(ToolError::InvalidCall(format!(
                    "missing required parameter '{required}' for tool '{}'",
                    self.name
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search_schema() -> ToolSchema {
        ToolSchema {
            name: "search_files".into(),
            description: "Search for files on the local system".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "what to look for" },
                    "path": { "type": "string", "description": "where to look" }
                },
                "required": ["query"]
            }),
        }
    }

    #[test]
    fn required_and_properties_are_read_from_the_schema() {
        let schema = search_schema();
        assert_eq!(schema.required(), vec!["query"]);
        assert_eq!(schema.properties(), vec!["query", "path"]);
    }

    #[test]
    fn needle_json_puts_name_first() {
        let json = search_schema().needle_json();
        assert!(
            json.starts_with("{\"name\":\"search_files\""),
            "expected name first, got: {json}"
        );
        let parsed: Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["name"], "search_files");
        assert_eq!(parsed["parameters"]["type"], "object");
    }

    #[test]
    fn validation_enforces_required_parameters() {
        let schema = search_schema();
        schema
            .validate_arguments(&json!({ "query": "abc.exe" }))
            .expect("valid call");
        let err = schema
            .validate_arguments(&json!({ "path": "/tmp" }))
            .expect_err("missing required query");
        assert!(err.to_string().contains("query"));
        let err = schema
            .validate_arguments(&json!([1, 2]))
            .expect_err("non-object arguments");
        assert!(err.to_string().contains("object"));
    }

    #[test]
    fn empty_schema_accepts_empty_arguments() {
        let schema = ToolSchema::new("ping", "No-op");
        schema
            .validate_arguments(&json!({}))
            .expect("no required parameters");
        schema
            .validate_arguments(&json!({ "extra": true }))
            .expect("extra keys are fine");
    }
}

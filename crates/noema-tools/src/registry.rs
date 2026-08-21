//! The central tool registry.
//!
//! Every tool Noema knows about is registered here. The registry owns the
//! three views of a tool described in [`crate::metadata`]:
//!
//! * [`ToolRegistry::gemma_summaries`] / [`ToolRegistry::gemma_tool_section`]
//!   — what the reasoning model sees (lightweight, schema-free).
//! * [`ToolRegistry::needle_tools_json`] / [`ToolRegistry::tool_needle_json`]
//!   — the complete schemas the tool-specific Needle agents bind to.
//! * [`ToolRegistry::execute`] — validated execution behind the
//!   [`NoemaTool`](crate::NoemaTool) contract.
//!
//! Adding a tool is registration, not code changes: a third-party `noema-*`
//! crate implements [`NoemaTool`](crate::NoemaTool) and is registered here.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::metadata::ToolSummary;
use crate::tool::{summarize, NoemaTool, ToolCall, ToolResult};
use crate::{Result, ToolError};

/// A registry of [`NoemaTool`] instances.
///
/// Tool names must be unique within a registry. The registry is cheap to
/// clone (tools are `Arc`d), so a runtime can hand the same registry to
/// every session.
#[derive(Debug, Default, Clone)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn NoemaTool>>,
}

impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool, taking ownership of it.
    ///
    /// Fails if a tool with the same name is already registered.
    pub fn register<T: NoemaTool + 'static>(&mut self, tool: T) -> Result<()> {
        self.register_shared(Arc::new(tool))
    }

    /// Registers a shared tool handle.
    ///
    /// Fails if a tool with the same name is already registered.
    pub fn register_shared(&mut self, tool: Arc<dyn NoemaTool>) -> Result<()> {
        let name = tool.metadata().name;
        if self.tools.contains_key(&name) {
            return Err(ToolError::Duplicate(name));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Merges every tool from `other` into this registry.
    ///
    /// Fails on the first name collision, leaving this registry unchanged.
    pub fn extend(&mut self, other: &ToolRegistry) -> Result<()> {
        for name in other.tools.keys() {
            if self.tools.contains_key(name) {
                return Err(ToolError::Duplicate(name.clone()));
            }
        }
        self.tools.extend(other.tools.clone());
        Ok(())
    }

    /// Looks up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn NoemaTool>> {
        self.tools.get(name).cloned()
    }

    /// The registered tool names, in sorted order.
    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Iterates over the registered tools in name order.
    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn NoemaTool>> {
        self.tools.values()
    }

    /// How many tools are registered.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The lightweight summaries Gemma sees.
    ///
    /// Exactly what the plan prescribes for the reasoning model: tool name,
    /// owning crate, and a one-line description — never the schema.
    pub fn gemma_summaries(&self) -> Vec<ToolSummary> {
        self.tools.values().map(|tool| summarize(tool.as_ref())).collect()
    }

    /// The dynamic "available tools" section for the Gemma system prompt.
    ///
    /// ```text
    /// Available tools:
    ///   search_files
    ///     crate: noema-filesearch
    ///     description: Search for files on the local system
    /// ```
    ///
    /// Schema-free by construction; the full schemas stay with Needle.
    pub fn gemma_tool_section(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }
        let mut section = String::from("Available tools:\n");
        for summary in self.gemma_summaries() {
            section.push_str(&format!(
                "  {}\n    crate: {}\n    description: {}\n",
                summary.name, summary.crate_name, summary.description
            ));
        }
        section
    }

    /// The complete schema JSON array for every registered tool, in the
    /// engine's tool format.
    ///
    /// `name` serializes before `description` (the engine's schema parser is
    /// key-order sensitive; see [`ToolSchema::needle_json`](crate::ToolSchema::needle_json)).
    pub fn needle_tools_json(&self) -> String {
        let tools: Vec<Value> = self
            .tools
            .values()
            .map(|tool| {
                let schema = tool.schema();
                json!({
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.parameters,
                })
            })
            .collect();
        let json = serde_json::to_string(&tools).expect("tool schemas serialize");
        if let Some(first) = self.tools.values().next() {
            debug_assert!(json.contains(&format!("\"name\":\"{}\"", first.schema().name)));
        }
        json
    }

    /// The engine-form schema JSON for a single tool.
    pub fn tool_needle_json(&self, name: &str) -> Result<String> {
        let tool = self.get(name).ok_or_else(|| ToolError::NotFound(name.into()))?;
        Ok(tool.schema().needle_json())
    }

    /// Validates a call against its tool's schema without executing it.
    ///
    /// Checks that the tool exists and that the arguments satisfy the
    /// schema's `required` list. Never execute an unvalidated call.
    pub fn validate_call(&self, call: &ToolCall) -> Result<()> {
        let tool = self
            .get(&call.tool)
            .ok_or_else(|| ToolError::NotFound(call.tool.clone()))?;
        tool.schema().validate_arguments(&call.arguments)
    }

    /// Validates and executes a tool call.
    pub async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        self.validate_call(&call)?;
        let tool = self.get(&call.tool).expect("validated call's tool exists");
        tool.execute(call).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::ToolMetadata;
    use crate::risk::RiskLevel;
    use async_trait::async_trait;

    #[derive(Debug)]
    struct EchoTool {
        name: &'static str,
        risk: RiskLevel,
    }

    #[async_trait]
    impl NoemaTool for EchoTool {
        fn metadata(&self) -> ToolMetadata {
            ToolMetadata {
                name: self.name.into(),
                crate_name: "noema-test".into(),
                description: "Echoes its message argument".into(),
                risk: self.risk,
            }
        }

        fn schema(&self) -> crate::ToolSchema {
            crate::ToolSchema {
                name: self.name.into(),
                description: "Echoes its message argument".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    },
                    "required": ["message"]
                }),
            }
        }

        async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
            let message = call
                .arguments
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("(none)");
            Ok(ToolResult::ok_with_data(format!("echo: {message}"), call.arguments))
        }
    }

    fn sample_registry() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry
            .register(EchoTool { name: "echo", risk: RiskLevel::None })
            .expect("register echo");
        registry
            .register(EchoTool { name: "delete", risk: RiskLevel::Critical })
            .expect("register delete");
        registry
    }

    #[test]
    fn registration_rejects_duplicates() {
        let mut registry = sample_registry();
        let err = registry
            .register(EchoTool { name: "echo", risk: RiskLevel::None })
            .expect_err("duplicate name");
        assert!(matches!(err, ToolError::Duplicate(_)));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn extend_merges_without_collisions() {
        let mut a = sample_registry();
        let mut b = ToolRegistry::new();
        b.register(EchoTool { name: "extra", risk: RiskLevel::Low }).expect("register");
        a.extend(&b).expect("merge");
        assert_eq!(a.names(), vec!["delete", "echo", "extra"]);
        assert!(a.get("extra").is_some());
    }

    #[test]
    fn gemma_summaries_never_contain_schemas() {
        let registry = sample_registry();
        let section = registry.gemma_tool_section();
        assert!(section.contains("echo"));
        assert!(section.contains("crate: noema-test"));
        assert!(!section.contains("parameters"));
        assert!(!section.contains("required"));
        assert!(!section.contains("properties"));
    }

    #[test]
    fn empty_registry_produces_empty_section_and_array() {
        let registry = ToolRegistry::new();
        assert!(registry.gemma_tool_section().is_empty());
        assert_eq!(registry.needle_tools_json(), "[]");
        assert!(registry.is_empty());
    }

    #[test]
    fn needle_json_puts_name_before_description() {
        let registry = sample_registry();
        let json = registry.needle_tools_json();
        let first_tool = json.split('{').nth(1).unwrap_or_default();
        let name_pos = first_tool.find("\"name\"").expect("name key");
        let description_pos = first_tool.find("\"description\"").expect("description key");
        assert!(
            name_pos < description_pos,
            "expected \"name\" before \"description\", got: {first_tool}"
        );
        let parsed: Value = serde_json::from_str(&json).expect("valid JSON array");
        assert_eq!(parsed.as_array().map(|a| a.len()), Some(2));
    }

    #[tokio::test]
    async fn execute_validates_before_running() {
        let registry = sample_registry();

        let result = registry
            .execute(ToolCall::with_arguments("echo", json!({ "message": "hi" })))
            .await
            .expect("valid call executes");
        assert!(result.success);
        assert_eq!(result.text, "echo: hi");

        let err = registry
            .execute(ToolCall::with_arguments("echo", json!({})))
            .await
            .expect_err("missing required message");
        assert!(matches!(err, ToolError::InvalidCall(_)));

        let err = registry
            .execute(ToolCall::new("nope"))
            .await
            .expect_err("unknown tool");
        assert!(matches!(err, ToolError::NotFound(_)));
    }

    #[test]
    fn single_tool_needle_json_is_the_engine_format() {
        let registry = sample_registry();
        let json = registry.tool_needle_json("echo").expect("tool json");
        assert!(json.starts_with("{\"name\":\"echo\""));
    }
}

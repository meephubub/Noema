//! Stub tool implementations for the bridge session.
//!
//! These are minimal placeholder tools that return basic results.
//! Replace them with real implementations as needed.

use async_trait::async_trait;
use noema_tools::{NoemaTool, RiskLevel, ToolCall, ToolMetadata, ToolResult, ToolSchema};
use serde_json::json;

/// A search tool stub.
#[derive(Debug)]
pub struct SearchTool;

#[async_trait]
impl NoemaTool for SearchTool {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "search".into(),
            crate_name: "noema-bridge".into(),
            description: "Search for information".into(),
            risk: RiskLevel::None,
        }
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "search".into(),
            description: "Search for information".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query" }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, call: ToolCall) -> noema_tools::Result<ToolResult> {
        let query = call.arguments["query"]
            .as_str()
            .unwrap_or("unknown");
        Ok(ToolResult::ok(format!("Search results for '{query}': [stub data]")))
    }
}

/// A calculate tool stub.
#[derive(Debug)]
pub struct CalculateTool;

#[async_trait]
impl NoemaTool for CalculateTool {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "calculate".into(),
            crate_name: "noema-bridge".into(),
            description: "Perform a calculation".into(),
            risk: RiskLevel::None,
        }
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "calculate".into(),
            description: "Perform a calculation".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": { "type": "string", "description": "The mathematical expression" }
                },
                "required": ["expression"]
            }),
        }
    }

    async fn execute(&self, call: ToolCall) -> noema_tools::Result<ToolResult> {
        let expr = call.arguments["expression"]
            .as_str()
            .unwrap_or("unknown");
        Ok(ToolResult::ok(format!("Result of '{expr}': 42 (stub)")))
    }
}

/// A translate tool stub.
#[derive(Debug)]
pub struct TranslateTool;

#[async_trait]
impl NoemaTool for TranslateTool {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "translate".into(),
            crate_name: "noema-bridge".into(),
            description: "Translate text between languages".into(),
            risk: RiskLevel::None,
        }
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "translate".into(),
            description: "Translate text between languages".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text to translate" },
                    "target_language": { "type": "string", "description": "Target language code" }
                },
                "required": ["text", "target_language"]
            }),
        }
    }

    async fn execute(&self, call: ToolCall) -> noema_tools::Result<ToolResult> {
        let text = call.arguments["text"]
            .as_str()
            .unwrap_or("unknown");
        let lang = call.arguments["target_language"]
            .as_str()
            .unwrap_or("xx");
        Ok(ToolResult::ok(format!("Translation to {lang}: '{text}' (stub)")))
    }
}

/// A summarize tool stub.
#[derive(Debug)]
pub struct SummarizeTool;

#[async_trait]
impl NoemaTool for SummarizeTool {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "summarize".into(),
            crate_name: "noema-bridge".into(),
            description: "Summarize text".into(),
            risk: RiskLevel::None,
        }
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "summarize".into(),
            description: "Summarize text".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text to summarize" }
                },
                "required": ["text"]
            }),
        }
    }

    async fn execute(&self, call: ToolCall) -> noema_tools::Result<ToolResult> {
        let text = call.arguments["text"]
            .as_str()
            .unwrap_or("unknown");
        Ok(ToolResult::ok(format!("Summary: {text}... (stub)")))
    }
}

/// A navigate tool stub.
#[derive(Debug)]
pub struct NavigateTool;

#[async_trait]
impl NoemaTool for NavigateTool {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "navigate".into(),
            crate_name: "noema-bridge".into(),
            description: "Navigate to a location or URL".into(),
            risk: RiskLevel::None,
        }
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "navigate".into(),
            description: "Navigate to a location or URL".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "destination": { "type": "string", "description": "The destination URL or location" }
                },
                "required": ["destination"]
            }),
        }
    }

    async fn execute(&self, call: ToolCall) -> noema_tools::Result<ToolResult> {
        let dest = call.arguments["destination"]
            .as_str()
            .unwrap_or("unknown");
        Ok(ToolResult::ok(format!("Navigated to: {dest} (stub)")))
    }
}

/// Returns a registry with all 5 stub tools pre-registered.
pub fn stub_registry() -> noema_tools::ToolRegistry {
    let mut registry = noema_tools::ToolRegistry::new();
    registry.register(SearchTool).expect("search");
    registry.register(CalculateTool).expect("calculate");
    registry.register(TranslateTool).expect("translate");
    registry.register(SummarizeTool).expect("summarize");
    registry.register(NavigateTool).expect("navigate");
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_registry_has_five_tools() {
        let registry = stub_registry();
        assert_eq!(registry.names().len(), 5);
        assert!(registry.get("search").is_some());
        assert!(registry.get("calculate").is_some());
        assert!(registry.get("translate").is_some());
        assert!(registry.get("summarize").is_some());
        assert!(registry.get("navigate").is_some());
    }

    #[tokio::test]
    async fn search_tool_executes() {
        let tool = SearchTool;
        let call = ToolCall::with_arguments("search", json!({ "query": "rust" }));
        let result = tool.execute(call).await.unwrap();
        assert!(result.success);
        assert!(result.text.contains("rust"));
    }

    #[tokio::test]
    async fn calculate_tool_executes() {
        let tool = CalculateTool;
        let call = ToolCall::with_arguments("calculate", json!({ "expression": "2+2" }));
        let result = tool.execute(call).await.unwrap();
        assert!(result.success);
        assert!(result.text.contains("2+2"));
    }
}

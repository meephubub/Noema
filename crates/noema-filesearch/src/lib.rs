//! The reference Noema tool: `search_files`.
//!
//! `noema-filesearch` is the first real tool crate (Phase 7), demonstrating
//! the tool contract end-to-end:
//!
//! * **Schema** — `search_files(query, path?)`, with the query required.
//! * **Risk** — [`RiskLevel::Low`]: a read-only filesystem walk.
//! * **Needle instructions** — bound to the tool's logical Needle agent so a
//!   semantic request formats into the exact structured call.
//! * **Execution** — a bounded, recursive name search over the local
//!   filesystem (case-insensitive).
//!
//! A third-party tool crate ships exactly this: implement
//! [`NoemaTool`], register it, and Noema handles everything else.
//!
//! # Example
//!
//! ```no_run
//! use noema_filesearch::Filesearch;
//! use noema_tools::ToolRegistry;
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(Filesearch::default()).expect("register");
//! ```

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use noema_tools::{
    NoemaTool, RiskLevel, ToolCall, ToolMetadata, ToolResult, ToolSchema, Result, ToolError,
};
use serde_json::{json, Value};

/// The tool name registered in the registry.
pub const TOOL_NAME: &str = "search_files";

/// Maximum number of matches returned in one call.
pub const MAX_RESULTS: usize = 25;

/// The maximum directory depth walked from the search root.
pub const MAX_DEPTH: usize = 8;

/// Directories skipped by default: build artifacts and VCS internals that
/// would otherwise dominate the results.
const SKIPPED_DIRS: &[&str] = &["target", ".git", "node_modules"];

/// The reference tool: search the local filesystem for files by name.
///
/// The search is case-insensitive and bounded (see [`MAX_RESULTS`] and
/// [`MAX_DEPTH`]).
#[derive(Debug, Clone, Default)]
pub struct Filesearch;

impl Filesearch {
    /// Searches `root` for files whose name contains `query` (ignoring
    /// case), returning up to [`MAX_RESULTS`] paths.
    pub fn search(root: &Path, query: &str) -> Vec<PathBuf> {
        let needle = query.to_lowercase();
        let mut matches = Vec::new();
        walk(root, &needle, 0, &mut matches);
        matches
    }
}

/// Recursively collects matching files under `dir`, skipping noisy
/// directories and stopping at [`MAX_DEPTH`] / [`MAX_RESULTS`].
fn walk(dir: &Path, needle: &str, depth: usize, matches: &mut Vec<PathBuf>) {
    if depth > MAX_DEPTH || matches.len() >= MAX_RESULTS {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        if matches.len() >= MAX_RESULTS {
            return;
        }
        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if path.is_dir() {
            if SKIPPED_DIRS.contains(&name.as_ref()) {
                continue;
            }
            walk(&path, needle, depth + 1, matches);
        } else if name.to_lowercase().contains(needle) {
            matches.push(path);
        }
    }
}

#[async_trait]
impl NoemaTool for Filesearch {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: TOOL_NAME.into(),
            crate_name: "noema-filesearch".into(),
            description: "Search for files on the local system".into(),
            risk: RiskLevel::Low,
        }
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: TOOL_NAME.into(),
            description: "Search for files on the local system".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the filename text to look for"
                    },
                    "path": {
                        "type": "string",
                        "description": "the directory to search (defaults to the current directory)"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    // No extra Needle instructions: the schema's own descriptions already
    // drive reliable formatting with the base engine (verified empirically —
    // appending instructions measurably lowered call confidence).

    async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
        let query = call
            .arguments
            .get("query")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|q| !q.is_empty())
            .ok_or_else(|| ToolError::InvalidCall("query must be a non-empty string".into()))?;

        let root = match call.arguments.get("path").and_then(Value::as_str) {
            Some(path) if !path.trim().is_empty() => PathBuf::from(path),
            _ => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        };

        let matches = Self::search(&root, query);
        if matches.is_empty() {
            return Ok(ToolResult::ok(format!(
                "no files matching '{query}' were found under {}",
                root.display()
            )));
        }

        let paths: Vec<String> = matches
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect();
        let mut text = format!(
            "found {} file{} matching '{query}' under {}:\n",
            paths.len(),
            if paths.len() == 1 { "" } else { "s" },
            root.display()
        );
        for path in &paths {
            text.push_str(&format!("- {path}\n"));
        }
        Ok(ToolResult::ok_with_data(
            text.trim_end().to_string(),
            json!({ "query": query, "root": root.to_string_lossy(), "files": paths }),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("docs/inflation")).expect("docs dir");
        std::fs::write(dir.path().join("notes.txt"), "hello").expect("notes.txt");
        std::fs::write(dir.path().join("Notes.md"), "upper").expect("Notes.md");
        std::fs::write(dir.path().join("other.md"), "other").expect("other.md");
        std::fs::write(dir.path().join("docs/readme.md"), "readme").expect("readme.md");
        std::fs::write(
            dir.path().join("docs/inflation/summary.txt"),
            "summary",
        )
        .expect("summary.txt");
        dir
    }

    #[test]
    fn search_finds_matches_case_insensitively_and_recursively() {
        let dir = sample_tree();
        // "notes" matches notes.txt, Notes.md (case-insensitive), and the
        // nested docs/study-notes.md below.
        std::fs::write(dir.path().join("docs/study-notes.md"), "study").expect("study");
        let matches = Filesearch::search(dir.path(), "notes");
        assert_eq!(matches.len(), 3);
        for path in &matches {
            let name = path.file_name().unwrap().to_string_lossy();
            assert!(
                name.to_lowercase().contains("notes"),
                "{name} should match 'notes'"
            );
        }

        let matches = Filesearch::search(&dir.path().join("docs"), "summary");
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn search_returns_nothing_when_no_file_matches() {
        let dir = sample_tree();
        assert!(Filesearch::search(dir.path(), "zzz-nonexistent").is_empty());
    }

    #[test]
    fn search_respects_the_result_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..40 {
            std::fs::write(dir.path().join(format!("file{i}.txt")), "x").expect("file");
        }
        let matches = Filesearch::search(dir.path(), "file");
        assert_eq!(matches.len(), MAX_RESULTS);
    }

    #[tokio::test]
    async fn execute_returns_paths_and_structured_data() {
        let dir = sample_tree();
        let tool = Filesearch::default();
        let call = ToolCall::with_arguments(
            TOOL_NAME,
            json!({ "query": "notes", "path": dir.path().to_string_lossy() }),
        );
        let result = tool.execute(call).await.expect("execute");
        assert!(result.success);
        assert!(result.text.contains("notes.txt"));
        let data = result.data.expect("structured data");
        assert_eq!(data["query"], "notes");
        assert!(data["files"].as_array().is_some_and(|a| !a.is_empty()));
    }

    #[tokio::test]
    async fn execute_reports_no_matches_without_error() {
        let dir = sample_tree();
        let tool = Filesearch::default();
        let call = ToolCall::with_arguments(
            TOOL_NAME,
            json!({ "query": "zzz", "path": dir.path().to_string_lossy() }),
        );
        let result = tool.execute(call).await.expect("execute");
        assert!(result.success);
        assert!(result.text.contains("no files matching"));
    }

    #[tokio::test]
    async fn execute_requires_a_query() {
        let tool = Filesearch::default();
        let call = ToolCall::with_arguments(TOOL_NAME, json!({ "path": "/tmp" }));
        let err = tool.execute(call).await.expect_err("missing query");
        assert!(matches!(err, ToolError::InvalidCall(_)));
    }
}

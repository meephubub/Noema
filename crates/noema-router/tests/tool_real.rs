//! Real-engine tool-formatting tests (ignored by default; need
//! `prebuilt/needle/`).
//!
//! ```text
//! cargo test -p noema-router --test tool_real -- --ignored --nocapture
//! ```

use noema_core::ToolFormatter;
use noema_router::NeedleToolFormatter;
use noema_tools::ToolSchema;
use serde_json::json;
use tokio_util::sync::CancellationToken;

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

#[tokio::test]
#[ignore = "needs the Needle engine"]
async fn semantic_request_formats_into_a_structured_call() {
    let formatter =
        NeedleToolFormatter::from_tool(&search_schema(), Some("Only search the local filesystem."))
            .expect("formatter loads");

    let call = formatter
        .format(search_schema(), "find the file abc.exe", CancellationToken::new())
        .await
        .expect("format");
    assert_eq!(call.tool, "search_files");
    assert_eq!(call.arguments["query"], "abc.exe");
    eprintln!("search_files {:?}", call.arguments);
}

#[tokio::test]
#[ignore = "needs the Needle engine"]
async fn unsupported_requests_are_refused() {
    let formatter = NeedleToolFormatter::from_tool(&search_schema(), None).expect("formatter loads");

    let err = formatter
        .format(
            search_schema(),
            "Write a poem about the ocean",
            CancellationToken::new(),
        )
        .await
        .expect_err("poem is not a file search");
    assert!(err.to_string().contains("refused"), "got: {err}");
}

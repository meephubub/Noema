//! The action registry: the simple application actions the router can emit.
//!
//! Every action is exposed to Needle as a tool with a name and description;
//! the router maps a request to a call of one of these tools. Requests no
//! tool can serve are refused by the engine (an empty call list), which the
//! router turns into an escalation.

use serde_json::{json, Value};

/// A simple application action the router can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionSpec {
    /// The registry id, e.g. `open_flashcards`. The frontend maps this onto
    /// the actual UI behaviour.
    pub id: &'static str,
    /// A short human description, used as the Needle tool description.
    pub description: &'static str,
}

/// The default action registry, covering the simple Agora actions from the
/// plan: opening, showing, and navigating — never answering questions.
#[derive(Debug, Clone)]
pub struct ActionRegistry {
    actions: Vec<ActionSpec>,
}

impl ActionRegistry {
    /// The built-in registry of application actions (see [`Default::default`]).
    pub fn builtin() -> Self {
        Self {
            // Descriptions verified against the real engine: each of these
            // phrasings maps to its action at high confidence, and arbitrary
            // questions are refused (escalated).
            actions: vec![
                ActionSpec {
                    id: "open_flashcards",
                    description: "Navigate to the user's flashcards",
                },
                ActionSpec {
                    id: "show_notes",
                    description: "Show the user's notes",
                },
                ActionSpec {
                    id: "open_pdfs",
                    description: "Navigate to the user's PDF documents",
                },
                ActionSpec {
                    id: "go_to_settings",
                    description: "Open application settings",
                },
                ActionSpec {
                    id: "start_revision",
                    description: "Begin a revision session",
                },
                ActionSpec {
                    id: "open_last_document",
                    description: "Open the last document the user worked on",
                },
            ],
        }
    }

    /// A registry with the given actions (for tests and custom builds).
    pub fn new(actions: Vec<ActionSpec>) -> Self {
        Self { actions }
    }

    /// The registered actions, in order.
    pub fn actions(&self) -> &[ActionSpec] {
        &self.actions
    }

    /// Looks up an action by registry id.
    pub fn get(&self, id: &str) -> Option<&ActionSpec> {
        self.actions.iter().find(|action| action.id == id)
    }

    /// Whether every action id in the registry is unique.
    pub fn ids_are_unique(&self) -> bool {
        let mut seen = std::collections::HashSet::new();
        self.actions.iter().all(|action| seen.insert(action.id))
    }

    /// The Needle tool schema JSON array for every registered action.
    ///
    /// Each action becomes a parameterless tool. The `name` key must come
    /// **before** `description` in the serialized JSON: the engine's
    /// grammar-driven schema parser is key-order sensitive, and with
    /// `description` first it mis-associates tools when the catalogue grows
    /// (verified empirically against the real engine).
    pub fn tools_json(&self) -> String {
        let tools: Vec<Value> = self
            .actions
            .iter()
            .map(|action| {
                json!({
                    "name": action.id,
                    "description": action.description,
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                })
            })
            .collect();
        let json = serde_json::to_string(&tools).expect("tool schema serializes");
        debug_assert!(json.contains("\"name\":\"open_flashcards\""));
        json
    }
}

impl Default for ActionRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_covers_the_plan_actions() {
        let registry = ActionRegistry::default();
        assert!(registry.ids_are_unique());
        for id in [
            "open_flashcards",
            "show_notes",
            "open_pdfs",
            "go_to_settings",
            "start_revision",
            "open_last_document",
        ] {
            assert!(registry.get(id).is_some(), "missing action {id}");
        }
    }

    #[test]
    fn tools_json_is_a_valid_needle_schema_array() {
        let registry = ActionRegistry::default();
        let json = registry.tools_json();
        let tools: Vec<Value> = serde_json::from_str(&json).expect("valid JSON array");
        assert_eq!(tools.len(), registry.actions().len());
        let first = &tools[0];
        assert!(first["name"].is_string());
        assert!(first["description"].is_string());
        assert_eq!(first["parameters"]["type"], "object");
    }

    #[test]
    fn tools_json_puts_name_before_description() {
        // The engine's schema parser is key-order sensitive (see
        // `tools_json`); `name` must serialize first.
        let registry = ActionRegistry::default();
        let json = registry.tools_json();
        let first_tool = json.split('{').nth(1).unwrap_or_default();
        let name_pos = first_tool.find("\"name\"").expect("name key");
        let description_pos = first_tool.find("\"description\"").expect("description key");
        assert!(
            name_pos < description_pos,
            "expected \"name\" before \"description\", got: {first_tool}"
        );
    }
}

//! Conversation message types (JSON Message protocol from LiteRT-LM).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Role of a conversation participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Model,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Model | Self::Assistant => "model",
            Self::Tool => "tool",
        }
    }
}

/// A multimodal content part inside a message.
///
/// Matches the formats handled by LiteRT-LM `data_utils.h`:
/// text, image (path/blob), audio (path/blob).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentPart {
    Text { text: String },
    Image {
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        blob: Option<String>,
    },
    Audio {
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        blob: Option<String>,
    },
    /// Escape hatch for model-specific / future part types.
    #[serde(untagged)]
    Other(Value),
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
        }
    }

    pub fn image_path(path: impl AsRef<Path>) -> Self {
        Self::Image {
            path: Some(path_string(path.as_ref())),
            blob: None,
        }
    }

    pub fn image_bytes(bytes: &[u8]) -> Self {
        use base64::Engine as _;
        Self::Image {
            path: None,
            blob: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
        }
    }

    pub fn audio_path(path: impl AsRef<Path>) -> Self {
        Self::Audio {
            path: Some(path_string(path.as_ref())),
            blob: None,
        }
    }

    pub fn audio_bytes(bytes: &[u8]) -> Self {
        use base64::Engine as _;
        Self::Audio {
            path: None,
            blob: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
        }
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// A chat message exchanged with [`crate::Conversation`].
///
/// Serializes to the LiteRT-LM `JsonMessage` shape (`role` + `content`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Value,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<Value>) -> Self {
        Self {
            role: role.as_str().to_owned(),
            content: content.into(),
            extra: serde_json::Map::new(),
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, Value::String(text.into()))
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::new(Role::System, Value::String(text.into()))
    }

    pub fn model(text: impl Into<String>) -> Self {
        Self::new(Role::Model, Value::String(text.into()))
    }

    pub fn tool(text: impl Into<String>) -> Self {
        Self::new(Role::Tool, Value::String(text.into()))
    }

    /// Multimodal user message with an array of content parts.
    pub fn user_parts(parts: impl IntoIterator<Item = ContentPart>) -> Result<Self> {
        let parts: Vec<ContentPart> = parts.into_iter().collect();
        Ok(Self::new(Role::User, serde_json::to_value(parts)?))
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Best-effort extraction of concatenated text from `content`.
    pub fn text(&self) -> Option<String> {
        match &self.content {
            Value::String(s) => Some(s.clone()),
            Value::Array(items) => {
                let mut out = String::new();
                for item in items {
                    if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                        out.push_str(t);
                    } else if let Some(t) = item.as_str() {
                        out.push_str(t);
                    }
                }
                if out.is_empty() {
                    None
                } else {
                    Some(out)
                }
            }
            Value::Object(map) => map.get("text").and_then(|v| v.as_str()).map(str::to_owned),
            _ => None,
        }
    }

    /// Tool calls attached to a model response, if present.
    pub fn tool_calls(&self) -> Option<&Value> {
        self.extra.get("tool_calls")
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(text) = self.text() {
            write!(f, "{text}")
        } else {
            write!(f, "{}", serde_json::to_string(self).unwrap_or_default())
        }
    }
}

impl From<&str> for Message {
    fn from(value: &str) -> Self {
        Message::user(value)
    }
}

impl From<String> for Message {
    fn from(value: String) -> Self {
        Message::user(value)
    }
}

/// Helper to build a Gemini-style tools JSON array entry.
pub fn tool_declaration(
    name: impl Into<String>,
    description: impl Into<String>,
    parameters: Value,
) -> Value {
    json!({
        "name": name.into(),
        "description": description.into(),
        "parameters": parameters,
    })
}

/// Convenience for loading image bytes from disk into a content part.
pub fn image_file(path: impl AsRef<Path>) -> Result<ContentPart> {
    let path = path.as_ref();
    if path.exists() {
        Ok(ContentPart::image_path(path))
    } else {
        Err(Error::Message(format!(
            "image path does not exist: {}",
            path.display()
        )))
    }
}

/// Convenience for loading audio bytes from disk into a content part.
pub fn audio_file(path: impl Into<PathBuf>) -> Result<ContentPart> {
    let path = path.into();
    if path.exists() {
        Ok(ContentPart::audio_path(&path))
    } else {
        Err(Error::Message(format!(
            "audio path does not exist: {}",
            path.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_json_shape() {
        let msg = Message::user_parts([
            ContentPart::text("Describe:"),
            ContentPart::image_path("/tmp/a.jpg"),
        ])
        .unwrap();
        let s = msg.to_json_string().unwrap();
        assert!(s.contains(r#""type":"text""#));
        assert!(s.contains(r#""type":"image""#));
        assert!(s.contains("/tmp/a.jpg"));
    }
}

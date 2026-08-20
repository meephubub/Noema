//! Conversion between Rig's provider-agnostic messages and Noema's model
//! messages.
//!
//! Rig's [`Message`] is the wire format rig agents build and replay; Noema's
//! [`noema_core::Message`] is what local models consume. The two directions
//! are intentionally lossy in opposite ways:
//!
//! * `rig_to_noema` flattens multimodal and tool content into the parts
//!   Noema's `Model` trait understands (text, image, audio). Tool calls and
//!   reasoning become their text renderings; documents and video are
//!   represented as text.
//! * `noema_to_rig` renders a Noema message back into rig's vocabulary,
//!   primarily for tests and echo adapters.

use base64::Engine as _;
use noema_core::{AudioData, ContentPart, ImageData, Message as NoemaMessage, Role};
use rig_core::completion::{CompletionError, Message as RigMessage};
use rig_core::message::{
    AssistantContent, Audio, AudioMediaType, DocumentSourceKind, Image, ImageMediaType, Text,
    ToolCallId, ToolResult, UserContent,
};

/// Converts a rig message into the Noema message a local model consumes.
pub fn rig_to_noema(message: &RigMessage) -> Result<NoemaMessage, CompletionError> {
    match message {
        RigMessage::System { content } => Ok(NoemaMessage::text(Role::System, content)),
        RigMessage::User { content } => {
            let parts = content
                .iter()
                .map(map_user_content)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(NoemaMessage::new(Role::User, parts))
        }
        RigMessage::Assistant { content, .. } => {
            let parts = content
                .iter()
                .map(map_assistant_content)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(NoemaMessage::new(Role::Assistant, parts))
        }
    }
}

/// Converts a Noema message back into rig's message vocabulary.
pub fn noema_to_rig(message: &NoemaMessage) -> RigMessage {
    let parts = &message.content;
    match message.role {
        Role::System => {
            let content = text_of(parts);
            RigMessage::System { content }
        }
        Role::Assistant => {
            let content = parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text(text) => AssistantContent::Text(Text::new(text)),
                    ContentPart::Image(image) => AssistantContent::Image(rig_image(image)),
                    ContentPart::Audio(_) => {
                        // Rig has no assistant audio block; render as text.
                        AssistantContent::Text(Text::new("<audio>"))
                    }
                })
                .collect();
            RigMessage::Assistant { id: None, content }
        }
        Role::User => {
            let content = parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text(text) => UserContent::Text(Text::new(text)),
                    ContentPart::Image(image) => UserContent::Image(rig_image(image)),
                    ContentPart::Audio(audio) => UserContent::Audio(rig_audio(audio)),
                })
                .collect();
            RigMessage::User { content }
        }
        Role::Tool => RigMessage::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: ToolCallId::new_or_mint("tool"),
                provider: None,
                name: "tool".into(),
                content: vec![rig_core::message::ToolResultContent::text(text_of(parts))],
            })],
        },
    }
}

/// The last user message of a rig chat history, if any.
pub fn last_user_message(messages: &[RigMessage]) -> Option<&RigMessage> {
    messages
        .iter()
        .rev()
        .find(|message| matches!(message, RigMessage::User { .. }))
}

/// The text of a Noema message: all text parts joined, non-text parts
/// rendered as placeholders.
fn text_of(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text(text) => text.clone(),
            ContentPart::Image(_) => "<image>".to_string(),
            ContentPart::Audio(_) => "<audio>".to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn map_user_content(content: &UserContent) -> Result<ContentPart, CompletionError> {
    match content {
        UserContent::Text(text) => Ok(ContentPart::Text(text.text.clone())),
        UserContent::ToolResult(result) => {
            let rendered = result
                .content
                .iter()
                .filter_map(|item| item.as_text().map(str::to_owned))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(ContentPart::Text(format!(
                "[tool result: {}]\n{}",
                result.name, rendered
            )))
        }
        UserContent::Image(image) => Ok(ContentPart::Image(ImageData {
            bytes: source_bytes(&image.data)?,
            mime: image.media_type.as_ref().map(mime_of_image).unwrap_or_else(|| {
                "application/octet-stream".to_string()
            }),
        })),
        UserContent::Audio(audio) => Ok(ContentPart::Audio(AudioData {
            bytes: source_bytes(&audio.data)?,
            mime: audio
                .media_type
                .as_ref()
                .map(mime_of_audio)
                .unwrap_or_else(|| "application/octet-stream".to_string()),
        })),
        UserContent::Video(video) => Ok(ContentPart::Text(format!(
            "[video: {}]",
            source_text(&video.data)
        ))),
        UserContent::Document(document) => Ok(ContentPart::Text(source_text(&document.data))),
    }
}

fn map_assistant_content(content: &AssistantContent) -> Result<ContentPart, CompletionError> {
    match content {
        AssistantContent::Text(text) => Ok(ContentPart::Text(text.text.clone())),
        AssistantContent::ToolCall(call) => Ok(ContentPart::Text(format!(
            "[tool call: {}({})]",
            call.function.name, call.function.arguments
        ))),
        AssistantContent::Reasoning(reasoning) => {
            Ok(ContentPart::Text(reasoning.display_text()))
        }
        AssistantContent::Image(image) => Ok(ContentPart::Image(ImageData {
            bytes: source_bytes(&image.data)?,
            mime: image
                .media_type
                .as_ref()
                .map(mime_of_image)
                .unwrap_or_else(|| "application/octet-stream".to_string()),
        })),
    }
}

/// Bytes for a rig content source: raw bytes pass through, base64 and
/// string/url/file-id sources are decoded to bytes.
fn source_bytes(source: &DocumentSourceKind) -> Result<Vec<u8>, CompletionError> {
    match source {
        DocumentSourceKind::Raw(bytes) => Ok(bytes.clone()),
        DocumentSourceKind::Base64(encoded) => base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|error| {
                CompletionError::RequestError(format!("invalid base64 content: {error}").into())
            }),
        DocumentSourceKind::Url(url)
        | DocumentSourceKind::FileId(url)
        | DocumentSourceKind::String(url) => Ok(url.as_bytes().to_vec()),
        DocumentSourceKind::Unknown => Ok(Vec::new()),
    }
}

/// A text rendering of a content source for unsupported binary kinds.
fn source_text(source: &DocumentSourceKind) -> String {
    match source {
        DocumentSourceKind::Raw(bytes) => format!("{} bytes", bytes.len()),
        DocumentSourceKind::Base64(encoded) => format!("{} base64 bytes", encoded.len()),
        DocumentSourceKind::Url(url) => url.clone(),
        DocumentSourceKind::FileId(id) => id.clone(),
        DocumentSourceKind::String(text) => text.clone(),
        DocumentSourceKind::Unknown => "<unknown>".to_string(),
    }
}

fn mime_of_image(media_type: &ImageMediaType) -> String {
    format!(
        "image/{}",
        match media_type {
            ImageMediaType::JPEG => "jpeg",
            ImageMediaType::PNG => "png",
            ImageMediaType::GIF => "gif",
            ImageMediaType::WEBP => "webp",
            ImageMediaType::HEIC => "heic",
            ImageMediaType::HEIF => "heif",
            ImageMediaType::SVG => "svg+xml",
        }
    )
}

fn mime_of_audio(media_type: &AudioMediaType) -> String {
    format!(
        "audio/{}",
        match media_type {
            AudioMediaType::WAV => "wav",
            AudioMediaType::MP3 => "mpeg",
            AudioMediaType::AIFF => "aiff",
            AudioMediaType::AAC => "aac",
            AudioMediaType::OGG => "ogg",
            AudioMediaType::FLAC => "flac",
            AudioMediaType::M4A => "mp4",
            AudioMediaType::PCM16 => "l16",
            AudioMediaType::PCM24 => "l24",
        }
    )
}

fn rig_image(image: &noema_core::ImageData) -> Image {
    Image {
        data: if image.mime.starts_with("image/") && is_binary_mime(&image.mime) {
            // Keep binary images as raw bytes; everything else travels as a
            // string.
            DocumentSourceKind::Raw(image.bytes.clone())
        } else {
            DocumentSourceKind::String(String::from_utf8_lossy(&image.bytes).into_owned())
        },
        media_type: image_mime_type(&image.mime),
        detail: None,
        additional_params: None,
    }
}

fn rig_audio(audio: &noema_core::AudioData) -> Audio {
    Audio {
        data: DocumentSourceKind::Raw(audio.bytes.clone()),
        media_type: None,
        additional_params: None,
    }
}

fn is_binary_mime(mime: &str) -> bool {
    matches!(
        mime,
        "image/png"
            | "image/jpeg"
            | "image/gif"
            | "image/webp"
            | "image/heic"
            | "image/heif"
    )
}

fn image_mime_type(mime: &str) -> Option<ImageMediaType> {
    Some(match mime {
        "image/jpeg" => ImageMediaType::JPEG,
        "image/png" => ImageMediaType::PNG,
        "image/gif" => ImageMediaType::GIF,
        "image/webp" => ImageMediaType::WEBP,
        "image/heic" => ImageMediaType::HEIC,
        "image/heif" => ImageMediaType::HEIF,
        "image/svg+xml" => ImageMediaType::SVG,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::message::{
        DocumentMediaType, DocumentSourceKind, Image as RigImage, ImageMediaType, ToolCall,
        ToolCallId, ToolFunction, ToolResultContent,
    };

    #[test]
    fn user_text_round_trips() {
        let rig = RigMessage::user("hello");
        let noema = rig_to_noema(&rig).expect("map");
        assert_eq!(noema.role, Role::User);
        assert_eq!(noema.content, vec![ContentPart::text("hello")]);

        let back = noema_to_rig(&noema);
        assert_eq!(back, RigMessage::user("hello"));
    }

    #[test]
    fn system_and_assistant_map_to_roles() {
        let system = rig_to_noema(&RigMessage::system("be brief")).expect("map");
        assert_eq!(system.role, Role::System);

        let assistant = rig_to_noema(&RigMessage::assistant("hi there")).expect("map");
        assert_eq!(assistant.role, Role::Assistant);
        assert_eq!(
            assistant.content,
            vec![ContentPart::text("hi there")]
        );
    }

    #[test]
    fn tool_result_renders_as_text() {
        let rig = RigMessage::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: ToolCallId::new_or_mint("call-1"),
                provider: None,
                name: "search_files".into(),
                content: vec![ToolResultContent::text("found it")],
            })],
        };
        let noema = rig_to_noema(&rig).expect("map");
        match &noema.content[0] {
            ContentPart::Text(text) => {
                assert!(text.contains("search_files"));
                assert!(text.contains("found it"));
            }
            other => panic!("expected text part, got {other:?}"),
        }
    }

    #[test]
    fn image_source_bytes_decode() {
        let rig = RigMessage::User {
            content: vec![UserContent::Image(RigImage {
                data: DocumentSourceKind::Raw(vec![1, 2, 3]),
                media_type: Some(ImageMediaType::PNG),
                detail: None,
                additional_params: None,
            })],
        };
        let noema = rig_to_noema(&rig).expect("map");
        match &noema.content[0] {
            ContentPart::Image(image) => {
                assert_eq!(image.bytes, vec![1, 2, 3]);
                assert_eq!(image.mime, "image/png");
            }
            other => panic!("expected image part, got {other:?}"),
        }
    }

    #[test]
    fn base64_image_decodes() {
        let encoded = base64::engine::general_purpose::STANDARD.encode(vec![9, 8, 7]);
        let rig = RigMessage::User {
            content: vec![UserContent::Image(RigImage {
                data: DocumentSourceKind::Base64(encoded),
                media_type: None,
                detail: None,
                additional_params: None,
            })],
        };
        let noema = rig_to_noema(&rig).expect("map");
        match &noema.content[0] {
            ContentPart::Image(image) => assert_eq!(image.bytes, vec![9, 8, 7]),
            other => panic!("expected image part, got {other:?}"),
        }
    }

    #[test]
    fn document_renders_as_source_text() {
        let rig = RigMessage::User {
            content: vec![UserContent::Document(
                rig_core::message::Document {
                    data: DocumentSourceKind::String("some content".into()),
                    media_type: None,
                    additional_params: None,
                },
            )],
        };
        let noema = rig_to_noema(&rig).expect("map");
        match &noema.content[0] {
            ContentPart::Text(text) => assert_eq!(text, "some content"),
            other => panic!("expected text part, got {other:?}"),
        }
    }

    #[test]
    fn assistant_tool_call_renders_as_json() {
        let rig = RigMessage::Assistant {
            id: None,
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: ToolCallId::new_or_mint("id-1"),
                provider: None,
                function: ToolFunction::new(
                    "set_lights".into(),
                    serde_json::json!({ "room": "living room" }),
                ),
                signature: None,
                additional_params: None,
            })],
        };
        let noema = rig_to_noema(&rig).expect("map");
        match &noema.content[0] {
            ContentPart::Text(text) => {
                assert!(text.contains("set_lights"));
                assert!(text.contains("living room"));
            }
            other => panic!("expected text part, got {other:?}"),
        }
    }

    #[test]
    fn last_user_message_finds_most_recent_user_turn() {
        let history = vec![
            RigMessage::user("first"),
            RigMessage::assistant("reply"),
            RigMessage::user("second"),
        ];
        let last = last_user_message(&history).expect("a user message");
        assert_eq!(last, &RigMessage::user("second"));
    }

    #[test]
    fn document_media_type_is_code() {
        assert!(DocumentMediaType::Javascript.is_code());
        assert!(!DocumentMediaType::PDF.is_code());
    }
}

//! Conversion between Noema's model-agnostic messages and the JSON message
//! format consumed by LiteRT-LM.

use litert_lm_rust::{ContentPart as NativePart, Message as NativeMessage};
use noema_core::{ContentPart, Message, NoemaError, Role};

/// Maps a Noema message (in particular the current user turn) onto a
/// LiteRT-LM message.
pub fn map_message(message: &Message) -> Result<NativeMessage, NoemaError> {
    let parts: Vec<NativePart> = message
        .content
        .iter()
        .map(map_part)
        .collect::<Result<_, _>>()?;

    match message.role {
        Role::User => {
            if let Some(text) = single_text(&parts) {
                Ok(NativeMessage::user(text))
            } else {
                NativeMessage::user_parts(parts).map_err(map_litert)
            }
        }
        Role::Assistant => Ok(NativeMessage::model(role_text(&parts, Role::Assistant)?)),
        Role::System => Ok(NativeMessage::system(role_text(&parts, Role::System)?)),
        Role::Tool => Ok(NativeMessage::tool(role_text(&parts, Role::Tool)?)),
    }
}

/// Maps a single Noema content part onto a LiteRT-LM content part.
pub fn map_part(part: &ContentPart) -> Result<NativePart, NoemaError> {
    match part {
        ContentPart::Text(text) => Ok(NativePart::text(text)),
        ContentPart::Image(image) => Ok(NativePart::image_bytes(&image.bytes)),
        ContentPart::Audio(audio) => Ok(NativePart::audio_bytes(&audio.bytes)),
    }
}

/// The message text when a message consists of exactly one text part.
fn single_text(parts: &[NativePart]) -> Option<String> {
    match parts {
        [NativePart::Text { text }] => Some(text.clone()),
        _ => None,
    }
}

/// Text for a non-user role: single text part, or joined text parts. Binary
/// parts are only supported for user messages.
fn role_text(parts: &[NativePart], role: Role) -> Result<String, NoemaError> {
    if let Some(text) = single_text(parts) {
        return Ok(text);
    }
    let mut text = String::new();
    for part in parts {
        match part {
            NativePart::Text { text: t } => text.push_str(t),
            other => {
                return Err(NoemaError::Model(format!(
                    "{role:?} messages cannot carry binary content parts ({other:?}); \
                     images and audio are supported for user turns"
                )));
            }
        }
    }
    Ok(text)
}

pub(crate) fn map_litert(error: litert_lm_rust::Error) -> NoemaError {
    NoemaError::Model(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_user_message() {
        let message = Message::text(Role::User, "hello");
        let native = map_message(&message).expect("map");
        assert_eq!(native.to_json_string().expect("json").as_str(), r#"{"role":"user","content":"hello"}"#);
    }

    #[test]
    fn multimodal_user_message_carries_blobs() {
        let message = Message::new(
            Role::User,
            vec![
                ContentPart::text("describe"),
                ContentPart::image(vec![1, 2, 3], "image/png"),
                ContentPart::audio(vec![4, 5], "audio/wav"),
            ],
        );
        let native = map_message(&message).expect("map");
        let json = native.to_json_string().expect("json");
        assert!(json.contains(r#""type":"text""#));
        assert!(json.contains(r#""type":"image""#));
        assert!(json.contains(r#""type":"audio""#));
        assert!(json.contains("AQID")); // base64 of [1, 2, 3]
    }

    #[test]
    fn assistant_and_system_map_to_roles() {
        let assistant = map_message(&Message::text(Role::Assistant, "hi")).expect("map");
        assert_eq!(
            assistant.to_json_string().expect("json").as_str(),
            r#"{"role":"model","content":"hi"}"#
        );
        let system = map_message(&Message::text(Role::System, "be brief")).expect("map");
        assert_eq!(
            system.to_json_string().expect("json").as_str(),
            r#"{"role":"system","content":"be brief"}"#
        );
    }

    #[test]
    fn binary_content_in_non_user_role_fails() {
        let message = Message::new(Role::Assistant, vec![ContentPart::image(vec![1], "image/png")]);
        assert!(map_message(&message).is_err());
    }
}

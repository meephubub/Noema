//! High-level Conversation API (recommended entry point).

use std::ffi::{CStr, CString};
use std::ptr::NonNull;

use serde_json::Value;

use crate::engine::{Engine, EngineLifetime};
use crate::error::{Error, Result};
use crate::ffi;
use crate::message::Message;
use crate::session::RawSessionConfig;
use crate::stream::{self, StreamEventReceiver};

/// Constraint type for constrained decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    None,
    Regex,
    JsonSchema,
}

impl ConstraintType {
    fn into_ffi(self) -> ffi::LiteRtLmConstraintType {
        match self {
            Self::None => ffi::LiteRtLmConstraintType_kLiteRtLmConstraintTypeNone,
            Self::Regex => ffi::LiteRtLmConstraintType_kLiteRtLmConstraintTypeRegex,
            Self::JsonSchema => ffi::LiteRtLmConstraintType_kLiteRtLmConstraintTypeJsonSchema,
        }
    }
}

/// Thinking / reasoning configuration.
#[derive(Debug, Clone, Default)]
pub struct ThinkingConfig {
    pub enable_thinking: Option<bool>,
    pub thinking_token_budget: Option<i32>,
}

/// Repetition penalty configuration.
#[derive(Debug, Clone, Default)]
pub struct RepetitionPenaltyConfig {
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub window_size: Option<i32>,
}

/// No-repeat n-gram configuration.
#[derive(Debug, Clone, Default)]
pub struct NoRepeatNgramConfig {
    pub no_repeat_ngram_size: Option<i32>,
    pub window_size: Option<i32>,
}

/// Suppress-tokens configuration.
#[derive(Debug, Clone, Default)]
pub struct SuppressTokensConfig {
    pub tokens: Vec<i32>,
}

/// Configuration for creating a [`Conversation`].
#[derive(Debug, Clone, Default)]
pub struct ConversationConfig {
    pub session: crate::SessionConfig,
    /// System instruction JSON message (or plain object).
    pub system_message: Option<Value>,
    /// Tools JSON array.
    pub tools: Option<Value>,
    /// Initial preface messages JSON array.
    pub messages: Option<Value>,
    /// Extra preface context (e.g. `{"enable_thinking": false}`).
    pub extra_context: Option<Value>,
    pub prompt_template: Option<String>,
    pub enable_constrained_decoding: Option<bool>,
    pub use_ll_guidance: Option<bool>,
    pub filter_channel_content_from_kv_cache: Option<bool>,
    pub stream_tool_calls: Option<(bool, String)>,
    pub thinking: Option<ThinkingConfig>,
}

/// Per-turn optional arguments for [`Conversation::send_message`].
#[derive(Debug, Clone, Default)]
pub struct ConversationOptionalArgs {
    pub repetition_penalty: Option<RepetitionPenaltyConfig>,
    pub no_repeat_ngram: Option<NoRepeatNgramConfig>,
    pub suppress_tokens: Option<SuppressTokensConfig>,
    pub visual_token_budget: Option<i32>,
    pub max_output_tokens: Option<i32>,
    pub thinking: Option<ThinkingConfig>,
    pub constraint_type: Option<ConstraintType>,
    pub constraint_string: Option<String>,
}

/// Options passed alongside a message.
#[derive(Debug, Clone, Default)]
pub struct SendOptions {
    pub extra_context: Option<Value>,
    pub optional_args: Option<ConversationOptionalArgs>,
}

/// Stateful chat conversation (manages Session + prompt template + multimodal preprocessing).
pub struct Conversation<'engine> {
    raw: NonNull<ffi::LiteRtLmConversation>,
    _engine: EngineLifetime<'engine>,
}

unsafe impl Send for Conversation<'_> {}

impl<'engine> Conversation<'engine> {
    pub(crate) fn new(engine: &'engine Engine, config: ConversationConfig) -> Result<Self> {
        let raw_config = RawConversationConfig::new(config)?;
        let raw = unsafe {
            ffi::litert_lm_conversation_create(engine.raw().as_ptr(), raw_config.as_mut_ptr())
        };
        let raw =
            NonNull::new(raw).ok_or(Error::NullPointer("litert_lm_conversation_create"))?;
        Ok(Self {
            raw,
            _engine: EngineLifetime::default(),
        })
    }

    /// Blocking send; returns the complete model [`Message`].
    pub fn send_message(&mut self, message: impl Into<Message>) -> Result<Message> {
        self.send_message_with(message, SendOptions::default())
    }

    pub fn send_message_with(
        &mut self,
        message: impl Into<Message>,
        options: SendOptions,
    ) -> Result<Message> {
        let message = message.into();
        let message_json = CString::new(message.to_json_string()?)?;
        let extra = match options.extra_context.as_ref() {
            Some(v) => Some(CString::new(serde_json::to_string(v)?)?),
            None => None,
        };
        let opt_args = options
            .optional_args
            .map(RawOptionalArgs::new)
            .transpose()?;

        let response = unsafe {
            ffi::litert_lm_conversation_send_message(
                self.raw.as_ptr(),
                message_json.as_ptr(),
                extra
                    .as_ref()
                    .map(|c| c.as_ptr())
                    .unwrap_or(std::ptr::null()),
                opt_args
                    .as_ref()
                    .map(|a| a.as_ptr())
                    .unwrap_or(std::ptr::null()),
            )
        };
        let response = NonNull::new(response)
            .ok_or(Error::NullPointer("litert_lm_conversation_send_message"))?;
        let json = unsafe {
            let ptr = ffi::litert_lm_json_response_get_string(response.as_ptr());
            let s = if ptr.is_null() {
                "{}".to_owned()
            } else {
                CStr::from_ptr(ptr).to_str()?.to_owned()
            };
            ffi::litert_lm_json_response_delete(response.as_ptr());
            s
        };
        Message::from_json_str(&json)
    }

    /// Non-blocking send; streams chunks via [`StreamEventReceiver`].
    pub fn send_message_stream(
        &mut self,
        message: impl Into<Message>,
    ) -> Result<StreamEventReceiver> {
        self.send_message_stream_with(message, SendOptions::default())
    }

    pub fn send_message_stream_with(
        &mut self,
        message: impl Into<Message>,
        options: SendOptions,
    ) -> Result<StreamEventReceiver> {
        let message = message.into();
        let message_json = CString::new(message.to_json_string()?)?;
        let extra = match options.extra_context.as_ref() {
            Some(v) => Some(CString::new(serde_json::to_string(v)?)?),
            None => None,
        };
        let opt_args = options
            .optional_args
            .map(RawOptionalArgs::new)
            .transpose()?;

        // Keep CStrings alive by moving into the stream starter; they only need
        // to live for the duration of the start call (native copies the JSON).
        stream::start_conversation_stream(
            self.raw,
            &message_json,
            extra.as_ref(),
            opt_args
                .as_ref()
                .map(|a| a.as_ptr())
                .unwrap_or(std::ptr::null()),
        )
    }

    pub fn clone_conversation(&self) -> Result<Conversation<'engine>> {
        let raw = unsafe { ffi::litert_lm_conversation_clone(self.raw.as_ptr()) };
        let raw =
            NonNull::new(raw).ok_or(Error::NullPointer("litert_lm_conversation_clone"))?;
        Ok(Conversation {
            raw,
            _engine: EngineLifetime::default(),
        })
    }

    pub fn cancel_process(&mut self) {
        unsafe { ffi::litert_lm_conversation_cancel_process(self.raw.as_ptr()) };
    }

    pub fn token_count(&self) -> Result<i32> {
        let n = unsafe { ffi::litert_lm_conversation_get_token_count(self.raw.as_ptr()) };
        if n < 0 {
            return Err(Error::NativeStatus(
                "litert_lm_conversation_get_token_count",
                n,
            ));
        }
        Ok(n)
    }

    pub fn render_message(&self, message: &Message) -> Result<String> {
        let json = CString::new(message.to_json_string()?)?;
        let ptr = unsafe {
            ffi::litert_lm_conversation_render_message_to_string(self.raw.as_ptr(), json.as_ptr())
        };
        if ptr.is_null() {
            return Err(Error::NullPointer(
                "litert_lm_conversation_render_message_to_string",
            ));
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_str()?.to_owned())
    }

    pub fn render_preface(&self) -> Result<String> {
        let ptr =
            unsafe { ffi::litert_lm_conversation_render_preface_to_string(self.raw.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::NullPointer(
                "litert_lm_conversation_render_preface_to_string",
            ));
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_str()?.to_owned())
    }

    pub fn benchmark_info(&self) -> Result<crate::BenchmarkInfo> {
        let raw = unsafe { ffi::litert_lm_conversation_get_benchmark_info(self.raw.as_ptr()) };
        let raw = NonNull::new(raw)
            .ok_or(Error::NullPointer("litert_lm_conversation_get_benchmark_info"))?;
        Ok(crate::BenchmarkInfo { raw })
    }
}

impl Drop for Conversation<'_> {
    fn drop(&mut self) {
        unsafe { ffi::litert_lm_conversation_delete(self.raw.as_ptr()) };
    }
}

struct RawConversationConfig {
    raw: Option<NonNull<ffi::LiteRtLmConversationConfig>>,
    _session: RawSessionConfig,
    _thinking: Option<RawThinkingConfig>,
}

impl RawConversationConfig {
    fn new(config: ConversationConfig) -> Result<Self> {
        let session = RawSessionConfig::new(config.session)?;
        let is_default = session.as_ptr().is_none()
            && config.system_message.is_none()
            && config.tools.is_none()
            && config.messages.is_none()
            && config.extra_context.is_none()
            && config.prompt_template.is_none()
            && config.enable_constrained_decoding.is_none()
            && config.use_ll_guidance.is_none()
            && config.filter_channel_content_from_kv_cache.is_none()
            && config.stream_tool_calls.is_none()
            && config.thinking.is_none();
        if is_default {
            return Ok(Self {
                raw: None,
                _session: session,
                _thinking: None,
            });
        }

        let raw = unsafe { ffi::litert_lm_conversation_config_create() };
        let raw = NonNull::new(raw)
            .ok_or(Error::NullPointer("litert_lm_conversation_config_create"))?;

        unsafe {
            if let Some(sc) = session.as_ptr() {
                ffi::litert_lm_conversation_config_set_session_config(raw.as_ptr(), sc.as_ptr());
            }
            if let Some(ref msg) = config.system_message {
                let c = CString::new(serde_json::to_string(msg)?)?;
                ffi::litert_lm_conversation_config_set_system_message(raw.as_ptr(), c.as_ptr());
            }
            if let Some(ref tools) = config.tools {
                let c = CString::new(serde_json::to_string(tools)?)?;
                ffi::litert_lm_conversation_config_set_tools(raw.as_ptr(), c.as_ptr());
            }
            if let Some(ref messages) = config.messages {
                let c = CString::new(serde_json::to_string(messages)?)?;
                ffi::litert_lm_conversation_config_set_messages(raw.as_ptr(), c.as_ptr());
            }
            if let Some(ref extra) = config.extra_context {
                let c = CString::new(serde_json::to_string(extra)?)?;
                ffi::litert_lm_conversation_config_set_extra_context(raw.as_ptr(), c.as_ptr());
            }
            if let Some(ref tmpl) = config.prompt_template {
                let c = CString::new(tmpl.as_str())?;
                ffi::litert_lm_conversation_config_set_prompt_template(raw.as_ptr(), c.as_ptr());
            }
            if let Some(v) = config.enable_constrained_decoding {
                ffi::litert_lm_conversation_config_set_enable_constrained_decoding(
                    raw.as_ptr(),
                    v,
                );
            }
            if config.use_ll_guidance == Some(true) {
                let provider = ffi::LiteRtLmConstraintProviderType_kLiteRtLmConstraintProviderTypeLlGuidance;
                ffi::litert_lm_conversation_config_set_constraint_provider(
                    raw.as_ptr(),
                    &provider,
                );
            }
            if let Some(v) = config.filter_channel_content_from_kv_cache {
                ffi::litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
                    raw.as_ptr(),
                    v,
                );
            }
            if let Some((enabled, ref channel)) = config.stream_tool_calls {
                let c = CString::new(channel.as_str())?;
                ffi::litert_lm_conversation_config_set_stream_tool_calls(
                    raw.as_ptr(),
                    enabled,
                    c.as_ptr(),
                );
            }
        }

        let thinking = if let Some(t) = config.thinking {
            let raw_t = RawThinkingConfig::new(t)?;
            unsafe {
                ffi::litert_lm_conversation_config_set_thinking_config(
                    raw.as_ptr(),
                    raw_t.as_ptr(),
                );
            }
            Some(raw_t)
        } else {
            None
        };

        Ok(Self {
            raw: Some(raw),
            _session: session,
            _thinking: thinking,
        })
    }

    fn as_mut_ptr(&self) -> *mut ffi::LiteRtLmConversationConfig {
        self.raw.map(|p| p.as_ptr()).unwrap_or(std::ptr::null_mut())
    }
}

impl Drop for RawConversationConfig {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe { ffi::litert_lm_conversation_config_delete(raw.as_ptr()) };
        }
    }
}

struct RawThinkingConfig {
    raw: NonNull<ffi::LiteRtLmThinkingConfig>,
}

impl RawThinkingConfig {
    fn new(config: ThinkingConfig) -> Result<Self> {
        let raw = unsafe { ffi::litert_lm_thinking_config_create() };
        let raw =
            NonNull::new(raw).ok_or(Error::NullPointer("litert_lm_thinking_config_create"))?;
        unsafe {
            if let Some(v) = config.enable_thinking {
                ffi::litert_lm_thinking_config_set_enable_thinking(raw.as_ptr(), v);
            }
            if let Some(v) = config.thinking_token_budget {
                ffi::litert_lm_thinking_config_set_thinking_token_budget(raw.as_ptr(), v);
            }
        }
        Ok(Self { raw })
    }

    fn as_ptr(&self) -> *mut ffi::LiteRtLmThinkingConfig {
        self.raw.as_ptr()
    }
}

impl Drop for RawThinkingConfig {
    fn drop(&mut self) {
        unsafe { ffi::litert_lm_thinking_config_delete(self.raw.as_ptr()) };
    }
}

struct RawOptionalArgs {
    raw: NonNull<ffi::LiteRtLmConversationOptionalArgs>,
    _owned: Vec<OwnedNative>,
}

enum OwnedNative {
    Repetition(NonNull<ffi::LiteRtLmRepetitionPenaltyConfig>),
    NoRepeat(NonNull<ffi::LiteRtLmNoRepeatNgramConfig>),
    Suppress(NonNull<ffi::LiteRtLmSuppressTokensConfig>),
    Thinking(RawThinkingConfig),
}

impl Drop for OwnedNative {
    fn drop(&mut self) {
        unsafe {
            match self {
                Self::Repetition(p) => {
                    ffi::litert_lm_repetition_penalty_config_delete(p.as_ptr());
                }
                Self::NoRepeat(p) => {
                    ffi::litert_lm_no_repeat_ngram_config_delete(p.as_ptr());
                }
                Self::Suppress(p) => {
                    ffi::litert_lm_suppress_tokens_config_delete(p.as_ptr());
                }
                Self::Thinking(_) => {}
            }
        }
    }
}

impl RawOptionalArgs {
    fn new(args: ConversationOptionalArgs) -> Result<Self> {
        let raw = unsafe { ffi::litert_lm_conversation_optional_args_create() };
        let raw = NonNull::new(raw).ok_or(Error::NullPointer(
            "litert_lm_conversation_optional_args_create",
        ))?;
        let mut owned = Vec::new();

        unsafe {
            if let Some(cfg) = args.repetition_penalty {
                let p = ffi::litert_lm_repetition_penalty_config_create();
                let p = NonNull::new(p).ok_or(Error::NullPointer(
                    "litert_lm_repetition_penalty_config_create",
                ))?;
                if let Some(v) = cfg.repetition_penalty {
                    ffi::litert_lm_repetition_penalty_config_set_repetition_penalty(p.as_ptr(), v);
                }
                if let Some(v) = cfg.presence_penalty {
                    ffi::litert_lm_repetition_penalty_config_set_presence_penalty(p.as_ptr(), v);
                }
                if let Some(v) = cfg.frequency_penalty {
                    ffi::litert_lm_repetition_penalty_config_set_frequency_penalty(p.as_ptr(), v);
                }
                if let Some(v) = cfg.window_size {
                    ffi::litert_lm_repetition_penalty_config_set_window_size(p.as_ptr(), v);
                }
                ffi::litert_lm_conversation_optional_args_set_repetition_penalty_config(
                    raw.as_ptr(),
                    p.as_ptr(),
                );
                owned.push(OwnedNative::Repetition(p));
            }
            if let Some(cfg) = args.no_repeat_ngram {
                let p = ffi::litert_lm_no_repeat_ngram_config_create();
                let p = NonNull::new(p).ok_or(Error::NullPointer(
                    "litert_lm_no_repeat_ngram_config_create",
                ))?;
                if let Some(v) = cfg.no_repeat_ngram_size {
                    ffi::litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(p.as_ptr(), v);
                }
                if let Some(v) = cfg.window_size {
                    ffi::litert_lm_no_repeat_ngram_config_set_window_size(p.as_ptr(), v);
                }
                ffi::litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
                    raw.as_ptr(),
                    p.as_ptr(),
                );
                owned.push(OwnedNative::NoRepeat(p));
            }
            if let Some(cfg) = args.suppress_tokens {
                let p = ffi::litert_lm_suppress_tokens_config_create();
                let p = NonNull::new(p).ok_or(Error::NullPointer(
                    "litert_lm_suppress_tokens_config_create",
                ))?;
                ffi::litert_lm_suppress_tokens_config_set_suppress_tokens(
                    p.as_ptr(),
                    cfg.tokens.as_ptr(),
                    cfg.tokens.len(),
                );
                ffi::litert_lm_conversation_optional_args_set_suppress_tokens_config(
                    raw.as_ptr(),
                    p.as_ptr(),
                );
                owned.push(OwnedNative::Suppress(p));
            }
            if let Some(v) = args.visual_token_budget {
                ffi::litert_lm_conversation_optional_args_set_visual_token_budget(raw.as_ptr(), v);
            }
            if let Some(v) = args.max_output_tokens {
                ffi::litert_lm_conversation_optional_args_set_max_output_tokens(raw.as_ptr(), v);
            }
            if let Some(t) = args.thinking {
                let raw_t = RawThinkingConfig::new(t)?;
                ffi::litert_lm_conversation_optional_args_set_thinking_config(
                    raw.as_ptr(),
                    raw_t.as_ptr(),
                );
                owned.push(OwnedNative::Thinking(raw_t));
            }
            if let Some(ty) = args.constraint_type {
                let s = args.constraint_string.unwrap_or_default();
                let c = CString::new(s)?;
                ffi::litert_lm_conversation_optional_args_set_constraint(
                    raw.as_ptr(),
                    ty.into_ffi(),
                    c.as_ptr(),
                );
            }
        }

        Ok(Self { raw, _owned: owned })
    }

    fn as_ptr(&self) -> *const ffi::LiteRtLmConversationOptionalArgs {
        self.raw.as_ptr()
    }
}

impl Drop for RawOptionalArgs {
    fn drop(&mut self) {
        unsafe { ffi::litert_lm_conversation_optional_args_delete(self.raw.as_ptr()) };
    }
}

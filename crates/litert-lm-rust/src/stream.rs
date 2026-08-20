//! Streaming callbacks bridged into a cross-thread channel.

use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use std::sync::mpsc::{self, Receiver, Sender};

use crate::error::{Error, Result};
use crate::ffi;
use crate::input::OwnedInputs;

/// A single streamed chunk from LiteRT-LM.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub text: Option<String>,
    pub error: Option<String>,
    pub is_final: bool,
}

/// Events produced by async generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Chunk(StreamChunk),
    /// Native stream failed to start (status != 0).
    StartFailed(i32),
}

struct CallbackState {
    tx: Sender<StreamEvent>,
}

extern "C" fn stream_callback(
    userdata: *mut std::os::raw::c_void,
    chunk: *const ffi::LiteRtLmStreamChunk,
) {
    if userdata.is_null() || chunk.is_null() {
        return;
    }
    let state = unsafe { &*(userdata as *const CallbackState) };
    let text = unsafe {
        let ptr = ffi::litert_lm_stream_chunk_get_text(chunk);
        if ptr.is_null() {
            None
        } else {
            CStr::from_ptr(ptr).to_str().ok().map(str::to_owned)
        }
    };
    let error = unsafe {
        let ptr = ffi::litert_lm_stream_chunk_get_error(chunk);
        if ptr.is_null() {
            None
        } else {
            CStr::from_ptr(ptr).to_str().ok().map(str::to_owned)
        }
    };
    let is_final = unsafe { ffi::litert_lm_stream_chunk_is_final(chunk) };
    let _ = state.tx.send(StreamEvent::Chunk(StreamChunk {
        text,
        error,
        is_final,
    }));
}

/// Holds native callback userdata until the receiver is dropped.
struct CallbackGuard {
    ptr: *mut CallbackState,
}

unsafe impl Send for CallbackGuard {}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                drop(Box::from_raw(self.ptr));
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Receiver half of a streaming generation call.
pub struct StreamEventReceiver {
    rx: Receiver<StreamEvent>,
    _inputs: Option<OwnedInputs>,
    _callback: CallbackGuard,
}

impl StreamEventReceiver {
    pub fn try_recv(&self) -> std::result::Result<StreamEvent, mpsc::TryRecvError> {
        self.rx.try_recv()
    }

    pub fn recv(&self) -> std::result::Result<StreamEvent, mpsc::RecvError> {
        self.rx.recv()
    }

    pub fn iter(&self) -> mpsc::Iter<'_, StreamEvent> {
        self.rx.iter()
    }

    /// Collect text until a final chunk (or error).
    pub fn collect_text(self) -> Result<String> {
        let mut out = String::new();
        for event in self.rx.iter() {
            match event {
                StreamEvent::StartFailed(code) => {
                    return Err(Error::NativeStatus("stream_start", code));
                }
                StreamEvent::Chunk(chunk) => {
                    if let Some(err) = chunk.error {
                        return Err(Error::Message(err));
                    }
                    if let Some(text) = chunk.text {
                        out.push_str(&text);
                    }
                    if chunk.is_final {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }
}

impl IntoIterator for StreamEventReceiver {
    type Item = StreamEvent;
    type IntoIter = mpsc::IntoIter<StreamEvent>;

    fn into_iter(self) -> Self::IntoIter {
        self.rx.into_iter()
    }
}

fn new_receiver(
    rx: Receiver<StreamEvent>,
    inputs: Option<OwnedInputs>,
    state_ptr: *mut CallbackState,
) -> StreamEventReceiver {
    StreamEventReceiver {
        rx,
        _inputs: inputs,
        _callback: CallbackGuard { ptr: state_ptr },
    }
}

pub(crate) fn start_session_stream(
    session: NonNull<ffi::LiteRtLmSession>,
    inputs: OwnedInputs,
) -> Result<StreamEventReceiver> {
    let (tx, rx) = mpsc::channel();
    let state = Box::new(CallbackState { tx: tx.clone() });
    let state_ptr = Box::into_raw(state);

    let status = unsafe {
        ffi::litert_lm_session_generate_content_stream(
            session.as_ptr(),
            inputs.as_ptr(),
            inputs.len(),
            Some(stream_callback),
            state_ptr.cast(),
        )
    };

    if status != 0 {
        unsafe {
            drop(Box::from_raw(state_ptr));
        }
        let _ = tx.send(StreamEvent::StartFailed(status));
        // No live callback; create a dummy guard.
        return Ok(StreamEventReceiver {
            rx,
            _inputs: Some(inputs),
            _callback: CallbackGuard {
                ptr: std::ptr::null_mut(),
            },
        });
    }

    Ok(new_receiver(rx, Some(inputs), state_ptr))
}

pub(crate) fn start_decode_stream(
    session: NonNull<ffi::LiteRtLmSession>,
) -> Result<StreamEventReceiver> {
    let (tx, rx) = mpsc::channel();
    let state = Box::new(CallbackState { tx });
    let state_ptr = Box::into_raw(state);
    let status = unsafe {
        ffi::litert_lm_session_run_decode_async(
            session.as_ptr(),
            Some(stream_callback),
            state_ptr.cast(),
        )
    };
    if status != 0 {
        unsafe {
            drop(Box::from_raw(state_ptr));
        }
        return Err(Error::NativeStatus(
            "litert_lm_session_run_decode_async",
            status,
        ));
    }
    Ok(new_receiver(rx, None, state_ptr))
}

pub(crate) struct CStringGuard(pub CString);

impl CStringGuard {
    pub(crate) fn new(s: &str) -> Result<Self> {
        Ok(Self(CString::new(s)?))
    }

    pub(crate) fn as_ptr(&self) -> *const std::os::raw::c_char {
        self.0.as_ptr()
    }
}

pub(crate) fn start_conversation_stream(
    conversation: NonNull<ffi::LiteRtLmConversation>,
    message_json: &CString,
    extra_context: Option<&CString>,
    optional_args: *const ffi::LiteRtLmConversationOptionalArgs,
) -> Result<StreamEventReceiver> {
    let (tx, rx) = mpsc::channel();
    let state = Box::new(CallbackState { tx });
    let state_ptr = Box::into_raw(state);

    let status = unsafe {
        ffi::litert_lm_conversation_send_message_stream(
            conversation.as_ptr(),
            message_json.as_ptr(),
            extra_context
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null()),
            optional_args,
            Some(stream_callback),
            state_ptr.cast(),
        )
    };
    if status != 0 {
        unsafe {
            drop(Box::from_raw(state_ptr));
        }
        return Err(Error::NativeStatus(
            "litert_lm_conversation_send_message_stream",
            status,
        ));
    }
    Ok(new_receiver(rx, None, state_ptr))
}

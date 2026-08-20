//! Multimodal input data for the Session API.

use std::ptr::NonNull;

use crate::error::{Error, Result};
use crate::ffi;

/// Multimodal input for [`crate::Session::generate_content`].
#[derive(Debug, Clone)]
pub enum InputData {
    Text(String),
    Image(Vec<u8>),
    ImageEnd,
    Audio(Vec<u8>),
    AudioEnd,
}

impl InputData {
    fn ffi_type(&self) -> ffi::LiteRtLmInputDataType {
        match self {
            Self::Text(_) => ffi::LiteRtLmInputDataType_kLiteRtLmInputDataTypeText,
            Self::Image(_) => ffi::LiteRtLmInputDataType_kLiteRtLmInputDataTypeImage,
            Self::ImageEnd => ffi::LiteRtLmInputDataType_kLiteRtLmInputDataTypeImageEnd,
            Self::Audio(_) => ffi::LiteRtLmInputDataType_kLiteRtLmInputDataTypeAudio,
            Self::AudioEnd => ffi::LiteRtLmInputDataType_kLiteRtLmInputDataTypeAudioEnd,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Self::Text(s) => s.as_bytes(),
            Self::Image(b) | Self::Audio(b) => b.as_slice(),
            Self::ImageEnd | Self::AudioEnd => &[],
        }
    }
}

/// Owned native input array kept alive for the duration of a call / stream.
pub(crate) struct OwnedInputs {
    ptrs: Vec<*const ffi::LiteRtLmInputData>,
    // Keep NonNull owners so Drop deletes each input.
    owned: Vec<NonNull<ffi::LiteRtLmInputData>>,
}

impl OwnedInputs {
    pub(crate) fn new(inputs: &[InputData]) -> Result<Self> {
        let mut owned = Vec::with_capacity(inputs.len());
        let mut ptrs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let bytes = input.bytes();
            let raw = unsafe {
                ffi::litert_lm_input_data_create(
                    input.ffi_type(),
                    bytes.as_ptr().cast(),
                    bytes.len(),
                )
            };
            let raw =
                NonNull::new(raw).ok_or(Error::NullPointer("litert_lm_input_data_create"))?;
            ptrs.push(raw.as_ptr() as *const _);
            owned.push(raw);
        }
        Ok(Self { ptrs, owned })
    }

    pub(crate) fn as_ptr(&self) -> *const *const ffi::LiteRtLmInputData {
        self.ptrs.as_ptr()
    }

    pub(crate) fn len(&self) -> usize {
        self.ptrs.len()
    }
}

impl Drop for OwnedInputs {
    fn drop(&mut self) {
        for raw in self.owned.drain(..) {
            unsafe { ffi::litert_lm_input_data_delete(raw.as_ptr()) };
        }
    }
}

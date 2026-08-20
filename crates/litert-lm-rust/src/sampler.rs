//! Sampler parameters for generation.

use std::ptr::NonNull;

use crate::error::{Error, Result};
use crate::ffi;

/// Sampler algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerType {
    TopK,
    TopP,
    Greedy,
}

impl SamplerType {
    fn into_ffi(self) -> ffi::LiteRtLmSamplerType {
        match self {
            Self::TopK => ffi::LiteRtLmSamplerType_kLiteRtLmSamplerTypeTopK,
            Self::TopP => ffi::LiteRtLmSamplerType_kLiteRtLmSamplerTypeTopP,
            Self::Greedy => ffi::LiteRtLmSamplerType_kLiteRtLmSamplerTypeGreedy,
        }
    }
}

/// Sampling hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerParams {
    pub sampler_type: SamplerType,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub temperature: Option<f32>,
    pub seed: Option<i32>,
}

impl SamplerParams {
    pub fn greedy() -> Self {
        Self {
            sampler_type: SamplerType::Greedy,
            top_k: None,
            top_p: None,
            temperature: None,
            seed: None,
        }
    }

    pub fn top_k(k: i32) -> Self {
        Self {
            sampler_type: SamplerType::TopK,
            top_k: Some(k),
            top_p: None,
            temperature: None,
            seed: None,
        }
    }

    pub fn top_p(p: f32) -> Self {
        Self {
            sampler_type: SamplerType::TopP,
            top_k: None,
            top_p: Some(p),
            temperature: None,
            seed: None,
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the top-k value alongside the chosen sampler type (the native
    /// top-p sampler also requires a positive `k`).
    pub fn with_top_k(mut self, k: i32) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn with_seed(mut self, seed: i32) -> Self {
        self.seed = Some(seed);
        self
    }

    pub(crate) fn create_ffi(self) -> Result<RawSamplerParams> {
        let raw = unsafe { ffi::litert_lm_sampler_params_create(self.sampler_type.into_ffi()) };
        let raw =
            NonNull::new(raw).ok_or(Error::NullPointer("litert_lm_sampler_params_create"))?;
        unsafe {
            if let Some(v) = self.top_k {
                ffi::litert_lm_sampler_params_set_top_k(raw.as_ptr(), v);
            }
            if let Some(v) = self.top_p {
                ffi::litert_lm_sampler_params_set_top_p(raw.as_ptr(), v);
            }
            if let Some(v) = self.temperature {
                ffi::litert_lm_sampler_params_set_temperature(raw.as_ptr(), v);
            }
            if let Some(v) = self.seed {
                ffi::litert_lm_sampler_params_set_seed(raw.as_ptr(), v);
            }
        }
        Ok(RawSamplerParams { raw })
    }
}

pub(crate) struct RawSamplerParams {
    raw: NonNull<ffi::LiteRtLmSamplerParams>,
}

impl RawSamplerParams {
    pub(crate) fn as_ptr(&self) -> *mut ffi::LiteRtLmSamplerParams {
        self.raw.as_ptr()
    }
}

impl Drop for RawSamplerParams {
    fn drop(&mut self) {
        unsafe { ffi::litert_lm_sampler_params_delete(self.raw.as_ptr()) };
    }
}

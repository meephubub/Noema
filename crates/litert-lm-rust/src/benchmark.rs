//! Benchmark information from sessions / conversations.

use std::ptr::NonNull;

use crate::ffi;

/// Timing and throughput metrics from a session or conversation.
pub struct BenchmarkInfo {
    pub(crate) raw: NonNull<ffi::LiteRtLmBenchmarkInfo>,
}

impl BenchmarkInfo {
    pub fn time_to_first_token(&self) -> f64 {
        unsafe { ffi::litert_lm_benchmark_info_get_time_to_first_token(self.raw.as_ptr()) }
    }

    pub fn total_init_time_secs(&self) -> f64 {
        unsafe { ffi::litert_lm_benchmark_info_get_total_init_time_in_second(self.raw.as_ptr()) }
    }

    pub fn num_prefill_turns(&self) -> i32 {
        unsafe { ffi::litert_lm_benchmark_info_get_num_prefill_turns(self.raw.as_ptr()) }
    }

    pub fn num_decode_turns(&self) -> i32 {
        unsafe { ffi::litert_lm_benchmark_info_get_num_decode_turns(self.raw.as_ptr()) }
    }

    pub fn prefill_token_count_at(&self, index: i32) -> i32 {
        unsafe {
            ffi::litert_lm_benchmark_info_get_prefill_token_count_at(self.raw.as_ptr(), index)
        }
    }

    pub fn decode_token_count_at(&self, index: i32) -> i32 {
        unsafe {
            ffi::litert_lm_benchmark_info_get_decode_token_count_at(self.raw.as_ptr(), index)
        }
    }

    pub fn prefill_tokens_per_sec_at(&self, index: i32) -> f64 {
        unsafe {
            ffi::litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(self.raw.as_ptr(), index)
        }
    }

    pub fn decode_tokens_per_sec_at(&self, index: i32) -> f64 {
        unsafe {
            ffi::litert_lm_benchmark_info_get_decode_tokens_per_sec_at(self.raw.as_ptr(), index)
        }
    }
}

impl Drop for BenchmarkInfo {
    fn drop(&mut self) {
        unsafe { ffi::litert_lm_benchmark_info_delete(self.raw.as_ptr()) };
    }
}

impl std::fmt::Debug for BenchmarkInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BenchmarkInfo")
            .field("time_to_first_token", &self.time_to_first_token())
            .field("total_init_time_secs", &self.total_init_time_secs())
            .field("num_prefill_turns", &self.num_prefill_turns())
            .field("num_decode_turns", &self.num_decode_turns())
            .finish()
    }
}

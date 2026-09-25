pub mod bonsai;
pub mod needle;
pub mod qwen;
pub mod qwen2vl;
pub mod qwen35;

use std::ffi::{c_char, c_int, c_void};
use std::sync::Arc;
use std::sync::OnceLock;

static SHARED_BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

unsafe extern "C" fn null_log_callback(
    _level: c_int,
    _text: *const c_char,
    _user_data: *mut c_void,
) {
}

unsafe extern "C" {
    fn llama_log_set(
        log_callback: Option<unsafe extern "C" fn(c_int, *const c_char, *mut c_void)>,
        user_data: *mut c_void,
    );
}

use llama_cpp_2::llama_backend::LlamaBackend;

pub fn shared_backend() -> &'static Arc<LlamaBackend> {
    SHARED_BACKEND.get_or_init(|| {
        unsafe {
            llama_log_set(Some(null_log_callback), std::ptr::null_mut());
        }
        Arc::new(LlamaBackend::init().expect("failed to initialize llama backend"))
    })
}

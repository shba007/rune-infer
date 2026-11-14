pub mod audio;
pub mod crispasr;
pub mod download;
pub mod encoder;
pub mod litert;
pub mod llama_server;
pub mod needle;
pub mod nemo_speech;
pub mod onnx;
pub(crate) mod registry;
pub(crate) mod release_cache;
pub mod sd_server;
pub mod whisper;

#[cfg_attr(not(feature = "tui"), allow(dead_code))]
pub(crate) mod update;

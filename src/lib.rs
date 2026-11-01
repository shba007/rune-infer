#![allow(clippy::too_many_arguments, clippy::collapsible_if)]

pub mod api;
pub mod audio_operations;
pub mod config;
pub mod detect;
pub mod home;
pub mod inference;
pub mod media;
pub mod migrate;
pub mod proxy;
pub mod schema;
pub mod server;
pub mod setup;
pub mod telemetry;
#[cfg(feature = "tui")]
pub mod tui;
pub mod types;

pub use api::{AppState, create_router};
pub use config::{
    Modality, ModelConfig, ModelRegistry, ProviderType, RateLimitConfig, ServerConfig,
};
pub use proxy::ProxyService;
pub use types::{
    ApiError, ChatCompletionRequest, Choice, ChoiceDelta, HealthResponse, LivezResponse, ModelInfo,
    ModelsResponse, ReadyzResponse, SseResponse,
};

/// ASCII rendering of public/logo.svg's concentric dot rings and central dot.
/// Coordinates are sampled at twice the horizontal resolution for terminal cells.
pub const CLI_LOGO: &str = "          .   .\n     . .         . .\n   .     o  o  o     .\n      o   o   o   o\n .  o   o       o   o  .\n.  o  o  O  O  O  o  o  .\n            *\n.  o  o  O  O  O  o  o  .\n .  o   o       o   o  .\n      o   o   o   o\n   .     o  o  o     .\n     . .         . .\n          .   .";

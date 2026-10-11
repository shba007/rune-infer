pub mod api;
pub mod config;
pub mod home;
pub mod inference;
pub mod media;
pub mod proxy;
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
    ApiError, ChatCompletionRequest, Choice, ChoiceDelta, HealthResponse, ModelInfo,
    ModelsResponse, SseResponse,
};

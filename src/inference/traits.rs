use super::types::{InferenceOutput, InferenceTaskRequest};
use std::error::Error;

pub trait InferenceEngine: Send + Sync {
    fn id(&self) -> &str;
    fn execute(
        &self,
        task: &InferenceTaskRequest,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>>;

    /// Signals the engine to abort any active generation or prefill and free occupied slots.
    fn cancel(&self) {}

    /// Whether this engine implements cancellation beyond the default no-op.
    fn supports_cancellation(&self) -> bool {
        false
    }

    /// Whether the cancellation hook can interrupt this specific task.
    fn supports_task_cancellation(&self, _task: &InferenceTaskRequest) -> bool {
        self.supports_cancellation()
    }
}

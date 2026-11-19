use super::types::{InferenceOutput, InferenceTaskRequest};
use std::error::Error;

pub trait InferenceEngine: Send + Sync {
    fn id(&self) -> &str;
    fn execute(
        &self,
        task: &InferenceTaskRequest,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>>;

    fn cancel(&self) {}

    fn supports_cancellation(&self) -> bool {
        false
    }

    fn supports_task_cancellation(&self, _task: &InferenceTaskRequest) -> bool {
        self.supports_cancellation()
    }
}

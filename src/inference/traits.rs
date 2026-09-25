use super::types::{InferenceOutput, InferenceTaskRequest};
use std::error::Error;

pub trait InferenceEngine: Send + Sync {
    fn id(&self) -> &str;
    fn execute(
        &self,
        task: &InferenceTaskRequest,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>>;
}

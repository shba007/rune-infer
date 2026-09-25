use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum Modality {
    Text,
    VisionText,
}

impl Modality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::VisionText => "vision-text",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub max_loaded_models: usize,
    pub idle_unload_seconds: u64,
    #[serde(default = "default_vram_budget_ratio")]
    pub vram_budget_ratio: f64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            api_key: None,
            max_loaded_models: 1,
            idle_unload_seconds: 300,
            vram_budget_ratio: 0.97,
        }
    }
}

fn default_vram_budget_ratio() -> f64 {
    0.97
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub schema_version: u32,
    pub server: ServerConfig,
    pub models: Vec<ModelConfig>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            schema_version: 1,
            server: ServerConfig::default(),
            models: Vec::new(),
        }
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = std::fs::read_to_string(path).context("Failed to read config file")?;
        let registry: ModelRegistry =
            serde_json::from_str(&raw).context("Failed to parse config as JSON")?;
        registry.validate()?;
        Ok(registry)
    }

    pub fn validate(&self) -> Result<()> {
        let mut seen_ids = std::collections::HashMap::new();
        for model in &self.models {
            if seen_ids.contains_key(&model.id) {
                anyhow::bail!("Duplicate model id '{}'", model.id);
            }
            seen_ids.insert(model.id.clone(), model.name.clone());
            model.validate()?;
        }
        Ok(())
    }

    pub fn find(&self, id: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|m| {
            m.id == id
                || m.id.eq_ignore_ascii_case(id)
                || m.name.eq_ignore_ascii_case(id)
                || (m.id == "ternary-Bonsai-2-27b" && id == "ternary-Bonsai-2-27b")
        })
    }

    pub fn find_owned(&self, id: &str) -> Option<ModelConfig> {
        self.find(id).cloned()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeConfig {
    #[serde(default = "default_context_length")]
    pub context_length: i64,
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: u32,
    #[serde(default)]
    pub extra_args: Option<String>,
}

fn default_context_length() -> i64 {
    8192
}

fn default_gpu_layers() -> u32 {
    99
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    pub modality: Modality,
    pub architecture: String,
    pub model_path: String,
    pub mmproj_path: Option<String>,
    pub vision: bool,
    pub description: String,
    #[serde(default)]
    pub bits_per_weight: Option<f64>,
    #[serde(default)]
    pub total_params: Option<u64>,
    #[serde(default)]
    pub max_context_length: Option<u32>,
    #[serde(default)]
    pub kv_bytes_per_token: Option<u64>,
    #[serde(default)]
    pub max_resolution: Option<String>,
    #[serde(default)]
    pub capabilities: Option<Vec<String>>,
    #[serde(default)]
    pub runtime: Option<RuntimeConfig>,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() {
            anyhow::bail!("Model '{}' has empty id", self.name);
        }
        Ok(())
    }

    pub fn resolved_capabilities(&self) -> Vec<String> {
        if let Some(ref caps) = self.capabilities {
            if !caps.is_empty() {
                return caps.clone();
            }
        }
        let mut caps = vec!["Chat".to_string(), "Tool Call".to_string()];
        if self.vision || self.modality == Modality::VisionText {
            caps.push("Vision".to_string());
        }
        caps
    }
}

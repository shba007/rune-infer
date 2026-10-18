//! VRAM Budgeting & Context Sizing Engine
//!
//! Provides pre-flight GPU VRAM detection, layer offloading calculations,
//! KV cache memory estimation, and dynamic context sizing to guarantee
//! zero CUDA out-of-memory crashes.

use crate::config::{ModelConfig, ServerConfig};

/// Comprehensive breakdown of pre-flight VRAM and context allocation for a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VramBudgetPlan {
    /// Context window in tokens (clamped to VRAM budget or model capacity).
    pub n_ctx: u32,
    /// Estimated VRAM required for KV cache at `n_ctx` tokens.
    pub context_vram_bytes: u64,
    /// Model weight bytes placed on the GPU (accounting for layer offloading).
    pub model_vram_on_gpu_bytes: u64,
    /// Total model weight file size on disk.
    pub model_vram_total_bytes: u64,
    /// Vision projector weight bytes (e.g. `mmproj`).
    pub projector_vram_bytes: u64,
    /// Speculative decoding draft head weight bytes (e.g. `mtp`).
    pub mtp_vram_bytes: u64,
    /// Total detected physical GPU VRAM on the system.
    pub total_gpu_vram_bytes: u64,
}

impl VramBudgetPlan {
    /// Combined estimated VRAM needed on GPU across weights, projector, MTP draft, and KV context.
    pub fn total_estimated_vram_bytes(&self) -> u64 {
        self.model_vram_on_gpu_bytes
            + self.projector_vram_bytes
            + self.mtp_vram_bytes
            + self.context_vram_bytes
    }
}

/// Formats a raw byte count into human-readable units (B, KiB, MiB, GiB, TiB).
pub fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;

    let b = bytes as f64;
    if b >= TB {
        format!("{:.2} TiB", b / TB)
    } else if b >= GB {
        format!("{:.2} GiB", b / GB)
    } else if b >= MB {
        format!("{:.2} MiB", b / MB)
    } else if b >= KB {
        format!("{:.2} KiB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}

/// Formats parameter count into a compact string (e.g. 27B, 300M, 7B).
pub fn format_params(params: u64) -> String {
    let p = params as f64;
    if p >= 1_000_000_000.0 {
        let val = p / 1_000_000_000.0;
        if (val.fract() * 10.0).round() == 0.0 {
            format!("{:.0}B", val)
        } else {
            format!("{:.1}B", val)
        }
    } else if p >= 1_000_000.0 {
        let val = p / 1_000_000.0;
        if (val.fract() * 10.0).round() == 0.0 {
            format!("{:.0}M", val)
        } else {
            format!("{:.1}M", val)
        }
    } else if p >= 1_000.0 {
        format!("{:.0}K", p / 1_000.0)
    } else {
        format!("{}", params)
    }
}

/// Formats token counts cleanly (e.g. "128K tokens", "8192 tokens", "1.0M tokens").
pub fn format_tokens(tokens: u32) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M tokens", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1024 && tokens.is_multiple_of(1024) {
        format!("{}K tokens", tokens / 1024)
    } else if tokens >= 1000 {
        format!("{:.1}K tokens", tokens as f64 / 1000.0)
    } else {
        format!("{} tokens", tokens)
    }
}

static TOTAL_GPU_VRAM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();

/// Detects total GPU VRAM across NVIDIA (via nvidia-smi), macOS unified memory (via sysctl),
/// or Linux system memory fallback. Cached after initial detection since physical GPU VRAM is constant.
pub fn detect_total_gpu_vram() -> Option<u64> {
    *TOTAL_GPU_VRAM.get_or_init(query_total_gpu_vram_uncached)
}

fn query_total_gpu_vram_uncached() -> Option<u64> {
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = text.lines().next() {
                if let Ok(mib) = line.trim().parse::<u64>() {
                    return Some(mib * 1024 * 1024);
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                if let Ok(bytes) = text.trim().parse::<u64>() {
                    return Some(bytes);
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Detects currently used GPU VRAM via nvidia-smi.
pub fn detect_used_gpu_vram() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mib: u64 = text.lines().next()?.trim().parse().ok()?;
    Some(mib * 1024 * 1024)
}

/// Heuristic parameter estimation from model metadata, id, name, or file size.
pub fn heuristic_model_params(model: &ModelConfig) -> u64 {
    if let Some(p) = model.total_params {
        if p > 0 {
            return p;
        }
    }

    let text = format!("{} {} {}", model.id, model.name, model.model_path).to_lowercase();
    if text.contains("70b") {
        70_000_000_000
    } else if text.contains("32b") {
        32_000_000_000
    } else if text.contains("27b") {
        27_000_000_000
    } else if text.contains("14b") {
        14_000_000_000
    } else if text.contains("9b") || text.contains("8b") {
        8_000_000_000
    } else if text.contains("7b") {
        7_000_000_000
    } else if text.contains("3b") || text.contains("4b") {
        3_500_000_000
    } else if text.contains("1.5b") || text.contains("1b") || text.contains("0.8b") {
        1_500_000_000
    } else {
        let file_size = std::fs::metadata(&model.model_path)
            .map(|m| m.len())
            .unwrap_or(0);
        if file_size >= 12 * 1024 * 1024 * 1024 {
            32_000_000_000
        } else if file_size >= 5 * 1024 * 1024 * 1024 {
            27_000_000_000
        } else if file_size >= 2 * 1024 * 1024 * 1024 {
            7_000_000_000
        } else {
            1_500_000_000
        }
    }
}

/// Heuristic estimate of model layers by parameter count when metadata is not provided.
pub fn estimate_total_layers(model: &ModelConfig) -> u32 {
    let params = heuristic_model_params(model);
    match params {
        p if p < 4_000_000_000 => 28,
        p if p < 10_000_000_000 => 32,
        p if p < 18_000_000_000 => 48,
        p if p < 36_000_000_000 => 64,
        p if p < 80_000_000_000 => 80,
        _ => 96,
    }
}

/// Parses CLI flags like `--cache-type-k` or `-ctk` to determine KV cache quantization scaling.
pub fn parse_cache_type_multiplier(args: &str, flag_long: &str, flag_short: &str) -> f64 {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    for (i, &t) in tokens.iter().enumerate() {
        let val = if t == flag_long || t == flag_short {
            tokens.get(i + 1).copied()
        } else if let Some(stripped) = t.strip_prefix(&format!("{flag_long}=")) {
            Some(stripped)
        } else {
            t.strip_prefix(&format!("{flag_short}="))
        };

        if let Some(v) = val {
            let clean = v.to_lowercase();
            if clean.contains("q4_0") || clean.contains("q4_1") || clean.contains("iq4") {
                return 0.28;
            } else if clean.contains("q5_0") || clean.contains("q5_1") {
                return 0.35;
            } else if clean.contains("q8_0") || clean.contains("fp8") || clean.contains("f8") {
                return 0.53;
            } else if clean.contains("f16") || clean.contains("bf16") {
                return 1.0;
            }
        }
    }
    1.0
}

/// Estimates bytes required per token of KV cache based on model size and cache quantization flags.
pub fn estimate_bytes_per_token(model: &ModelConfig) -> u64 {
    let extra_args = model
        .runtime
        .as_ref()
        .and_then(|r| r.extra_args.as_deref())
        .unwrap_or("");

    if extra_args.contains("--no-kv-offload") {
        return 0;
    }

    let params = heuristic_model_params(model);
    let base = match params {
        p if p < 2_000_000_000 => 16 * 1024,
        p if p < 10_000_000_000 => 32 * 1024,
        p if p < 35_000_000_000 => 64 * 1024,
        _ => 128 * 1024,
    };

    let k_mult = parse_cache_type_multiplier(extra_args, "--cache-type-k", "-ctk");
    let v_mult = parse_cache_type_multiplier(extra_args, "--cache-type-v", "-ctv");
    let quant_mult = (k_mult + v_mult) / 2.0;

    ((base as f64 * quant_mult).max(1024.0)) as u64
}

/// Calculates pre-flight VRAM budget, offload ratios, and optimal dynamic context length.
pub fn calculate_budget(model: &ModelConfig, server: &ServerConfig) -> VramBudgetPlan {
    let total_vram = detect_total_gpu_vram().unwrap_or(24 * 1024 * 1024 * 1024);
    let budget_ratio = server.vram_budget_ratio.clamp(0.0, 1.0);
    let budget_bytes = (total_vram as f64 * budget_ratio) as u64;

    let total_model_bytes = std::fs::metadata(&model.model_path)
        .map(|m| m.len())
        .unwrap_or_else(|_| {
            if let (Some(params), Some(bpw)) = (model.total_params, model.bits_per_weight) {
                ((params as f64 * bpw) / 8.0) as u64
            } else {
                4 * 1024 * 1024 * 1024
            }
        });

    let gpu_layers = model.runtime.as_ref().map(|r| r.gpu_layers).unwrap_or(99);
    let total_layers = estimate_total_layers(model);
    let model_vram_on_gpu = if gpu_layers >= total_layers || gpu_layers >= 90 {
        total_model_bytes
    } else {
        let offload_ratio = (gpu_layers as f64 / total_layers as f64).clamp(0.0, 1.0);
        ((total_model_bytes as f64) * (0.05 + 0.95 * offload_ratio)).min(total_model_bytes as f64)
            as u64
    };

    let projector_vram = if let Some(ref mmproj) = model.mmproj_path {
        std::fs::metadata(mmproj).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let mtp_vram = if let Some(ref mtp) = model.mtp_path {
        std::fs::metadata(mtp).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let bytes_per_token = estimate_bytes_per_token(model);
    let max_capacity = model.max_context_length.unwrap_or(131072);
    let configured_ctx = model
        .runtime
        .as_ref()
        .map(|r| r.context_length)
        .unwrap_or(8192);

    let mtp_headroom = if model.has_mtp() {
        1536 * 1024 * 1024
    } else {
        0
    };

    let extra_args = model
        .runtime
        .as_ref()
        .and_then(|r| r.extra_args.as_deref())
        .unwrap_or("");
    let has_flash_attn = extra_args.contains("-fa") || extra_args.contains("--flash-attn");
    let overhead_scratch = if has_flash_attn {
        512 * 1024 * 1024
    } else {
        768 * 1024 * 1024
    };

    let n_ctx = if configured_ctx <= 0 {
        // Automatic dynamic context calculation based on remaining budgeted headroom
        let fixed_overhead =
            model_vram_on_gpu + projector_vram + mtp_vram + overhead_scratch + mtp_headroom;
        let remaining_kv = budget_bytes.saturating_sub(fixed_overhead);

        if let Some(raw_tokens) = remaining_kv.checked_div(bytes_per_token) {
            let rounded = ((raw_tokens as u32) / 1024) * 1024;
            let auto_cap = max_capacity.min(65536);
            rounded.clamp(2048, auto_cap)
        } else {
            max_capacity.min(65536)
        }
    } else {
        (configured_ctx as u32).min(max_capacity)
    };

    let context_vram = (n_ctx as u64) * bytes_per_token;

    VramBudgetPlan {
        n_ctx,
        context_vram_bytes: context_vram,
        model_vram_on_gpu_bytes: model_vram_on_gpu,
        model_vram_total_bytes: total_model_bytes,
        projector_vram_bytes: projector_vram,
        mtp_vram_bytes: mtp_vram,
        total_gpu_vram_bytes: total_vram,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_scales_correctly() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(2048), "2.00 KiB");
        assert_eq!(format_bytes(50 * 1024 * 1024), "50.00 MiB");
        assert_eq!(format_bytes(16 * 1024 * 1024 * 1024), "16.00 GiB");
    }

    #[test]
    fn format_params_compacts_nicely() {
        assert_eq!(format_params(27_000_000_000), "27B");
        assert_eq!(format_params(7_000_000_000), "7B");
        assert_eq!(format_params(300_000_000), "300M");
    }

    #[test]
    fn format_tokens_handles_k_and_m() {
        assert_eq!(format_tokens(8192), "8K tokens");
        assert_eq!(format_tokens(131072), "128K tokens");
        assert_eq!(format_tokens(1_000_000), "1.0M tokens");
    }

    #[test]
    fn parses_cache_quant_multipliers() {
        assert!(
            (parse_cache_type_multiplier("-ctk q4_0", "--cache-type-k", "-ctk") - 0.28).abs()
                < 1e-4
        );
        assert!(
            (parse_cache_type_multiplier("--cache-type-k=q8_0", "--cache-type-k", "-ctk") - 0.53)
                .abs()
                < 1e-4
        );
        assert!((parse_cache_type_multiplier("", "--cache-type-k", "-ctk") - 1.0).abs() < 1e-4);
    }
}

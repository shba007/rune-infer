pub mod install;
pub mod tools;

use crate::inference::process::{ProcessGuard, configure_death_signal};
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ImageCaptionResponse, Usage};
use base64::Engine;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::io::BufRead;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeFlavor {
    Upstream,
    Prism,
}

#[derive(Default, Debug, Clone)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

pub struct LlamaServerEngine {
    id: String,
    port: u16,
    client: reqwest::blocking::Client,
    _process: Arc<Mutex<ProcessGuard>>,
}

impl LlamaServerEngine {
    pub fn new(
        id: String,
        model_path: impl AsRef<Path>,
        mmproj_path: Option<impl AsRef<Path>>,
        mtp_path: Option<impl AsRef<Path>>,
        n_ctx: Option<u32>,
        n_gpu_layers: Option<u32>,
        mtp_heads: Option<u32>,
        extra_args: Option<String>,
        flavor: RuntimeFlavor,
    ) -> Result<Self, Box<dyn Error>> {
        let bin_path = install::ensure_binary_installed(flavor)?;
        let port = Self::find_free_port()?;

        println!(
            "[LlamaServerEngine] Launching managed llama-server ({:?}) for '{}' on port {}...",
            flavor, id, port
        );

        let mut cmd = Command::new(&bin_path);
        configure_death_signal(&mut cmd);

        cmd.arg("-m")
            .arg(model_path.as_ref())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-c")
            .arg(n_ctx.unwrap_or(32768).to_string())
            .arg("-ngl")
            .arg(n_gpu_layers.unwrap_or(99).to_string())
            .arg("--jinja")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        if let Some(ref mmproj) = mmproj_path {
            let path = mmproj.as_ref();
            if path.exists() {
                cmd.arg("--mmproj").arg(path);
            }
        }

        if let Some(ref mtp) = mtp_path {
            let path = mtp.as_ref();
            if path.exists() {
                let heads = mtp_heads.unwrap_or(2).max(1);
                cmd.arg("--spec-draft-model").arg(path);
                cmd.arg("--spec-type").arg("draft-mtp");
                cmd.arg("--spec-draft-n-max").arg(heads.to_string());
            }
        } else if let Some(heads) = mtp_heads {
            if heads > 0 {
                cmd.arg("--spec-type").arg("draft-mtp");
                cmd.arg("--spec-draft-n-max").arg(heads.to_string());
            }
        }

        if let Some(extras) = extra_args {
            for arg in extras.split_whitespace() {
                cmd.arg(arg);
            }
        }

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to spawn llama-server binary at '{}': {e}",
                bin_path.display()
            )
        })?;

        let stderr_pipe = child.stderr.take();
        let last_stderr_lines = Arc::new(Mutex::new(Vec::new()));
        if let Some(pipe) = stderr_pipe {
            let lines_clone = last_stderr_lines.clone();
            let label = format!("llama-server-{}", id);
            std::thread::spawn(move || {
                let reader = std::io::BufReader::new(pipe);
                for line in reader.lines().map_while(Result::ok) {
                    tracing::info!(target: "llama_server", "[{}] {}", label, line);
                    let mut l = lines_clone.lock().unwrap();
                    if l.len() >= 50 {
                        l.remove(0);
                    }
                    l.push(line);
                }
            });
        }

        let guard = ProcessGuard::new(child, format!("llama-server-{}", id));
        Self::wait_for_server(port, &guard, &last_stderr_lines)?;

        let client = reqwest::blocking::Client::builder()
            .tcp_nodelay(true)
            .pool_max_idle_per_host(10)
            .timeout(Duration::from_secs(300))
            .build()?;

        Ok(Self {
            id,
            port,
            client,
            _process: guard,
        })
    }

    fn find_free_port() -> Result<u16, Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        Ok(listener.local_addr()?.port())
    }

    fn wait_for_server(
        port: u16,
        guard_mutex: &Arc<Mutex<ProcessGuard>>,
        stderr_lines: &Arc<Mutex<Vec<String>>>,
    ) -> Result<(), Box<dyn Error>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()?;
        let health_url = format!("http://127.0.0.1:{port}/health");
        let start = Instant::now();
        let timeout = Duration::from_secs(120);

        while start.elapsed() < timeout {
            if let Ok(mut guard) = guard_mutex.try_lock() {
                if let Ok(Some(status)) = guard.try_wait() {
                    let logs = stderr_lines.lock().unwrap().join("\n");
                    return Err(
                        format!("Server terminated prematurely with {status}: {logs}").into(),
                    );
                }
            }
            if let Ok(resp) = client.get(&health_url).send() {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        Err("Timed out waiting for managed llama-server".into())
    }
}

impl InferenceEngine for LlamaServerEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn execute(
        &self,
        task: &InferenceTaskRequest,
        mut on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        match task {
            InferenceTaskRequest::ToolCall {
                prompt,
                schema,
                images,
                messages,
                max_tokens,
                temperature,
                top_p,
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
                let stream_mode = on_token.is_some();
                let mut server_messages = Vec::new();

                if !messages.is_empty() {
                    for (m_idx, msg) in messages.iter().enumerate() {
                        let (text, _) = msg.split_text_and_media();
                        let has_images = !images.is_empty()
                            && msg.role == "user"
                            && m_idx == messages.iter().position(|m| m.role == "user").unwrap_or(0);

                        let mut msg_obj = serde_json::json!({ "role": msg.role });
                        if has_images {
                            let mut parts = Vec::new();
                            if !text.is_empty() {
                                parts.push(serde_json::json!({ "type": "text", "text": text }));
                            }
                            for frame in images.iter() {
                                let mime = if frame.starts_with(b"\x89PNG") {
                                    "image/png"
                                } else {
                                    "image/jpeg"
                                };
                                let b64 = base64::engine::general_purpose::STANDARD.encode(frame);
                                parts.push(serde_json::json!({
                                    "type": "image_url",
                                    "image_url": { "url": format!("data:{mime};base64,{b64}") }
                                }));
                            }
                            msg_obj["content"] = serde_json::Value::Array(parts);
                        } else if msg.role == "assistant" && msg.tool_calls.is_some() {
                            msg_obj["content"] = if text.is_empty() {
                                serde_json::Value::Null
                            } else {
                                serde_json::Value::String(text)
                            };
                            msg_obj["tool_calls"] = serde_json::json!(msg.tool_calls);
                        } else {
                            msg_obj["content"] = serde_json::Value::String(text);
                        }
                        server_messages.push(msg_obj);
                    }
                } else {
                    let clean_prompt = prompt
                        .replace("<|im_start|>", "")
                        .replace("<|im_end|>", "")
                        .trim()
                        .to_string();
                    server_messages
                        .push(serde_json::json!({ "role": "user", "content": clean_prompt }));
                }

                let mut body = serde_json::json!({
                    "messages": server_messages,
                    "temperature": temperature.unwrap_or(1.0),
                    "top_p": top_p.unwrap_or(0.95),
                    "stream": stream_mode,
                    "max_tokens": max_tokens.unwrap_or(16384)
                });

                if schema.is_array() && !schema.as_array().is_none_or(|a| a.is_empty()) {
                    body["tools"] = schema.clone();
                } else if schema.is_object() && !schema.as_object().is_none_or(|o| o.is_empty()) {
                    body["response_format"] = serde_json::json!({
                        "type": "json_schema",
                        "json_schema": { "schema": schema.clone() }
                    });
                }

                let resp = self.client.post(&endpoint).json(&body).send()?;
                if !resp.status().is_success() {
                    return Err(
                        format!("llama-server error: {}", resp.text().unwrap_or_default()).into(),
                    );
                }

                if stream_mode {
                    let mut full_output = String::new();
                    let mut in_reasoning = false;
                    let mut tool_accumulators: Vec<ToolCallAccumulator> = Vec::new();
                    let reader = std::io::BufReader::new(resp);

                    for line in reader.lines().map_while(Result::ok) {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data.trim() == "[DONE]" {
                                break;
                            }
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                                let delta = &v["choices"][0]["delta"];

                                // 1. Forward reasoning content if emitted by llama-server (e.g. Qwen thinking mode)
                                if let Some(reasoning) =
                                    delta.get("reasoning_content").and_then(|r| r.as_str())
                                {
                                    if !reasoning.is_empty() {
                                        if !in_reasoning {
                                            in_reasoning = true;
                                            full_output.push_str("<think>\n");
                                            if let Some(ref mut cb) = on_token {
                                                let _ = cb("<think>");
                                            }
                                        }
                                        full_output.push_str(reasoning);
                                        if let Some(ref mut cb) = on_token {
                                            if !cb(reasoning) {
                                                break;
                                            }
                                        }
                                    }
                                }

                                // 2. Forward regular content
                                if let Some(content) = delta.get("content").and_then(|c| c.as_str())
                                {
                                    if in_reasoning {
                                        in_reasoning = false;
                                        full_output.push_str("\n</think>\n");
                                        if let Some(ref mut cb) = on_token {
                                            let _ = cb("</think>");
                                        }
                                    }
                                    full_output.push_str(content);
                                    if let Some(ref mut cb) = on_token {
                                        if !cb(content) {
                                            break;
                                        }
                                    }
                                }

                                // 3. Correctly accumulate native tool-call deltas across streaming chunks
                                if let Some(tc_array) =
                                    delta.get("tool_calls").and_then(|t| t.as_array())
                                {
                                    for tc in tc_array {
                                        let idx =
                                            tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0)
                                                as usize;
                                        while tool_accumulators.len() <= idx {
                                            tool_accumulators.push(ToolCallAccumulator {
                                                id: String::new(),
                                                name: String::new(),
                                                arguments: String::new(),
                                            });
                                        }

                                        let target = &mut tool_accumulators[idx];
                                        if let Some(id) = tc.get("id").and_then(|s| s.as_str()) {
                                            if !id.is_empty() {
                                                target.id = id.to_string();
                                            }
                                        }
                                        if let Some(func) = tc.get("function") {
                                            if let Some(name) =
                                                func.get("name").and_then(|s| s.as_str())
                                            {
                                                target.name.push_str(name);
                                            }
                                            if let Some(args) =
                                                func.get("arguments").and_then(|s| s.as_str())
                                            {
                                                target.arguments.push_str(args);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if in_reasoning {
                        full_output.push_str("\n</think>\n");
                        if let Some(ref mut cb) = on_token {
                            let _ = cb("</think>");
                        }
                    }

                    // If native tool calls were accumulated across the stream, return them
                    let valid_accumulated: Vec<serde_json::Value> = tool_accumulators
                        .into_iter()
                        .filter(|t| !t.name.trim().is_empty())
                        .map(|t| {
                            let args = if t.arguments.trim().is_empty() {
                                "{}".to_string()
                            } else {
                                t.arguments
                            };
                            serde_json::json!({
                                "id": if t.id.is_empty() { format!("call_{}", std::process::id()) } else { t.id },
                                "type": "function",
                                "function": {
                                    "name": t.name,
                                    "arguments": args
                                }
                            })
                        })
                        .collect();

                    if !valid_accumulated.is_empty() {
                        let tokens = (full_output.len() / 4).max(1) as u32;
                        return Ok(InferenceOutput {
                            response: InferenceTaskResponse::ToolCall {
                                content: if full_output.trim().is_empty() {
                                    None
                                } else {
                                    Some(full_output.trim().to_string())
                                },
                                tool_calls: serde_json::Value::Array(valid_accumulated),
                            },
                            usage: Usage::new(tokens, tokens),
                        });
                    }

                    // Otherwise, check if full_output contains inline tool calls (e.g. XML format)
                    if let Some((clean_content, parsed_tools)) =
                        tools::extract_tool_calls(&full_output)
                    {
                        let tokens = (full_output.len() / 4).max(1) as u32;
                        return Ok(InferenceOutput {
                            response: InferenceTaskResponse::ToolCall {
                                content: clean_content,
                                tool_calls: parsed_tools,
                            },
                            usage: Usage::new(tokens, tokens),
                        });
                    }

                    let tokens = (full_output.len() / 4).max(1) as u32;
                    return Ok(InferenceOutput {
                        response: InferenceTaskResponse::Text(full_output),
                        usage: Usage::new(tokens, tokens),
                    });
                }

                let result: serde_json::Value = resp.json()?;
                let choice = &result["choices"][0];

                if let Some(tc) = choice["message"]
                    .get("tool_calls")
                    .filter(|v| v.is_array() && !v.as_array().unwrap().is_empty())
                {
                    let content_str = choice["message"]["content"].as_str().map(|s| s.to_string());
                    return Ok(InferenceOutput {
                        response: InferenceTaskResponse::ToolCall {
                            content: content_str,
                            tool_calls: tc.clone(),
                        },
                        usage: Usage::new(10, 10),
                    });
                }

                let reasoning = choice["message"]
                    .get("reasoning_content")
                    .and_then(|r| r.as_str())
                    .unwrap_or("")
                    .trim();

                let raw_content = choice["message"]["content"].as_str().unwrap_or("").trim();

                let combined_content = if !reasoning.is_empty() && !raw_content.is_empty() {
                    format!("<think>\n{}\n</think>\n{}", reasoning, raw_content)
                } else if !reasoning.is_empty() {
                    format!("<think>\n{}\n</think>", reasoning)
                } else {
                    raw_content.to_string()
                };

                if let Some((clean_content, parsed_tools)) =
                    tools::extract_tool_calls(&combined_content)
                {
                    return Ok(InferenceOutput {
                        response: InferenceTaskResponse::ToolCall {
                            content: clean_content,
                            tool_calls: parsed_tools,
                        },
                        usage: Usage::new(10, 10),
                    });
                }

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Text(combined_content),
                    usage: Usage::new(10, 10),
                })
            }
            InferenceTaskRequest::ImageCaption {
                image_bytes,
                detail,
                max_tokens,
                ..
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
                let mime = if image_bytes.starts_with(b"\x89PNG") {
                    "image/png"
                } else {
                    "image/jpeg"
                };
                let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
                let prompt_text = if detail == "detailed" {
                    "Provide a detailed description of this image."
                } else {
                    "Provide a concise, single-sentence caption for this image."
                };

                let body = serde_json::json!({
                    "messages": [{
                        "role": "user",
                        "content": [
                            { "type": "text", "text": prompt_text },
                            { "type": "image_url", "image_url": { "url": format!("data:{mime};base64,{b64}") } }
                        ]
                    }],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "stream": false
                });

                let resp = self.client.post(&endpoint).json(&body).send()?;
                let result: serde_json::Value = resp.json()?;
                let caption = result["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Caption(ImageCaptionResponse {
                        id: format!(
                            "cap_{}",
                            hex::encode(&Sha256::digest(caption.as_bytes())[..8])
                        ),
                        object: "image.caption".to_string(),
                        created,
                        model: self.id.clone(),
                        caption,
                        usage: Usage::new(85, 12),
                    }),
                    usage: Usage::new(85, 12),
                })
            }
            _ => Err("LlamaServerEngine does not support this task type".into()),
        }
    }
}

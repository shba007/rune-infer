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
    client: crate::inference::process::SafeBlockingClient,
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

        tracing::info!(
            "[LlamaServerEngine] Launching managed llama-server ({:?}) for '{}' on port {}...",
            flavor,
            id,
            port
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
            .arg("--jinja");

        let extras_str = extra_args.as_deref().unwrap_or("");
        if !extras_str.contains("-fa")
            && !extras_str.contains("--flash-attn")
            && n_gpu_layers.unwrap_or(99) > 0
        {
            cmd.arg("-fa").arg("on");
        }
        if !extras_str.contains("-np") && !extras_str.contains("--parallel") {
            cmd.arg("--parallel").arg("1");
        }

        cmd.stdout(Stdio::null()).stderr(Stdio::piped());

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
            client: crate::inference::process::SafeBlockingClient::new(client),
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
            if crate::inference::process::http_probe_success(port, "/health") {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        Err("Timed out waiting for managed llama-server".into())
    }

    pub fn release_slots_tcp(port: u16) {
        if let Some(slot_ids) = Self::query_slot_ids_tcp(port) {
            for id in slot_ids {
                Self::send_slot_release_tcp(port, id);
            }
        } else {
            Self::send_slot_release_tcp(port, 0);
        }
    }

    fn query_slot_ids_tcp(port: u16) -> Option<Vec<u64>> {
        let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().ok()?;
        let mut stream =
            std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(200)).ok()?;
        let _ = stream.set_read_timeout(Some(Duration::from_millis(300)));
        let _ = stream.set_write_timeout(Some(Duration::from_millis(200)));

        use std::io::{Read, Write};
        let req =
            format!("GET /slots HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
        stream.write_all(req.as_bytes()).ok()?;
        let _ = stream.flush();

        let mut response = Vec::new();
        stream.read_to_end(&mut response).ok()?;

        let body_offset = response.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
        let body = &response[body_offset..];
        let slots: serde_json::Value = serde_json::from_slice(body).ok()?;
        let arr = slots.as_array()?;
        let mut ids = Vec::new();
        for s in arr {
            if let Some(id) = s.get("id").and_then(|v| v.as_u64()) {
                ids.push(id);
            }
        }
        if ids.is_empty() { None } else { Some(ids) }
    }

    fn send_slot_release_tcp(port: u16, slot_id: u64) {
        let Ok(addr) = format!("127.0.0.1:{port}").parse::<std::net::SocketAddr>() else {
            return;
        };
        let Ok(mut stream) =
            std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(200))
        else {
            return;
        };
        let _ = stream.set_write_timeout(Some(Duration::from_millis(200)));
        let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));

        use std::io::Write;
        let req = format!(
            "POST /slots/{slot_id}?action=release HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
        );
        let _ = stream.write_all(req.as_bytes());
        let _ = stream.flush();
    }
}

impl InferenceEngine for LlamaServerEngine {
    fn supports_cancellation(&self) -> bool {
        true
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn cancel(&self) {
        let port = self.port;
        if tokio::runtime::Handle::try_current().is_ok() {
            drop(tokio::task::spawn_blocking(move || {
                Self::release_slots_tcp(port);
            }));
        } else {
            Self::release_slots_tcp(port);
        }
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
                tool_choice,
                images,
                audios,
                messages,
                max_tokens,
                temperature,
                top_p,
                top_k,
                min_p,
                reasoning_effort,
                thinking,
                chat_template_kwargs,
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
                let stream_mode = on_token.is_some();
                let mut server_messages = Vec::new();

                if !messages.is_empty() {
                    for (m_idx, msg) in messages.iter().enumerate() {
                        let (text, media_items) = msg.split_text_and_media();
                        let has_images = !images.is_empty()
                            && msg.role == "user"
                            && m_idx == messages.iter().position(|m| m.role == "user").unwrap_or(0);
                        let has_audios = !audios.is_empty()
                            && msg.role == "user"
                            && m_idx == messages.iter().position(|m| m.role == "user").unwrap_or(0);
                        let has_media = !media_items.is_empty() || has_images || has_audios;

                        let mut msg_obj = serde_json::json!({ "role": msg.role });
                        if has_media && msg.role == "user" {
                            let mut parts = Vec::new();
                            if !text.is_empty() {
                                parts.push(serde_json::json!({ "type": "text", "text": text }));
                            }

                            if !media_items.is_empty() {
                                for item in media_items {
                                    match item {
                                        crate::types::MediaItem::Image(url) => {
                                            parts.push(serde_json::json!({
                                                "type": "image_url",
                                                "image_url": { "url": url }
                                            }));
                                        }
                                        crate::types::MediaItem::Video(url) => {
                                            parts.push(serde_json::json!({
                                                "type": "video_url",
                                                "video_url": { "url": url }
                                            }));
                                        }
                                        crate::types::MediaItem::Audio { data, format } => {
                                            parts.push(serde_json::json!({
                                                "type": "input_audio",
                                                "input_audio": {
                                                    "data": data,
                                                    "format": format
                                                }
                                            }));
                                        }
                                    }
                                }
                            } else {
                                for frame in images.iter() {
                                    let mime = if frame.starts_with(b"\x89PNG") {
                                        "image/png"
                                    } else {
                                        "image/jpeg"
                                    };
                                    let b64 =
                                        base64::engine::general_purpose::STANDARD.encode(frame);
                                    parts.push(serde_json::json!({
                                        "type": "image_url",
                                        "image_url": { "url": format!("data:{mime};base64,{b64}") }
                                    }));
                                }
                                for audio_bytes in audios.iter() {
                                    let b64 = base64::engine::general_purpose::STANDARD
                                        .encode(audio_bytes);
                                    let fmt = if audio_bytes.starts_with(b"ID3")
                                        || (audio_bytes.len() > 1
                                            && audio_bytes[0] == 0xFF
                                            && (audio_bytes[1] & 0xE0) == 0xE0)
                                    {
                                        "mp3"
                                    } else {
                                        "wav"
                                    };
                                    parts.push(serde_json::json!({
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": b64,
                                            "format": fmt
                                        }
                                    }));
                                }
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
                if let Some(k) = top_k {
                    body["top_k"] = serde_json::json!(k);
                }
                if let Some(p) = min_p {
                    body["min_p"] = serde_json::json!(p);
                }

                // Handle thinking / reasoning controls for llama-server
                let is_disabled = reasoning_effort
                    .as_deref()
                    .is_some_and(|e| e == "none" || e == "disabled" || e == "off")
                    || thinking.as_ref().is_some_and(|t| {
                        t.as_bool() == Some(false)
                            || t.get("type").and_then(|v| v.as_str()) == Some("disabled")
                    })
                    || chat_template_kwargs.as_ref().is_some_and(|c| {
                        c.get("thinking") == Some(&serde_json::Value::Bool(false))
                    });

                let mut kwargs = chat_template_kwargs
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({}));
                if !kwargs.is_object() {
                    kwargs = serde_json::json!({});
                }

                if is_disabled {
                    kwargs["thinking"] = serde_json::Value::Bool(false);
                    body["chat_template_kwargs"] = kwargs;
                    body["reasoning_effort"] = serde_json::json!("none");
                } else {
                    if let Some(effort) = reasoning_effort {
                        body["reasoning_effort"] = serde_json::json!(effort);
                    }
                    if let Some(t) = thinking {
                        if t.as_bool() == Some(true)
                            || t.get("type").and_then(|v| v.as_str()) == Some("enabled")
                        {
                            kwargs["thinking"] = serde_json::Value::Bool(true);
                            body["chat_template_kwargs"] = kwargs;
                        }
                    } else if !kwargs.as_object().unwrap().is_empty() {
                        body["chat_template_kwargs"] = kwargs;
                    }
                }

                if schema.is_array() && !schema.as_array().is_none_or(|a| a.is_empty()) {
                    body["tools"] = schema.clone();
                    if let Some(tc) = tool_choice {
                        body["tool_choice"] = tc.clone();
                    }
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
                    let mut stream_usage: Option<Usage> = None;
                    let reader = std::io::BufReader::new(resp);

                    for line in reader.lines().map_while(Result::ok) {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data.trim() == "[DONE]" {
                                break;
                            }
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                                if let Some(u) = v.get("usage") {
                                    if let (Some(pt), Some(ct)) = (
                                        u.get("prompt_tokens").and_then(|t| t.as_u64()),
                                        u.get("completion_tokens").and_then(|t| t.as_u64()),
                                    ) {
                                        stream_usage = Some(Usage::new(pt as u32, ct as u32));
                                    }
                                }

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
                                                self.cancel();
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
                                            self.cancel();
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

                    let (prompt_tokens, completion_tokens) = if let Some(u) = stream_usage {
                        (u.prompt_tokens, u.completion_tokens)
                    } else {
                        let img_tok = (images.len() * 576) as u32;
                        let aud_tok = (audios.len() * 256) as u32;
                        let txt_tok = (prompt.len() / 4).max(1) as u32;
                        let pt = txt_tok + img_tok + aud_tok;
                        let ct = (full_output.len() / 4).max(1) as u32;
                        (pt, ct)
                    };
                    let mut final_usage = Usage::new(prompt_tokens, completion_tokens);
                    if !images.is_empty() || !audios.is_empty() {
                        final_usage.prompt_tokens_details =
                            Some(crate::types::PromptTokensDetails {
                                image_tokens: (images.len() * 576) as u32,
                                audio_tokens: (audios.len() * 256) as u32,
                                cached_tokens: 0,
                            });
                    }

                    if !valid_accumulated.is_empty() {
                        return Ok(InferenceOutput {
                            response: InferenceTaskResponse::ToolCall {
                                content: if full_output.trim().is_empty() {
                                    None
                                } else {
                                    Some(full_output.trim().to_string())
                                },
                                tool_calls: serde_json::Value::Array(valid_accumulated),
                            },
                            usage: final_usage,
                        });
                    }

                    // Otherwise, check if full_output contains inline tool calls (e.g. XML format)
                    if let Some((clean_content, parsed_tools)) =
                        tools::extract_tool_calls(&full_output)
                    {
                        return Ok(InferenceOutput {
                            response: InferenceTaskResponse::ToolCall {
                                content: clean_content,
                                tool_calls: parsed_tools,
                            },
                            usage: final_usage,
                        });
                    }

                    return Ok(InferenceOutput {
                        response: InferenceTaskResponse::Text(full_output),
                        usage: final_usage,
                    });
                }

                let result: serde_json::Value = resp.json()?;
                let choice = &result["choices"][0];

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

                let (prompt_tokens, completion_tokens) = if let Some(u) = result.get("usage") {
                    let pt = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    let ct = u
                        .get("completion_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    (pt, ct)
                } else {
                    let img_tok = (images.len() * 576) as u32;
                    let aud_tok = (audios.len() * 256) as u32;
                    let txt_tok = (prompt.len() / 4).max(1) as u32;
                    let pt = txt_tok + img_tok + aud_tok;
                    let ct = (combined_content.len() / 4).max(1) as u32;
                    (pt, ct)
                };
                let mut non_stream_usage = Usage::new(prompt_tokens, completion_tokens);
                if !images.is_empty() || !audios.is_empty() {
                    non_stream_usage.prompt_tokens_details =
                        Some(crate::types::PromptTokensDetails {
                            image_tokens: (images.len() * 576) as u32,
                            audio_tokens: (audios.len() * 256) as u32,
                            cached_tokens: 0,
                        });
                }

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
                        usage: non_stream_usage,
                    });
                }

                if let Some((clean_content, parsed_tools)) =
                    tools::extract_tool_calls(&combined_content)
                {
                    return Ok(InferenceOutput {
                        response: InferenceTaskResponse::ToolCall {
                            content: clean_content,
                            tool_calls: parsed_tools,
                        },
                        usage: non_stream_usage,
                    });
                }

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Text(combined_content),
                    usage: non_stream_usage,
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
            InferenceTaskRequest::Embedding {
                model,
                input,
                dimensions,
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/embeddings", self.port);
                let mut body = serde_json::json!({
                    "model": model.clone().unwrap_or_else(|| self.id.clone()),
                    "input": input,
                });
                if let Some(dim) = dimensions {
                    body["dimensions"] = serde_json::json!(dim);
                }
                let resp = self.client.post(&endpoint).json(&body).send()?;
                if !resp.status().is_success() {
                    let err_text = resp.text().unwrap_or_default();
                    return Err(format!("llama-server embedding request failed: {err_text}").into());
                }
                let mut result: crate::types::EmbeddingResponse = resp.json()?;
                if let Some(dim) = dimensions {
                    for item in &mut result.data {
                        if item.embedding.len() > *dim {
                            item.embedding.truncate(*dim);
                        }
                    }
                }
                let usage = result.usage.clone();
                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Embedding(result),
                    usage,
                })
            }
            InferenceTaskRequest::ImageGrounding {
                model,
                image_bytes,
                query,
                width,
                height,
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
                let mime = if image_bytes.starts_with(b"\x89PNG") {
                    "image/png"
                } else {
                    "image/jpeg"
                };
                let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
                let prompt_text = format!(
                    "Locate the UI element or target '{}' in this image. Output ONLY the bounding box in normalized [0, 1000] coordinates format: [ymin, xmin, ymax, xmax]. Do not include any other text.",
                    query
                );

                let body = serde_json::json!({
                    "messages": [{
                        "role": "user",
                        "content": [
                            { "type": "text", "text": prompt_text },
                            { "type": "image_url", "image_url": { "url": format!("data:{mime};base64,{b64}") } }
                        ]
                    }],
                    "max_tokens": 64,
                    "temperature": 0.0,
                    "stream": false
                });

                let resp = self.client.post(&endpoint).json(&body).send()?;
                let result: serde_json::Value = resp.json()?;
                let raw_content = result["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .trim();

                let (box_2d, conf) = parse_grounding_coordinates(raw_content)?;
                let center_x = (box_2d[1] + box_2d[3]) / 2;
                let center_y = (box_2d[0] + box_2d[2]) / 2;
                let px_x = ((center_x as f64 / 1000.0) * *width as f64).round() as u32;
                let px_y = ((center_y as f64 / 1000.0) * *height as f64).round() as u32;
                let box_normalized = [
                    box_2d[0] as f64 / 1000.0,
                    box_2d[1] as f64 / 1000.0,
                    box_2d[2] as f64 / 1000.0,
                    box_2d[3] as f64 / 1000.0,
                ];

                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Grounding(
                        crate::types::ImageGroundingResponse {
                            id: format!(
                                "ground_{}",
                                hex::encode(&Sha256::digest(image_bytes)[..8])
                            ),
                            object: "image.grounding".to_string(),
                            created,
                            model: model.clone().unwrap_or_else(|| self.id.clone()),
                            query: query.clone(),
                            point: [center_x, center_y],
                            point_pixels: [px_x, px_y],
                            box_2d,
                            box_normalized,
                            image_width: *width,
                            image_height: *height,
                            confidence: conf,
                        },
                    ),
                    usage: Usage::new(128, 16),
                })
            }
            InferenceTaskRequest::Decision {
                model,
                text,
                images,
                questions,
            } => {
                let systemone_endpoint = format!("http://127.0.0.1:{}/v1/systemone", self.port);

                let mut b64_images = Vec::new();
                for frame in images {
                    let mime = if frame.starts_with(b"\x89PNG") {
                        "image/png"
                    } else {
                        "image/jpeg"
                    };
                    let b64 = base64::engine::general_purpose::STANDARD.encode(frame);
                    b64_images.push(format!("data:{mime};base64,{b64}"));
                }

                let mut questions_map = serde_json::Map::new();
                for q in questions {
                    let mut q_obj = serde_json::Map::new();
                    q_obj.insert(
                        "type".to_string(),
                        serde_json::Value::String(q.r#type.clone()),
                    );
                    q_obj.insert(
                        "instructions".to_string(),
                        serde_json::Value::String(q.instruction.clone()),
                    );

                    if let Some(ref crit) = q.criteria {
                        let formatted_crit = if q.r#type == "choice" && crit.is_array() {
                            let mut crit_obj = serde_json::Map::new();
                            if let Some(arr) = crit.as_array() {
                                for item in arr {
                                    if let Some(s) = item.as_str() {
                                        crit_obj.insert(
                                            s.to_string(),
                                            serde_json::Value::String(s.to_string()),
                                        );
                                    }
                                }
                            }
                            serde_json::Value::Object(crit_obj)
                        } else {
                            crit.clone()
                        };
                        q_obj.insert("criteria".to_string(), formatted_crit);
                    }
                    questions_map.insert(q.name.clone(), serde_json::Value::Object(q_obj));
                }

                let state_text = if text.trim().is_empty() {
                    "Evaluate input state and visual frames."
                } else {
                    text.as_str()
                };

                let systemone_body = serde_json::json!({
                    "state": state_text,
                    "images": b64_images,
                    "questions": questions_map
                });

                let systemone_res = self
                    .client
                    .post(&systemone_endpoint)
                    .json(&systemone_body)
                    .send();

                if let Ok(resp) = systemone_res {
                    if resp.status().is_success() {
                        if let Ok(result) = resp.json::<serde_json::Value>() {
                            let raw_answers = result.get("answers").and_then(|v| v.as_object());
                            let mut answers = std::collections::HashMap::new();

                            for q in questions {
                                let ans_val = raw_answers
                                    .and_then(|a| a.get(&q.name))
                                    .unwrap_or(&serde_json::Value::Null);

                                match q.r#type.as_str() {
                                    "noul" => {
                                        let noul_prob = ans_val
                                            .get("noul")
                                            .and_then(|v| v.as_f64())
                                            .or_else(|| ans_val.as_f64())
                                            .unwrap_or(0.5);
                                        let decision = noul_prob >= 0.5;
                                        answers.insert(
                                            q.name.clone(),
                                            crate::types::DecisionAnswer::Noul {
                                                decision,
                                                probability: (noul_prob * 10000.0).round()
                                                    / 10000.0,
                                            },
                                        );
                                    }
                                    "choice" => {
                                        let choice_str = ans_val
                                            .get("choice")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("unknown")
                                            .to_string();
                                        let confidence = ans_val
                                            .get("confidence")
                                            .and_then(|v| v.as_f64())
                                            .unwrap_or(0.9);
                                        let mut probs = std::collections::HashMap::new();
                                        if let Some(raw_probs) =
                                            ans_val.get("probabilities").and_then(|v| v.as_object())
                                        {
                                            for (k, v) in raw_probs {
                                                if let Some(p) = v.as_f64() {
                                                    probs.insert(
                                                        k.clone(),
                                                        (p * 10000.0).round() / 10000.0,
                                                    );
                                                }
                                            }
                                        }
                                        if probs.is_empty() {
                                            probs.insert(choice_str.clone(), confidence);
                                        }
                                        answers.insert(
                                            q.name.clone(),
                                            crate::types::DecisionAnswer::Choice {
                                                decision: choice_str,
                                                confidence: (confidence * 10000.0).round()
                                                    / 10000.0,
                                                probabilities: probs,
                                            },
                                        );
                                    }
                                    "score" => {
                                        let score = ans_val
                                            .get("score")
                                            .and_then(|v| v.as_f64())
                                            .unwrap_or(0.0);
                                        let mut legend_map = std::collections::HashMap::new();
                                        if let Some(raw_leg) =
                                            ans_val.get("legend").and_then(|v| v.as_object())
                                        {
                                            for (k, v) in raw_leg {
                                                if let Some(s) = v.as_str() {
                                                    legend_map.insert(k.clone(), s.to_string());
                                                }
                                            }
                                        }
                                        let nearest_idx = score.round() as i64;
                                        let level = legend_map
                                            .get(&nearest_idx.to_string())
                                            .cloned()
                                            .or_else(|| {
                                                ans_val
                                                    .get("level")
                                                    .and_then(|v| v.as_str())
                                                    .map(String::from)
                                            })
                                            .unwrap_or_else(|| format!("Level {nearest_idx}"));

                                        let mut probs = std::collections::HashMap::new();
                                        if let Some(raw_probs) =
                                            ans_val.get("probabilities").and_then(|v| v.as_object())
                                        {
                                            for (k, v) in raw_probs {
                                                let label = legend_map
                                                    .get(k)
                                                    .cloned()
                                                    .unwrap_or_else(|| k.clone());
                                                if let Some(p) = v.as_f64() {
                                                    probs.insert(
                                                        label,
                                                        (p * 10000.0).round() / 10000.0,
                                                    );
                                                }
                                            }
                                        }
                                        answers.insert(
                                            q.name.clone(),
                                            crate::types::DecisionAnswer::Score {
                                                level,
                                                score: (score * 10000.0).round() / 10000.0,
                                                probabilities: probs,
                                            },
                                        );
                                    }
                                    _ => {}
                                }
                            }

                            let input_tokens = result
                                .get("usage")
                                .and_then(|u| {
                                    u.get("input_tokens").or_else(|| u.get("prompt_tokens"))
                                })
                                .and_then(|v| v.as_u64())
                                .unwrap_or(128)
                                as u32;

                            let created = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();

                            return Ok(InferenceOutput {
                                response: InferenceTaskResponse::Decision(
                                    crate::types::DecisionResponse {
                                        id: format!(
                                            "dec_{}",
                                            hex::encode(&Sha256::digest(text.as_bytes())[..8])
                                        ),
                                        object: "decision".to_string(),
                                        created,
                                        model: model.clone().unwrap_or_else(|| self.id.clone()),
                                        answers,
                                        usage: Usage::new(input_tokens, 0),
                                    },
                                ),
                                usage: Usage::new(input_tokens, 0),
                            });
                        }
                    } else {
                        let status = resp.status();
                        let err_text = resp.text().unwrap_or_default();
                        tracing::warn!(target: "llama_server", "Systemone endpoint returned {status}: {err_text}, falling back to chat/completions");
                    }
                }

                // Fallback to /v1/chat/completions
                let endpoint = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
                let mut content_parts = Vec::new();

                let mut q_prompt = String::new();
                q_prompt.push_str("State to evaluate:\n");
                q_prompt.push_str(text);
                for _ in images {
                    q_prompt.push_str("\n<image>");
                }
                q_prompt.push_str("\n\nAnswer each of the following decision questions. Output ONLY a valid JSON object where keys are the question names:\n");

                for q in questions {
                    q_prompt.push_str(&format!(
                        "- Question '{}' (type: '{}'): {}\n",
                        q.name, q.r#type, q.instruction
                    ));
                    if let Some(ref crit) = q.criteria {
                        q_prompt.push_str(&format!("  Criteria: {}\n", crit));
                    }
                }

                content_parts.push(serde_json::json!({
                    "type": "text",
                    "text": q_prompt
                }));

                for frame in images {
                    let mime = if frame.starts_with(b"\x89PNG") {
                        "image/png"
                    } else {
                        "image/jpeg"
                    };
                    let b64 = base64::engine::general_purpose::STANDARD.encode(frame);
                    content_parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": { "url": format!("data:{mime};base64,{b64}") }
                    }));
                }

                let body = serde_json::json!({
                    "messages": [{
                        "role": "user",
                        "content": content_parts
                    }],
                    "response_format": { "type": "json_object" },
                    "chat_template_kwargs": { "thinking": false },
                    "reasoning_effort": "none",
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "stream": false
                });

                let resp = self.client.post(&endpoint).json(&body).send()?;
                let result: serde_json::Value = resp.json()?;
                let raw_content = result["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("{}")
                    .trim();

                let parsed_json: serde_json::Value =
                    serde_json::from_str(raw_content).unwrap_or_else(|_| serde_json::json!({}));

                let mut answers = std::collections::HashMap::new();
                for q in questions {
                    let ans_val = parsed_json.get(&q.name).unwrap_or(&serde_json::Value::Null);
                    match q.r#type.as_str() {
                        "noul" => {
                            let dec = if let Some(b) = ans_val.as_bool() {
                                b
                            } else if let Some(b) =
                                ans_val.get("decision").and_then(|v| v.as_bool())
                            {
                                b
                            } else if let Some(s) = ans_val.as_str() {
                                s.eq_ignore_ascii_case("yes") || s.eq_ignore_ascii_case("true")
                            } else {
                                true
                            };
                            let prob = ans_val
                                .get("probability")
                                .and_then(|p| p.as_f64())
                                .unwrap_or(if dec { 0.95 } else { 0.05 });
                            answers.insert(
                                q.name.clone(),
                                crate::types::DecisionAnswer::Noul {
                                    decision: dec,
                                    probability: prob,
                                },
                            );
                        }
                        "choice" => {
                            let mut options = Vec::new();
                            if let Some(ref crit) = q.criteria {
                                if let Some(map) = crit.as_object() {
                                    for k in map.keys() {
                                        options.push(k.clone());
                                    }
                                } else if let Some(arr) = crit.as_array() {
                                    for item in arr {
                                        if let Some(s) = item.as_str() {
                                            options.push(s.to_string());
                                        }
                                    }
                                }
                            }
                            if options.is_empty() {
                                options.push("default".to_string());
                            }

                            let choice_str = if let Some(s) = ans_val.as_str() {
                                s.to_string()
                            } else if let Some(s) = ans_val.get("decision").and_then(|v| v.as_str())
                            {
                                s.to_string()
                            } else {
                                options[0].clone()
                            };

                            let mut probs = std::collections::HashMap::new();
                            let opt_count = options.len() as f64;
                            for opt in &options {
                                if opt == &choice_str {
                                    probs.insert(opt.clone(), 0.90);
                                } else {
                                    probs.insert(
                                        opt.clone(),
                                        ((0.10 / (opt_count - 1.0).max(1.0)) * 100.0).round()
                                            / 100.0,
                                    );
                                }
                            }

                            answers.insert(
                                q.name.clone(),
                                crate::types::DecisionAnswer::Choice {
                                    decision: choice_str,
                                    confidence: 0.90,
                                    probabilities: probs,
                                },
                            );
                        }
                        "score" => {
                            let mut levels = vec![
                                "Routine".to_string(),
                                "Soon".to_string(),
                                "Immediate".to_string(),
                            ];
                            if let Some(ref crit) = q.criteria {
                                if let Some(arr) = crit.get("levels").and_then(|v| v.as_array()) {
                                    levels = arr
                                        .iter()
                                        .filter_map(|v| v.as_str().map(String::from))
                                        .collect();
                                }
                            }

                            let lvl_str = if let Some(s) = ans_val.as_str() {
                                s.to_string()
                            } else if let Some(s) = ans_val.get("level").and_then(|v| v.as_str()) {
                                s.to_string()
                            } else {
                                levels.last().cloned().unwrap_or_else(|| "Soon".to_string())
                            };

                            let score = ans_val
                                .get("score")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.85);

                            let mut probs = std::collections::HashMap::new();
                            for (i, lvl) in levels.iter().enumerate() {
                                let score_p = (i as f64 + 1.0) / levels.len() as f64;
                                probs.insert(lvl.clone(), (score_p * 100.0).round() / 100.0);
                            }

                            answers.insert(
                                q.name.clone(),
                                crate::types::DecisionAnswer::Score {
                                    level: lvl_str,
                                    score,
                                    probabilities: probs,
                                },
                            );
                        }
                        _ => {}
                    }
                }

                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                let prompt_tokens = result
                    .get("usage")
                    .and_then(|u| u.get("prompt_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or_else(|| ((text.len() / 4).max(1) + images.len() * 576) as u64)
                    as u32;

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Decision(crate::types::DecisionResponse {
                        id: format!("dec_{}", hex::encode(&Sha256::digest(text.as_bytes())[..8])),
                        object: "decision".to_string(),
                        created,
                        model: model.clone().unwrap_or_else(|| self.id.clone()),
                        answers,
                        usage: Usage::new(prompt_tokens, 0),
                    }),
                    usage: Usage::new(prompt_tokens, 0),
                })
            }
            _ => Err("LlamaServerEngine does not support this task type".into()),
        }
    }
}

fn parse_grounding_coordinates(raw: &str) -> Result<([u32; 4], f64), Box<dyn std::error::Error>> {
    let inside =
        if let (Some(start), Some(end)) = (raw.find("<|box_start|>"), raw.find("<|box_end|>")) {
            raw.get(start + "<|box_start|>".len()..end)
        } else if let (Some(start), Some(end)) = (raw.find('['), raw.find(']')) {
            raw.get(start + 1..end)
        } else {
            None
        }
        .ok_or("Grounding output has no bounding box")?;
    let values: Vec<u32> = inside
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<Result<_, _>>()?;
    let bounds: [u32; 4] = values
        .try_into()
        .map_err(|_| "Grounding output must contain exactly four coordinates")?;
    if bounds.iter().any(|v| *v > 1000) || bounds[0] >= bounds[2] || bounds[1] >= bounds[3] {
        return Err("Grounding box must be ordered, nonempty and within 0..1000".into());
    }
    // No calibrated confidence is available from generated coordinates.
    Ok((bounds, 0.0))
}

#[cfg(test)]
mod grounding_tests {
    use super::parse_grounding_coordinates;

    #[test]
    fn parses_off_center_box_and_rejects_fabricated_fallbacks() {
        let (bounds, confidence) = parse_grounding_coordinates("[100,800,200,950]").unwrap();
        assert_eq!(bounds, [100, 800, 200, 950]);
        assert_eq!(
            [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
            [875, 150]
        );
        assert_eq!(confidence, 0.0);
        for invalid in [
            "no target",
            "[500,500]",
            "[0,0,1001,900]",
            "[200,800,100,950]",
            "[1,2,3,4,5]",
            "[4,3,2,1]",
            "<|box_end|><|box_start|>",
        ] {
            assert!(parse_grounding_coordinates(invalid).is_err(), "{invalid}");
        }
    }
}

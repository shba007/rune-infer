use std::io::{self, Write};

use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::{GenerationConfig, LlmModelType};
use crane::prelude::*;

const MODEL_PATH: &str = "D:/Models/huggingface/hub/models--mradermacher--OxCoder-9B-i1-GGUF/snapshots/e3335e20fc13b04618c821f9b99d1f8ef5cd103d/OxCoder-9B.i1-Q6_K.gguf";

/// Handles real-time token streaming and differentiates reasoning from response.
struct ThinkingStreamer {
    in_thinking: bool,
    buffer: String,
    header_printed: bool,
}

impl ThinkingStreamer {
    fn new() -> Self {
        Self {
            in_thinking: true,
            buffer: String::new(),
            header_printed: false,
        }
    }

    fn on_chunk(&mut self, chunk: &str) {
        if !self.in_thinking {
            // Already past the thinking block: stream directly to stdout
            print!("{}", chunk);
            io::stdout().flush().ok();
            return;
        }

        // Print thinking banner once at the start of generation
        if !self.header_printed {
            print!("\x1b[1;36m🧠 Thinking Process:\x1b[0m\n\x1b[90m");
            io::stdout().flush().ok();
            self.header_printed = true;
        }

        // Strip leading <think> tag if emitted as a token
        let chunk = chunk.strip_prefix("<think>").unwrap_or(chunk);
        self.buffer.push_str(chunk);

        if let Some(idx) = self.buffer.find("</think>") {
            // Output remaining thinking text before the tag
            let (thinking, rest) = self.buffer.split_at(idx);
            print!("{}", thinking);

            // Reset color, print response header, and output remainder
            print!("\x1b[0m\n\n\x1b[1;32m💬 OxCoder:\x1b[0m\n");
            let after_think = &rest["</think>".len()..];
            print!("{}", after_think.trim_start());
            io::stdout().flush().ok();

            self.in_thinking = false;
            self.buffer.clear();
        } else {
            // Keep up to 8 bytes in buffer in case "</think>" is split across chunk boundaries
            const HOLD: usize = 8;
            if self.buffer.len() > HOLD {
                let print_len = self.buffer.len() - HOLD;
                let safe_len = self
                    .buffer
                    .char_indices()
                    .map(|(i, _)| i)
                    .take_while(|&i| i <= print_len)
                    .last()
                    .unwrap_or(0);

                if safe_len > 0 {
                    let to_print = self.buffer[..safe_len].to_string();
                    print!("{}", to_print);
                    io::stdout().flush().ok();
                    self.buffer.drain(..safe_len);
                }
            }
        }
    }

    fn finish(&mut self) {
        if self.in_thinking && !self.buffer.is_empty() {
            print!("{}", self.buffer);
        }
        // Always reset terminal styling at the end of the turn
        println!("\x1b[0m\n");
        io::stdout().flush().ok();
    }
}

fn main() -> CraneResult<()> {
    #[cfg(feature = "cuda")]
    let (device, dtype) = (DeviceConfig::Cuda(0), DataType::F16);
    #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
    let (device, dtype) = (DeviceConfig::Rocm(0), DataType::F16);
    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), target_os = "macos"))]
    let (device, dtype) = (DeviceConfig::Metal, DataType::F16);
    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(target_os = "macos")))]
    let (device, dtype) = (DeviceConfig::Cpu, DataType::F32);

    let config = ChatConfig {
        common: CommonConfig {
            model_path: MODEL_PATH.to_string(),
            model_type: LlmModelType::Qwen35,
            device,
            dtype,
            max_memory: None,
        },
        generation: GenerationConfig {
            max_new_tokens: 1024,
            temperature: Some(0.2),
            top_p: Some(0.95),
            ..Default::default()
        },
        max_history_turns: 4,
        enable_streaming: true,
    };

    println!("Initializing OxCoder-9B from: {MODEL_PATH}");
    let mut chat_client = ChatClient::new(config)?;

    println!("\nOxCoder ready. Enter your prompt (type 'exit' or 'quit' to stop).\n");

    let stdin = io::stdin();
    loop {
        print!("You: ");
        io::stdout().flush().ok();

        let mut input = String::new();
        if stdin.read_line(&mut input).unwrap_or(0) == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }

        let streamer = std::sync::Mutex::new(ThinkingStreamer::new());

        // Stream tokens live via callback using interior mutability
        chat_client.send_message_streaming(input, |chunk| {
            if let Ok(mut s) = streamer.lock() {
                s.on_chunk(chunk);
            }
        })?;

        if let Ok(mut s) = streamer.lock() {
            s.finish();
        }
    }

    println!("Goodbye!");
    Ok(())
}

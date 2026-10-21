# Changelog

All notable changes to this project will be documented in this file.

## [0.7.0] - 2026-10-21

### 🚀 Features

- *(inference)* Enhance audio engine with mock model support and command environment configuration

- Add support for audio inputs in multimodal chat

- Add example and enhance descriptions for chat completions and image grounding requests

- *(gateway)* Implement rune studio contract, residency controls, honest usage, and dot TUI

- *(gateway)* Implement background preemption, route attribution, and spec TUI


### 🐛 Bug Fixes

- *(engine)* Eliminate blocking runtime drop panic in async context


### 🚜 Refactor

- Image generation and processing handlers to use ModelExecutionGate for engine loading and task execution


## [0.6.0] - 2026-10-19

### 🚀 Features

- Eliminate model load lock contention, enforce bounded SSE backpressure, and fix audio8-asr-infinite model path by @shba007

- Add audio diarization functionality and update models by @shba007

- Add health and readiness endpoints, implement SSRF protection, and enhance model configuration tests by @shba007

- *(inference)* Support per-request thinking mode and reasoning effort controls by @shba007

- *(decision)* Integrate liquid d1-3b multimodal system 1 decision model by @shba007

- *(decision)* Enable native systemone vision ingestion and robust multimodal payload parsing by @shba007


### 🐛 Bug Fixes

- Update actions/checkout and other dependencies in workflows, upgraded performence of model load/unload by @shba007


### ⚡ Performance

- *(inference)* Optimize token speed and prevent VRAM PCIe spill with Flash Attention and smart context budgeting by @shba007


## [0.5.0] - 2026-10-16

### 🚀 Features

- Add VRAM budgeting and context sizing engine by @shba007


### 🚜 Refactor

- Model discovery and setup wizard by @shba007


## [0.4.0] - 2026-10-11

### 🚀 Features

- *(schema)* Introduce new model registry schema version 2 by @shba007


## [0.3.0] - 2026-10-11

### 🚀 Features

- Add terminal dashboard, setup wizard and installers; honor sampling and idle unload by @shba007


### 🐛 Bug Fixes

- *(config)* Update server port and adjust model parameters for improved performance by @shba007


### 📚 Documentation

- Document arm64 support and the remaining x64-only engine fallbacks by @shba007


### ⚙️ Miscellaneous Tasks

- *(install)* Make install.sh executable and pin shell scripts to LF by @shba007


## [0.2.0] - 2026-10-07

### 🚀 Features

- Add NemoSpeechEngine for audio transcription and translation by @shba007


### 🐛 Bug Fixes

- *(chat)* Regession fix chat completions handler with structured prompts and tool validation by @shba007


## [0.1.0] - 2026-10-06

### 🚀 Features

- Init project by @shba007

- Add Qwen2Vl and Qwen35 inference engines with tool call support by @shba007

- Enhance inference engine with tool call support and update model configurations by @shba007

- Enhance model configuration and inference engine by @shba007

- Add support for structured output in chat completions by @shba007

- Add dotenv support for environment variable management and enhance S3 configuration handling by @shba007

- Implement download and extraction utilities for inference engines by @shba007

- Enhance image generation request with additional parameters and structured logging by @shba007

- Add audio transcription and speech synthesis endpoints with CrispASR integration by @shba007

- Add image generation API and media handling by @shba007

- *(api)* Add image generation and moderation endpoints by @shba007

- Add new scripts and update api request examples by @shba007

- Add image captioning and related inference tasks by @shba007

- *(proxy)* Add gemini adapter for chat completions by @shba007

- *(chat)* Enhance chat completion response with reasoning content extraction by @shba007

- *(proxy)* Implement automatic retries for chat completion requests with configurable parameters by @shba007

- *(openrouter)* Add Apodex and Dots3-Note models with configuration parsing and integration tests by @shba007

- *(models)* Add Space Bunny Alpha model with multimodal capabilities by @shba007

- *(hooks)* Add commit-msg and pre-commit hooks for message verification and secret scanning by @shba007

- *(workflows)* Enhance CI/CD configurations for improved deployment and integration by @shba007


### 🚜 Refactor

- Inference engine response handling and add usage tracking by @shba007

- Inference engines and introduce process management by @shba007


### ⚙️ Miscellaneous Tasks

- Make cargo-husky hook scripts executable by @shba007

- Add initial_tag configuration for versioning by @shba007

- Update dependencies and improve documentation in docker-compose.yml by @shba007




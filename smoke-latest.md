| Model | Task | Result | Seconds | Detail |
|---|---|---|---|---|
| gemma-4-12b-it | text-to-text | PASS | 10.3 |  |
| gemma-4-12b-it | text-image-to-text | PASS | 1.4 |  |
| gemma-4-12b-it | text-video-to-text | SKIP | 0.0 | video chat input not probed |
| minicpm-o-4-5-q8-0 | text-to-text | PASS | 9.0 |  |
| minicpm-o-4-5-q8-0 | text-image-to-text | PASS | 1.3 |  |
| minicpm-o-4-5-q8-0 | text-video-to-text | SKIP | 0.0 | video chat input not probed |
| deepseek-r1-distill-qwen-32b | text-to-text | PASS | 19.7 |  |
| huihui-qwen3.8-27b-abliterated | text-to-text | PASS | 16.4 |  |
| huihui-qwen3.8-27b-abliterated | text-image-to-text | PASS | 2.2 |  |
| huihui-qwen3.8-27b-abliterated | text-video-to-text | SKIP | 0.0 | video chat input not probed |
| scadcoder-qwen3.5-9b-v11-q5-k-m | text-to-text | PASS | 6.4 |  |
| oxcoder-9b | text-to-text | PASS | 6.3 |  |
| qwen3.5-4b-q4 | text-to-text | PASS | 4.6 |  |
| qwen3.5-4b-q4 | text-image-to-text | PASS | 0.7 |  |
| qwen3.5-4b-q4 | text-video-to-text | SKIP | 0.0 | video chat input not probed |
| qwen3.5-0.8b | text-to-text | PASS | 2.7 |  |
| lfm3-omni-2b | text-to-text | PASS | 4.8 |  |
| lfm3-omni-2b | text-image-to-text | PASS | 0.2 |  |
| lfm3-omni-2b | text-audio-to-text | SKIP | 0.0 | audio chat input not probed |
| lfm3-omni-2b | text-video-to-text | SKIP | 0.0 | video chat input not probed |
| lfm3-omni-2b | text-to-audio | PASS | 0.4 | 99884 bytes audio/wav |
| liquid-d1-3b | text-to-decision | PASS | 7.8 |  |
| holo3.1-9b-q4 | text-to-text | PASS | 9.4 |  |
| holo3.1-9b-q4 | text-image-to-text | PASS | 1.0 |  |
| ternary-bonsai-2-27b-pq2-0 | text-to-text | PASS | 12.7 |  |
| ternary-bonsai-2-27b-pq2-0 | text-image-to-text | PASS | 1.8 |  |
| ternary-bonsai-2-27b-pq2-0 | text-video-to-text | SKIP | 0.0 | video chat input not probed |
| qwen-image-2.1-uncensored | text-to-image | PASS | 24.8 |  |
| qwen-image-2.1-uncensored | text-image-to-image | PASS | 25.5 |  |
| qwen-image-2.1-uncensored | image-to-image | SKIP | 0.0 | covered by text-image-to-image |
| qwen-image-2.1-turbo | text-to-image | PASS | 9.7 |  |
| qwen-image-edit-2511 | text-image-to-image | PASS | 90.5 |  |
| qwen-image-edit-2511 | image-to-image | SKIP | 0.0 | covered by text-image-to-image |
| moss-image-sft | text-to-image | PASS | 29.0 |  |
| h3-image | text-to-image | PASS | 14.5 |  |
| h3-image | text-image-to-image | PASS | 69.6 |  |
| h3-image | image-to-image | SKIP | 0.0 | covered by text-image-to-image |
| wan2.2-ti2v-5b | text-to-video | PASS | 21.1 |  |
| ltx-2.3-22b-distilled | text-to-video | FAIL | 27.1 | HTTP 500: {"error":{"message":"sd-server failed during weight preparation: [E] ltxav segment 23/50 (ltxav.transformer_blocks.21) failed during weight preparatio |
| fish-audio-s2-pro | text-to-audio | PASS | 15.0 | 143404 bytes audio/wav |
| voxcpm2-audiocpp | text-to-audio | PASS | 4.8 | 245804 bytes audio/wav |
| qwen3-tts-1.7b-customvoice | text-to-audio | PASS | 4.2 | 96044 bytes audio/wav |
| omnivoice | text-to-audio | FAIL | 0.0 | HTTP 503: {"error":{"message":"Model 'omnivoice' cannot execute speech_generation: speech_generation has not passed a live check with 'audio-cpp'","type":"inval |
| acestep-v1.5-turbo | text-to-audio | FAIL | 0.0 | HTTP 400: {"error":{"message":"Model 'acestep-v1.5-turbo' cannot execute speech_generation: Engine 'sd-server' does not implement speech_generation","type":"inv |
| whisper-large-v3 | audio-to-text | PASS | 15.5 |  |
| parakeet-tdt-0.6b-v3 | audio-to-text | PASS | 5.5 |  |
| nemotron-3-diarization | audio-to-diarization | PASS | 14.4 |  |
| google/embeddinggemma-2 | text-to-embedding | PASS | 3.2 |  |
| qwen3-embedding-4b | text-to-embedding | PASS | 2.6 |  |
| dinov2-giant | image-to-embedding | FAIL | 0.0 | HTTP 400: {"error":{"message":"Engine 'encoder' does not implement image-embedding","type":"invalid_request_error","param":null,"code":"engine_task_unsupported" |
| yolo26x | image-to-detection | PASS | 7.0 |  |
| yolo26x-pose | image-to-detection | PASS | 5.6 |  |
| yolo26x-seg | image-to-detection | PASS | 5.6 |  |
| yolo26x-obb | image-to-detection | PASS | 5.5 |  |
| yolo26x-sem-ade20k | image-to-segmentation | PASS | 5.0 |  |
| grounding-dino-base | text-image-to-detection | FAIL | 0.0 | HTTP 400: {"error":{"message":"Engine 'encoder' does not implement grounding","type":"invalid_request_error","param":null,"code":"engine_task_unsupported"}} |
| sam2.1-hiera-large | image-to-segmentation | FAIL | 0.0 | HTTP 400: {"error":{"message":"Engine 'encoder' does not implement segmentation","type":"invalid_request_error","param":null,"code":"engine_task_unsupported"}} |
| birefnet-matting | image-to-matting | PASS | 13.0 |  |
| depth-anything-v2-large | image-to-depth | PASS | 9.3 |  |
| scrfd-34g-gnkps | image-to-detection | PASS | 5.0 |  |
| glintr100-arcface | image-to-embedding | PASS | 4.5 |  |
| w600k-r50-arcface | image-to-embedding | PASS | 4.4 |  |
| mivolo-v2-age-gender | image-to-detection | FAIL | 0.3 | HTTP 500: {"error":{"message":"Failed to load model 'mivolo-v2-age-gender': onnx: ONNX model is not a reviewed YOLO26x detection, pose, segmentation or oriented |
| fairface-demographics | image-to-classification | PASS | 4.3 |  |
| sixdrepnet-head-pose | image-to-classification | PASS | 4.4 |  |
| enet-b2-emotion | image-to-classification | PASS | 5.1 |  |
| segformer-face-parsing | image-to-segmentation | PASS | 6.0 |  |
| resnet50-gaze-estimation | image-to-classification | PASS | 4.4 |  |
| minifasnet-v2-liveness | image-to-classification | PASS | 4.4 |  |
| aesthetic-score-mlp | image-to-detection | FAIL | 0.1 | HTTP 500: {"error":{"message":"Failed to load model 'aesthetic-score-mlp': onnx: ONNX model is not a reviewed YOLO26x detection, pose, segmentation or oriented- |
| gliguard-LLMGuardrails-300M | text-to-moderation | FAIL | 0.0 | HTTP 400: {"error":{"message":"Engine 'encoder' does not implement moderation","type":"invalid_request_error","param":null,"code":"engine_task_unsupported"}} |
| mmbert32k-intent-classifier-merged | nlu | FAIL | 0.0 | HTTP 400: {"error":{"message":"Engine 'encoder' does not implement nlu","type":"invalid_request_error","param":null,"code":"engine_task_unsupported"}} |
| cactus-needle-3 | text-to-text | PASS | 0.4 |  |
| gemini-4-hydrogen | - | SKIP | 0.0 | cloud model (use --include-cloud) |
| gemini-4-carbon | - | SKIP | 0.0 | cloud model (use --include-cloud) |
| gemini-3.8-flash | - | SKIP | 0.0 | cloud model (use --include-cloud) |
| nemotron-3.5-super-vl | - | SKIP | 0.0 | cloud model (use --include-cloud) |
| apodex-1.1-mini | - | SKIP | 0.0 | cloud model (use --include-cloud) |
| dots-3-note-preview | - | SKIP | 0.0 | cloud model (use --include-cloud) |
| space-bunny-alpha | - | SKIP | 0.0 | cloud model (use --include-cloud) |

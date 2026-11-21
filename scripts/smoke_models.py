"""Smoke-test every configured model through a running Rune Infer server.

For each model in the catalog, send one small request per task it declares
(chat, vision chat, embeddings, image generation, detection, speech, ...) and
report pass/fail, latency and the first error line. Uses only the standard
library and models already on disk; it downloads nothing.

  rune-infer serve --headless          # in one terminal
  python scripts/smoke_models.py       # in another

  python scripts/smoke_models.py --only yolo26x,qwen3.5-0.8b --out smoke.md
  python scripts/smoke_models.py --binary target/debug/rune-infer.exe   # spawns the server

Cloud models are skipped unless --include-cloud is given (they cost money).
"""
import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "static/artifacts/images/test.png"
AUDIO = ROOT / "static/artifacts/audios/sample-1-original-bengali-converted.wav"


def tiny_png(size=64):
    """A valid gradient PNG, used when the sample image is missing."""
    rows = b"".join(
        b"\x00" + bytes(v for x in range(size) for v in (x * 4 % 256, y * 4 % 256, 128))
        for y in range(size)
    )

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def image_bytes():
    return IMAGE.read_bytes() if IMAGE.is_file() else tiny_png()


def data_uri():
    return "data:image/png;base64," + base64.b64encode(image_bytes()).decode()


def tone_wav(seconds=2, rate=16000):
    import math
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / rate))) for i in range(rate * seconds)
    )
    return (
        b"RIFF" + struct.pack("<I", 36 + len(frames)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(frames)) + frames
    )


def audio_bytes():
    return AUDIO.read_bytes() if AUDIO.is_file() else tone_wav()


def multipart(fields, files):
    boundary = uuid.uuid4().hex
    out = bytearray()
    for key, value in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
    for key, (name, data, mime) in files.items():
        out += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode() + data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def probes(model):
    """task -> (path, body-or-(bytes, content-type), validator) for one model."""
    mid = model["id"]
    png = image_bytes()
    prompt = "Say hello in one short sentence."
    table = {
        "text-to-intent": ("/v1/nlu/intents", {
            "model": mid, "text": "How do neural networks learn from labeled data?"},
            lambda r: bool(r.get("intent", {}).get("name"))),
        "text-to-text": ("/v1/chat/completions", {
            "model": mid, "max_tokens": 32, "stream": False,
            "messages": [{"role": "user", "content": prompt}]}, lambda r: bool(r["choices"])),
        "text-image-to-text": ("/v1/chat/completions", {
            "model": mid, "max_tokens": 32, "stream": False,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this image in five words."},
                {"type": "image_url", "image_url": {"url": data_uri()}}]}]}, lambda r: bool(r["choices"])),
        "text-to-embedding": ("/v1/embeddings", {
            "model": mid, "input": "Rune Infer smoke test", "encoding_format": "float"},
            lambda r: len(r["data"][0]["embedding"]) > 0),
        "text-to-moderation": ("/v1/moderations", {"model": mid, "input": "I want to hurt someone."},
                               lambda r: bool(r["results"])),
        "text-to-decision": ("/v1/decisions", {
            "model": mid, "state": {"text": "Deploy the container build to production."},
            "questions": [{"name": "route", "type": "choice", "instruction": "Pick a path",
                           "criteria": ["canary_rollout", "blue_green"]}]}, lambda r: True),
        "text-to-image": ("/v1/images/generations", {
            "model": mid, "prompt": "A red apple on a table", "n": 1, "size": "512x512",
            "response_format": "b64_json"}, lambda r: bool(r["data"][0].get("b64_json"))),
        "text-image-to-image": ("/v1/images/edits", multipart(
            {"model": mid, "prompt": "Make the picture black and white", "size": "512x512"},
            {"image[]": ("input.png", png, "image/png")}), lambda r: bool(r["data"])),
        "text-to-video": ("/v1/videos/generations", {
            "model": mid, "prompt": "A red fox running through snow", "size": "512x320",
            "frames": 9, "fps": 8, "steps": 4}, lambda r: True),
        "text-to-audio": ("/v1/audio/speech", {
            "model": mid, "input": "Rune Infer smoke test.", "response_format": "wav",
            # llama-server speech takes female/male; other engines use their default voice.
            **({"voice": "female"} if "llama-server" in str(model.get("engine")) else {})},
            None),
        "audio-to-text": ("/v1/audio/transcriptions", multipart(
            {"model": mid, "response_format": "json"}, {"file": ("sample.wav", audio_bytes(), "audio/wav")}),
            lambda r: "text" in r),
        "audio-to-diarization": ("/v1/audio/diarizations", multipart(
            {"model": mid}, {"file": ("sample.wav", audio_bytes(), "audio/wav")}), lambda r: True),
        "image-to-detection": ("/v1/images/detections", {
            "model": mid, "image": data_uri(), "confidence_threshold": 0.05, "features": ["boxes"]},
            lambda r: True),
        "text-image-to-detection": ("/v1/images/grounding", {
            "model": mid, "image": data_uri(), "query": "a button"}, lambda r: True),
        "image-to-segmentation": ("/v1/images/segmentations", {"model": mid, "image": data_uri()}, lambda r: True),
        "image-to-classification": ("/v1/images/classifications", {
            "model": mid, "image": data_uri(), "top_k": 3}, lambda r: bool(r["heads"] or r.get("measurements"))),
        "image-to-embedding": ("/v1/images/embeddings", {
            "model": mid, "image": data_uri(), "encoding_format": "float"}, lambda r: bool(r["data"])),
        "image-to-matting": ("/v1/images/portrait-mattings", {"model": mid, "image": data_uri()},
                             lambda r: "alpha_matte" in r),
        "image-to-depth": (f"/v1/inference?engine={mid}", {
            "type": "image_depth", "image_bytes": list(png), "preview": False}, lambda r: True),
    }
    return table


# Tasks that are variants of a primary probe, or that have no public endpoint.
SKIP_TASKS = {
    "image-to-image": "covered by text-image-to-image",
    "audio-to-audio": "no public endpoint (only /v1/audio/speech-to-speech)",
    "text-audio-to-text": "audio chat input not probed",
    "text-video-to-text": "video chat input not probed",
}

# Models whose declared task has no public endpoint (checked per id, not per task).
SKIP_MODELS = {
    "acestep-v1.5-turbo": "music generation: no public endpoint (text-to-audio is speech)",
}


def request(base, key, path, body, timeout):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if isinstance(body, tuple):
        data, ctype = body
    else:
        data, ctype = json.dumps(body).encode(), "application/json"
    req = urllib.request.Request(base + path, data=data, headers={**headers, "Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")


def run_probe(base, key, model, task, timeout):
    path, body, check = probes(model)[task]
    started = time.time()
    try:
        status, raw, ctype = request(base, key, path, body, timeout)
        if check is None:  # binary audio
            ok = status == 200 and len(raw) > 100
            detail = f"{len(raw)} bytes {ctype}"
        else:
            parsed = json.loads(raw)
            ok = status == 200 and bool(check(parsed))
            detail = "" if ok else raw[:200].decode("utf-8", "replace")
        return ok, time.time() - started, detail
    except urllib.error.HTTPError as err:
        text = err.read()[:300].decode("utf-8", "replace").replace("\n", " ")
        return False, time.time() - started, f"HTTP {err.code}: {text}"
    except Exception as err:  # timeout, connection reset, bad JSON
        return False, time.time() - started, f"{type(err).__name__}: {err}"


def wait_ready(base, seconds=60):
    for _ in range(seconds):
        try:
            with urllib.request.urlopen(base + "/readyz", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.environ.get("RUNE_API_KEY", ""))
    parser.add_argument("--catalog", type=Path, default=ROOT / "config/catalog.json")
    parser.add_argument("--only", help="comma-separated model ids")
    parser.add_argument("--skip", default="", help="comma-separated model ids")
    parser.add_argument("--include-cloud", action="store_true")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per request")
    parser.add_argument("--binary", type=Path, help="start this rune-infer headless for the run")
    parser.add_argument("--out", type=Path, help="write a markdown report here")
    args = parser.parse_args()

    server = None
    if args.binary:
        server = subprocess.Popen([str(args.binary), "serve", "--headless"], cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_ready(args.base_url):
            sys.exit(f"No ready Rune Infer server at {args.base_url}")
        models = json.loads(args.catalog.read_text(encoding="utf-8"))["models"]
        only = set(filter(None, (args.only or "").split(",")))
        skip = set(filter(None, args.skip.split(",")))
        rows = []
        for model in models:
            mid = model["id"]
            if (only and mid not in only) or mid in skip:
                continue
            if "remote" in model and not args.include_cloud:
                rows.append((mid, "-", "SKIP", 0, "cloud model (use --include-cloud)"))
                continue
            if mid in SKIP_MODELS:
                rows.append((mid, "-", "SKIP", 0, SKIP_MODELS[mid]))
                continue
            tasks = model.get("tasks") or []
            available = probes(model)
            for task in tasks:
                if task in SKIP_TASKS:
                    rows.append((mid, task, "SKIP", 0, SKIP_TASKS[task]))
                    continue
                if task not in available:
                    rows.append((mid, task, "SKIP", 0, "no probe for this task"))
                    continue
                print(f"{mid} / {task} ...", flush=True)
                ok, seconds, detail = run_probe(args.base_url, args.api_key, model, task, args.timeout)
                rows.append((mid, task, "PASS" if ok else "FAIL", seconds, detail))
                print(f"  {'PASS' if ok else 'FAIL'} {seconds:.1f}s {detail[:120]}", flush=True)
        counts = {s: sum(1 for r in rows if r[2] == s) for s in ("PASS", "FAIL", "SKIP")}
        print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
        if args.out:
            lines = ["| Model | Task | Result | Seconds | Detail |", "|---|---|---|---|---|"]
            for mid, task, status, seconds, detail in rows:
                lines.append(f"| {mid} | {task} | {status} | {seconds:.1f} | {detail.replace('|', '/')[:160]} |")
            args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        sys.exit(1 if counts["FAIL"] else 0)
    finally:
        if server:
            server.terminate()


if __name__ == "__main__":
    main()

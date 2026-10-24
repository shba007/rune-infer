"""Opt-in real local inference after automatic home migration; no model downloads."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary, llama = args.binary.resolve(), args.llama.resolve()
    model = next(m for m in json.loads(args.catalog.read_text(encoding="utf-8"))["models"] if m["id"] == args.model)
    original_weight = Path(model["local"]["path"])
    assert original_weight.is_file() and llama.is_file(), "Model and engine must already exist"
    fixture = args.output.resolve().parent / ("migration owner spaces " + uuid.uuid4().hex)
    old = fixture / ".rune-infer"
    new = fixture / ".rune/infer"
    cwd = fixture / "unrelated working directory"
    for path in [old / "config", old / "bin", fixture / ".rune/tools/plugins", cwd]:
        path.mkdir(parents=True, exist_ok=True)
    (fixture / ".rune/rune-kit.exe").write_bytes(b"kit-owned sentinel")
    (fixture / ".rune/tools/plugins/keep.txt").write_bytes(b"kit plugin sentinel")
    (old / "bin/custom-state").write_bytes(b"retained engine customization")
    try:
        reference = os.path.relpath(original_weight, old).replace("\\", "/")
    except ValueError:
        reference = str(original_weight)
    model["local"]["path"] = "{{MIGRATION_TRIAL_MODEL_FILE}}"
    model["local"]["context_length"] = 4096
    model["sampling"]["max_tokens"] = 64
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = {"schema_version": 3, "server": {"host": "127.0.0.1", "port": port,
              "api_key": "{{MIGRATION_TRIAL_API_KEY}}", "max_loaded_models": 1,
              "idle_unload_seconds": 300, "vram_budget_ratio": 0.95}}
    (old / "config/server.json").write_text(json.dumps(server), encoding="utf-8")
    (old / "config/catalog.json").write_text(json.dumps({"schema_version": 3, "models": [model]}), encoding="utf-8")
    env_bytes = (f"MIGRATION_TRIAL_MODEL_FILE='{reference}'\nLLAMA_SERVER_PATH='{llama.as_posix()}'\n"
                 "MIGRATION_TRIAL_API_KEY=local-fixture-only\n").encode()
    (old / ".env").write_bytes(env_bytes)
    originals = {p.name: p.read_bytes() for p in (old / "config").iterdir()}
    environment = os.environ.copy()
    for key in list(environment):
        if key.upper().startswith("RUNE_") or key.upper() == "LLAMA_SERVER_PATH":
            environment.pop(key)
    environment.update(USERPROFILE=str(fixture), HOME=str(fixture))
    log = open(fixture / "server.log", "wb")
    process = subprocess.Popen([str(binary), "--headless"], cwd=cwd, env=environment,
                               stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        for _ in range(150):
            assert process.poll() is None, "Server exited; inspect the isolated fixture log"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    health = json.load(response)
                break
            except (OSError, urllib.error.URLError):
                time.sleep(.1)
        else:
            raise RuntimeError("Server did not become healthy")
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                  data=json.dumps({"model": args.model, "messages": [{"role": "user", "content": "Reply READY only."}],
                                  "thinking": False, "reasoning_effort": "none", "max_tokens": 64}).encode(),
                  headers={"Content-Type": "application/json", "Authorization": "Bearer local-fixture-only"})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
            assert response.status == 200
            assert response.headers["x-model-served"] == args.model
            assert response.headers["x-model-locality"] == "local"
        assert result["choices"][0]["message"]["content"]
        assert (new / ".env").read_bytes() == env_bytes == (old / ".env").read_bytes()
        for name, content in originals.items():
            assert (old / "config" / name).read_bytes() == content
        assert (new / "bin/custom-state").read_bytes() == b"retained engine customization"
        assert (fixture / ".rune/rune-kit.exe").read_bytes() == b"kit-owned sentinel"
        assert (fixture / ".rune/tools/plugins/keep.txt").read_bytes() == b"kit plugin sentinel"
        assert original_weight.is_file() and (new / "logs/rune-infer.log").is_file()
        evidence = {"version": health["version"], "backend": health["backend"], "model": args.model,
                    "status": 200, "reply": result["choices"][0]["message"]["content"],
                    "fixture": str(fixture), "selected_home": str(new), "originals_preserved": True,
                    "environment_preserved": True, "kit_preserved": True, "weights_not_moved": True}
        args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()


if __name__ == "__main__":
    main()

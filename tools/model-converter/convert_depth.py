"""Export the pinned Large depth checkpoint; never modify Infer's model catalog.

Run with --help for arguments. Conversion dependencies live beside this script,
separately from the native gateway. Retain this tool and its reports for reuse.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path

SOURCE = "depth-anything/Depth-Anything-V2-Large-hf"
REVISION = "7581137eff8d4e94f6e796d3baea0e9fa79b22d2"
SOURCE_SHA256 = "4e01e34ed5549b529b70b92d53226bc370f03041977b390d3dde45d47f516cf9"
WIDTH, HEIGHT = 686, 518


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_environment():
    lock = Path(__file__).with_name("requirements.lock")
    for line in lock.read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            package, expected = line.split("==", 1)
            if importlib.metadata.version(package) != expected:
                raise ValueError(f"Installed {package} does not match requirements.lock")
    return sha256(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New directory for the candidate, raw parity fixtures and report")
    parser.add_argument("--image", required=True, action="append", type=Path,
                        help="Real RGB image fixture; repeat for additional parity checks")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="Use only the existing pinned Hugging Face cache")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("The pinned converter requires Linux x86_64; the exported model targets CPU.")
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be between 1 and 32")
    if any(not image.is_file() for image in args.image):
        parser.error("Every --image must name an existing encoded image")
    # Exclusive creation preserves prior conversions and never overwrites weights.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "source": SOURCE, "revision": REVISION, "license": "cc-by-nc-4.0",
        "input": {"shape": [1, 3, HEIGHT, WIDTH], "dtype": "float32", "layout": "NCHW"},
        "output": {"shape": [1, HEIGHT, WIDTH], "dtype": "float32", "units": "relative"},
        "preprocessing": {"resize": "bicubic", "keep_aspect_ratio": False,
                          "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "tolerance": {"atol": 0.001, "rtol": 0.001},
        "status": "failed", "checks": [],
    }
    started = time.monotonic()
    try:
        report["dependency_lock_sha256"] = verify_environment()
        import numpy as np
        import torch
        import litert_torch
        from PIL import Image
        from huggingface_hub import snapshot_download
        from transformers import DepthAnythingForDepthEstimation, DPTImageProcessor
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator
        from ai_edge_litert.options import CpuOptions, Options

        report["packages"] = {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "torchao", "litert-torch", "litert-converter",
            "ai-edge-litert", "transformers", "numpy", "Pillow")}
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        print("Downloading pinned Large safetensors checkpoint", flush=True)
        snapshot = Path(snapshot_download(SOURCE, revision=REVISION, local_files_only=args.offline, allow_patterns=[
            "config.json", "preprocessor_config.json", "model.safetensors"]))
        report["source_weights_sha256"] = sha256(snapshot / "model.safetensors")
        if report["source_weights_sha256"] != SOURCE_SHA256:
            raise ValueError("Source checkpoint checksum does not match the reviewed revision")
        report["source_config_sha256"] = {
            name: sha256(snapshot / name) for name in ("config.json", "preprocessor_config.json")}
        processor = DPTImageProcessor.from_pretrained(snapshot, local_files_only=True)
        model = DepthAnythingForDepthEstimation.from_pretrained(
            snapshot, local_files_only=True, use_safetensors=True,
            attn_implementation="eager").float().eval()

        class Depth(torch.nn.Module):
            def __init__(self, depth_model):
                super().__init__()
                self.depth_model = depth_model

            def forward(self, pixels):
                return self.depth_model(pixel_values=pixels).predicted_depth

        wrapped = Depth(model).eval()
        candidate = args.output_dir / "depth_anything_v2_large.candidate.tflite"
        print("Converting full FP32 depth network", flush=True)
        with torch.no_grad():
            edge_model = litert_torch.convert(wrapped, (torch.zeros(1, 3, HEIGHT, WIDTH),))
            edge_model.export(str(candidate))
        report["candidate"] = {"file": candidate.name, "sha256": sha256(candidate),
                               "size_bytes": candidate.stat().st_size}
        compiled = CompiledModel.from_file(str(candidate), options=Options(
            hardware_accelerators=HardwareAccelerator.CPU,
            cpu_options=CpuOptions(num_threads=args.threads)))
        inputs, outputs = compiled.create_input_buffers(0), compiled.create_output_buffers(0)
        for index, path in enumerate(args.image):
            print(f"Checking fixture {index}: {path.name}", flush=True)
            with Image.open(path) as image:
                pixels = processor(images=image.convert("RGB"),
                                   size={"height": HEIGHT, "width": WIDTH},
                                   keep_aspect_ratio=False, return_tensors="np")["pixel_values"]
            pixels = np.ascontiguousarray(pixels, dtype=np.float32)
            if pixels.shape != (1, 3, HEIGHT, WIDTH):
                raise ValueError(f"Unexpected preprocessed shape: {pixels.shape}")
            with torch.no_grad():
                reference = wrapped(torch.from_numpy(pixels)).numpy()
            check_started = time.monotonic()
            inputs[0].write(pixels.ravel())
            compiled.run_by_index(0, inputs, outputs)
            actual = outputs[0].read(HEIGHT * WIDTH, np.float32).reshape(1, HEIGHT, WIDTH).copy()
            elapsed = time.monotonic() - check_started
            if reference.shape != actual.shape or not np.isfinite(actual).all():
                raise ValueError("Invalid depth shape or non-finite runtime output")
            if not np.isfinite(reference).all() or np.ptp(reference) <= 0:
                raise ValueError("Invalid reference depth output")
            difference = np.abs(actual - reference)
            passed = bool(np.allclose(actual, reference, **report["tolerance"]))
            prefix = f"fixture-{index}"
            pixels.astype("<f4").tofile(args.output_dir / f"{prefix}.input.f32")
            reference.astype("<f4").tofile(args.output_dir / f"{prefix}.reference.f32")
            actual.astype("<f4").tofile(args.output_dir / f"{prefix}.litert.f32")
            report["checks"].append({
                "image": path.name, "image_sha256": sha256(path),
                "input_file": f"{prefix}.input.f32", "reference_file": f"{prefix}.reference.f32",
                "passed": passed, "max_absolute_error": float(difference.max()),
                "mean_absolute_error": float(difference.mean()), "cpu_seconds": elapsed,
                "reference_min": float(reference.min()), "reference_max": float(reference.max())})
            if not passed:
                raise ValueError(f"Fixture {index} exceeds the recorded FP32 parity tolerance")
        report["status"] = "linux_cpu_parity_passed"
        print("Linux parity passed; Windows native runtime validation is still required.", flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (args.output_dir / "conversion-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

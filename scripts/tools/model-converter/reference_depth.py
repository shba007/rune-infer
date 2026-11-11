"""Generate strict Large PyTorch references for images decoded by Infer.

Run inside the pinned converter container. This performs no conversion and uses
only the existing local checkpoint cache. The output directory must be new.
"""
import argparse
import json
from pathlib import Path
from convert_depth import SOURCE, REVISION, SOURCE_SHA256, WIDTH, HEIGHT, sha256, verify_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    lock_hash = verify_environment()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import snapshot_download
    from transformers import DepthAnythingForDepthEstimation, DPTImageProcessor
    torch.set_num_threads(4)
    snapshot = Path(snapshot_download(SOURCE, revision=REVISION, local_files_only=True))
    if sha256(snapshot / "model.safetensors") != SOURCE_SHA256:
        raise ValueError("Source checkpoint checksum mismatch")
    processor = DPTImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = DepthAnythingForDepthEstimation.from_pretrained(
        snapshot, local_files_only=True, use_safetensors=True, attn_implementation="eager").float().eval()
    checks = []
    for index, path in enumerate(args.image):
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("Expected an RGB PNG exported by decode_image")
            pixels = processor(images=image, size={"height": HEIGHT, "width": WIDTH},
                               keep_aspect_ratio=False, return_tensors="np")["pixel_values"]
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        with torch.no_grad():
            depth = model(pixel_values=torch.from_numpy(pixels)).predicted_depth.numpy()
        if pixels.shape != (1, 3, HEIGHT, WIDTH) or depth.shape != (1, HEIGHT, WIDTH):
            raise ValueError("Incorrect reference shape")
        if not np.isfinite(depth).all():
            raise ValueError("Non-finite reference output")
        pixels.astype("<f4").tofile(args.output_dir / f"fixture-{index}.input.f32")
        depth.astype("<f4").tofile(args.output_dir / f"fixture-{index}.reference.f32")
        checks.append({"image": path.name, "image_sha256": sha256(path)})
        print(f"Reference saved: {path.name}", flush=True)
    (args.output_dir / "reference-report.json").write_text(json.dumps({
        "source": SOURCE, "revision": REVISION, "decoded_by": "Infer image crate", "dependency_lock_sha256": lock_hash,
        "checks": checks}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

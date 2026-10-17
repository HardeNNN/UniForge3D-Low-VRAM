"""Run stage 1 several times in one process, with identical noise, and compare the results.

Repeating stage 1 in one process has failed twice with an intermittent
``CUDA error: an illegal memory access`` (in the web UI's low-RAM mode on 2026-10-16, and in
the A/B harness on 2026-10-17), and once with ``Stage1 decoded an empty sparse structure``.
This is the harness that looks for it: it seeds the CPU noise before every call, so every
call gets the same input, and captures the sampler's latent and the decoder's occupancy
logits to say which of the two changes when it goes wrong.

    python scripts\\check_stage1_repeat.py --calls 3

Report, not a gate: it exits non-zero only if a call raises. A clean run is one where the
latents match each other, which is not the same as the bug being gone — it is intermittent.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import uniforge3d.pipelines.uniforge3d_image_to_3d as pipeline_module  # noqa: E402
from uniforge3d import UniForge3DImageTo3DPipeline  # noqa: E402

CONFIG = ROOT / "weights" / "pipeline.local.json"
IMAGES = sorted((ROOT / "assets" / "case01").glob("*.png"))


def stats(tensor: torch.Tensor) -> str:
    values = tensor.float()
    return (
        f"{tuple(tensor.shape)} {tensor.dtype} finite={bool(torch.isfinite(values).all())} "
        f"min={values.min().item():+.3f} max={values.max().item():+.3f} "
        f"absmax={values.abs().max().item():.3f} mean={values.mean().item():+.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-low-vram", action="store_true", help="keep models on the card")
    args = parser.parse_args()

    captured: dict[str, torch.Tensor | None] = {"latent": None, "logits": None}
    original_sample = pipeline_module.sample_sparse_structure
    original_decode = pipeline_module.decode_sparse_structure

    def capturing_sample(*call_args, **call_kwargs):
        captured["latent"] = original_sample(*call_args, **call_kwargs)
        return captured["latent"]

    def capturing_decode(*call_args, **call_kwargs):
        captured["logits"] = original_decode(*call_args, **call_kwargs)
        return captured["logits"]

    pipeline_module.sample_sparse_structure = capturing_sample
    pipeline_module.decode_sparse_structure = capturing_decode

    pipeline = UniForge3DImageTo3DPipeline.from_config(
        CONFIG, low_vram=not args.no_low_vram, weights_dtype=torch.float16, prefetch=False
    )
    batch = pipeline._preprocess_images(IMAGES)
    features = pipeline._encode_dino(batch)["sparse_structure"]

    latents: list[torch.Tensor] = []
    for call in range(1, args.calls + 1):
        captured["latent"] = captured["logits"] = None
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        outcome = "ok"
        structure = None
        try:
            structure = pipeline._generate_structure(batch, features)
        except Exception as error:  # noqa: BLE001 - report what came out
            outcome = f"FAILED {type(error).__name__}: {error}"
        elapsed = time.perf_counter() - started

        latent = captured["latent"]
        print(f"[repro] call {call} in {elapsed:5.1f}s -> {outcome}", flush=True)
        if latent is None:
            print("[repro]   no latent captured", flush=True)
            continue
        latents.append(latent.cpu())
        print(f"[repro]   latent {stats(latent)}", flush=True)
        if captured["logits"] is not None:
            logits = captured["logits"]
            occupied = int((logits > 0).sum())
            print(
                f"[repro]   logits {stats(logits)} | cells over 0: {occupied}",
                flush=True,
            )
        if structure is not None:
            print(f"[repro]   voxels {structure.coords32.shape[0]}", flush=True)

    for index in range(1, len(latents)):
        difference = (latents[index] - latents[0]).abs().max().item()
        print(
            f"[repro] latent {index + 1} vs 1: max abs difference {difference:.6f} "
            f"{'(identical)' if difference == 0 else ''}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure what the fp32-pinned weights cost stage 1: A/B, one arm per process.

Arm A is the shipped dtype policy: the Linear/Conv/Sparse weights of the flow models are fp16
and 388 MiB of the structure model — normalisations, LayerScale gammas, the VGGT raw
attention projections, LoRA — stays fp32. Arm B converts every floating parameter and buffer
of those models to fp16, which is what arm A's pins cost, and which would also change the
results (they are not bit-exact). Each arm runs one warm-up call and then `--calls` timed
calls, all with the same seeded CPU noise, so the only difference between the arms is the
weight dtype.

One arm per process on purpose: two structure models in one process has produced an
intermittent CUDA illegal access (see `check_stage1_repeat.py` and section 5 of the notes).

    python scripts\\bench_fp16_stage1.py --arm shipped  --calls 2
    python scripts\\bench_fp16_stage1.py --arm all-fp16 --calls 2

Measured 2026-10-17 on the RTX 3070 Ti, two runs per arm: 20.0 and 20.0 s against 19.7 and
19.7 s per call, 5.15 against 4.95 GB of peak VRAM, 388 against 0 MiB fp32, output 1568
against 1567 voxels. About 1%.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from uniforge3d import UniForge3DImageTo3DPipeline  # noqa: E402
from uniforge3d.models import loaders  # noqa: E402

CONFIG = ROOT / "weights" / "pipeline.local.json"
IMAGES = sorted((ROOT / "assets" / "case01").glob("*.png"))


def convert_everything(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """Halve every floating parameter and buffer, not just the mix-precision modules."""
    for parameter in module.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype)
    for buffer in module.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype)
    for submodule in module.modules():
        if isinstance(getattr(submodule, "dtype", None), torch.dtype):
            submodule.dtype = dtype


def fp32_pinned_mib(model: torch.nn.Module) -> float:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.dtype not in (torch.float16, torch.bfloat16)
    ) / 2**20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("shipped", "all-fp16"), required=True)
    parser.add_argument("--calls", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pipeline = UniForge3DImageTo3DPipeline.from_config(
        CONFIG, weights_dtype=torch.float16, prefetch=False
    )
    if args.arm == "all-fp16":
        # The store builds groups lazily, so the patch has to be in place when the group is
        # first asked for, not when the pipeline is constructed.
        original = loaders.convert_model_to_dtype
        loaders.convert_model_to_dtype = convert_everything
        try:
            model = pipeline.structure_model
        finally:
            loaders.convert_model_to_dtype = original
    else:
        model = pipeline.structure_model
    print(f"[ab] {args.arm}: {fp32_pinned_mib(model):.0f} MiB of the structure model is fp32", flush=True)

    batch = pipeline._preprocess_images(IMAGES)
    features = pipeline._encode_dino(batch)["sparse_structure"]

    for call in range(args.calls + 1):
        warmup = call == 0
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            structure = pipeline._generate_structure(batch, features)
        except Exception as error:  # noqa: BLE001 - report what came out
            print(
                f"[ab] {args.arm} {'warm-up' if warmup else f'call {call}'}: "
                f"FAILED after {time.perf_counter() - started:.1f}s: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            return 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        print(
            f"[ab] {args.arm:<9} {'warm-up' if warmup else f'call {call}':<8} {elapsed:6.1f}s | "
            f"voxels {structure.coords32.shape[0]:>6} | "
            f"peak vram {torch.cuda.max_memory_allocated() / 2**30:5.2f} GB",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

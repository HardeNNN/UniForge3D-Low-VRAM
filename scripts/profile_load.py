"""Attribute the model-load wall clock to each loader step.

Loading the whole stack costs about two minutes, so it is worth knowing which checkpoint
and which construction step it goes into. This wraps the loader functions with timers and
then asks the pipeline for every stage's weights, so the numbers come from the same
checkpoint resolution and group order as a normal run.

CPU only: no device is touched, so it can run beside a GPU job.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uniforge3d.models.loaders as loaders  # noqa: E402
from uniforge3d import UniForge3DImageTo3DPipeline  # noqa: E402


def _timed(label: str, call):
    """Wrap a loader function or class in a wall-clock report."""

    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        result = call(*args, **kwargs)
        print(f"[profile] {label:<36} {time.perf_counter() - started:6.1f}s", flush=True)
        return result

    return wrapper


def _timed_class(label: str, cls):
    """Wrap a model class so its constructor is reported on its own line."""

    class Timed(cls):  # noqa: N801 - a transparent stand-in for the wrapped class
        def __init__(self, *args, **kwargs):
            started = time.perf_counter()
            super().__init__(*args, **kwargs)
            print(f"[profile] {label:<36} {time.perf_counter() - started:6.1f}s (construct)", flush=True)

    Timed.__name__ = cls.__name__
    Timed.__qualname__ = cls.__qualname__
    return Timed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "weights" / "pipeline.local.json")
    parser.add_argument("--weights-dtype", default="float16")
    args = parser.parse_args()

    if not args.config.is_file():
        print(
            f"[profile] no config at {args.config}: "
            "run .venv\\Scripts\\python.exe scripts\\make_local_config.py first"
        )
        return 1

    for name in (
        "_read_weights",
        "_load_structure_weights",
        "_load_flow_model",
        "load_sparse_structure_decoder",
        "load_shape_decoder",
        "load_texture_decoder",
        "DinoV3Encoder",
        "StructuredLatentConditioner",
    ):
        setattr(loaders, name, _timed(name, getattr(loaders, name)))
    loaders.get_model_class = _timed("get_model_class", loaders.get_model_class)
    loaders.VGGTConditionedSLatFlowModel = _timed_class(
        "VGGTConditionedSLatFlowModel", loaders.VGGTConditionedSLatFlowModel
    )

    started = time.perf_counter()
    # No prefetch: this wants one clean timeline for the whole stack, in the order a case
    # reaches it.
    pipeline = UniForge3DImageTo3DPipeline.from_config(
        args.config, weights_dtype=args.weights_dtype, prefetch=False
    )
    pipeline.load_all()
    print(f"[profile] {'TOTAL':<36} {time.perf_counter() - started:6.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

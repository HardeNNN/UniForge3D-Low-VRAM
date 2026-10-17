"""Check that each stage's weights load when that stage runs, and not before.

The pipeline builds one group of models per stage: when the stage asks for them, or ahead
of time on the store's worker thread while the GPU runs the stage before. This walks the
same accessors a case does, in the same order, releasing exactly what a case releases, and
reports the groups, the wall clock and the process memory at every step. CPU only: no
device is touched, so it can run beside a GPU job.

    .venv\\Scripts\\python.exe scripts\\check_stage_loading.py
    .venv\\Scripts\\python.exe scripts\\check_stage_loading.py --low-ram
    .venv\\Scripts\\python.exe scripts\\check_stage_loading.py --prefetch

Without ``--prefetch`` the walk is deterministic and shows what each stage costs on its
own. With it, the groups appear without anyone asking for them, which is the loading that
a run hides behind the previous stage's compute. ``--low-ram`` releases each stage's
weights as the case finishes with them, so what stays resident is one group instead of the
whole stack.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uniforge3d import UniForge3DImageTo3DPipeline  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover - psutil arrives with gradio
    psutil = None

# One step per stage of a case: the accessors the stage touches with the group each
# belongs to, and the groups a case releases once it is done with them (the pipeline only
# acts on that in low-RAM mode).
STEPS = (
    (
        "dino features",
        (("remove_background", "preprocess"), ("dino", "preprocess")),
        ("preprocess",),
    ),
    (
        "stage1 decode",
        (("structure_model", "structure"), ("structure_decoder", "structure")),
        ("structure",),
    ),
    (
        "stage2 shape512 decode",
        (
            ("conditioner", "conditioner"),
            ("shape512_model", "shape512"),
            ("shape512_decoder", "shape512"),
        ),
        ("shape512",),
    ),
    (
        "shape1024",
        (("shape1024_model", "shape1024"),),
        ("shape1024",),
    ),
    (
        "texture export",
        (
            ("shape1024_decoder", "texture"),
            ("texture_model", "texture"),
            ("texture_decoder", "texture"),
        ),
        ("texture", "conditioner"),
    ),
)


def memory() -> str:
    """Committed and working-set memory, when psutil is around to report them.

    The peak here is committed private bytes, which is what the machine has to hold. The
    peak of the working set would also count the file-backed pages of the memory-mapped
    checkpoints, and the OS can drop those under pressure.
    """
    if psutil is None:
        return "memory n/a"
    info = psutil.Process().memory_info()
    parts = [f"commit {info.pagefile / 2**30:5.2f} GB", f"peak commit {info.peak_pagefile / 2**30:5.2f} GB"]
    if getattr(info, "rss", 0):
        parts.append(f"working set {info.rss / 2**30:5.2f} GB")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "weights" / "pipeline.local.json")
    parser.add_argument("--weights-dtype", default="float16")
    parser.add_argument("--low-ram", action="store_true", help="release each stage after a case is done with it")
    parser.add_argument("--prefetch", action="store_true", help="load later stages on the worker thread")
    args = parser.parse_args()

    weights_dtype = None if args.weights_dtype == "auto" else args.weights_dtype
    pipeline = UniForge3DImageTo3DPipeline.from_config(
        args.config, weights_dtype=weights_dtype, low_ram=args.low_ram, prefetch=args.prefetch
    )
    resident = pipeline._models.loaded()  # white-box: this checks residency itself
    print(f"[check] pipeline built, resident groups {resident or '()'} | {memory()}", flush=True)
    problems: list[str] = []
    if resident:
        problems.append(f"building the pipeline loaded {resident} before any stage asked")

    if args.prefetch:
        time.sleep(5)
        print(f"[check] 5s later, nobody asked yet: {pipeline._models.loaded()} | {memory()}", flush=True)

    expected: set[str] = set()
    for label, accessors, released in STEPS:
        started = time.perf_counter()
        for accessor, group in accessors:
            getattr(pipeline, accessor)
            expected.add(group)
        stage_seconds = time.perf_counter() - started
        pipeline._release(*released)
        expected -= set(released)
        resident = pipeline._models.loaded()
        if set(resident) != expected:
            problems.append(f"{label}: resident {resident}, expected {sorted(expected)}")
        print(
            f"[check] {label:<22} loaded in {stage_seconds:6.1f}s | "
            f"resident {resident or '()'} | {memory()}",
            flush=True,
        )

    if args.low_ram and resident:
        problems.append(f"low-RAM mode left {resident} resident at the end of the walk")

    print(f"\n[check] {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"[check] {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

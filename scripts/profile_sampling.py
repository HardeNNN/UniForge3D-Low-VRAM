"""Profile where a case's sampling time goes, at a reduced step count.

Builds the pipeline from a copy of the config with `steps` lowered (so the profile is short
and this script is cheap to re-run), runs one case with the CUDA profiler attached, and
prints which operators dominate device time, with input shapes. A sampling forward that is
compute-bound shows up as large `aten::mm`/attention entries; if instead the device time sits
in elementwise, index and copy kernels, or if summed kernel time is far below wall time, the
cost is overhead rather than arithmetic and batching work will not pay.

Usage: python scripts\\profile_sampling.py assets\\case05 [--steps 3] [--output outputs\\_prof]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def reduced_config(config_path: Path, steps: int) -> Path:
    """Write a sibling config with every stage's step count lowered.

    It has to stay in the same directory: checkpoint paths in the config are relative to it.
    The counts live at `<stage>.sampler.params.steps`, so walk the tree.
    """
    config = json.loads(config_path.read_text())

    def lower(node: Any) -> int:
        lowered = 0
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "steps" and isinstance(value, int):
                    node[key] = steps
                    lowered += 1
                else:
                    lowered += lower(value)
        elif isinstance(node, list):
            for value in node:
                lowered += lower(value)
        return lowered

    if lower(config) == 0:
        raise SystemExit(f"no 'steps' key found in {config_path}")
    path = config_path.with_name(f"{config_path.stem}.profile{config_path.suffix}")
    path.write_text(json.dumps(config, indent=2))
    return path


def report(profiler: torch.profiler.profile, wall: float, top: int) -> None:
    events = profiler.key_averages(group_by_input_shape=True)
    totals = sorted(
        ((event.self_device_time_total, event.count, event.key, event.input_shapes) for event in events),
        key=lambda row: row[0],
        reverse=True,
    )
    device = sum(row[0] for row in totals)
    print(f"\nsummed device time {device / 1e6:.1f}s against {wall:.1f}s wall ({100 * device / 1e6 / wall:.0f}%)")

    for title, wanted in (("library operators", True), ("raw kernels", False)):
        print(f"\n--- top {top} {title} by self device time ---")
        shown = 0
        for device_us, count, key, shapes in totals:
            if key.startswith("aten::") != wanted:
                continue
            share = 100 * device_us / device if device else 0.0
            print(f"{device_us / 1e6:8.2f}s {share:5.1f}%  n={count:<7} {key}")
            if shapes and shown < 8:
                print(f"{'':>17} shapes {shapes}")
            shown += 1
            if shown >= top:
                break


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path, help="a case directory of images")
    parser.add_argument("--config", type=Path, default=ROOT / "weights" / "pipeline.local.json")
    parser.add_argument("--steps", type=int, default=3, help="steps per sampling stage")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "_prof")
    parser.add_argument("--trace", type=Path, default=ROOT / "logs" / "sampling_trace.json")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    from uniforge3d.pipelines.uniforge3d_image_to_3d import UniForge3DImageTo3DPipeline

    config = reduced_config(args.config, args.steps)
    print(f"[profile] {args.steps} steps per stage via {config.name}", flush=True)
    try:
        pipeline = UniForge3DImageTo3DPipeline.from_config(config, device=0, low_vram=True, weights_dtype=None)
        images = sorted(path for path in args.case.iterdir() if path.is_file())
        print(f"[profile] running {args.case.name} on {len(images)} image(s)", flush=True)
        started = time.perf_counter()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            pipeline.run(images, args.output / args.case.name / "prediction.glb", overwrite=True)
        wall = time.perf_counter() - started
    finally:
        config.unlink(missing_ok=True)
        torch.cuda.empty_cache()

    report(profiler, wall, args.top)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.trace))
    print(f"\ntrace written to {args.trace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

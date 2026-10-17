"""Compare the models loaded through the original checkpoints and their half-precision copies.

``scripts\\convert_weights_fp16.py`` writes fp16 copies of the fp32 checkpoints the pipeline
halves anyway. This builds both configs itself — one pointing at the originals, one at the
copies — loads every stage's models through each, with the same ``--weights-dtype float16``
request a run uses, and reports the tensors that come out different: how many, how many
values, and the largest absolute and relative difference. Bit-identical everywhere is the
expected result, because the pipeline halves these tensors in memory anyway, and anything
else is a finding.

    .venv\\Scripts\\python.exe scripts\\check_fp16_weights.py

CPU only, no device is touched, so it can run beside a GPU job. It builds twelve models and
takes about five minutes; the conversion tool's own read-back check is the fast one.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import make_local_config  # noqa: E402
from uniforge3d import UniForge3DImageTo3DPipeline  # noqa: E402
from uniforge3d.models.loaders import MODEL_GROUPS  # noqa: E402


def state_of(config: Path, group: str, weights_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """One group's models through ``config``, as a flat name -> tensor mapping."""
    pipeline = UniForge3DImageTo3DPipeline.from_config(
        config, weights_dtype=weights_dtype, prefetch=False
    )
    models = pipeline._models.get(group)  # white-box: this compares whole models
    return {
        f"{name}.{key}": tensor
        for name, model in models.items()
        # `preprocess` holds the background remover, which is not a torch module.
        if isinstance(model, torch.nn.Module)
        for key, tensor in model.state_dict().items()
    }


def compare(group: str, reference: dict, converted: dict) -> list[str]:
    if set(reference) != set(converted):
        missing = sorted(set(reference) - set(converted))[:5]
        extra = sorted(set(converted) - set(reference))[:5]
        return [f"{group}: tensor names differ (missing {missing}, extra {extra})"]

    started = time.perf_counter()
    different = []
    for key, tensor in reference.items():
        other = converted[key]
        if tensor.dtype != other.dtype or tensor.shape != other.shape:
            different.append((key, float("inf"), float("inf")))
            continue
        if torch.equal(tensor, other):
            continue
        difference = (tensor.float() - other.float()).abs()
        scale = tensor.float().abs().max().item() or 1.0
        different.append((key, difference.max().item(), difference.max().item() / scale))

    values = sum(reference[key].numel() for key, *_ in different)
    print(
        f"[check] {group:<12} {len(reference):>4} tensors, {len(different):>3} different "
        f"({values / 1e6:.2f}M of {sum(t.numel() for t in reference.values()) / 1e6:.2f}M "
        f"values) in {time.perf_counter() - started:5.1f}s",
        flush=True,
    )
    for key, absolute, relative in sorted(different, key=lambda row: -row[2])[:5]:
        print(f"[check]   {key:<58} max abs {absolute:.3e} rel {relative:.3e}")
    return [f"{group}: {len(different)} tensors differ"] if different else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()

    copies = make_local_config.half_precision_copies()
    if not copies:
        print("[check] no half-precision copies found: nothing to compare")
        print("[check] run scripts\\convert_weights_fp16.py first")
        return 0

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as directory:
        original_config = Path(directory) / "originals.json"
        copy_config = Path(directory) / "copies.json"
        make_local_config.generate(prefer_half_precision=False, out=original_config)
        make_local_config.generate(prefer_half_precision=True, out=copy_config)

        for group in MODEL_GROUPS:
            reference = state_of(original_config, group, torch.float16)
            converted = state_of(copy_config, group, torch.float16)
            problems += compare(group, reference, converted)

    print(f"\n[check] {'PASS' if not problems else 'FAIL'} over {len(copies)} half-precision copies")
    for problem in problems:
        print(f"[check] {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

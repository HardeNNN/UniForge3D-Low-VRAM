"""Write half-precision copies of the fp32 checkpoints the pipeline halves anyway.

``--weights-dtype float16`` (the default under ``--low-vram``) converts those weights in
memory after loading them. Doing that conversion once on disk instead costs nothing in
precision — it applies the same rounding, earlier — while every run reads, copies and holds
half as many bytes for those tensors. The three denoisers, the VGGT backbone, the Stage-1
checkpoint and the three adapters come to 21.45 GiB; the copies are 11.47 GiB.

    python scripts/convert_weights_fp16.py     # convert, verify, and switch over
    python scripts/convert_weights_fp16.py --revert

The copies go to ``weights/fp16/`` with a manifest recording each source and its size, and
``make_local_config.generate`` is then re-run so ``weights/pipeline.local.json`` — the one
config every tool and launcher reads — points at them. Originals are never modified. To go
back: ``--revert``, or delete ``weights/fp16/`` and re-run ``make_local_config.py``.

It is not a whole-file conversion. The pipeline halves the Linear/Conv/Sparse weights of the
flow models and keeps the VGGT attention projections, the LoRA matrices, the normalisations
and the LayerScale gammas in fp32. Which tensors those are is read off each built model, so
the copies hold exactly what a run would have held, and the tensors a run would have held in
fp32 stay fp32 in the copy. Each copy is read back and compared tensor by tensor as it is
written, and ``scripts\\check_fp16_weights.py`` is the slower end-to-end gate: it loads every
stage through both configs and fails on a single differing bit.

Two checkpoints are not touched: the VGGT conditioner stays in the precision it was published
with, and the decoders and the base DiT are already f16/bf16 on disk.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import make_local_config  # noqa: E402
from uniforge3d import UniForge3DImageTo3DPipeline  # noqa: E402

# The config section that holds each flow model, and the group name it is built under.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("sparse_structure", "structure"),
    ("structured_latent_512", "shape512"),
    ("structured_latent_1024", "shape1024"),
    ("texture", "texture"),
)

# The Stage-1 checkpoint names the fusion blocks the way the training code did.
FUSION_PREFIX = re.compile(r"^fusion_groups\.(\d+)\.f_from_z\.")

MANIFEST = "manifest.json"


def mappings(section: str) -> dict[str, Callable[[str], str]]:
    """How each of a section's checkpoint files names its tensors in the built model.

    Mirrors ``_load_flow_model`` and ``_load_structure_weights``, the only two places that
    read these files.
    """
    if section == "sparse_structure":
        return {
            "dit_base": lambda key: f"dit.{key}",
            "vggt_base": lambda key: f"vggt.{key.removeprefix('model.')}",
            "checkpoint": lambda key: FUSION_PREFIX.sub(r"dit_to_vggt.\1.", key),
        }
    return {
        "checkpoint": lambda key: f"denoiser.{key}",
        "base_checkpoint": lambda key: f"denoiser.{key}",
        "adapter_checkpoint": lambda key: f"adapter.{key}",
    }


def read_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    if checkpoint.suffix == ".safetensors":
        return load_file(str(checkpoint), device="cpu")
    state = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=True)
    return state["model"] if "model" in state else state


def wanted_dtype(name: str, dtypes: dict[str, torch.dtype]) -> torch.dtype | None:
    """The dtype the built model holds ``name`` in, following peft's wrapping of base layers.

    ``attach_stage1_lora`` replaces the targeted Linear layers with LoRA layers whose own
    weight sits at ``<module>.base.weight``, so a base checkpoint's ``<module>.weight`` is the
    model's ``<module>.base.weight``.
    """
    dtype = dtypes.get(name)
    if dtype is not None:
        return dtype
    head, _, tail = name.rpartition(".")
    return dtypes.get(f"{head}.base.{tail}") if head else None


def write_half_copy(
    path: Path, name_of: Callable[[str], str], dtypes: dict[str, torch.dtype], target: Path
) -> tuple[int, bool]:
    """Halve the tensors the built model holds in fp16, leaving its fp32 ones alone.

    Returns how many tensors were halved and whether the copy read back equals what was
    written, which is the cheap half of the verification.
    """
    state = read_state(path)
    copy: dict[str, torch.Tensor] = {}
    halved = kept = unmapped = 0
    for file_key, tensor in state.items():
        wanted = wanted_dtype(name_of(file_key), dtypes)
        if wanted is None:
            # Dropped by the loader: the unused camera head, the unused z_from_f heads, the
            # mask token, or a buffer the model builds for itself.
            unmapped += 1
        if wanted == torch.float16 and tensor.dtype == torch.float32:
            copy[file_key] = tensor.half().contiguous()
            halved += 1
        else:
            copy[file_key] = tensor.contiguous()
            kept += 1
    del state
    if halved == 0:
        # Nothing this run would have halved, so a copy would only waste disk.
        target.unlink(missing_ok=True)
        print(f"[fp16]   {path.name:<53} already half precision, not copied", flush=True)
        return 0, True
    save_file(copy, str(target))

    written = load_file(str(target), device="cpu")
    verified = set(written) == set(copy) and all(
        torch.equal(written[key], tensor) for key, tensor in copy.items()
    )
    del written, copy
    print(
        f"[fp16]   {path.name:<53} halve {halved:>4}  keep fp32 {kept:>4}  unmapped {unmapped:>3}  "
        f"{path.stat().st_size / 2**30:5.2f} -> {target.stat().st_size / 2**30:5.2f} GiB  "
        f"{'read back equal' if verified else 'MISMATCH'}",
        flush=True,
    )
    return halved, verified


def convert(config_path: Path, root: Path, out_root: Path) -> int:
    started = time.perf_counter()
    pipeline = UniForge3DImageTo3DPipeline.from_config(
        config_path, weights_dtype=torch.float16, prefetch=False
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    entries: dict[str, dict] = {}
    source_total = copy_total = 0
    verified = True
    for section, group in SECTIONS:
        weights = config[section].get("weights", {})
        name_of = mappings(section)
        files = {
            key: value
            for key, value in weights.items()
            if isinstance(value, str) and key in name_of
        }
        if not files:
            continue

        built = pipeline._models.get(group)
        dtypes = {
            name: tensor.dtype for name, tensor in built[f"{group}_model"].state_dict().items()
        }
        del built

        for key, source in files.items():
            path = Path(source)
            path = path if path.is_absolute() else root / path
            if not path.is_file():
                print(f"[fp16] missing {path}")
                return 1
            relative = path.relative_to(root) if path.is_relative_to(root) else Path(path.name)
            target = out_root / relative.with_suffix(".safetensors")
            target.parent.mkdir(parents=True, exist_ok=True)

            halved, copy_verified = write_half_copy(path, name_of[key], dtypes, target)
            verified = verified and copy_verified
            if not halved:
                continue
            source_total += path.stat().st_size
            copy_total += target.stat().st_size
            copy_relative = (
                str(target.relative_to(root)).replace("\\", "/")
                if target.is_relative_to(root)
                else str(target)
            )
            entries[str(path.relative_to(root)).replace("\\", "/")] = {
                "copy": copy_relative,
                "size": path.stat().st_size,
            }

        pipeline._models.release(group)

    if not entries:
        print("[fp16] no checkpoints needed converting")
        return 1

    (out_root / MANIFEST).write_text(
        json.dumps({"sources": entries}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\n[fp16] {len(entries)} files in {time.perf_counter() - started:.1f}s | "
        f"{source_total / 2**30:.2f} GiB -> {copy_total / 2**30:.2f} GiB, "
        f"saving {(source_total - copy_total) / 2**30:.2f} GiB of disk (originals untouched)"
    )
    if not verified:
        print("[fp16] a copy did not read back equal: not switching the config over")
        return 1
    # One owner for that file: this only asks it for the half-precision preference.
    return make_local_config.generate(prefer_half_precision=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=ROOT / "weights" / "pipeline.local.json",
        help="the config whose weight paths are converted",
    )
    parser.add_argument(
        "--weights-root", type=Path, default=ROOT / "weights",
        help="the directory the config's weight paths are relative to",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="where the copies go (default: fp16/ under the weights root)",
    )
    parser.add_argument(
        "--revert", action="store_true",
        help="delete the copies and point the config back at the originals",
    )
    args = parser.parse_args()

    root = args.weights_root.resolve()
    out_root = (args.out or root / "fp16").resolve()

    if args.revert:
        if out_root.is_dir():
            shutil.rmtree(out_root)
            print(f"[fp16] removed {out_root}")
        else:
            print(f"[fp16] nothing to remove at {out_root}")
        return make_local_config.generate(prefer_half_precision=False)

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"[fp16] no config at {config_path}")
        return 1
    return convert(config_path, root, out_root)


if __name__ == "__main__":
    raise SystemExit(main())

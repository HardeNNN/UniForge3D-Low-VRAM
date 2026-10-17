"""Generate the local-weights pipeline config.

The upstream ``pipeline.json`` mixes two kinds of checkpoint reference: paths inside
the UniForge3D repository, and Hugging Face Hub files (the TRELLIS.2 decoders and the
base DiT). This script rewrites the Hub references to paths inside ``weights/`` and
verifies that every file they name is present, so the runtime never touches the Hub.

    python scripts/make_local_config.py [--check] [--fp32] [--out PATH]

Writes ``weights/pipeline.local.json``, used with::

    python infer.py --config weights/pipeline.local.json

Half-precision copies made by ``scripts/convert_weights_fp16.py`` are preferred when they
exist and still match their sources, so one conversion switches over every tool that reads
that file, launchers included. ``--fp32`` writes the checkpoint-faithful config instead,
which is what ``scripts/check_fp16_weights.py`` and the ``--weights-dtype float32`` and
``auto`` comparisons want.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
SOURCE = ROOT / "uniforge3d" / "configs" / "pipeline.json"
TARGET = WEIGHTS / "pipeline.local.json"
# Where convert_weights_fp16.py keeps the copies and the record of what they came from.
HALF_PRECISION = WEIGHTS / "fp16"
MANIFEST = "manifest.json"

# Hub repository -> directory inside weights/. The UniForge3D repository is mirrored
# as the weights root itself, so its own in-repo paths need no rewriting.
HUB_DIRS = {
    "microsoft/TRELLIS.2-4B": "trellis2-4b",
    "microsoft/TRELLIS-image-large": "trellis-image-large",
    "xibushu/UniForge3D": ".",
}

# Model ids that are passed straight to transformers, not through the weights root.
EXTERNAL_DIRS = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3",
    "briaai/RMBG-2.0": "RMBG-2.0",
}

KEYS = (
    "checkpoint", "checkpoint_path", "adapter_checkpoint", "dit_base", "vggt_base",
    "base_checkpoint", "shape_decoder_checkpoint", "texture_decoder_checkpoint",
)


def _local_path(reference):
    """Rewrite a Hub reference as a weights-relative path; leave strings alone."""
    if not isinstance(reference, dict):
        return reference
    repo, filename = reference["repo_id"], reference["filename"]
    if repo not in HUB_DIRS:
        raise KeyError(f"no local mirror configured for {repo}")
    return f"{HUB_DIRS[repo]}/{filename}"


def _rewrite(node, collected):
    for key, value in node.items():
        if isinstance(value, dict) and "repo_id" not in value:
            _rewrite(value, collected)
            continue
        if key in KEYS:
            local = _local_path(value)
            node[key] = local
            collected.append((f"{key}", local))


def _walk_checkpoints(node, found):
    """Every checkpoint path in the config, including nested weight dictionaries."""
    for key, value in node.items():
        if isinstance(value, dict):
            _walk_checkpoints(value, found)
        elif key in KEYS and isinstance(value, str):
            found.append(value)


def half_precision_copies() -> dict[str, str]:
    """Source-relative path -> copy-relative path, for copies whose source still matches.

    The manifest records each source's size, so a checkpoint that has been downloaded again
    since the conversion is not silently served from an older copy.
    """
    manifest_path = HALF_PRECISION / MANIFEST
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copies: dict[str, str] = {}
    for source, entry in manifest.get("sources", {}).items():
        original = WEIGHTS / source
        copy = WEIGHTS / entry["copy"]
        if not copy.is_file():
            print(f"[config] ignoring {entry['copy']}: not found, re-run the conversion")
            continue
        if not original.is_file() or original.stat().st_size != entry["size"]:
            print(f"[config] ignoring {entry['copy']}: {source} changed since the conversion")
            continue
        copies[source] = entry["copy"]
    return copies


def _use_copies(node, copies, used) -> None:
    for key, value in list(node.items()):
        if isinstance(value, dict):
            _use_copies(value, copies, used)
        elif key in KEYS and isinstance(value, str) and value in copies:
            node[key] = copies[value]
            used.add(value)


def generate(
    *, prefer_half_precision: bool = True, out: Path = TARGET, check_only: bool = False
) -> int:
    """Write the local config, preferring half-precision copies. Returns 1 on missing files."""
    if not SOURCE.is_file():
        raise SystemExit(f"missing upstream config: {SOURCE}")
    config = json.loads(SOURCE.read_text(encoding="utf-8"))
    root = WEIGHTS
    collected: list[tuple[str, str]] = []
    _rewrite(config, collected)

    for key, value in list(config["external_models"].items()):
        directory = EXTERNAL_DIRS.get(value)
        if directory:
            config["external_models"][key] = str((ROOT / "models" / directory).resolve())
    rembg = EXTERNAL_DIRS.get(config["image_preprocess"]["rembg_model"])
    if rembg:
        config["image_preprocess"]["rembg_model"] = str((ROOT / "models" / rembg).resolve())

    copies = half_precision_copies() if prefer_half_precision else {}
    used: set[str] = set()
    _use_copies(config, copies, used)
    # The pipeline reads this to warn when a full-precision request is pointed at the copies.
    config["half_precision_copies"] = bool(used)

    references: list[str] = []
    _walk_checkpoints(config, references)
    missing = []
    for relative in references:
        base = root / relative
        # Decoder references name a stem: <stem>.json describes it, <stem>.safetensors holds it.
        candidates = [base] if base.suffix else [base.with_suffix(".json"), Path(f"{base}.safetensors")]
        missing += [str(path) for path in candidates if not path.is_file()]
    for value in (config["external_models"]["dino"], config["image_preprocess"]["rembg_model"]):
        path = Path(value)
        target = path / "config.json"
        if not target.is_file():
            missing.append(str(target))

    if missing:
        print(f"[config] {len(missing)} weight file(s) not downloaded yet:")
        for path in missing:
            print(f"[config]   {path}")
    if check_only:
        return 1 if missing else 0

    if used:
        print(
            f"[config] {len(used)} of {len(collected)} checkpoint references use the "
            f"half-precision copies in weights/{HALF_PRECISION.name}/"
        )
    else:
        print("[config] using the original checkpoints")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[config] wrote {out}")
    return 1 if missing else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report missing files")
    parser.add_argument(
        "--fp32", action="store_true",
        help="use the original checkpoints even where half-precision copies exist",
    )
    parser.add_argument("--out", type=Path, default=TARGET, help="where to write the config")
    args = parser.parse_args()
    return generate(
        prefer_half_precision=not args.fp32, out=args.out.resolve(), check_only=args.check
    )


if __name__ == "__main__":
    raise SystemExit(main())

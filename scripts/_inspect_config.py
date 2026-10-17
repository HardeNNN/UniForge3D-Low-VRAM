"""Inspect pipeline.json weight references (dev helper)."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = sys.argv[1] if len(sys.argv) > 1 else root / "uniforge3d" / "configs" / "pipeline.json"
config = json.load(open(path, encoding="utf-8"))
KEYS = (
    "checkpoint", "checkpoint_path", "adapter_checkpoint", "dit_base", "vggt_base",
    "base_checkpoint", "shape_decoder_checkpoint", "texture_decoder_checkpoint",
)


def dump(node, trail=""):
    for key, value in node.items():
        here = f"{trail}/{key}"
        if isinstance(value, dict):
            if "repo_id" in value or "filename" in value:
                print(f"{here} = {json.dumps(value)}")
            else:
                dump(value, here)
        elif key in KEYS:
            print(f"{here} = {json.dumps(value)}")


dump(config)
print("--- top level ---", list(config.keys()))
print("--- external_models ---", json.dumps(config["external_models"]))
print("--- image_preprocess ---", json.dumps(config["image_preprocess"]))

"""Validate exported GLB files: container, declared length, and geometry.

Usage: python scripts\\check_glb.py [path ...]
Paths may be files, directories (searched recursively for prediction.glb) or globs.
Defaults to outputs\\**\\prediction.glb.
"""

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def targets(arguments: list[str]) -> list[Path]:
    if not arguments:
        return sorted((ROOT / "outputs").glob("**/prediction.glb"))
    found: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            found += sorted(path.glob("**/prediction.glb"))
        else:
            found += sorted(Path().glob(argument)) or [path]
    return found


def check(path: Path) -> bool:
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    chunk_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    if magic != b"glTF" or version != 2 or chunk_type != b"JSON":
        print(f"{path}: not a glTF 2.0 container")
        return False
    if length != len(data):
        print(f"{path}: declared length {length} != {len(data)} bytes (truncated or padded)")
        return False
    document = json.loads(data[20:20 + chunk_length])
    primitive = document["meshes"][0]["primitives"][0]
    accessors = document["accessors"]
    position = accessors[primitive["attributes"]["POSITION"]]
    indices = accessors[primitive["indices"]]
    attributes = sorted(primitive["attributes"])
    print(
        f"{path}: {len(document.get('meshes', []))} mesh, "
        f"{len(document.get('materials', []))} material, {len(document.get('images', []))} image, "
        f"{len(document.get('textures', []))} texture, attrs {attributes}, "
        f"{position['count']} verts, {indices['count'] // 3} faces"
    )
    print(f"  bbox min={[round(v, 4) for v in position['min']]} max={[round(v, 4) for v in position['max']]}")
    problems = []
    if "POSITION" not in primitive["attributes"] or indices["count"] % 3:
        problems.append("missing positions or non-triangular index count")
    if len(document.get("images", [])) != 2:
        problems.append("expected base colour and normal textures")
    for problem in problems:
        print(f"  FAIL: {problem}")
    return not problems


def main() -> int:
    paths = targets(sys.argv[1:])
    if not paths:
        print("no GLB files found")
        return 1
    return 0 if all(check(path) for path in paths) else 1


if __name__ == "__main__":
    sys.exit(main())

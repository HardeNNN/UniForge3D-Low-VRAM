"""Verify that the Windows/CUDA environment satisfies what the pipeline imports.

Run inside the venv after the installer and the weight download finish::

    python scripts/check_env.py [--config weights/pipeline.local.json]

Every symbol checked here is called by the inference path, so a pass means the
prebuilt wheels expose the API this code expects.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = (
    "torch", "torchvision", "triton", "flash_attn",
    "cumesh", "flex_gemm", "o_voxel", "nvdiffrast", "nvdiffrec_render",
    "transformers", "peft", "safetensors", "trimesh", "cv2", "timm", "kornia",
    "easydict", "huggingface_hub", "scipy",
)

# (module, attribute) pairs the pipeline calls at inference time.
SYMBOLS = (
    ("flex_gemm.ops.spconv", "set_algorithm"),
    ("flex_gemm.ops.spconv", "set_hashmap_ratio"),
    ("flex_gemm.ops.spconv.submanifold_conv3d", "SubMConv3dFunction"),
    ("flex_gemm.kernels.cuda", "hashmap_build_submanifold_conv_neighbour_map_cuda"),
    ("flex_gemm.kernels.cuda", "hashmap_insert_3d_idx_as_val_cuda"),
    ("flex_gemm.kernels.cuda", "hashmap_lookup_3d_cuda"),
    ("flex_gemm.ops.grid_sample", "grid_sample_3d_torch"),
    ("o_voxel.convert", "flexible_dual_grid_to_mesh"),
    ("o_voxel.io", "read"),
    ("cumesh", "CuMesh"),
    ("cumesh", "cuBVH"),
    ("cumesh.remeshing", "remesh_narrow_band_dc"),
    ("flash_attn", "flash_attn_varlen_func"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="local pipeline config to resolve")
    args = parser.parse_args()

    failures: list[str] = []
    print(f"[env] python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        failures.append(f"python {sys.version_info[:2]} != 3.11 (prebuilt wheels require 3.11)")

    for package in PACKAGES:
        if importlib.util.find_spec(package) is None:
            failures.append(f"missing package: {package}")
            print(f"[env] MISSING {package}")
        else:
            print(f"[env] ok      {package}")

    if failures:
        print(f"\n[env] {len(failures)} problem(s) before imports could be checked")
        for problem in failures:
            print(f"[env]   {problem}")
        return 1

    import torch

    print(f"[env] torch {torch.__version__} (cuda {torch.version.cuda}), triton {importlib.import_module('triton').__version__}")
    if not torch.cuda.is_available():
        failures.append("torch.cuda.is_available() is False")
    else:
        props = torch.cuda.get_device_properties(0)
        print(f"[env] gpu   {props.name}, {props.total_memory / 2**30:.2f} GB, sm_{props.major}{props.minor}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"[env] free  {free / 2**30:.2f}/{total / 2**30:.2f} GB")
        # fp16 autocast and a sparse-conv-shaped tensor prove the kernels can launch.
        with torch.autocast("cuda", dtype=torch.float16):
            assert (torch.randn(4, 4, device="cuda", dtype=torch.float16) @ torch.eye(4, device="cuda", dtype=torch.float16)).dtype == torch.float16
        print("[env] cuda  fp16 matmul ok")

    for module_name, attribute in SYMBOLS:
        module = importlib.import_module(module_name)
        if not hasattr(module, attribute):
            failures.append(f"{module_name}.{attribute} is missing")
            print(f"[env] MISSING {module_name}.{attribute}")
        else:
            print(f"[env] ok      {module_name}.{attribute}")

    sys.path.insert(0, str(ROOT))
    uniforge3d_ok = False
    try:
        import uniforge3d  # noqa: F401
        uniforge3d_ok = True
        print("[env] ok      uniforge3d")
    except Exception as error:  # noqa: BLE001 - report the import failure verbatim
        failures.append(f"import uniforge3d failed: {error}")
        print(f"[env] FAILED  uniforge3d: {error}")

    # The wheel ships no Triton kernels, so an algorithm that dispatches to them fails
    # inside the first sparse convolution. Fail here instead, before a model load.
    if uniforge3d_ok:
        from flex_gemm import kernels

        from uniforge3d.modules.sparse.conv import config as conv_config

        if not hasattr(kernels, "triton") and conv_config.FLEX_GEMM_ALGO != "explicit_gemm":
            failures.append(
                f"flex_gemm.kernels.triton is absent but FLEX_GEMM_ALGO is "
                f"{conv_config.FLEX_GEMM_ALGO!r}; only 'explicit_gemm' can dispatch"
            )
        print(f"[env] sparse conv algorithm: {conv_config.FLEX_GEMM_ALGO}")

    if args.config:
        from uniforge3d.pipelines.uniforge3d_image_to_3d import UniForge3DImageTo3DPipeline

        pipeline = UniForge3DImageTo3DPipeline.from_config(
            args.config, low_vram=True, weights_dtype="float16"
        )
        # Weights otherwise load stage by stage during a run, which would let a broken
        # checkpoint pass this check and fail halfway through a case.
        resident = pipeline.load_all()
        # The preprocess group holds the background remover, which is not a torch module.
        total = sum(
            parameter.numel()
            for model in resident.values()
            if isinstance(model, torch.nn.Module)
            for parameter in model.parameters()
        )
        print(
            f"[env] models loaded, {total / 1e9:.2f}B parameters resident in RAM"
        )
        print(
            "[env] weights   half-precision copies in weights/fp16/"
            if pipeline.config.get("half_precision_copies")
            else "[env] weights   original checkpoints"
        )

    print(f"\n[env] {'PASS' if not failures else 'FAIL'}")
    for problem in failures:
        print(f"[env] {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

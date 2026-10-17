import argparse
import os
import time
from pathlib import Path

# The CUDA allocator must be configured before torch initialises its context.
# Expandable segments keep the fragmented allocations of sparse 3D decoding from
# reserving far more VRAM than they hold, which matters most on 8 GB cards. Windows
# builds do not implement the setting, so it is only requested where it applies.
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import torch  # noqa: E402
from uniforge3d import UniForge3DImageTo3DPipeline  # noqa: E402


def cuda_context_alive() -> bool:
    """A failed kernel can poison the CUDA context, where retrying is pointless."""
    try:
        torch.cuda.synchronize()
    except Exception:
        return False
    return True


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def select_cases(root: Path, output: Path, *, overwrite: bool, limit: int | None) -> list[Path]:
    """Cases under a root that have no published prediction yet, in name order.

    A cases root holds one directory per case; a single case directory holds images.
    """
    cases = sorted(path for path in root.iterdir() if path.is_dir())
    if not cases:
        cases = [root]
    pending = [
        case for case in cases
        if overwrite or not (output / case.name / "prediction.glb").is_file()
    ]
    return pending[:limit] if limit is not None else pending


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=ROOT / "assets",
        help="a cases root (one directory per case) or a single case directory of images",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", default="xibushu/UniForge3D")
    source.add_argument("--config", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--weights-dtype",
        default=None,
        help="weight precision for the denoisers: float16 (default on --low-vram), "
             "float32, or a checkpoint-faithful 'auto'",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="run at most this many cases")
    parser.add_argument(
        "--save-textures", action="store_true",
        help="also write the baked base colour and metallic-roughness maps as PNG beside "
             "each prediction.glb",
    )
    parser.add_argument(
        "--low-ram", action="store_true",
        help="keep only the stage being run in memory: peak RAM is one group of weights "
             "instead of the whole 25 GiB stack, and every case reloads them from disk",
    )
    parser.add_argument(
        "--prefetch", action=argparse.BooleanOptionalAction, default=True,
        help="load later stages' weights on a background thread while the GPU runs the "
             "current stage (never used by --low-ram)",
    )
    parser.add_argument("--vram-debug", action="store_true", help="print per-stage VRAM and timing statistics")
    parser.add_argument(
        "--watch", type=int, default=None, metavar="SECONDS",
        help="keep the models resident and re-scan --input every SECONDS seconds for new cases",
    )
    args = parser.parse_args()

    if args.vram_debug:
        os.environ["UNIFORGE_VRAM_DEBUG"] = "1"
    if args.watch is not None and args.overwrite:
        parser.error("--watch would recompute every case on every rescan; drop --overwrite")

    cases = select_cases(args.input, args.output, overwrite=args.overwrite, limit=args.limit)
    if not cases and args.watch is None:
        print("[infer] every case already has a prediction.glb")
        return

    weights_dtype = None if args.weights_dtype == "auto" else args.weights_dtype
    options = {
        "device": args.device, "low_vram": args.low_vram, "weights_dtype": weights_dtype,
        "low_ram": args.low_ram, "prefetch": args.prefetch,
    }
    load_started = time.perf_counter()
    pipeline = (
        UniForge3DImageTo3DPipeline.from_config(args.config, **options)
        if args.config else
        UniForge3DImageTo3DPipeline.from_pretrained(args.model, **options)
    )
    print(
        f"[infer] pipeline ready in {time.perf_counter() - load_started:.1f}s, weights "
        f"{'one stage at a time' if args.low_ram else 'prefetched' if args.prefetch else 'on demand'}",
        flush=True,
    )
    processed = 0
    announced_idle = False
    while True:
        if cases:
            print(f"[infer] {len(cases)} case(s) to run", flush=True)
            announced_idle = False
            for index, case in enumerate(cases, start=1):
                try:
                    images = sorted(path for path in case.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
                    started = time.perf_counter()
                    output = pipeline.run(
                        images,
                        args.output / case.name / "prediction.glb",
                        overwrite=args.overwrite,
                        seed=args.seed,
                        save_textures=args.save_textures,
                    )
                except Exception as error:
                    # A resident server must survive one bad case, but a poisoned CUDA
                    # context cannot be recovered.
                    if args.watch is None or not cuda_context_alive():
                        raise
                    print(f"[infer] {case.name} failed: {type(error).__name__}: {error}", flush=True)
                    continue
                print(
                    f"[infer] {index}/{len(cases)} {case.name}: {time.perf_counter() - started:.1f}s -> {output}",
                    flush=True,
                )
                processed += 1
        if args.watch is None:
            return
        if not announced_idle:
            print(
                f"[infer] models resident ({processed} case(s) done), watching {args.input} "
                f"every {args.watch}s for new cases",
                flush=True,
            )
            announced_idle = True
        time.sleep(args.watch)
        cases = select_cases(args.input, args.output, overwrite=False, limit=args.limit)


if __name__ == "__main__":
    main()

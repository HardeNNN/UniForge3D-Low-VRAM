"""Multi-connection weight downloader with stall detection and per-file resume.

The Hub and pip clients stalled on long transfers over this link, so every file is
split into independent HTTP range chunks. A chunk that stops making progress is
aborted and retried from the byte it actually reached, so a stalled connection
costs seconds instead of the whole transfer.

Examples:
    python scripts/fetch.py weights          # hub weights into weights/
    python scripts/fetch.py wheels           # torch + flash-attn wheels into whl/
    python scripts/fetch.py extras           # licence-gated zips into models/
    python scripts/fetch.py all --jobs 16
    python scripts/fetch.py status           # what is complete, partial, missing
    python scripts/fetch.py import           # move files from another download client into place
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HF = "https://huggingface.co/{repo}/resolve/{revision}/{path}"

# Wall-clock seconds a chunk may go without receiving a byte before it is retried.
STALL_SECONDS = 45


class Job:
    """One file to fetch, split into ranges written to sibling part files."""

    MAX_PARTS = 128

    def __init__(self, url: str, dest: Path, size: int, *, chunk_mb: int = 16, post: str = "none"):
        self.url = url
        self.dest = dest
        self.size = int(size)
        self.chunk_mb = chunk_mb
        self.post = post
        self.chunk_bytes = self._plan_chunk_bytes(chunk_mb)
        self.parts = max(1, -(-self.size // self.chunk_bytes))
        self.done = [0] * self.parts
        self.finished = False
        self.lock = threading.Lock()

    def _plan_chunk_bytes(self, chunk_mb: int) -> int:
        """Chunk size that still covers the whole file within the part budget."""
        requested = max(1, int(chunk_mb)) * 1024 * 1024
        parts = min(self.MAX_PARTS, max(1, -(-self.size // requested)))
        return max(1, -(-self.size // parts))

    def chunk_bounds(self, index: int) -> tuple[int, int]:
        start = index * self.chunk_bytes
        return start, min(start + self.chunk_bytes, self.size) - 1

    def part_path(self, index: int) -> Path:
        return self.dest.with_name(f"{self.dest.name}.part{index:03d}")

    def existing_bytes(self) -> int:
        total = 0
        for index in range(self.parts):
            path = self.part_path(index)
            if path.exists():
                total += path.stat().st_size
        return total


class Manifest:
    """Aggregate progress reporting across every job."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: list[Job] = []
        self.bytes_done = 0
        self.started = time.time()
        self.last_print = 0.0
        self.errors: list[str] = []

    def add(self, job: Job) -> None:
        self.jobs.append(job)

    def report(self, force: bool = False) -> None:
        now = time.time()
        with self.lock:
            if not force and now - self.last_print < 20:
                return
            self.last_print = now
            total = sum(j.size for j in self.jobs)
            done = sum(j.existing_bytes() for j in self.jobs)
            elapsed = max(1e-6, now - self.started)
            rate = done / elapsed / 1024 / 1024
            eta = f"{(total - done) / (rate * 1024 * 1024) / 60:.0f} min" if rate > 0.05 else "--"
            print(
                f"[fetch] {done / 1e9:.2f}/{total / 1e9:.2f} GB "
                f"({100 * done / max(1, total):.1f}%) {rate:.2f} MB/s eta {eta}",
                flush=True,
            )


def download_chunk(job: Job, index: int, retries: int, timeout: int, stats: Manifest) -> bool:
    start, end = job.chunk_bounds(index)
    part = job.part_path(index)
    attempt = 0
    while attempt <= retries:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        if have >= end - start + 1:
            return True
        request = urllib.request.Request(
            job.url, headers={"Range": f"bytes={start + have}-{end}", "User-Agent": "uniforge3d-lowvram/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status not in (200, 206):
                    raise OSError(f"HTTP {response.status}")
                mode = "ab" if have else "wb"
                last_data = time.time()
                with open(part, mode) as handle:
                    while True:
                        block = response.read(1024 * 512)
                        if not block:
                            break
                        handle.write(block)
                        last_data = time.time()
                        if time.time() - last_data > STALL_SECONDS:
                            raise TimeoutError("stalled")
            if part.stat().st_size >= end - start + 1:
                return True
            raise OSError(f"short read {part.stat().st_size}/{end - start + 1}")
        except KeyboardInterrupt:
            raise
        except Exception as error:  # noqa: BLE001 - any transport failure is retryable
            if attempt > retries:
                stats.errors.append(f"{job.dest.name} part {index}: {error}")
                return False
            time.sleep(min(15, 1.5 * attempt))
    return False


def fetch_job(job: Job, pool: ThreadPoolExecutor, args: argparse.Namespace, stats: Manifest) -> bool:
    job.dest.parent.mkdir(parents=True, exist_ok=True)
    if job.dest.exists() and job.dest.stat().st_size == job.size:
        stats.report()
        return True

    print(
        f"[fetch] {job.dest.relative_to(ROOT)} {job.size / 1e6:.0f} MB in {job.parts} parts "
        f"({job.existing_bytes() / 1e6:.0f} MB already on disk)",
        flush=True,
    )
    futures = [pool.submit(download_chunk, job, i, args.retries, args.timeout, stats) for i in range(job.parts)]
    ok = all(future.result() for future in futures)
    if not ok:
        return False

    # Assemble atomically: a half-written destination must never look complete, and the
    # parts are only removed once the finished file is in place.
    temporary = job.dest.with_name(f"{job.dest.name}.assembling")
    with open(temporary, "wb") as out:
        for index in range(job.parts):
            part = job.part_path(index)
            with open(part, "rb") as handle:
                while True:
                    block = handle.read(1 << 22)
                    if not block:
                        break
                    out.write(block)
    actual = temporary.stat().st_size
    if actual != job.size:
        temporary.unlink(missing_ok=True)
        stats.errors.append(f"{job.dest.name}: assembled {actual} != expected {job.size}")
        return False
    os.replace(temporary, job.dest)
    for index in range(job.parts):
        job.part_path(index).unlink(missing_ok=True)
    print(f"[fetch] done {job.dest.relative_to(ROOT)} ({actual / 1e6:.0f} MB)", flush=True)
    return True


# ---------------------------------------------------------------- job groups

HUB_FILES = {
    ("xibushu/UniForge3D", "main"): [
        "pipeline.json",
        "uniforge3d/stage1/uniforge3d_stage1_structure_joint_model.pt",
        "uniforge3d/shape512/uniforge3d_stage2_shape512_denoiser.pt",
        "uniforge3d/shape512/uniforge3d_stage2_shape512_vggt_adapter.pt",
        "uniforge3d/shape1024/uniforge3d_stage2_shape1024_denoiser.pt",
        "uniforge3d/shape1024/uniforge3d_stage2_shape1024_vggt_adapter.pt",
        "uniforge3d/texture/uniforge3d_stage3_texture1024_denoiser.pt",
        "uniforge3d/texture/uniforge3d_stage3_texture1024_vggt_adapter.pt",
        "uniforge3d/vggt/uniforge3d_shape_texture_vggt_conditioner.pt",
        "uniforge3d/vggt/uniforge3d_stage1_structure_vggt_backbone.pt",
    ],
    ("microsoft/TRELLIS.2-4B", "af44b45f2e35a493886929c6d786e563ec68364d"): [
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/shape_dec_next_dc_f16c32_fp16.json",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/tex_dec_next_dc_f16c32_fp16.json",
    ],
    ("microsoft/TRELLIS-image-large", "25e0d31ffbebe4b5a97464dd851910efc3002d96"): [
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "ckpts/ss_dec_conv3d_16l8_fp16.json",
    ],
}

HUB_DEST = {
    "xibushu/UniForge3D": ROOT / "weights",
    "microsoft/TRELLIS.2-4B": ROOT / "weights" / "trellis2-4b",
    "microsoft/TRELLIS-image-large": ROOT / "weights" / "trellis-image-large",
}

WHEEL_FILES = [
    # (url, dest). Sizes always come from the server, so a truncated transfer cannot pass.
    (
        "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-win_amd64.whl",
        ROOT / "whl" / "torch-2.8.0+cu128-cp311-cp311-win_amd64.whl",
    ),
    (
        "https://download.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp311-cp311-win_amd64.whl",
        ROOT / "whl" / "torchvision-0.23.0+cu128-cp311-cp311-win_amd64.whl",
    ),
    (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/"
        "flash_attn-2.8.2%2Bcu128torch2.8-cp311-cp311-win_amd64.whl",
        ROOT / "whl" / "flash_attn-2.8.2+cu128torch2.8-cp311-cp311-win_amd64.whl",
    ),
]

# The CUDA extensions are the prebuilt cp311 wheels that the TRELLIS.2-stableprojectorz project
# publishes in its own whl/ directory. They need no nvcc and no build step, and nothing else
# publishes them, so they are fetched from that repository by path.
EXTENSION_WHEEL_NAMES = (
    "cumesh-0.0.1-cp311-cp311-win_amd64.whl",
    "flex_gemm-0.0.1-cp311-cp311-win_amd64.whl",
    "o_voxel-0.0.1-cp311-cp311-win_amd64.whl",
    "nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl",
    "nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl",
)
EXTENSION_WHEELS = [
    (
        "https://raw.githubusercontent.com/IgorAherne/TRELLIS.2-stableprojectorz/main/whl/"
        f"{name}",
        ROOT / "whl" / name,
    )
    for name in EXTENSION_WHEEL_NAMES
]

EXTRA_ZIPS = [
    (
        "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/dinov3.zip",
        ROOT / "models" / "dinov3.zip",
        "dinov3",
    ),
    (
        "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/RMBG-2.0.zip",
        ROOT / "models" / "RMBG-2.0.zip",
        "RMBG-2.0",
    ),
]


def hub_sizes(repo: str, revision: str, paths: list[str]) -> dict[str, int]:
    """Ask the Hub for authoritative file sizes."""
    url = f"https://huggingface.co/api/models/{repo}?blobs=true&revision={revision}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    known = {sibling["rfilename"]: sibling.get("size", 0) for sibling in payload["siblings"]}
    missing = [path for path in paths if not known.get(path)]
    if missing:
        raise RuntimeError(f"{repo}: missing sizes for {missing}")
    return {path: int(known[path]) for path in paths}


def head_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "uniforge3d-lowvram/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers.get("Content-Length", 0))


def pypi_wheel_url(package: str, version: str) -> str:
    """Resolve the Windows wheel URL for a package from the PyPI JSON API.

    Prefers the wheel built for the interpreter running this script; compiled
    Windows packages publish one cpXY wheel per Python version.
    """
    tag = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=60) as response:
        payload = json.load(response)
    wheels = [f for f in payload["urls"] if f["filename"].endswith("win_amd64.whl")]
    for file in wheels:
        if tag in file["filename"]:
            return file["url"]
    if not wheels:
        raise RuntimeError(f"{package} {version}: no Windows wheel on PyPI")
    return wheels[0]["url"]


def build_jobs(group: str, manifest: Manifest, chunk_mb: int) -> list[Job]:
    jobs: list[Job] = []
    if group in ("weights", "all"):
        for (repo, revision), paths in HUB_FILES.items():
            sizes = hub_sizes(repo, revision, paths)
            dest_root = HUB_DEST[repo]
            for path in paths:
                jobs.append(
                    Job(
                        HF.format(repo=repo, revision=revision, path=path),
                        dest_root / path,
                        sizes[path],
                        chunk_mb=chunk_mb,
                    )
                )
    if group in ("wheels", "all"):
        # Every wheel here is cp311, and the triton wheel is resolved by the running
        # interpreter's own tag, so a different Python would assemble an environment that
        # cannot run. Say so now rather than 3 GB into the download.
        if sys.version_info[:2] != (3, 11):
            raise SystemExit(
                "these wheels are cp311: run this with Python 3.11, not "
                f"{sys.version_info[0]}.{sys.version_info[1]} ({sys.executable})"
            )
        for url, dest in WHEEL_FILES + EXTENSION_WHEELS:
            jobs.append(Job(url, dest, head_size(url), chunk_mb=chunk_mb))
        triton_url = pypi_wheel_url("triton-windows", "3.4.0.post21")
        jobs.append(Job(triton_url, ROOT / "whl" / triton_url.rsplit("/", 1)[-1], head_size(triton_url)))
    if group in ("extras", "all"):
        for url, dest, member in EXTRA_ZIPS:
            # The archive is only a delivery mechanism: once its directory exists the
            # artifact is present and nothing needs fetching. A leftover archive (for
            # example from an interrupted run) is still kept so it gets extracted.
            if (ROOT / "models" / member).is_dir() and not dest.exists():
                continue
            jobs.append(Job(url, dest, head_size(url), chunk_mb=chunk_mb))
    for job in jobs:
        manifest.add(job)
    return jobs


def _local_bytes(job: Job) -> int:
    """Bytes already on disk for this job, whether assembled or still in parts."""
    if job.dest.is_file():
        return job.dest.stat().st_size
    return sum(
        path.stat().st_size
        for path in (job.part_path(index) for index in range(job.parts))
        if path.is_file()
    )


def report_status(jobs: list[Job]) -> int:
    """Print what is complete, partial and missing, before fetching anything."""
    complete = partial = missing = 0
    total = done = 0
    for job in sorted(jobs, key=lambda item: -item.size):
        local = _local_bytes(job)
        total += job.size
        done += min(local, job.size)
        if local == job.size and (job.size > 0 or job.dest.is_file()):
            state, complete = "complete", complete + 1
        elif local:
            state, partial = f"partial {100 * local / job.size:.0f}%", partial + 1
        else:
            state, missing = "missing", missing + 1
        print(
            f"[status] {state:>12}  {local / 1e6:9.1f}/{job.size / 1e6:9.1f} MB  "
            f"{job.dest.relative_to(ROOT)}",
            flush=True,
        )
    print(f"[status] {complete} complete, {partial} partial, {missing} missing", flush=True)
    print(
        f"[status] {done / 1e9:.2f}/{total / 1e9:.2f} GB on disk "
        f"({100 * done / max(1, total):.1f}%); {(total - done) / 1e9:.2f} GB to fetch",
        flush=True,
    )
    return 0


def import_local(source: Path, *, copy: bool = False) -> int:
    """Move files fetched by another client into their manifest destinations.

    Every file is matched by name against the manifest and verified by exact byte size,
    so a truncated or proxied download cannot enter the tree. Anything that does not
    match a manifest entry is reported and left alone.
    """
    if not source.is_dir():
        print(f"[import] no such directory: {source}", file=sys.stderr)
        return 1
    jobs = build_jobs("weights", Manifest(), 16)
    by_name = {job.dest.name: job for job in jobs}
    imported = skipped = 0
    mismatched: list[str] = []
    unknown: list[str] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        if path.name.endswith((".part", ".crdownload", ".tmp", ".assembling")):
            continue
        job = by_name.get(path.name)
        if job is None:
            unknown.append(path.name)
            continue
        actual = path.stat().st_size
        if actual != job.size:
            mismatched.append(f"{path.name}: {actual} bytes on disk, {job.size} expected")
            continue
        if job.dest.is_file() and job.dest.stat().st_size == job.size:
            skipped += 1
            continue
        job.dest.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(path, job.dest)
        else:
            shutil.move(str(path), str(job.dest))
        imported += 1
        print(f"[import] {job.dest.relative_to(ROOT)}  ({actual / 1e6:.1f} MB)", flush=True)

    print(f"[import] {imported} imported, {skipped} already complete", flush=True)
    for name in unknown:
        print(f"[import] not in the manifest, left alone: {name}", flush=True)
    for problem in mismatched:
        print(f"[import] SIZE MISMATCH {problem}", file=sys.stderr)
    return 1 if mismatched else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group", choices=["weights", "wheels", "extras", "all", "status", "import"],
    )
    parser.add_argument("--jobs", type=int, default=1, help="files fetched concurrently")
    parser.add_argument("--connections", type=int, default=12, help="range chunks in flight")
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="fetch only files whose destination contains this substring (repeatable), "
             "so an external download client can take the rest of the group without overlap",
    )
    parser.add_argument(
        "--source", type=Path, default=Path.home() / "Downloads",
        help="directory to import from (group=import)",
    )
    parser.add_argument(
        "--copy", action="store_true", help="copy instead of moving (group=import)",
    )
    args = parser.parse_args()

    if args.group == "status":
        return report_status(build_jobs("all", Manifest(), args.chunk_mb))
    if args.group == "import":
        return import_local(args.source, copy=args.copy)

    manifest = Manifest()
    jobs = build_jobs(args.group, manifest, args.chunk_mb)
    if args.only:
        jobs = [
            job for job in jobs
            if any(token in str(job.dest) for token in args.only)
        ]
        if not jobs:
            print(f"[fetch] --only matched no file in group {args.group}", file=sys.stderr)
            return 1
    # Biggest first: the long pole starts immediately.
    jobs.sort(key=lambda job: -job.size)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.connections) as pool:
        if args.jobs <= 1:
            for job in jobs:
                if not fetch_job(job, pool, args, manifest):
                    failures += 1
                manifest.report(force=True)
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as file_pool:
                results = [file_pool.submit(fetch_job, job, pool, args, manifest) for job in jobs]
                for result in results:
                    if not result.result():
                        failures += 1

    manifest.report(force=True)
    if args.group in ("extras", "all") and failures == 0:
        for _url, dest, member in EXTRA_ZIPS:
            if not dest.exists():
                continue
            print(f"[fetch] extracting {dest.name} -> models/{member}", flush=True)
            with zipfile.ZipFile(dest) as archive:
                archive.extractall(ROOT / "models")
            dest.unlink()
    for error in manifest.errors:
        print(f"[fetch] ERROR {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download the Hugging Face weights UniForge3D needs into the local HF cache.

Resumable: interrupted runs restart from the partial files already on disk.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download  # noqa: E402

TRELLIS2_REV = "af44b45f2e35a493886929c6d786e563ec68364d"
TRELLIS_IMAGE_LARGE_REV = "25e0d31ffbebe4b5a97464dd851910efc3002d96"

# (repo_id, revision, allow_patterns)
REPOS: list[tuple[str, str | None, list[str] | None]] = [
    ("xibushu/UniForge3D", None, ["pipeline.json", "uniforge3d/**/*.pt"]),
    (
        "microsoft/TRELLIS.2-4B",
        TRELLIS2_REV,
        [
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.*",
            "ckpts/shape_dec_next_dc_f16c32_fp16.*",
            "ckpts/tex_dec_next_dc_f16c32_fp16.*",
        ],
    ),
    (
        "microsoft/TRELLIS-image-large",
        TRELLIS_IMAGE_LARGE_REV,
        ["ckpts/ss_dec_conv3d_16l8_fp16.*"],
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--workers", type=int, default=8, help="parallel file downloads")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--repo", action="append", default=None, help="limit to a repo id")
    args = parser.parse_args()

    repos = [r for r in REPOS if not args.repo or r[0] in args.repo]
    failed: list[str] = []

    for repo_id, revision, patterns in repos:
        start = time.time()
        print(f"[weights] {repo_id}@{revision or 'main'} -> {args.cache_dir or 'default cache'}", flush=True)
        for attempt in range(1, args.retries + 1):
            try:
                path = snapshot_download(
                    repo_id,
                    revision=revision,
                    cache_dir=args.cache_dir,
                    allow_patterns=patterns,
                    max_workers=args.workers,
                )
                print(f"[weights] {repo_id} ready in {time.time() - start:.0f}s: {path}", flush=True)
                break
            except KeyboardInterrupt:
                raise
            except Exception as error:  # noqa: BLE001 - retry any transport failure
                print(f"[weights] {repo_id} attempt {attempt}/{args.retries} failed: {error}", flush=True)
                if attempt == args.retries:
                    failed.append(repo_id)
                else:
                    time.sleep(min(60, 5 * attempt))

    if failed:
        print(f"[weights] FAILED: {failed}", file=sys.stderr)
        return 1
    print("[weights] all repositories downloaded", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""List model URLs for an external download client (IDM, aria2, a proxy).

Modes:
  --all       every file with its destination and server-side size (default)
  --missing   files that are not complete yet (includes partially fetched ones)
  --absent    files with no bytes on disk at all: the ones only the external client
              can usefully take, so it cannot collide with a resumable part file

Print a batch list for IDM with::

    python scripts\\_list_urls.py --absent > weights-urls.txt

Then place each file at the destination printed by --all and run
`python scripts/fetch.py weights` to verify sizes and finish anything short.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch  # noqa: E402


def keep(job, mode: str) -> bool:
    """Whether this job belongs in the list for the requested mode."""
    local = fetch._local_bytes(job)
    if mode == "absent":
        return local == 0
    if mode == "missing":
        complete = local == job.size and (job.size > 0 or job.dest.is_file())
        return not complete
    return True


def job_group(group: str, mode: str) -> list:
    jobs = fetch.build_jobs(group, fetch.Manifest(), 16)
    return [job for job in jobs if keep(job, mode)]


def main() -> int:
    mode = "all"
    for candidate in ("all", "missing", "absent"):
        if f"--{candidate}" in sys.argv:
            mode = candidate
    jobs = job_group("weights", mode) + job_group("extras", mode)
    for job in sorted(jobs, key=lambda item: -item.size):
        if mode != "all":
            print(job.url)
            continue
        local = fetch._local_bytes(job)
        note = f"  [{local / 1e6:.1f} MB already on disk]" if local else ""
        print(f"{job.size / 1e6:9.1f} MB  {job.dest.relative_to(fetch.ROOT)}{note}")
        print(f"{'':9}     {job.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

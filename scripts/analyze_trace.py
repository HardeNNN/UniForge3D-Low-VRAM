"""Summarise a chrome trace written by scripts\\profile_sampling.py.

`key_averages()` double counts device time: a CUDA kernel shows up once on its owning
operator and again as its own `kernel` record, so the profiler report's "summed device
time" is an upper bound. This script reads the trace and sums only real GPU work
(`kernel`, `gpu_memcpy`, `gpu_memset`), merges the intervals per stream to get true busy
time and gap structure, and buckets all of it by time so stage boundaries are visible.

Usage: python scripts\\analyze_trace.py logs\\sampling_trace.json [--bucket 10]
"""

import argparse
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["traceEvents"]


def merge(intervals: list[tuple[int, int]]) -> int:
    """Union length of [start, end) intervals, in microseconds."""
    total = 0
    start = end = None
    for lo, hi in sorted(intervals):
        if end is None or lo > end:
            if end is not None:
                total += end - start
            start, end = lo, hi
        else:
            end = max(end, hi)
    return total + (end - start if end is not None else 0)


def span_index(events: list[dict]) -> tuple[list[tuple[float, float, str]], list[float]]:
    """CPU operator spans, sorted by start, plus their start times for bisecting."""
    spans = sorted(
        (event["ts"], event["ts"] + event.get("dur", 0), event["name"])
        for event in events
        if event.get("cat") == "cpu_op"
    )
    return spans, [span[0] for span in spans]


def owning_op(spans: list[tuple[float, float, str]], starts: list[float], ts: float) -> str | None:
    """Innermost CPU operator whose span contains `ts`, found by walking back from it."""
    index = bisect_right(starts, ts) - 1
    for _ in range(2000):
        if index < 0:
            return None
        start, end, name = spans[index]
        if end > ts:
            return name
        index -= 1
    return None


def attribute_copies(
    events: list[dict], spans: list[tuple[float, float, str]], starts: list[float]
) -> Counter[tuple[str, str]]:
    """Count host<->device copies per owning operator.

    A pageable copy blocks the thread that launched it, so the innermost operator whose span
    contains the copy is the call site.
    """
    owners: Counter[tuple[str, str]] = Counter()
    for event in events:
        if event.get("cat") != "gpu_memcpy":
            continue
        direction = "HtoD" if "HtoD" in event["name"] else "DtoH"
        name = owning_op(spans, starts, event["ts"])
        if name is not None:
            owners[(name, direction)] += 1
    return owners


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--bucket", type=float, default=10.0, help="bucket width in seconds")
    args = parser.parse_args()

    events = load(args.trace)
    print(f"{len(events)} events from {args.trace}")
    spans, starts = span_index(events)

    per_stream: dict[int, list[tuple[float, float]]] = defaultdict(list)
    by_name: Counter[str] = Counter()
    name_time: Counter[str] = Counter()
    buckets: Counter[int] = Counter()
    cpu_only: Counter[str] = Counter()
    cpu_count: Counter[str] = Counter()
    bucket_name: dict[int, Counter[str]] = defaultdict(Counter)
    width = args.bucket * 1e6
    lo = min(event["ts"] for event in events)
    hi = max(event["ts"] + event.get("dur", 0) for event in events)

    for event in events:
        cat = event.get("cat")
        dur = event.get("dur", 0)
        if cat in GPU_CATS:
            start, end = event["ts"], event["ts"] + dur
            per_stream[event["tid"]].append((start, end))
            by_name[event["name"]] += 1
            name_time[event["name"]] += dur
            index = int((start - lo) // width)
            buckets[index] += dur
            bucket_name[index][event["name"]] += dur
        elif cat == "cpu_op":
            cpu_only[event["name"]] += dur
            cpu_count[event["name"]] += 1

    span = hi - lo
    busy = merge([interval for intervals in per_stream.values() for interval in intervals])
    print(f"\nspan {span / 1e6:.1f}s   gpu busy (union) {busy / 1e6:.1f}s = {100 * busy / span:.0f}%")
    for tid, intervals in sorted(per_stream.items(), key=lambda row: -merge(row[1])):
        print(f"  stream {tid:>4}  busy {merge(intervals) / 1e6:7.1f}s  n={len(intervals):<7}")

    print(f"\n--- gpu time by name (top 20 of {len(by_name)}) ---")
    for name, total in name_time.most_common(20):
        print(f"{total / 1e6:8.2f}s {100 * total / busy:5.1f}%  n={by_name[name]:<7} {name[:100]}")

    print(f"\n--- top 20 cpu_op self time (of {len(cpu_only)}) ---")
    for name, total in cpu_only.most_common(20):
        print(f"{total / 1e6:8.2f}s  n={cpu_count[name]:<7} {name[:100]}")

    print("\n--- every memcpy and memset kernel by name ---")
    for name, count in by_name.most_common():
        if "Memcpy" in name or "Memset" in name:
            print(f"n={count:<7} {name_time[name] / 1e6:7.2f}s  {name}")

    owners = attribute_copies(events, spans, starts)
    per_op: Counter[str] = Counter()
    splits: dict[str, Counter[str]] = defaultdict(Counter)
    for (name, direction), count in owners.items():
        per_op[name] += count
        splits[name][direction] += count
    print(f"\n--- host<->device copies by owning cpu_op (top 15 of {len(per_op)}) ---")
    for name, count in per_op.most_common(15):
        split = " ".join(f"{direction}={splits[name][direction]}" for direction in ("HtoD", "DtoH"))
        print(f"n={count:<7} {split:<20} {name[:90]}")

    for wanted in ("cudaStreamSynchronize", "cudaMalloc", "cudaFree"):
        by_owner: Counter[str] = Counter()
        for event in events:
            if event.get("cat") == "cuda_runtime" and event["name"] == wanted:
                name = owning_op(spans, starts, event["ts"])
                if name is not None:
                    by_owner[name] += 1
        if not by_owner:
            continue
        print(f"\n--- {wanted} by owning cpu_op (top 8 of {len(by_owner)}) ---")
        for name, count in by_owner.most_common(8):
            print(f"n={count:<7} {name[:90]}")

    runtime: Counter[str] = Counter()
    runtime_time: Counter[str] = Counter()
    for event in events:
        if event.get("cat") == "cuda_runtime":
            runtime[event["name"]] += 1
            runtime_time[event["name"]] += event.get("dur", 0)
    print(f"\n--- cuda_runtime calls (top 10 of {len(runtime)}) ---")
    for name, count in runtime.most_common(10):
        print(f"n={count:<7} {runtime_time[name] / 1e6:7.2f}s  {name}")

    print(f"\n--- copy-owning ops by {args.bucket:g}s bucket (counts) ---")
    for name, _ in per_op.most_common(4):
        counts: Counter[int] = Counter()
        for event in events:
            if event.get("cat") == "cpu_op" and event["name"] == name:
                counts[int((event["ts"] - lo) // width)] += 1
        cells = " ".join(f"{index * args.bucket:.0f}s:{counts[index]}" for index in sorted(counts))
        print(f"{name[:28]:28} {cells}")

    print(f"\n--- copies and syncs by {args.bucket:g}s bucket ---")
    for name in (
        "Memcpy HtoD (Pageable -> Device)",
        "Memcpy DtoH (Device -> Pageable)",
        "Memcpy DtoH (Device -> Pinned)",
        "cudaStreamSynchronize",
        "cudaMemcpyAsync",
        "cudaMalloc",
    ):
        category = "cuda_runtime" if name.startswith("cuda") else "gpu_memcpy"
        counts: Counter[int] = Counter()
        for event in events:
            if event.get("cat") == category and event["name"] == name:
                counts[int((event["ts"] - lo) // width)] += 1
        cells = " ".join(f"{index * args.bucket:.0f}s:{counts[index]}" for index in sorted(counts))
        print(f"{name[:28]:28} {cells}")

    print(f"\n--- gpu busy per {args.bucket:g}s bucket ---")
    for index in sorted(buckets):
        fill = int(40 * buckets[index] / width)
        top = ", ".join(name[:38] for name, _ in bucket_name[index].most_common(2))
        print(f"{index * args.bucket:7.1f}s {buckets[index] / width:5.0%} {'#' * fill} {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

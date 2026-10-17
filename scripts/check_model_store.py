"""Check the stage model store: lazy builds, prefetch order, release, and single-build races.

``StageModelStore`` holds one group of models per pipeline stage, and builds them either when
a stage asks for them or ahead of time on a worker thread. The pipeline depends on three
behaviours that a real weight load cannot demonstrate cheaply: a group is only built when
something asks for it or the worker reaches it, a release really drops it, and two callers
that ask for the same group at once share one build instead of starting a second.

    .venv\\Scripts\\python.exe scripts\\check_model_store.py

CPU only and instant: the fake builders sleep instead of loading anything.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uniforge3d.models.store import StageModelStore  # noqa: E402

ORDER = ("a", "b", "c")


class Recorder:
    """Builders that record which group was asked for and take a fixed time to answer."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def groups(self) -> dict[str, object]:
        def builder(name: str):
            def build():
                with self._lock:
                    self.calls.append(name)
                time.sleep(self.delay)
                return {name: object()}

            return build

        return {name: builder(name) for name in ORDER}

    def built(self) -> list[str]:
        with self._lock:
            return list(self.calls)


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline and not predicate():
        time.sleep(0.02)
    return predicate()


def main() -> int:
    problems: list[str] = []

    def check(ok: bool, description: str, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        print(f"[check] {'ok  ' if ok else 'FAIL'} {description}{suffix}", flush=True)
        if not ok:
            problems.append(f"{description}{suffix}")

    recorder = Recorder(0.01)
    lazy = StageModelStore(recorder.groups(), order=ORDER, prefetch=False)
    check(recorder.built() == [], "nothing is built without prefetch")
    lazy.get("c")
    check(recorder.built() == ["c"], "get builds only the group it was asked for", f"built {recorder.built()}")
    check(lazy.loaded() == ("c",), "loaded reports what is resident", f"resident {lazy.loaded()}")
    time.sleep(0.2)
    check(recorder.built() == ["c"], "no other group is built while nothing asks", f"built {recorder.built()}")

    # The worker starts first, so it reaches "a" before the caller asks for "b"; "b"'s own
    # build races the worker's and its position in the record is not fixed.
    recorder = Recorder(0.2)
    store = StageModelStore(recorder.groups(), order=ORDER, prefetch=True)
    group = store.get("b")
    check(list(group) == ["b"], "get returns its own group while the worker runs", f"got {list(group)}")
    check(
        # Wait for the builds to publish, not just to start: the recorder counts a call as
        # soon as the builder is entered.
        wait_until(lambda: len(store.loaded()) == len(ORDER)),
        "the worker builds the rest",
        f"resident {store.loaded()}",
    )
    recorded = recorder.built()
    check(set(recorded) == set(ORDER), "every group ended up built", f"built {recorded}")
    check([name for name in recorded if name != "b"] == ["a", "c"], "the worker builds in stage order", f"built {recorded}")
    check(len(recorded) == len(set(recorded)), "no group is built twice", f"built {recorded}")
    store.release("a")
    check(store.loaded() == ("b", "c"), "release drops the group", f"resident {store.loaded()}")
    store.get("a")
    check(recorder.built() == sorted(recorded) + ["a"], "get after release rebuilds it", f"built {recorder.built()}")

    recorder = Recorder(0.2)
    store = StageModelStore(recorder.groups(), order=ORDER, prefetch=False)
    threads = [threading.Thread(target=store.get, args=("a",)) for _ in range(4)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    check(
        recorder.built().count("a") == 1 and elapsed < 1.0,
        "four concurrent gets share one build",
        f"{recorder.built().count('a')} build in {elapsed:.2f}s",
    )

    print(f"\n[check] {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"[check] {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

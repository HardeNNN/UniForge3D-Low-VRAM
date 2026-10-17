"""Stage-by-stage weight residency for the inference pipeline.

The stack is 25 GiB of checkpoints and the first stage needs a small part of it. Loading
everything up front makes a case wait ~100 s before any work starts, and keeps all four
denoisers resident for the life of the process. This store builds one group at a time
instead: on demand when a stage asks for it, or ahead of time on a worker thread while the
GPU runs the stage before it.

``release`` drops a group again, which is what the pipeline's low-RAM mode is built from:
peak resident memory becomes the largest single group instead of their sum. Groups are
always built on CPU; moving them to the GPU is the caller's business.
"""

from __future__ import annotations

import gc
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import torch

Group = dict[str, Any]
Builder = Callable[[], Group]


class StageModelStore:
    """Owns one lazily built group of models per pipeline stage.

    A group is built at most once at a time: a caller that asks for a group the prefetch
    worker is already building waits for that build instead of starting a second one.
    """

    def __init__(
        self,
        groups: Mapping[str, Builder],
        *,
        order: Sequence[str] | None = None,
        prefetch: bool = True,
    ) -> None:
        self._builders = dict(groups)
        # The order is the order in which the pipeline first needs the groups, which is
        # also the order that hides the most loading behind the most compute.
        self._order = tuple(groups if order is None else order)
        unknown = set(self._order) - set(self._builders)
        if unknown:
            raise ValueError(f"Unknown stage groups: {sorted(unknown)}")
        self._loaded: dict[str, Group] = {}
        self._building: set[str] = set()
        # A build reads 2-20 GiB from disk and takes seconds, so the lock only ever guards
        # this bookkeeping: the worker has to keep building while a caller waits.
        self._condition = threading.Condition()
        if prefetch and len(self._order) > 1:
            threading.Thread(
                target=self._prefetch, name="stage-model-prefetch", daemon=True
            ).start()

    def get(self, group: str) -> Group:
        """The group's models, built now if nobody else is already building them."""
        with self._condition:
            while group in self._building:
                self._condition.wait()
            if group in self._loaded:
                return self._loaded[group]
            self._building.add(group)
        try:
            models = self._build(group)
        except BaseException:
            with self._condition:
                self._building.discard(group)
                self._condition.notify_all()
            raise
        # Publishing under the same lock as clearing the marker is what keeps a waiter
        # from starting a second build of the same group in between.
        with self._condition:
            self._loaded[group] = models
            self._building.discard(group)
            self._condition.notify_all()
        return models

    def release(self, group: str) -> None:
        """Drop a group's models; the next ``get`` builds them again."""
        with self._condition:
            while group in self._building:
                self._condition.wait()
            models = self._loaded.pop(group, None)
        if models is not None:
            models.clear()
            gc.collect()

    def loaded(self) -> tuple[str, ...]:
        """The resident groups, in the order the pipeline needs them."""
        with self._condition:
            return tuple(name for name in self._order if name in self._loaded)

    def _build(self, group: str) -> Group:
        started = time.perf_counter()
        # The default device is a process-wide setting, so a builder that leaves it alone
        # would still construct its weights on the calling thread's device.
        with torch.device("cpu"):
            models = self._builders[group]()
        print(f"[load] {group} ready in {time.perf_counter() - started:.1f}s", flush=True)
        return models

    def _prefetch(self) -> None:
        """Build every group in stage order, skipping the ones already handled."""
        for group in self._order:
            with self._condition:
                if group in self._loaded or group in self._building:
                    continue
            try:
                self.get(group)
            except Exception as error:  # noqa: BLE001 - the stage that needs it re-raises
                print(
                    f"[load] cannot load {group}: {type(error).__name__}: {error} "
                    "(the stage that needs it will try again)",
                    flush=True,
                )
                return

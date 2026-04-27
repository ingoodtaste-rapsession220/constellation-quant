"""Performance profiling helpers.

Context manager and decorator for timing code blocks. Logs wall-clock time to
the standard logger — useful for identifying whether training is bottlenecked
on data loading, graph construction, or forward/backward passes.

    with Timer("build_graph"):
        graph = builder.build(date)

    @timed
    def forward(self, x):
        ...
"""

from __future__ import annotations

import time
from contextlib import ContextDecorator
from functools import wraps
from typing import Any, Callable, Optional

from constellation_quant.utils.logger import get_logger

_log = get_logger(__name__)


class Timer(ContextDecorator):
    """Context manager / decorator that logs elapsed wall time.

    Args:
        name: Label used in the log line.
        log_fn: Callable receiving the formatted message. Defaults to logger.info.
    """

    def __init__(self, name: str, log_fn: Optional[Callable[[str], None]] = None):
        self.name = name
        self.log_fn = log_fn or _log.info
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        self.log_fn(f"[timer] {self.name}: {self.elapsed:.3f}s")


def timed(fn: Callable) -> Callable:
    """Decorator: log the runtime of a function using its qualified name."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with Timer(fn.__qualname__):
            return fn(*args, **kwargs)

    return wrapper

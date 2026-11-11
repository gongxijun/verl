# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enflame GCU tracing markers and topsprof backend.

``topstx`` is the GCU counterpart of NVTX/MSTX: it annotates a trace with
named ranges. ``topsprof`` is the system-level profiler that consumes those
ranges; it is enabled through ``TOPSPTI_*`` environment variables (set by
``verl/utils/profiler/tops_runtime.py`` for both the PPO ``global_profiler``
and the SFT ``profiler`` entry point) rather than through a
Python API, so :class:`TopsProfiler.start` only adds a range around the
profiled step.
"""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Optional

from torch_gcu.gcu import topstx

logger = logging.getLogger(__name__)


def mark_start_range(message: Optional[str] = None) -> int:
    """Start a marker range in the profiler.

    Args:
        message (str, optional): The message to display in the trace.

    Returns:
        int: A range handle to pass to :func:`mark_end_range`.
    """
    return topstx.range_start(message or "")


def mark_end_range(range_id: int) -> None:
    """End a marker range started by :func:`mark_start_range`.

    Args:
        range_id (int): The handle returned by :func:`mark_start_range`.
    """
    topstx.range_end(range_id)


def mark_annotate(message: Optional[str] = None) -> Callable:
    """Decorate a function to annotate its lifetime with a marker range.

    Args:
        message (str, optional): The message to display in the trace. Defaults to
            the decorated function's name.
    """

    def decorator(func: Callable) -> Callable:
        profile_message = message or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            handle = mark_start_range(profile_message)
            try:
                return func(*args, **kwargs)
            finally:
                mark_end_range(handle)

        return wrapper

    return decorator


@contextmanager
def marked_timer(name: str, timing_raw: dict[str, float], *args: Any, **kwargs: Any):
    """Context manager that times a block and marks it in the profiler.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (dict[str, float]): Dictionary to store timing information.
    """
    if args:
        logger.warning("Args are not supported in topstx_profile, but received: %s", args)
    if kwargs:
        logger.warning("Kwargs are not supported in topstx_profile, but received: %s", kwargs)
    handle = mark_start_range(name)
    from .performance import _timer

    try:
        yield from _timer(name, timing_raw)
    finally:
        mark_end_range(handle)


class TopsProfiler:
    """Thin wrapper around the system-level ``topsprof`` collection.

    Recording is controlled by ``TOPSPTI_AUTO_DUMP``/``TOPSPTI_HANDLE_SIGNAL``;
    this class only brackets the profiled step with a topstx range so that the
    dump can be correlated with verl's step boundaries.
    """

    def __init__(self, rank: int = 0, config=None, tool_config=None, **kwargs) -> None:
        self.rank = rank
        self.config = config
        self.tool_config = tool_config
        self._range_id: Optional[int] = None

    def start(self, **kwargs) -> None:
        role = kwargs.get("role")
        if self._range_id is None:
            self._range_id = mark_start_range(f"topsprof:{role or 'step'}")

    def annotate(self, message: Optional[str] = None, role: Optional[str] = None, **kwargs_outer) -> Callable:
        """Decorate a Worker stage method with a topstx range.

        ``DistProfiler.annotate`` only forwards stage annotations to backends that
        expose ``annotate``; without this method the ``@DistProfiler.annotate``
        decorators on the engine-worker stages would be silently ignored. Prefer
        the stage label (``role``) over the method name, matching the NPU backend.
        """
        return mark_annotate(message or role)

    def stop(self) -> None:
        if self._range_id is not None:
            mark_end_range(self._range_id)
            self._range_id = None

    def step(self) -> None:
        return

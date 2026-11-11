# Copyright (c) 2026 BAAI. All rights reserved.
"""Ray runtime-environment wiring for the Enflame ``topsprof`` profiler.

Recording on GCU is driven by the TOPS runtime, not by verl: ``topsprof`` reads
``TOPSPTI_AUTO_DUMP`` / ``TOPSPTI_HANDLE_SIGNAL`` from each worker's environment
and takes its command-line flags from a ``topsprof`` entry in Ray's runtime env.
This module owns that mapping so every ``ray.init`` entry point (PPO and SFT)
applies it identically.
"""

import logging
from typing import Any

from omegaconf.errors import MissingMandatoryValue

from verl.plugin.platform import get_platform
from verl.utils.profiler.config import TopsprofOptions

logger = logging.getLogger(__name__)

TOPSPROF_TOOL = "topsprof"

# Marks a field that is absent, so it joins the same "use the default" branch as
# an explicit None and is kept apart from a value of the wrong type.
_MISSING = object()

# Option field on ``TopsprofOptions`` -> the ``topsprof`` command-line flag it feeds.
TOPSPROF_OPTION_FIELDS = (
    ("enable_activities", "enable-activities"),
    ("export_visual_profiler", "export"),
    ("trace", "trace"),
    ("export_rawdata", "export-rawdata"),
)


def configure_topsprof_runtime_env(profiler_config: Any, runtime_env: dict[str, Any]) -> dict[str, Any]:
    """Add the GCU ``topsprof`` entries to ``runtime_env`` in place and return it.

    The injection is skipped unless the active platform is GCU and the profiler
    tool is ``topsprof``, so a shared config cannot arm TOPS tracing on CUDA or
    NPU hosts. Existing entries are left untouched, which lets a user-provided
    value win.

    PPO passes ``config.global_profiler``, whose options live under
    ``global_tool_config.topsprof``. SFT has no ``global_profiler`` node: its
    ``profiler`` config carries the options under ``tool_config.topsprof``. Both
    layouts are accepted.

    Args:
        profiler_config: The ``global_profiler`` (PPO) or ``profiler`` (SFT) config node.
        runtime_env: The runtime env dict ``ray.init`` will receive.

    Returns:
        The same ``runtime_env`` object, for call-site convenience.
    """
    if get_platform().device_name != "gcu":
        return runtime_env
    if _safe_getattr(profiler_config, "tool") != TOPSPROF_TOOL:
        return runtime_env

    # Recording itself is switched on by these two variables: the TOPS runtime
    # reads them when a worker process starts, and selecting the tool above is
    # the user asking for a trace. Inject them even when the options node is
    # broken, so a partial config still records; only the CLI flags then fall
    # back to the documented defaults.
    env_vars = runtime_env.setdefault("env_vars", {})
    env_vars.setdefault("TOPSPTI_AUTO_DUMP", "1")
    env_vars.setdefault("TOPSPTI_HANDLE_SIGNAL", "1")

    options = _resolve_topsprof_options(profiler_config)
    topsprof_runtime = runtime_env.setdefault(TOPSPROF_TOOL, {})
    for field_name, flag in TOPSPROF_OPTION_FIELDS:
        value = getattr(options, field_name)
        # "none" is the sentinel for "no raw-data export", not a usable CLI value.
        if flag == "export-rawdata" and value == "none":
            continue
        topsprof_runtime.setdefault(flag, value)
    return runtime_env


def _resolve_topsprof_options(profiler_config: Any) -> TopsprofOptions:
    """Return the option values to forward, falling back to the defaults.

    A raw ``DictConfig`` reaches this hook without ``TopsprofOptions.__post_init__``
    having validated it, and an incomplete config may omit the node entirely. Each
    field is therefore read defensively: a missing value, an unresolved ``???``
    mandatory value, or a non-string value (for example ``trace: true`` as a bool)
    is replaced by the default and logged, rather than being forwarded as an
    invalid command-line value.
    """

    defaults = TopsprofOptions()
    topsprof_config = _resolve_topsprof_config(profiler_config)
    if topsprof_config is None:
        logger.warning(
            "profiler tool is %r but neither global_profiler.global_tool_config.%s nor "
            "profiler.tool_config.%s is present; using the default topsprof options",
            TOPSPROF_TOOL,
            TOPSPROF_TOOL,
            TOPSPROF_TOOL,
        )
        return defaults

    configured = _safe_getattr(topsprof_config, "topsprof_options")
    if configured is None:
        logger.warning(
            "profiler tool is %r but the %s config node has no topsprof_options; using the default topsprof options",
            TOPSPROF_TOOL,
            TOPSPROF_TOOL,
        )
        return defaults

    values = {}
    for field_name, _ in TOPSPROF_OPTION_FIELDS:
        value = _safe_getattr(configured, field_name, _MISSING)
        if not isinstance(value, str):
            default = getattr(defaults, field_name)
            if value is _MISSING or value is None:
                logger.warning(
                    "topsprof_options.%s is not set; using the default %r",
                    field_name,
                    default,
                )
            else:
                logger.warning(
                    "topsprof_options.%s must be a string, got %r; using the default %r",
                    field_name,
                    value,
                    default,
                )
            value = default
        values[field_name] = value
    return TopsprofOptions(**values)


def _resolve_topsprof_config(profiler_config: Any) -> Any:
    """Return the ``topsprof`` config node for either the PPO or the SFT layout.

    ``global_tool_config`` is preferred when present: in a PPO config that is the
    documented location, and SFT's own node only has ``tool_config``. Nothing else
    in verl reads the topsprof options, so this hook is their only consumer.
    """

    global_topsprof = _safe_getattr(_safe_getattr(profiler_config, "global_tool_config"), TOPSPROF_TOOL)
    if global_topsprof is not None:
        return global_topsprof
    return _safe_getattr(_safe_getattr(profiler_config, "tool_config"), TOPSPROF_TOOL)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr`` that also treats OmegaConf's mandatory-missing as absent.

    A ``???`` value raises ``MissingMandatoryValue`` on access, which is not an
    ``AttributeError``; without this, a config that spells out ``trace: ???`` or
    ``global_tool_config: ???`` would crash ``ray.init`` instead of taking the
    documented "warn and use the default" path. ``None`` is returned for an
    object that is not there at all, which is what every caller here expects.
    """

    if obj is None:
        return default
    try:
        return getattr(obj, name, default)
    except MissingMandatoryValue:
        return default

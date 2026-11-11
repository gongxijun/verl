# Copyright (c) 2026 BAAI. All rights reserved.
"""Enflame GCU platform implementation.

Enflame GCU (``torch_gcu``) is a PyTorch ``PrivateUse1`` backend:

* the device type is ``"gcu"`` (``torch.utils.rename_privateuse1_backend("gcu")``),
* the device module is ``torch.gcu``,
* the collective backend is ``eccl``,
* visible devices are selected through ``TOPS_VISIBLE_DEVICES``.

Unlike ROCm, GCU does **not** reuse the CUDA API surface, so this
class extends :class:`PlatformBase` directly instead of subclassing
``PlatformCUDA``.
"""

import logging
import os
import shutil
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)

#: Environment variable overriding the Enflame system-management binary name.
SMI_ENV_VAR = "VERL_ENFLAME_SMI"
#: Default Enflame system-management binary.
DEFAULT_SMI = "efsmi"
#: Default minimum vLLM version accepted on the GCU rollout backend.
DEFAULT_VLLM_MIN_VERSION = "0.14.1"

# Common install locations checked when the SMI binary is not on PATH.
_SMI_FALLBACK_DIRS = ("/usr/bin", "/usr/local/bin", "/opt/tops/bin")


def _ensure_torch_gcu() -> bool:
    """Try to import ``torch_gcu`` so that ``torch.gcu`` becomes available.

    Returns True if ``torch.gcu`` is usable after the attempt.
    """
    if hasattr(torch, "gcu"):
        return True
    try:
        import torch_gcu  # noqa: F401

        return hasattr(torch, "gcu")
    except Exception as e:  # noqa: BLE001 - a broken runtime must not break platform probing
        logger.debug("The current machine has no torch.gcu, because: %s", e)
    return False


def _smi_path(name: str) -> Optional[str]:
    """Return an executable path for the Enflame SMI binary, or None."""
    found = shutil.which(name)
    if found is not None:
        return found
    for directory in _SMI_FALLBACK_DIRS:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


_ensure_torch_gcu()  # Attempt to import torch_gcu at module load time so availability checks are fast


@PlatformRegistry.register(platform="enflame")
class PlatformEnflame(PlatformBase):
    """Platform backend for Enflame GCU accelerators."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "gcu"

    @property
    def vendor_name(self) -> str:
        return "enflame"

    @property
    def device_module(self) -> ModuleType:
        if not _ensure_torch_gcu():
            raise RuntimeError(
                "torch.gcu is not available in this process. Load the TOPS runtime "
                "(e.g. `source /opt/tops/bin/set_env.sh`) and make sure torch_gcu is installed, "
                "or unset VERL_PLATFORM=enflame to let verl auto-detect the platform."
            )
        return torch.gcu

    def is_available(self) -> bool:
        return _ensure_torch_gcu() and torch.gcu.is_available()

    def is_platform_available(self, use_smi_check=False) -> bool:
        """Return True if this platform is available on this host.

        Used during auto-detection. When ``use_smi_check=True`` (CPU-only Ray
        actors, where ``torch.gcu.is_available()`` can be False even though the
        node has GCUs), fall back to the Enflame SMI binary.

        An importable ``torch_gcu`` alone is deliberately *not* treated as
        evidence: the package can be installed (and its runtime loaded) on a
        host without GCUs, and reporting such a host as Enflame would switch the
        device type to ``"gcu"`` and the checkpoint engine to ECCL. Without an
        SMI binary to ask, require a device to be actually visible.
        """
        if not _ensure_torch_gcu():
            return False
        if not callable(getattr(torch.gcu, "is_available", None)):
            return False
        if use_smi_check:
            smi = os.environ.get(SMI_ENV_VAR, DEFAULT_SMI)
            smi_path = _smi_path(smi)
            if smi_path is not None:
                # Pass the resolved path: check_smi_command only searches PATH and
                # the CUDA-oriented common dirs, not /opt/tops/bin.
                return self.check_smi_command(smi_path)
        return torch.gcu.is_available()

    def current_device(self) -> int:
        return torch.gcu.current_device()

    def device_count(self) -> int:
        return torch.gcu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.gcu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.gcu.synchronize(device_index)

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.gcu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.gcu.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        try:
            torch.gcu.memory._set_allocator_settings(settings)
        except Exception:
            logger.warning(
                "Current version of torch_gcu does not support `_set_allocator_settings`, "
                "please upgrade torch_gcu to 2.11.0 or later"
            )

    def empty_cache(self) -> None:
        torch.gcu.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # Mirrors PlatformNPU: some torch_gcu builds do not expose this method, and
        # `verl.trainer.constants_ppo` calls it at import time, so it must never raise.
        if not _ensure_torch_gcu():
            return (None, None)
        cap = getattr(torch.gcu, "get_device_capability", None)
        if cap is None:
            return (None, None)
        try:
            result = cap(device_index)
        except Exception as e:  # noqa: BLE001 - the driver decides, and callers run at import time
            # CPU-only actors (the case the SMI fallback exists for) can still
            # see a visible-device list that the device module cannot query.
            logger.debug("torch.gcu.get_device_capability(%s) failed: %s", device_index, e)
            return (None, None)
        # torch.gcu.get_device_capability may return None instead of a tuple
        if result is None:
            return (None, None)
        return result

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "eccl"

    def visible_devices_envvar(self) -> str:
        return "TOPS_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        # GCU is scheduled through Ray's generic GPU resource, like other
        # CUDA-compatible accelerators.
        return "GPU"

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES"]

    def get_ray_init_kwargs(self) -> dict[str, Any]:
        """Tell Ray not to write ``TOPS_VISIBLE_DEVICES`` for our workers.

        Whether Ray knows about GCUs depends on the Ray build: upstream Ray only
        manages ``CUDA_VISIBLE_DEVICES`` for the ``GPU`` resource, so without this
        flag every worker on a node sees every GCU and ``current_device()`` stays
        at 0, making all local ranks collide on device 0. With the flag set,
        ``Worker._setup_env_cuda_visible_devices`` assigns the device explicitly
        from Ray's accelerator ids, which is correct for both stock and
        Enflame-patched Ray. Users can still override the entry through
        ``ray_kwargs.ray_init.runtime_env``.
        """
        return {"runtime_env": {"env_vars": {"RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES": "1"}}}

    def rollout_env_vars(self) -> dict[str, str]:
        # Keep parity with the CUDA/NPU rollout paths: weight synchronization
        # between actor and rollout must disable NCCL's CUDA memory pool reuse.
        return {"NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0")}

    # ------------------------------------------------------------------
    # vLLM integration
    # ------------------------------------------------------------------

    def vllm_min_version(self) -> str:
        """Minimum vLLM version accepted by the GCU rollout backend.

        The GCU vLLM wheel tracks an older upstream release than the floor
        verl requires for NVIDIA, so the floor is platform-provided instead of
        hard-coded in ``verl/third_party/vllm``.
        """
        return os.environ.get("VERL_GCU_VLLM_MIN_VERSION", DEFAULT_VLLM_MIN_VERSION)

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        # torch's cross-process tensor sharing (`reduce_tensor` -> `rebuild_ipc`)
        # only knows a CUDA IPC handle and CUDA (or meta) storages: everything
        # else goes through `rebuild_tensor`, whose argument tuple is 3 elements
        # long, while `rebuild_ipc` rewrites index 6 to the receiver's device id.
        # Claiming IPC support would therefore make the first weight update fail
        # with IndexError. Return False so `vllm_rollout` transfers weights
        # through shared memory instead (slower, but correct).
        return False

    # ------------------------------------------------------------------
    # Collective communication
    # ------------------------------------------------------------------

    def get_collective_module(self) -> Any:
        # ECCL is registered directly with torch.distributed; there is no cupy
        # nccl-style module for the NCCL checkpoint engine to use on GCU.
        return None

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    def profiler_markers(self):
        """Return topstx-based marker callables for stage annotation."""
        try:
            from verl.utils.profiler.topstx_profile import (
                mark_annotate,
                mark_end_range,
                mark_start_range,
                marked_timer,
            )
        except Exception as e:  # noqa: BLE001 - markers are optional
            logger.debug("topstx profiler markers unavailable: %s", e)
            return None
        return (mark_start_range, mark_end_range, mark_annotate, marked_timer)

    @contextmanager
    def nvtx_range(self, msg: str):
        # GCU has no NVTX; use the topstx range so traces still contain stage
        # markers when a profiler is attached.
        try:
            from torch_gcu.gcu import topstx
        except Exception:
            logger.debug("topstx range (no-op on GCU without topstx): %s", msg)
            yield
            return
        try:
            handle = topstx.range_start(str(msg))
        except Exception:
            logger.debug("topstx range (no-op on GCU): %s", msg)
            yield
            return
        try:
            yield
        finally:
            try:
                topstx.range_end(handle)
            except Exception:
                pass

    def profiler_start(self) -> None:
        profiler = getattr(torch.gcu, "profiler", None)
        if profiler is not None and callable(getattr(profiler, "start", None)):
            profiler.start()

    def profiler_stop(self) -> None:
        profiler = getattr(torch.gcu, "profiler", None)
        if profiler is not None and callable(getattr(profiler, "stop", None)):
            profiler.stop()

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return None

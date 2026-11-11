# Copyright (c) 2026 BAAI. All rights reserved.
"""The CUDA-only NCCL checkpoint engine must be skipped on an Enflame GCU host.

``NCCLCheckpointEngine`` and ``ECCLCheckpointEngine`` both register the ``nccl``
backend name. The arbitration lives in ``verl/checkpoint_engine/__init__.py`` so
that the vendor-neutral NCCL engine does not have to know about GCU; these tests
re-import that package with the GCU probe patched. The decision is taken before
the ``cupy`` import, so it holds even on hosts where cupy is missing.
"""

import importlib
import sys

import pytest

pytest.importorskip("torch")

# Importing the package executes the guarded engine in a real import, which needs
# the verl runtime dependencies (ray among them). Skip, do not error, when they
# are absent: this file is meant to run on any host without a GCU, but not on a
# host without verl's requirements.
try:
    # Import the package before any probe is patched, so the baseline registry
    # state (which engine owns "nccl" on this host) is recorded outside the tests.
    import verl.checkpoint_engine  # noqa: E402,F401
    from verl.checkpoint_engine import CheckpointEngineRegistry  # noqa: E402
except ImportError as e:  # pragma: no cover - depends on the environment
    pytest.skip(f"verl runtime dependencies unavailable: {e}", allow_module_level=True)

PACKAGE = "verl.checkpoint_engine"
MODULE = "verl.checkpoint_engine.nccl_checkpoint_engine"
GUARD_MESSAGE = "Enflame GCU"
_ABSENT = object()


def _reload_package():
    # Re-run the package __init__ so the patched probe decides whether the NCCL
    # engine is imported. The registry is process-global: a reload that gets past
    # the guard would overwrite the "nccl" entry, so snapshot and restore it as
    # well. Popping the engine module makes the import-time decision re-run, and
    # every attribute the reload rebinds is restored for the other test modules.
    # The decision is captured before the restore, which would otherwise put the
    # previous engine back before the caller can assert on it.
    package = sys.modules[PACKAGE]
    saved_module = sys.modules.pop(MODULE, None)
    saved_attr = getattr(package, "nccl_checkpoint_engine", _ABSENT)
    saved_nccl = getattr(package, "NCCLCheckpointEngine", _ABSENT)
    saved_all = list(package.__all__)
    saved_registry = dict(CheckpointEngineRegistry._registry)
    saved_errors = dict(CheckpointEngineRegistry._import_errors)
    try:
        importlib.reload(package)
        return {
            "engine": package.NCCLCheckpointEngine,
            "all": list(package.__all__),
            "registry": dict(CheckpointEngineRegistry._registry),
            "errors": dict(CheckpointEngineRegistry._import_errors),
        }
    finally:
        sys.modules.pop(MODULE, None)
        if saved_module is not None:
            sys.modules[MODULE] = saved_module
        if saved_attr is _ABSENT:
            if hasattr(package, "nccl_checkpoint_engine"):
                delattr(package, "nccl_checkpoint_engine")
        else:
            package.nccl_checkpoint_engine = saved_attr
        if saved_nccl is _ABSENT:
            if hasattr(package, "NCCLCheckpointEngine"):
                delattr(package, "NCCLCheckpointEngine")
        else:
            package.NCCLCheckpointEngine = saved_nccl
        package.__all__ = saved_all
        CheckpointEngineRegistry._registry.clear()
        CheckpointEngineRegistry._registry.update(saved_registry)
        CheckpointEngineRegistry._import_errors.clear()
        CheckpointEngineRegistry._import_errors.update(saved_errors)


def test_guard_skips_nccl_on_gcu(monkeypatch):
    # Arrange
    import verl.utils.device as device

    monkeypatch.setattr(device, "is_torch_gcu_available", lambda *args, **kwargs: True)

    # Act
    snapshot = _reload_package()

    # Assert
    assert snapshot["engine"] is None
    assert "NCCLCheckpointEngine" not in snapshot["all"]
    error = snapshot["errors"]["nccl_checkpoint_engine"]
    assert GUARD_MESSAGE in str(error)
    # The message has to name the replacement, otherwise the user cannot act on it.
    assert "eccl" in str(error)
    # The skipped engine must not leave a stale "nccl" entry behind: when the ECCL
    # engine imported successfully it must own that backend name instead.
    if "eccl_checkpoint_engine" not in snapshot["errors"]:
        assert snapshot["registry"]["nccl"] is not None


def test_guard_requires_an_actual_device(monkeypatch):
    # Arrange: the probe must ask for a device, not just for the package, so that
    # a CUDA host with torch_gcu installed keeps the NCCL engine.
    import verl.utils.device as device

    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(device, "is_torch_gcu_available", spy)

    # Act
    _reload_package()

    # Assert
    assert calls, "the package must probe for a GCU before importing the NCCL engine"
    assert all(args == () and kwargs == {} for args, kwargs in calls)


def test_guard_does_not_fire_without_a_device(monkeypatch):
    # Arrange: with no GCU present the import must proceed past the guard. It may
    # still fail on a missing transport dependency (cupy, ray, zmq), so only the
    # guard's own message is asserted absent.
    import verl.utils.device as device

    monkeypatch.setattr(device, "is_torch_gcu_available", lambda *args, **kwargs: False)

    # Act
    snapshot = _reload_package()

    # Assert
    error = snapshot["errors"].get("nccl_checkpoint_engine")
    if error is None:
        # No error at all: the engine imported, which is the expected outcome here.
        assert snapshot["engine"] is not None
    else:
        # The engine can still fail on a missing transport dependency (cupy, ray,
        # zmq); only the guard's own message is asserted absent.
        assert GUARD_MESSAGE not in str(error), "the guard fired even though no GCU is available"

# Copyright (c) 2026 BAAI. All rights reserved.
"""Unit tests for the Enflame GCU platform backend."""

import sys
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from verl.plugin.platform import platform_enflame as enflame_mod  # noqa: E402
from verl.plugin.platform.platform_cuda import PlatformCUDA  # noqa: E402
from verl.plugin.platform.platform_enflame import PlatformEnflame  # noqa: E402
from verl.plugin.platform.platform_manager import PlatformRegistry, _detect_platform_name  # noqa: E402

_MISSING = object()


class _FakeGcu:
    """Minimal stand-in for the ``torch.gcu`` module."""

    def __init__(self, available=True, with_memory=False, capability=_MISSING):
        self.available = available
        self.calls = []
        if with_memory:
            self.memory = mock.Mock()
        if capability is not _MISSING:
            self.get_device_capability = mock.Mock(return_value=capability)

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def device_count(self):
        return 8

    def set_device(self, index):
        self.calls.append(("set_device", index))

    def synchronize(self, index=None):
        self.calls.append(("synchronize", index))

    def manual_seed(self, seed):
        self.calls.append(("manual_seed", seed))

    def manual_seed_all(self, seed):
        self.calls.append(("manual_seed_all", seed))

    def empty_cache(self):
        self.calls.append(("empty_cache",))


@pytest.fixture
def fake_gcu(monkeypatch):
    fake = _FakeGcu()
    monkeypatch.delattr(torch, "gcu", raising=False)
    monkeypatch.setattr(torch, "gcu", fake, raising=False)
    return fake


@pytest.fixture
def platform():
    return PlatformEnflame()


class TestRegistrationAndDetection:
    def test_registered_under_enflame_name(self):
        assert PlatformRegistry.get("enflame") is PlatformEnflame

    def test_env_override_selects_enflame(self, monkeypatch):
        monkeypatch.setenv("VERL_PLATFORM", "enflame")
        assert _detect_platform_name() == "enflame"

    def test_cuda_defers_to_gcu(self, fake_gcu):
        # torch_gcu exposes a PrivateUse1 backend, so CUDA must not claim the host.
        assert PlatformCUDA().is_platform_available(use_smi_check=False) is False

    def test_cuda_kept_when_torch_gcu_has_no_device(self, monkeypatch):
        # A CUDA host with an importable but idle torch_gcu must keep CUDA.
        monkeypatch.delattr(torch, "gcu", raising=False)
        monkeypatch.setattr(torch, "gcu", _FakeGcu(available=False), raising=False)
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            assert PlatformCUDA().is_platform_available(use_smi_check=False) is True


class TestPlatformIdentity:
    def test_device_and_vendor_names(self, platform):
        assert platform.device_name == "gcu"
        assert platform.vendor_name == "enflame"

    def test_device_module_is_torch_gcu(self, platform, fake_gcu):
        assert platform.device_module is fake_gcu

    def test_communication_and_visible_device_settings(self, platform):
        assert platform.communication_backend_name() == "eccl"
        assert platform.visible_devices_envvar() == "TOPS_VISIBLE_DEVICES"
        assert platform.ray_resource_name() == "GPU"
        assert platform.ray_noset_envvars() == ["RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES"]

    def test_ray_scheduling_options(self, platform):
        # GCUs consume Ray's generic GPU resource.
        assert platform.ray_resource_options(4) == {"num_gpus": 4}

    def test_ray_init_kwargs_disable_tops_visible_devices(self, platform):
        # Without this flag a stock Ray never writes TOPS_VISIBLE_DEVICES and every
        # local rank stays on device 0, so placement must be assigned explicitly.
        assert platform.get_ray_init_kwargs() == {
            "runtime_env": {"env_vars": {"RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES": "1"}}
        }

    def test_ray_noset_visible_devices_detects_tops_var(self, monkeypatch):
        from verl.utils.ray_utils import ray_noset_visible_devices

        monkeypatch.delenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES", "1")
        assert ray_noset_visible_devices() is True

    def test_ipc_and_collective_module(self, platform):
        # torch's cross-process reduce path only understands CUDA IPC handles, so
        # claiming IPC support would break the first weight update; the rollout
        # must fall back to shared memory instead.
        assert platform.is_ipc_supported() is False
        assert platform.get_collective_module() is None
        assert platform.cudart() is None


class TestAvailability:
    def test_unavailable_without_torch_gcu(self, monkeypatch):
        monkeypatch.delattr(torch, "gcu", raising=False)
        # `import torch_gcu` must fail: None in sys.modules makes the import raise.
        monkeypatch.setitem(sys.modules, "torch_gcu", None)
        assert PlatformEnflame().is_platform_available(use_smi_check=False) is False
        assert PlatformEnflame().is_available() is False

    def test_available_through_device_probe(self, fake_gcu):
        assert PlatformEnflame().is_platform_available(use_smi_check=False) is True

    def test_unavailable_when_device_reports_no_gcu(self, monkeypatch):
        monkeypatch.delattr(torch, "gcu", raising=False)
        monkeypatch.setattr(torch, "gcu", _FakeGcu(available=False), raising=False)
        assert PlatformEnflame().is_platform_available(use_smi_check=False) is False

    def test_smi_check_honours_env_override(self, platform, fake_gcu, monkeypatch):
        monkeypatch.setenv("VERL_ENFLAME_SMI", "my-efsmi")
        with (
            mock.patch.object(enflame_mod, "_smi_path", return_value="/opt/tops/bin/my-efsmi") as smi_path,
            mock.patch.object(PlatformEnflame, "check_smi_command", return_value=True) as smi_check,
        ):
            assert platform.is_platform_available(use_smi_check=True) is True
        smi_path.assert_called_once_with("my-efsmi")
        # The resolved path is passed on: check_smi_command does not look in /opt/tops/bin.
        smi_check.assert_called_once_with("/opt/tops/bin/my-efsmi")

    def test_smi_check_without_smi_requires_visible_device(self, platform, fake_gcu):
        with mock.patch.object(enflame_mod, "_smi_path", return_value=None):
            assert platform.is_platform_available(use_smi_check=True) is True

    def test_importable_torch_gcu_without_device_is_not_enflame(self, monkeypatch):
        # A host that merely has torch_gcu installed must not be detected as GCU.
        monkeypatch.delattr(torch, "gcu", raising=False)
        monkeypatch.setattr(torch, "gcu", _FakeGcu(available=False), raising=False)
        with mock.patch.object(enflame_mod, "_smi_path", return_value=None):
            assert PlatformEnflame().is_platform_available(use_smi_check=True) is False


class TestDeviceManagement:
    def test_device_queries(self, platform, fake_gcu):
        assert platform.current_device() == 0
        assert platform.device_count() == 8
        # The fake has no get_device_capability: the platform must return (None, None)
        # instead of raising at import time of verl.trainer.constants_ppo.
        assert platform.get_device_capability(0) == (None, None)

    def test_device_capability_forwarded(self, platform, monkeypatch):
        fake = _FakeGcu(capability=(8, 0))
        monkeypatch.delattr(torch, "gcu", raising=False)
        monkeypatch.setattr(torch, "gcu", fake, raising=False)

        assert platform.get_device_capability(2) == (8, 0)
        fake.get_device_capability.assert_called_once_with(2)

    def test_device_capability_none_result(self, platform, monkeypatch):
        fake = _FakeGcu(capability=None)
        monkeypatch.delattr(torch, "gcu", raising=False)
        monkeypatch.setattr(torch, "gcu", fake, raising=False)

        assert platform.get_device_capability(0) == (None, None)

    def test_device_capability_failure_does_not_raise(self, platform, monkeypatch):
        # constants_ppo queries this at import time, so a driver error must not
        # escape: CPU-only actors may still have a device module that cannot
        # answer for the visible device.
        fake = _FakeGcu()
        fake.get_device_capability = mock.Mock(side_effect=RuntimeError("driver error"))
        monkeypatch.delattr(torch, "gcu", raising=False)
        monkeypatch.setattr(torch, "gcu", fake, raising=False)

        assert platform.get_device_capability(0) == (None, None)

    def test_device_operations_are_forwarded(self, platform, fake_gcu):
        # Arrange / Act
        platform.set_device(3)
        platform.synchronize()
        platform.manual_seed(7)
        platform.manual_seed_all(11)
        platform.empty_cache()

        # Assert
        assert fake_gcu.calls == [
            ("set_device", 3),
            ("synchronize", None),
            ("manual_seed", 7),
            ("manual_seed_all", 11),
            ("empty_cache",),
        ]

    def test_allocator_settings_forwarded(self, platform, monkeypatch):
        fake = _FakeGcu(with_memory=True)
        monkeypatch.delattr(torch, "gcu", raising=False)
        monkeypatch.setattr(torch, "gcu", fake, raising=False)

        platform.set_allocator_settings("expandable_segments:True")

        fake.memory._set_allocator_settings.assert_called_once_with("expandable_segments:True")

    def test_allocator_settings_warns_when_unsupported(self, platform, fake_gcu, caplog):
        with caplog.at_level("WARNING", logger=enflame_mod.logger.name):
            platform.set_allocator_settings("expandable_segments:True")
        assert "does not support" in caplog.text


class TestVllmIntegration:
    def test_default_min_version(self, platform, monkeypatch):
        monkeypatch.delenv("VERL_GCU_VLLM_MIN_VERSION", raising=False)
        assert platform.vllm_min_version() == "0.14.1"

    def test_min_version_env_override(self, platform, monkeypatch):
        monkeypatch.setenv("VERL_GCU_VLLM_MIN_VERSION", "0.9.0")
        assert platform.vllm_min_version() == "0.9.0"

    def test_rollout_env_vars_disable_cumem(self, platform, monkeypatch):
        monkeypatch.delenv("NCCL_CUMEM_ENABLE", raising=False)
        assert platform.rollout_env_vars() == {"NCCL_CUMEM_ENABLE": "0"}

    def test_device_uuid_prefers_visible_devices(self, platform, monkeypatch):
        monkeypatch.setenv("TOPS_VISIBLE_DEVICES", "4,5")
        assert platform.get_device_uuid(1) == "GPU5"

    def test_device_uuid_falls_back_to_index(self, platform, monkeypatch):
        monkeypatch.delenv("TOPS_VISIBLE_DEVICES", raising=False)
        assert platform.get_device_uuid(2) == "GPU-2"


class TestProfilerHelpers:
    def test_profiler_markers_none_without_topstx(self, platform, monkeypatch):
        monkeypatch.setitem(sys.modules, "verl.utils.profiler.topstx_profile", None)
        assert platform.profiler_markers() is None

    def test_nvtx_range_is_noop_without_topstx(self, platform, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_gcu", None)
        with platform.nvtx_range("stage"):
            pass

    def test_profiler_start_stop_without_profiler(self, platform, fake_gcu):
        # No `profiler` attribute on torch.gcu: both calls must be no-ops.
        platform.profiler_start()
        platform.profiler_stop()
        assert fake_gcu.calls == []

    def test_profiler_start_stop_forwarded(self, platform, monkeypatch):
        fake = _FakeGcu()
        fake.profiler = mock.Mock()
        monkeypatch.delattr(torch, "gcu", raising=False)
        monkeypatch.setattr(torch, "gcu", fake, raising=False)

        platform.profiler_start()
        platform.profiler_stop()

        fake.profiler.start.assert_called_once_with()
        fake.profiler.stop.assert_called_once_with()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

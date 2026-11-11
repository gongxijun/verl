# Copyright (c) 2026 BAAI. All rights reserved.
"""Unit tests for the Enflame ``topsprof`` tool configuration."""

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("omegaconf")

from verl.utils.profiler.config import TopsprofOptions, TopsToolConfig  # noqa: E402

# Distinguishes "the package has no such attribute" from "the attribute is None".
_ABSENT = object()


class TestTopsprofOptions:
    def test_defaults(self):
        # Arrange / Act
        options = TopsprofOptions()

        # Assert
        assert options.enable_activities == "kernel"
        assert options.export_visual_profiler == "true"
        assert options.trace == "true"
        assert options.export_rawdata == "none"
        assert options.name == "topsprof_options"

    @pytest.mark.parametrize(
        "field_name",
        ["enable_activities", "export_visual_profiler", "trace", "export_rawdata"],
    )
    def test_rejects_non_string_values(self, field_name):
        with pytest.raises(AssertionError, match=field_name):
            TopsprofOptions(**{field_name: True})

    def test_accepts_valid_values(self):
        options = TopsprofOptions(enable_activities="memory", export_rawdata="csv")
        assert options.enable_activities == "memory"
        assert options.export_rawdata == "csv"


class TestTopsToolConfig:
    def test_defaults(self):
        config = TopsToolConfig()

        assert config.name == "topsprof"
        assert config.discrete is False
        assert config.contents == []
        assert config.profile_token_start is None
        assert config.profile_token_end is None
        assert isinstance(config.topsprof_options, TopsprofOptions)

    def test_accepts_supported_contents(self):
        config = TopsToolConfig(contents=["gcu", "memory", "cpu"])
        assert config.contents == ["gcu", "memory", "cpu"]

    def test_rejects_unknown_contents(self):
        with pytest.raises(AssertionError, match="only supports"):
            TopsToolConfig(contents=["cuda"])

    def test_rejects_non_list_contents(self):
        with pytest.raises(AssertionError, match="must be of type list"):
            TopsToolConfig(contents="gcu")

    def test_rejects_negative_token_index(self):
        with pytest.raises(AssertionError, match=">= 0"):
            TopsToolConfig(profile_token_start=-1)

    def test_rejects_non_int_token_index(self):
        with pytest.raises(AssertionError, match="must be int or None"):
            TopsToolConfig(profile_token_end="10")

    def test_rejects_non_increasing_token_range(self):
        with pytest.raises(AssertionError, match="must be >"):
            TopsToolConfig(profile_token_start=10, profile_token_end=10)

    def test_accepts_increasing_token_range(self):
        config = TopsToolConfig(profile_token_start=0, profile_token_end=16)
        assert (config.profile_token_start, config.profile_token_end) == (0, 16)


class TestTopsProfilerAnnotate:
    """``DistProfiler.annotate`` only decorates backends that expose ``annotate``."""

    @pytest.fixture
    def topsprof_module(self, monkeypatch):
        import importlib
        import sys
        from unittest import mock

        topstx = mock.Mock()
        topstx.range_start.return_value = 7
        gcu_pkg = mock.Mock(topstx=topstx)
        monkeypatch.setitem(sys.modules, "torch_gcu", mock.Mock(gcu=gcu_pkg))
        monkeypatch.setitem(sys.modules, "torch_gcu.gcu", gcu_pkg)

        package = sys.modules["verl.utils.profiler"]
        module_name = "verl.utils.profiler.topstx_profile"
        # Snapshot both the sys.modules entry and the attribute the package binds
        # for it. `from package import submodule` returns that attribute, so a
        # leftover binding would hand later tests a stale module carrying this
        # fixture's dead mocks; monkeypatch.delattr is a no-op (and leaves no
        # rollback) when the entry is absent on the first test.
        saved_module = sys.modules.pop(module_name, None)
        saved_attr = getattr(package, "topstx_profile", _ABSENT)
        try:
            yield importlib.import_module(module_name), topstx
        finally:
            sys.modules.pop(module_name, None)
            if saved_module is not None:
                sys.modules[module_name] = saved_module
            if saved_attr is _ABSENT:
                if hasattr(package, "topstx_profile"):
                    delattr(package, "topstx_profile")
            else:
                package.topstx_profile = saved_attr

    def test_annotate_uses_role_as_range_name(self, topsprof_module):
        # Arrange
        module, topstx = topsprof_module
        profiler = module.TopsProfiler()

        # Act
        @profiler.annotate(role="actor_update")
        def stage():
            return "done"

        result = stage()

        # Assert
        assert result == "done"
        topstx.range_start.assert_called_once_with("actor_update")
        topstx.range_end.assert_called_once_with(7)

    def test_annotate_falls_back_to_function_name(self, topsprof_module):
        module, topstx = topsprof_module
        profiler = module.TopsProfiler()

        @profiler.annotate()
        def rollout_stage():
            return None

        rollout_stage()

        topstx.range_start.assert_called_once_with("rollout_stage")

    def test_marked_timer_marks_range(self, topsprof_module):
        module, topstx = topsprof_module
        timing_raw = {}

        with module.marked_timer("step", timing_raw):
            pass

        assert "step" in timing_raw
        assert topstx.range_start.call_count == 1
        assert topstx.range_end.call_count == 1


class TestTopsprofRuntimeEnv:
    """Ray runtime-env wiring shared by the PPO and SFT entry points."""

    @staticmethod
    def _config(tool="topsprof", **option_overrides):
        options = TopsprofOptions(**option_overrides)
        tool_config = TopsToolConfig(topsprof_options=options)
        global_tool_config = SimpleNamespace(topsprof=tool_config)
        return SimpleNamespace(tool=tool, global_tool_config=global_tool_config)

    @staticmethod
    def _sft_config(tool="topsprof", **option_overrides):
        """The SFT layout: options live under ``profiler.tool_config.topsprof``."""
        options = TopsprofOptions(**option_overrides)
        tool_config = SimpleNamespace(topsprof=TopsToolConfig(topsprof_options=options))
        return SimpleNamespace(tool=tool, tool_config=tool_config)

    @staticmethod
    def _platform(device_name):
        return SimpleNamespace(device_name=device_name)

    def test_injects_env_and_cli_options_on_gcu(self, monkeypatch):
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        runtime_env = {}

        returned = tops_runtime.configure_topsprof_runtime_env(self._config(), runtime_env)

        assert returned is runtime_env
        assert runtime_env["env_vars"] == {"TOPSPTI_AUTO_DUMP": "1", "TOPSPTI_HANDLE_SIGNAL": "1"}
        assert runtime_env["topsprof"] == {
            "enable-activities": "kernel",
            "export": "true",
            "trace": "true",
        }
        # export_rawdata defaults to "none": not a usable CLI value, so it is omitted.
        assert "export-rawdata" not in runtime_env["topsprof"]

    def test_includes_export_rawdata_when_set(self, monkeypatch):
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        runtime_env = {}

        tops_runtime.configure_topsprof_runtime_env(self._config(export_rawdata="csv"), runtime_env)

        assert runtime_env["topsprof"]["export-rawdata"] == "csv"

    def test_reads_options_from_tool_config_for_sft(self, monkeypatch):
        # SFT composes `config.profiler` (no `global_profiler` node), so the
        # options are reached through `tool_config.topsprof`.
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        runtime_env = {}

        tops_runtime.configure_topsprof_runtime_env(self._sft_config(enable_activities="memory"), runtime_env)

        assert runtime_env["topsprof"]["enable-activities"] == "memory"
        assert runtime_env["env_vars"]["TOPSPTI_AUTO_DUMP"] == "1"

    def test_injects_defaults_when_topsprof_node_is_missing(self, monkeypatch):
        # A `topsprof` selection without an options node is an incomplete config;
        # the recording switch must still be armed, and the CLI flags fall back
        # to the documented defaults.
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        runtime_env = {}

        tops_runtime.configure_topsprof_runtime_env(
            SimpleNamespace(tool="topsprof", tool_config=SimpleNamespace()), runtime_env
        )

        assert runtime_env["env_vars"] == {"TOPSPTI_AUTO_DUMP": "1", "TOPSPTI_HANDLE_SIGNAL": "1"}
        assert runtime_env["topsprof"] == {
            "enable-activities": "kernel",
            "export": "true",
            "trace": "true",
        }

    def test_defaults_when_topsprof_options_are_missing(self, monkeypatch):
        # Same fallback when the node exists but carries no `topsprof_options`.
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        runtime_env = {}

        tops_runtime.configure_topsprof_runtime_env(
            SimpleNamespace(tool="topsprof", tool_config=SimpleNamespace(topsprof=SimpleNamespace())),
            runtime_env,
        )

        assert runtime_env["env_vars"] == {"TOPSPTI_AUTO_DUMP": "1", "TOPSPTI_HANDLE_SIGNAL": "1"}
        assert runtime_env["topsprof"]["enable-activities"] == "kernel"

    def test_non_string_option_falls_back_to_default(self, monkeypatch, caplog):
        # A raw DictConfig bypasses `TopsprofOptions.__post_init__`, so a bool
        # `trace: true` must not be forwarded as a CLI value.
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        runtime_env = {}

        with caplog.at_level(logging.WARNING, logger="verl.utils.profiler.tops_runtime"):
            tops_runtime.configure_topsprof_runtime_env(
                SimpleNamespace(
                    tool="topsprof",
                    tool_config=SimpleNamespace(topsprof=SimpleNamespace(topsprof_options=SimpleNamespace(trace=True))),
                ),
                runtime_env,
            )

        assert runtime_env["topsprof"]["trace"] == "true"
        # The fallback must be reported, not silent: this is the only consumer
        # of the options and the user cannot see the resulting CLI values.
        assert "topsprof_options.trace must be a string" in caplog.text
        # Fields that were never set take the other branch of the same fallback.
        assert "topsprof_options.enable_activities is not set" in caplog.text

    def test_mandatory_missing_values_fall_back(self, monkeypatch, caplog):
        # `???` raises `MissingMandatoryValue` on access, which is not an
        # `AttributeError`; it must still warn and use defaults instead of
        # crashing `ray.init`.
        omega_conf = pytest.importorskip("omegaconf")
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        config = omega_conf.OmegaConf.create(
            {
                "tool": "topsprof",
                "tool_config": {"topsprof": {"topsprof_options": {"trace": "???", "enable_activities": "memory"}}},
            }
        )
        runtime_env = {}

        with caplog.at_level(logging.WARNING, logger="verl.utils.profiler.tops_runtime"):
            tops_runtime.configure_topsprof_runtime_env(config, runtime_env)

        assert runtime_env["topsprof"]["trace"] == "true"
        assert runtime_env["topsprof"]["enable-activities"] == "memory"
        assert "topsprof_options.trace is not set" in caplog.text

    def test_mandatory_missing_tool_is_not_topsprof(self, monkeypatch):
        # `tool: ???` must behave like "no tool selected", not crash the run.
        omega_conf = pytest.importorskip("omegaconf")
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        runtime_env = {}

        tops_runtime.configure_topsprof_runtime_env(omega_conf.OmegaConf.create({"tool": "???"}), runtime_env)

        assert runtime_env == {}

    def test_user_values_are_preserved(self, monkeypatch):
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        runtime_env = {"env_vars": {"TOPSPTI_AUTO_DUMP": "0"}, "topsprof": {"trace": "false"}}

        tops_runtime.configure_topsprof_runtime_env(self._config(), runtime_env)

        assert runtime_env["env_vars"]["TOPSPTI_AUTO_DUMP"] == "0"
        assert runtime_env["topsprof"]["trace"] == "false"
        assert runtime_env["env_vars"]["TOPSPTI_HANDLE_SIGNAL"] == "1"

    def test_noop_on_other_platforms(self, monkeypatch):
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("nvidia"))
        runtime_env = {}

        tops_runtime.configure_topsprof_runtime_env(self._config(), runtime_env)

        assert runtime_env == {}

    def test_noop_for_other_tools(self, monkeypatch):
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        runtime_env = {}

        tops_runtime.configure_topsprof_runtime_env(self._config(tool="torch"), runtime_env)

        assert runtime_env == {}

    def test_noop_without_global_profiler(self, monkeypatch):
        from verl.utils.profiler import tops_runtime

        monkeypatch.setattr(tops_runtime, "get_platform", lambda: self._platform("gcu"))
        runtime_env = {}

        tops_runtime.configure_topsprof_runtime_env(None, runtime_env)

        assert runtime_env == {}


class TestTopsprofPlatformGate:
    """``tool=topsprof`` must not build a GCU profiler on another platform.

    A CUDA or NPU host can import ``torch_gcu`` (shared image, stale install), so
    the import probe alone is not enough to decide that topstx is usable there.
    """

    @staticmethod
    def _dist_profiler(monkeypatch, device_name, enable):
        pytest.importorskip("orjson")  # imported by ``verl.utils.tracking`` via profile.py

        from verl.plugin import platform
        from verl.utils import import_utils
        from verl.utils.profiler.config import ProfilerConfig
        from verl.utils.profiler.profile import DistProfiler

        monkeypatch.setattr(platform, "get_platform", lambda: SimpleNamespace(device_name=device_name))
        monkeypatch.setattr(import_utils, "is_topstx_available", lambda: True)
        config = ProfilerConfig(tool="topsprof", enable=enable, save_path="/tmp/verl-profiler-test")
        return DistProfiler(rank=0, config=config)

    def test_rejects_non_gcu_platform_when_enabled(self, monkeypatch):
        # Arrange: topstx imports fine, but the active platform is not GCU.
        # Act / Assert
        with pytest.raises(RuntimeError, match="Enflame GCU platform"):
            self._dist_profiler(monkeypatch, device_name="nvidia", enable=True)

    def test_falls_back_to_noop_when_disabled(self, monkeypatch):
        # Arrange / Act
        profiler = self._dist_profiler(monkeypatch, device_name="nvidia", enable=False)

        # Assert
        assert type(profiler._impl).__name__ == "_NoOpProfiler"

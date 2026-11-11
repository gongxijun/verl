# Copyright (c) 2026 BAAI. All rights reserved.
"""Determinism of the seeded value-head initialization.

Enflame GCU does not reproduce CUDA's RNG stream across ranks, so a randomly
initialized critic value head differs per rank unless it is seeded. These tests
pin the seeding contract: the same seed yields bit-identical parameters, the
bias is zeroed (it is not drawn from the generator), and a different seed
actually changes the weights.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from torch import nn  # noqa: E402

from verl.utils.model import _get_valuehead_module, _init_valuehead_parameters  # noqa: E402


class _Head(nn.Module):
    """Minimal stand-in for a HuggingFace value head."""

    def __init__(self):
        super().__init__()
        self.summary = nn.Linear(8, 1)

    def forward(self, hidden_states):
        return self.summary(hidden_states)


def _build_head():
    with torch.no_grad():
        head = _Head()
        # Make sure the two instances start from different values, so equality
        # after seeding cannot be a coincidence.
        for parameter in head.parameters():
            parameter.fill_(1.0)
    return head


class TestSeededInitialization:
    def test_same_seed_is_bitwise_identical(self):
        # Arrange
        config = SimpleNamespace(initializer_range=0.02)
        first, second = _build_head(), _build_head()

        # Act
        _init_valuehead_parameters(first, config, value_head_seed=42)
        _init_valuehead_parameters(second, config, value_head_seed=42)

        # Assert
        first_params = dict(first.named_parameters())
        second_params = dict(second.named_parameters())
        assert first_params.keys() == second_params.keys()
        for name, parameter in first_params.items():
            assert torch.equal(parameter, second_params[name]), name

    def test_bias_is_zeroed(self):
        head = _build_head()

        _init_valuehead_parameters(head, SimpleNamespace(initializer_range=0.02), value_head_seed=7)

        assert torch.equal(head.summary.bias, torch.zeros_like(head.summary.bias))

    def test_weight_is_not_left_at_the_original_value(self):
        head = _build_head()

        _init_valuehead_parameters(head, SimpleNamespace(initializer_range=0.02), value_head_seed=7)

        assert not torch.allclose(head.summary.weight, torch.ones_like(head.summary.weight))

    def test_different_seeds_differ(self):
        config = SimpleNamespace(initializer_range=0.02)
        first, second = _build_head(), _build_head()

        _init_valuehead_parameters(first, config, value_head_seed=1)
        _init_valuehead_parameters(second, config, value_head_seed=2)

        assert not torch.equal(first.summary.weight, second.summary.weight)

    def test_falls_back_to_default_std_without_initializer_range(self):
        head = _build_head()

        # Act: no `initializer_range` attribute at all.
        _init_valuehead_parameters(head, SimpleNamespace(), value_head_seed=3)

        # Assert: the documented fallback (0.02) is used, and the init is finite.
        assert torch.isfinite(head.summary.weight).all()
        assert head.summary.weight.std().item() < 0.1


class TestValueheadModuleResolution:
    def test_prefers_score(self):
        model = SimpleNamespace(score=nn.Linear(2, 1), classifier=nn.Linear(2, 1))

        assert _get_valuehead_module(model, ("score", "classifier")) is model.score

    def test_falls_back_to_classifier(self):
        model = SimpleNamespace(classifier=nn.Linear(2, 1))

        assert _get_valuehead_module(model, ("score", "classifier")) is model.classifier

    def test_ignores_parameterless_modules(self):
        # TRL wraps the head in `v_head.summary`; an empty module must not match.
        model = SimpleNamespace(score=nn.Identity())

        with pytest.raises(RuntimeError, match="exposes none of"):
            _get_valuehead_module(model, ("score", "classifier"))

    def test_error_names_the_candidates(self):
        with pytest.raises(RuntimeError, match="`score` / `classifier`"):
            _get_valuehead_module(SimpleNamespace(), ("score", "classifier"))

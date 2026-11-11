# Copyright (c) 2026 BAAI. All rights reserved.
"""Unit tests for the ECCL checkpoint engine.

The engine imports ``ray``, ``zmq`` and ``torch_gcu``, and only registers when
an Enflame GCU device is present; the whole module is skipped otherwise (i.e.
on non-GCU CI hosts).
"""

import dataclasses
from unittest import mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("ray")
pytest.importorskip("zmq")
pytest.importorskip("torch_gcu")

from verl.utils.device import is_torch_gcu_available  # noqa: E402

if not is_torch_gcu_available():
    # The engine raises ImportError at import time without a device, so a
    # collection-time skip is the only way to keep non-GCU hosts green.
    pytest.skip("no Enflame GCU device available", allow_module_level=True)

import torch  # noqa: E402

from verl.checkpoint_engine import eccl_checkpoint_engine as ecc_mod  # noqa: E402
from verl.checkpoint_engine.base import CheckpointEngineRegistry  # noqa: E402
from verl.checkpoint_engine.eccl_checkpoint_engine import (  # noqa: E402
    BroadcastOperation,
    ECCLCheckpointEngine,
    MasterMetadata,
)


def _make_engine(**kwargs):
    defaults = {"bucket_size": 1024, "is_master": False}
    defaults.update(kwargs)
    with mock.patch.object(ecc_mod, "get_device_id", return_value=0):
        return ECCLCheckpointEngine(**defaults)


class TestRegistration:
    def test_registered_as_nccl_backend(self):
        assert CheckpointEngineRegistry.get("nccl") is ECCLCheckpointEngine

    def test_master_metadata_has_no_dist_endpoint(self):
        # ECCL joins through ray.util.collective, so unlike the HCCL engine the
        # master only advertises its zeromq endpoint.
        names = {f.name for f in dataclasses.fields(MasterMetadata)}
        assert names == {"zmq_ip", "zmq_port"}


class TestBuildTopology:
    def test_actor_and_rollout_ranks(self):
        metadata = [MasterMetadata(zmq_ip="10.0.0.1", zmq_port=1234)]

        actor_kwargs, rollout_kwargs = ECCLCheckpointEngine.build_topology(4, 2, metadata)

        # Arrange: one sending actor rank (0) and three consuming ones (-1).
        assert actor_kwargs["rank"] == [0, -1, -1, -1]
        assert actor_kwargs["world_size"] == [3, 3, 3, 3]
        assert actor_kwargs["master_metadata"] == metadata * 4
        # Rollout ranks are numbered after the single actor sender.
        assert rollout_kwargs["rank"] == [1, 2]
        assert rollout_kwargs["world_size"] == [3, 3]
        assert rollout_kwargs["master_metadata"] == metadata * 2


class TestPrepare:
    def test_allocates_two_uint8_bucket_buffers(self, monkeypatch):
        engine = _make_engine(bucket_size=512)
        allocations = []

        def fake_zeros(size, dtype=None, device=None):
            allocations.append((size, dtype, device))
            return torch.empty(0)

        monkeypatch.setattr(ecc_mod.torch, "zeros", fake_zeros)
        monkeypatch.setattr(ecc_mod, "get_device_name", lambda: "gcu")

        assert engine.prepare() is None  # non-master has no metadata to publish
        assert allocations == [(512, torch.uint8, "gcu:0"), (512, torch.uint8, "gcu:0")]


class TestInitProcessGroup:
    def test_non_sender_rank_skips_collective(self, monkeypatch):
        engine = _make_engine()
        init_group = mock.Mock()
        monkeypatch.setattr(ecc_mod.collective, "init_collective_group", init_group)
        monkeypatch.setattr(ecc_mod.collective, "barrier", mock.Mock())

        engine.init_process_group(rank=-1, world_size=3, master_metadata=None)

        assert engine.rank == -1
        assert engine.world_size == 3
        init_group.assert_not_called()

    def test_worker_joins_group_and_connects_zmq(self, monkeypatch):
        engine = _make_engine()
        metadata = MasterMetadata(zmq_ip="10.0.0.1", zmq_port=1234)
        init_group = mock.Mock()
        barrier = mock.Mock()
        connect = mock.Mock()
        monkeypatch.setattr(ecc_mod.collective, "is_group_initialized", lambda name: False)
        monkeypatch.setattr(ecc_mod.collective, "init_collective_group", init_group)
        monkeypatch.setattr(ecc_mod.collective, "barrier", barrier)
        monkeypatch.setattr(engine, "_connect_zmq_client", connect)

        engine.init_process_group(rank=1, world_size=3, master_metadata=metadata)

        init_group.assert_called_once_with(3, 1, "eccl", "default")
        connect.assert_called_once_with(metadata)
        barrier.assert_called_once_with("default")


class TestBroadcastOperation:
    def test_source_publishes_metadata_then_broadcasts(self, monkeypatch):
        broadcast = mock.Mock()
        monkeypatch.setattr(ecc_mod.collective, "broadcast", broadcast)
        socket = mock.Mock()
        bucket = torch.empty(8, dtype=torch.uint8)
        metadata = {"bucket_meta": {}, "is_last": True}

        BroadcastOperation(rank=0, group_name="default", bucket=bucket, metadata=metadata, socket=socket, topic="t")

        socket.send_string.assert_called_once_with("t", flags=ecc_mod.zmq.SNDMORE)
        socket.send_pyobj.assert_called_once_with(metadata)
        broadcast.assert_called_once_with(bucket, src_rank=0, group_name="default")

    def test_worker_receives_metadata_then_broadcasts(self, monkeypatch):
        broadcast = mock.Mock()
        monkeypatch.setattr(ecc_mod.collective, "broadcast", broadcast)
        socket = mock.Mock()
        socket.recv_pyobj.return_value = {"bucket_meta": {"w": "meta"}, "is_last": False}
        bucket = torch.empty(8, dtype=torch.uint8)

        op = BroadcastOperation(rank=1, group_name="default", bucket=bucket, metadata=None, socket=socket, topic="t")

        socket.recv_string.assert_called_once_with()
        assert op.metadata == {"bucket_meta": {"w": "meta"}, "is_last": False}
        broadcast.assert_called_once_with(bucket, src_rank=0, group_name="default")


class TestFinalize:
    def test_releases_buffers_and_cache(self, monkeypatch):
        engine = _make_engine()
        engine.send_buf = torch.empty(8, dtype=torch.uint8)
        engine.recv_buf = torch.empty(8, dtype=torch.uint8)
        fake_device = mock.Mock()
        monkeypatch.setattr(ecc_mod, "get_torch_device", lambda: fake_device)

        engine.finalize()

        assert engine.send_buf is None
        assert engine.recv_buf is None
        fake_device.empty_cache.assert_called_once_with()

    def test_rebuild_group_destroys_collective_group(self, monkeypatch):
        engine = _make_engine(rebuild_group=True)
        engine.rank = 1
        engine.world_size = 3
        destroy = mock.Mock()
        monkeypatch.setattr(ecc_mod.collective, "destroy_collective_group", destroy)
        monkeypatch.setattr(ecc_mod, "get_torch_device", mock.Mock())

        engine.finalize()

        destroy.assert_called_once_with("default")
        assert engine.rank is None
        assert engine.world_size is None


class TestSendWeights:
    @pytest.mark.asyncio
    async def test_non_sender_rank_consumes_weights_without_broadcasting(self, monkeypatch):
        engine = _make_engine()
        engine.rank = -1
        broadcast = mock.Mock()
        monkeypatch.setattr(ecc_mod.collective, "broadcast", broadcast)

        consumed = []

        def weights():
            for name in ("a", "b"):
                consumed.append(name)
                yield name, torch.zeros(1)

        await engine.send_weights(weights())

        assert consumed == ["a", "b"]
        broadcast.assert_not_called()

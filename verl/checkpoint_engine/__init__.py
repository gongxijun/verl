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

from verl.utils.device import is_torch_gcu_available

from .base import (
    CheckpointEngine,
    CheckpointEngineManager,
    CheckpointEngineRegistry,
    CheckpointEngineWorker,
    ColocatedCheckpointEngine,
    TensorMeta,
)

__all__ = [
    "CheckpointEngine",
    "CheckpointEngineRegistry",
    "TensorMeta",
    "ColocatedCheckpointEngine",
    "CheckpointEngineManager",
    "CheckpointEngineWorker",
]

# Every engine below is optional: it carries its own transport dependency, and
# the vendor engines are mutually exclusive by construction (nccl, hccl and eccl
# all register the "nccl" backend name, and each checks for the presence of its
# own accelerator before importing). Failures are recorded so that asking for a
# backend whose module did not import reports the missing dependency instead of
# just "not registered".
#
# The NCCL engine is skipped here rather than guarding itself, so that no
# vendor-neutral engine has to know about a specific accelerator, and so that its
# CUDA-only cupy dependency is never imported on a GCU host. Skipping it is safe
# because the ECCL engine registers the same "nccl" backend name below.
if is_torch_gcu_available():
    CheckpointEngineRegistry.record_import_error(
        "nccl_checkpoint_engine",
        ImportError("NCCLCheckpointEngine is unavailable on Enflame GCU; the eccl engine provides the nccl backend."),
    )
    NCCLCheckpointEngine = None
else:
    try:
        from .nccl_checkpoint_engine import NCCLCheckpointEngine

        __all__ += ["NCCLCheckpointEngine"]
    except ImportError as e:
        CheckpointEngineRegistry.record_import_error("nccl_checkpoint_engine", e)
        NCCLCheckpointEngine = None

try:
    from .hccl_checkpoint_engine import HCCLCheckpointEngine

    __all__ += ["HCCLCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("hccl_checkpoint_engine", e)
    HCCLCheckpointEngine = None

try:
    from .eccl_checkpoint_engine import ECCLCheckpointEngine

    __all__ += ["ECCLCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("eccl_checkpoint_engine", e)
    ECCLCheckpointEngine = None

try:
    from .nixl_checkpoint_engine import NIXLCheckpointEngine

    __all__ += ["NIXLCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("nixl_checkpoint_engine", e)
    NIXLCheckpointEngine = None

try:
    from .kimi_checkpoint_engine import KIMICheckpointEngine

    __all__ += ["KIMICheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("kimi_checkpoint_engine", e)
    KIMICheckpointEngine = None

try:
    from .mooncake_checkpoint_engine import MooncakeCheckpointEngine

    __all__ += ["MooncakeCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("mooncake_checkpoint_engine", e)
    MooncakeCheckpointEngine = None

try:
    from .delta_checkpoint_engine import DeltaShardedCheckpointEngine

    __all__ += ["DeltaShardedCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("delta_checkpoint_engine", e)
    DeltaShardedCheckpointEngine = None

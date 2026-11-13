"""Model loading / generation / checkpoint helpers.

Nothing here imports torch, transformers or peft at module import time: the
heavy imports live inside the functions so prompt/eval code stays importable on
a CPU-only box with stdlib only.
"""

from __future__ import annotations

from lifemate.models.checkpoint import (
    CheckpointInfo,
    load_checkpoint_meta,
    merge_lora_adapter,
    resolve_checkpoint,
    save_checkpoint,
)
from lifemate.models.generation import CallableGenerator, GenerationStats, HFGenerator
from lifemate.models.loader import (
    ModelBundle,
    count_parameters,
    load_model_and_tokenizer,
)

__all__ = [
    "CallableGenerator",
    "CheckpointInfo",
    "GenerationStats",
    "HFGenerator",
    "ModelBundle",
    "count_parameters",
    "load_checkpoint_meta",
    "load_model_and_tokenizer",
    "merge_lora_adapter",
    "resolve_checkpoint",
    "save_checkpoint",
]

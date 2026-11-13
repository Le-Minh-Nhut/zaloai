"""Training utilities: SFT collator, full/LoRA configuration, trainer, preflight.

Nothing here imports torch/transformers/peft at module import time; every heavy
import is function-local, so ``import lifemate.training`` works on a
stdlib-only box (CI, data preparation, tests).
"""

from __future__ import annotations

from .callbacks import LoggingCallback, NullTracker, StopController
from .collator import (
    SupervisedCollator,
    assert_loss_masking,
    build_supervised_example,
    render_prompt_text,
    to_torch_batch,
)
from .full_finetuning import build_optimizer_and_scheduler, configure_full_finetuning
from .lora_finetuning import configure_lora, discover_target_modules, load_adapter, merge_adapter, save_adapter
from .memory_preflight import format_report, run_memory_preflight
from .trainer import (
    CheckpointManager,
    SFTTrainer,
    TrainResult,
    TrainingInterrupted,
    read_training_state,
    set_seed,
)

__all__ = [
    "CheckpointManager",
    "LoggingCallback",
    "NullTracker",
    "SFTTrainer",
    "StopController",
    "SupervisedCollator",
    "TrainResult",
    "TrainingInterrupted",
    "assert_loss_masking",
    "build_optimizer_and_scheduler",
    "build_supervised_example",
    "configure_full_finetuning",
    "configure_lora",
    "discover_target_modules",
    "format_report",
    "load_adapter",
    "merge_adapter",
    "read_training_state",
    "render_prompt_text",
    "run_memory_preflight",
    "save_adapter",
    "set_seed",
    "to_torch_batch",
]

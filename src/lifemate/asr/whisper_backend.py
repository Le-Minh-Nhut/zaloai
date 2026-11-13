"""Whisper-family backend (PhoWhisper / Whisper) via transformers.

Heavy imports are deferred to ``__init__`` so importing this module is free.
Weights are NEVER downloaded by this module: when ``HF_HUB_OFFLINE=1`` or
``offline=True`` we pass ``local_files_only=True`` to the loaders, and a missing
cache produces an actionable :class:`AsrError` instead of a network fetch.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .interface import (
    DEFAULT_MAX_AUDIO_SECONDS,
    TARGET_SAMPLE_RATE,
    AsrError,
    TranscriptResult,
    load_audio_16k_mono,
)

__all__ = ["WhisperBackend"]

#: Rough token budget: Vietnamese Whisper uses ~4-6 tokens/second. 12 is a safe
#: upper bound that still bounds runaway generation; +32 covers short clips.
_TOKENS_PER_SECOND = 12
_MIN_NEW_TOKENS = 64
_MAX_NEW_TOKENS = 448


class WhisperBackend:
    """Deterministic greedy Whisper inference on CPU or a given device."""

    name = "whisper"
    is_oracle = False

    def __init__(
        self,
        model_name: str = "vinai/PhoWhisper-small",
        device: str = "auto",
        language: str = "vi",
        torch_dtype: str | None = "float16",
        *,
        offline: bool | None = None,
        max_audio_seconds: float | None = DEFAULT_MAX_AUDIO_SECONDS,
        local_files_only: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.language = language or None
        self.max_audio_seconds = max_audio_seconds
        self._requested_dtype = torch_dtype
        self._offline = (
            os.environ.get("HF_HUB_OFFLINE", "0") == "1" if offline is None else bool(offline)
        )
        self._local_files_only = (
            self._offline if local_files_only is None else bool(local_files_only)
        )
        self._torch: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"
        self._dtype: Any = None
        self._load(device)

    # ------------------------------------------------------------------
    def _load(self, device: str) -> None:
        try:
            import torch  # noqa: PLC0415 - intentional lazy import
            from transformers import (  # noqa: PLC0415
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )
        except ImportError as exc:  # pragma: no cover - env dependent
            raise AsrError(
                "WhisperBackend requires the 'torch' and 'transformers' packages "
                f"(missing: {getattr(exc, 'name', 'unknown')}). Install with "
                "`pip install torch transformers` or use backend='oracle' for "
                "transcript-based evaluation. No network access was attempted."
            ) from exc

        torch_mod = torch
        self._torch = torch_mod
        if device == "auto":
            if torch_mod.cuda.is_available():
                device = "cuda"
            elif getattr(torch_mod.backends, "mps", None) is not None and torch_mod.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype = None
        if self._requested_dtype:
            try:
                dtype = getattr(torch_mod, self._requested_dtype)
            except AttributeError as exc:
                raise AsrError(
                    f"unknown torch dtype {self._requested_dtype!r}; use float16/float32/bfloat16 or None"
                ) from exc
        if device == "cpu" and dtype is not None and dtype == torch_mod.float16:
            # fp16 matmuls are not reliably supported on CPU; fall back loudly.
            dtype = torch_mod.float32

        try:
            self._processor = WhisperProcessor.from_pretrained(
                self.model_name, local_files_only=self._local_files_only
            )
            load_kwargs: dict[str, Any] = {"local_files_only": self._local_files_only}
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            self._model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name, **load_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - normalized into AsrError
            hint = (
                " (offline mode: weights must already be in the HF cache)"
                if self._local_files_only
                else " (set HF_HUB_OFFLINE=1 to forbid downloads)"
            )
            raise AsrError(
                f"failed to load ASR model {self.model_name!r}{hint}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._device = device
        self._dtype = dtype
        self._model.eval()
        self._model.to(device)

    # ------------------------------------------------------------------
    @property
    def device(self) -> str:
        return self._device

    def parameter_count(self) -> int | None:
        """Total parameter count (competition model-size accounting)."""
        if self._model is None:
            return None
        try:
            return int(sum(p.numel() for p in self._model.parameters()))
        except Exception:  # pragma: no cover
            return None

    def model_size_report(self) -> dict[str, Any]:
        n = self.parameter_count()
        return {
            "backend": self.name,
            "model_name": self.model_name,
            "device": self._device,
            "torch_dtype": str(self._dtype).replace("torch.", "") if self._dtype else None,
            "parameters": n,
            "parameters_millions": round(n / 1e6, 2) if n else None,
            "parameters_by_precision": (
                {
                    label: round(n * bytes_per / 1e6, 1)
                    for label, bytes_per in (("float32_mb", 4), ("float16_mb", 2), ("int8_mb", 1))
                }
                if n
                else None
            ),
            "local_files_only": self._local_files_only,
            "language": self.language,
        }

    # ------------------------------------------------------------------
    def _max_new_tokens(self, duration_s: float | None) -> int:
        if not duration_s or duration_s <= 0:
            return _MIN_NEW_TOKENS
        return max(_MIN_NEW_TOKENS, min(_MAX_NEW_TOKENS, int(duration_s * _TOKENS_PER_SECOND) + 32))

    def transcribe(
        self, audio_path: str | Path, *, sample_id: str | None = None, lenient: bool = False
    ) -> TranscriptResult:
        t0 = time.perf_counter()
        info_duration: float | None = None
        try:
            samples, rate = load_audio_16k_mono(
                audio_path, target_rate=TARGET_SAMPLE_RATE, max_seconds=self.max_audio_seconds
            )
            info_duration = len(samples) / rate if rate else None
            torch = self._torch
            inputs = self._processor(
                samples, sampling_rate=rate, return_tensors="pt"
            )
            input_features = inputs.input_features.to(self._device)
            if self._dtype is not None and self._device != "cpu":
                input_features = input_features.to(self._dtype)
            gen_kwargs: dict[str, Any] = {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self._max_new_tokens(info_duration),
            }
            if self.language:
                gen_kwargs["language"] = self.language
                gen_kwargs["task"] = "transcribe"
            with torch.inference_mode():
                ids = self._model.generate(input_features, **gen_kwargs)
            raw = self._processor.batch_decode(ids, skip_special_tokens=True)
            text = raw[0].strip() if raw else ""
            return TranscriptResult(
                text=text,
                raw_text=raw[0] if raw else None,
                language=self.language,
                duration_s=info_duration,
                latency_s=time.perf_counter() - t0,
                backend=self.name,
                model_name=self.model_name,
            )
        except AsrError:
            raise
        except Exception as exc:  # noqa: BLE001 - inference-time failures
            msg = f"{type(exc).__name__}: {exc}"
            if not lenient:
                raise AsrError(f"transcription failed for {audio_path}: {msg}") from exc
            return TranscriptResult(
                text="",
                duration_s=info_duration,
                latency_s=time.perf_counter() - t0,
                backend=self.name,
                model_name=self.model_name,
                error=msg,
                language=self.language,
            )

    def __repr__(self) -> str:  # pragma: no cover
        return f"WhisperBackend(model_name={self.model_name!r}, device={self._device!r})"

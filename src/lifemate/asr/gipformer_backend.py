"""Gipformer-68M-RNNT backend via sherpa-onnx.

Default speech recognition backend for LifeMate (Zalo AI Challenge 2026 Track 1).
Uses the transducer ONNX models exported from Zipformer:
  encoder: encoder.int8.onnx (or encoder.onnx)
  decoder: decoder.int8.onnx (or decoder.onnx)
  joiner:  joiner.int8.onnx  (or joiner.onnx)
  tokens:  tokens.txt

Heavy packages (sherpa_onnx, soundfile) are imported lazily inside __init__
and transcribe so importing this module is free on standard Python.
"""

from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path
from typing import Any

from ..errors import ProhibitedAccessError
from .interface import (
    DEFAULT_MAX_AUDIO_SECONDS,
    TARGET_SAMPLE_RATE,
    AsrError,
    TranscriptResult,
    load_audio_16k_mono,
    resample_linear,
)

logger = logging.getLogger(__name__)

__all__ = ["GipformerBackend", "DEFAULT_MODEL_ID", "PINNED_REVISION", "lower_if_uppercase_only"]

DEFAULT_MODEL_ID = "g-group-ai-lab/gipformer-68M-rnnt"
PINNED_REVISION = "318322b009200f8e77a3751fa64638e5cc85a7dc"

_INT8_FILENAMES = {
    "encoder": "encoder.int8.onnx",
    "decoder": "decoder.int8.onnx",
    "joiner": "joiner.int8.onnx",
    "tokens": "tokens.txt",
}

_FP32_FILENAMES = {
    "encoder": "encoder.onnx",
    "decoder": "decoder.onnx",
    "joiner": "joiner.onnx",
    "tokens": "tokens.txt",
}


def lower_if_uppercase_only(text: str) -> str:
    """Lowercase ``text`` only when it carries no case information.

    g-group-ai-lab/gipformer-68M-rnnt emits ALL UPPERCASE (3 lowercase entries out
    of 2000 tokens (25,847 bytes) in its tokens file), so its casing is an
    artifact of the export, not
    a prediction. Training transcripts are sentence-case, so feeding uppercase to
    the agent shifts the prompt's request field on every sample. Output that does
    contain lowercase is returned unchanged, so a mixed-case recognizer keeps its
    proper nouns.
    """
    s = str(text or "")
    if s and not any(ch.islower() for ch in s) and any(ch.isupper() for ch in s):
        return s.lower()
    return s


class GipformerBackend:
    """Offline Zipformer-Transducer ASR backend via sherpa-onnx."""

    name = "gipformer"
    is_oracle = False

    def __init__(
        self,
        model_dir: str | Path | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = PINNED_REVISION,
        quantization: str = "int8",
        device: str = "cpu",
        num_threads: int = 4,
        decoding_method: str = "modified_beam_search",
        offline: bool = True,
        max_audio_seconds: float | None = DEFAULT_MAX_AUDIO_SECONDS,
        lenient: bool = False,
    ) -> None:
        if "1.5" in model_id or "gipformer1.5" in model_id:
            raise AsrError(
                f"model_id {model_id!r} is rejected: gipformer1.5 is a different model. "
                f"LifeMate target is {DEFAULT_MODEL_ID}"
            )

        quant = str(quantization).lower()
        if quant not in ("int8", "fp32"):
            raise AsrError(f"unknown quantization {quantization!r}; choose 'int8' or 'fp32'")

        self.model_id = model_id
        self.revision = revision
        self.quantization = quant
        self.model_dir = Path(model_dir) if model_dir else None
        self.num_threads = int(num_threads)
        method = str(decoding_method).strip().lower()
        if method not in ("greedy_search", "modified_beam_search"):
            raise AsrError(
                f"unknown decoding_method {decoding_method!r}; "
                "choose 'modified_beam_search' (default, matches published WER) or "
                "'greedy_search' (fast/experimental)"
            )
        self.decoding_method = method
        self.offline = bool(offline)
        self.max_audio_seconds = max_audio_seconds
        self.lenient = lenient

        self._device = self._resolve_device(device)
        self._resolved_paths: dict[str, Path] = {}
        self._recognizer: Any = None
        self._load()

    def _resolve_device(self, device: str) -> str:
        dev = str(device).lower().strip()
        if dev == "auto":
            has_cuda = False
            try:
                import torch  # noqa: PLC0415

                has_cuda = torch.cuda.is_available()
            except ImportError:
                pass
            return "cuda" if has_cuda else "cpu"
        if dev in ("cuda", "gpu"):
            return "cuda"
        if dev == "cpu":
            return "cpu"
        raise AsrError(f"unknown device {device!r}; use 'cpu', 'cuda', or 'auto'")

    def _load(self) -> None:
        try:
            import sherpa_onnx  # noqa: PLC0415
        except ImportError as exc:
            missing_pkg = getattr(exc, "name", "sherpa_onnx")
            raise AsrError(
                f"GipformerBackend requires 'sherpa-onnx' and 'sherpa-onnx-core' (missing: {missing_pkg}). "
                "Install with: pip install \"sherpa-onnx>=1.10\" \"sherpa-onnx-core>=1.10\" soundfile "
                "or pip install -e \".[asr]\"."
            ) from exc

        target_files = _INT8_FILENAMES if self.quantization == "int8" else _FP32_FILENAMES
        other_files = _FP32_FILENAMES if self.quantization == "int8" else _INT8_FILENAMES

        if self.offline:
            search_dir = self.model_dir if self.model_dir else Path("models/gipformer-68M-rnnt")
            missing: list[str] = []
            resolved: dict[str, Path] = {}
            for role, filename in target_files.items():
                p = search_dir / filename
                if not p.is_file():
                    missing.append(filename)
                else:
                    resolved[role] = p
            if missing:
                opposite_found = (search_dir / other_files["encoder"]).is_file()
                opposite_hint = (
                    f" (found {'fp32' if self.quantization == 'int8' else 'int8'} files, "
                    f"but quantization={self.quantization!r} was requested; silent fallback is forbidden)"
                    if opposite_found
                    else ""
                )
                raise AsrError(
                    f"missing Gipformer ({self.quantization}) model files in {search_dir}: {', '.join(missing)}{opposite_hint}. "
                    f"Download with: python scripts/download_asr_model.py --output-dir {search_dir} --quantization {self.quantization}"
                )
            self._resolved_paths = resolved
        else:
            try:
                import huggingface_hub  # noqa: PLC0415
            except ImportError as exc:
                raise AsrError(
                    "online model resolution (offline=False) requires 'huggingface_hub'. "
                    "Install with: pip install \"huggingface-hub>=0.20.0\""
                ) from exc
            resolved = {}
            for role, filename in target_files.items():
                local_path = huggingface_hub.hf_hub_download(
                    repo_id=self.model_id,
                    filename=filename,
                    revision=self.revision,
                )
                resolved[role] = Path(local_path)
            logger.info("Gipformer online resolution: %s", resolved)
            self._resolved_paths = resolved

        # Build recognizer
        sig = inspect.signature(sherpa_onnx.OfflineRecognizer.from_transducer)
        kwargs: dict[str, Any] = {
            "encoder": str(self._resolved_paths["encoder"]),
            "decoder": str(self._resolved_paths["decoder"]),
            "joiner": str(self._resolved_paths["joiner"]),
            "tokens": str(self._resolved_paths["tokens"]),
            "num_threads": self.num_threads,
            "sample_rate": TARGET_SAMPLE_RATE,
            "feature_dim": 80,
            "decoding_method": self.decoding_method,
        }
        if self._device == "cuda":
            if "provider" in sig.parameters:
                kwargs["provider"] = "cuda"
            else:
                raise AsrError(
                    "installed sherpa-onnx OfflineRecognizer.from_transducer does not accept provider='cuda'. "
                    "Use a CUDA-compatible sherpa-onnx build or pass device='cpu'."
                )

        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)
        except Exception as exc:
            raise AsrError(f"failed to initialize sherpa-onnx recognizer: {exc}") from exc

    @property
    def recognizer(self) -> Any:
        return self._recognizer

    @property
    def device(self) -> str:
        return self._device

    def parameter_count(self) -> int:
        """Gipformer-68M class budget."""
        return 68_000_000

    def model_size_report(self) -> dict[str, Any]:
        """On-disk and parameter size accounting."""
        total_bytes = 0
        file_sizes: dict[str, int] = {}
        for role, path in self._resolved_paths.items():
            if path.is_file():
                sz = path.stat().st_size
                total_bytes += sz
                file_sizes[path.name] = sz
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "revision": self.revision,
            "quantization": self.quantization,
            "device": self._device,
            "parameter_class": "68M",
            "parameters": 68_000_000,
            "on_disk_bytes": total_bytes,
            "on_disk_mb": round(total_bytes / (1024 * 1024), 2),
            "files": file_sizes,
            "offline": self.offline,
        }

    def _load_audio(self, audio_path: str | Path) -> tuple[list[float], int, float]:
        path = Path(audio_path)
        from ..compliance import assert_path_allowed

        path = assert_path_allowed(path)  # *_environment paths are never opened
        assert_path_allowed(path.resolve())  # ...and not via a symlink target either
        if not path.is_file():
            raise AsrError(f"audio file not found: {path}")

        samples: list[float] = []
        rate: int = TARGET_SAMPLE_RATE
        loaded_sf = False

        try:
            import soundfile as sf  # noqa: PLC0415

            # Bound the READ, not just the result: max_audio_seconds is a memory
            # guard, so a 1-hour file must never be fully materialised first.
            frame_budget = -1
            if self.max_audio_seconds and self.max_audio_seconds > 0:
                info = sf.info(str(path))
                if info.samplerate:
                    frame_budget = max(1, int(self.max_audio_seconds * info.samplerate))
            data, rate = sf.read(str(path), dtype="float32", frames=frame_budget)
            if data.size == 0:
                raise AsrError("empty audio")
            if data.ndim == 2:
                samples = data.mean(axis=1).tolist()
            elif data.ndim == 1:
                samples = data.tolist()
            else:
                raise AsrError(f"unsupported audio dimension: {data.ndim}")
            loaded_sf = True
        except AsrError:
            raise
        except Exception:
            loaded_sf = False

        if not loaded_sf:
            try:
                samples, rate = load_audio_16k_mono(
                    path, target_rate=TARGET_SAMPLE_RATE, max_seconds=self.max_audio_seconds
                )
            except Exception as exc:
                raise AsrError(f"failed to load audio {path}: {exc}") from exc

        if not samples:
            raise AsrError("empty audio")

        if rate != TARGET_SAMPLE_RATE:
            samples = resample_linear(samples, src_rate=rate, dst_rate=TARGET_SAMPLE_RATE)
            rate = TARGET_SAMPLE_RATE

        if self.max_audio_seconds and self.max_audio_seconds > 0:
            max_len = int(self.max_audio_seconds * rate)
            if len(samples) > max_len:
                samples = samples[:max_len]

        duration_s = len(samples) / float(rate)
        if duration_s == 0:
            raise AsrError("empty audio")
        return samples, rate, duration_s

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        sample_id: str | None = None,
        lenient: bool = False,
    ) -> TranscriptResult:
        t0 = time.perf_counter()
        duration_s: float | None = None
        effective_lenient = lenient or self.lenient

        try:
            samples, rate, duration_s = self._load_audio(audio_path)

            try:
                import numpy as np  # noqa: PLC0415

                wave_input = np.asarray(samples, dtype=np.float32)
            except ImportError:
                wave_input = samples

            stream = self._recognizer.create_stream()
            stream.accept_waveform(rate, wave_input)
            self._recognizer.decode_streams([stream])
            raw_text = getattr(getattr(stream, "result", None), "text", "") or ""
            raw_text = str(raw_text).strip()

            # ``raw_text`` keeps the recognizer string verbatim (this export emits
            # ALL UPPERCASE — its tokens file has 3 lowercase entries out of 2000
            # tokens (25,847 bytes)).
            # ``text`` is what the agent sees: an uppercase-only string carries no
            # case information, while every training transcript is sentence-case, so
            # passing it through would shift the prompt's most important field on the
            # whole corpus. Mixed-case output is left untouched.
            text = lower_if_uppercase_only(raw_text)

            # LLM normalization still happens once, in the inference pipeline, so
            # every backend is fed to the agent identically.
            return TranscriptResult(
                text=text,
                raw_text=raw_text,
                language="vi",
                duration_s=duration_s,
                latency_s=time.perf_counter() - t0,
                backend=self.name,
                model_name=self.model_id,
            )
        except ProhibitedAccessError:
            raise  # compliance violations are never converted to a result
        except AsrError as exc:
            if not effective_lenient:
                raise
            return TranscriptResult(
                text="",
                error=str(exc),
                duration_s=duration_s,
                latency_s=time.perf_counter() - t0,
                backend=self.name,
                model_name=self.model_id,
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if not effective_lenient:
                raise AsrError(f"transcription failed for {audio_path}: {msg}") from exc
            return TranscriptResult(
                text="",
                error=msg,
                duration_s=duration_s,
                latency_s=time.perf_counter() - t0,
                backend=self.name,
                model_name=self.model_id,
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GipformerBackend(model_id={self.model_id!r}, "
            f"quantization={self.quantization!r}, device={self._device!r})"
        )

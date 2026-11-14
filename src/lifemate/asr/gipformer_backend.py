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
from .model_specs import (
    DEFAULT_SPEC_KEY,
    GIPFORMER_SPECS,
    download_hint,
    artifacts_for,
    resolve_gipformer_spec,
    spec_key,
    validate_artifacts,
)
from .nbest import NBestResult, NBestUnsupportedError, require_nbest_support, validate_nbest_size

logger = logging.getLogger(__name__)

__all__ = ["GipformerBackend", "DEFAULT_MODEL_ID", "PINNED_REVISION", "lower_if_uppercase_only"]

DEFAULT_MODEL_ID = "g-group-ai-lab/gipformer-68M-rnnt"
PINNED_REVISION = "318322b009200f8e77a3751fa64638e5cc85a7dc"

#: Filename sets now live in the spec table (``lifemate.asr.model_specs``); the
#: aliases stay so anything importing them keeps working.
_INT8_FILENAMES = artifacts_for(GIPFORMER_SPECS[DEFAULT_SPEC_KEY], "int8")
_FP32_FILENAMES = artifacts_for(GIPFORMER_SPECS[DEFAULT_SPEC_KEY], "fp32")


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
        nbest_size: int = 0,
    ) -> None:
        quant = str(quantization).lower()
        if quant not in ("int8", "fp32"):
            raise AsrError(f"unknown quantization {quantization!r}; choose 'int8' or 'fp32'")

        # Spec-driven identity: the model id (or the model directory's name) must
        # name a known checkpoint. An unknown id is an error — there is no silent
        # fallback to the 68M default, and "1.5" is no longer special-cased.
        self.spec = resolve_gipformer_spec(model_id, kind=model_dir)
        if self.spec.model_id != model_id and spec_key(model_id) != self.spec.key:
            raise AsrError(
                f"model_id {model_id!r} does not match the resolved spec "
                f"{self.spec.model_id!r}"
            )

        if not self.spec.verified:
            # The spec table knows this id, but no artifact set was ever observed
            # for it, so no recognizer can be built from it honestly. Derived from
            # the spec, not from a substring of the model id.
            raise AsrError(
                f"model_id {model_id!r} is rejected: spec {self.spec.key!r} is [UNVERIFIED] "
                f"and declares no verified artifact set. {self.spec.notes}"
            )

        self.model_id = self.spec.model_id
        # ``nbest_size`` is recorded, never able to change the decode: the pinned
        # runtime returns one hypothesis (see lifemate.asr.nbest).
        self.nbest_size = int(nbest_size or 0)
        # Revision provenance: the spec's own pinned revision when it has one.
        # An [UNVERIFIED] spec has none, and borrowing the 68M commit would be a
        # fabricated provenance, so it stays empty unless the caller names one.
        if self.spec.revision:
            self.revision = self.spec.revision
        elif revision and revision != PINNED_REVISION:
            self.revision = revision
        else:
            self.revision = ""
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
        # Artifact resolution runs BEFORE the sherpa import: a missing/unverified
        # model must produce the actionable "which file is missing" error even on
        # a box where sherpa-onnx is not installed at all.
        target_files = artifacts_for(self.spec, self.quantization)
        other_quant = "fp32" if self.quantization == "int8" else "int8"
        other_files = (
            artifacts_for(self.spec, other_quant) if other_quant in self.spec.quantizations else {}
        )

        if self.offline:
            search_dir = self.model_dir if self.model_dir else Path("models/gipformer-68M-rnnt")
            problems = validate_artifacts(self.spec, search_dir, quantization=self.quantization)
            if problems:
                opposite_found = bool(other_files) and (
                    search_dir / other_files["encoder"]
                ).is_file()
                opposite_hint = (
                    f" (found {'fp32' if self.quantization == 'int8' else 'int8'} files, "
                    f"but quantization={self.quantization!r} was requested; silent fallback is forbidden)"
                    if opposite_found
                    else ""
                )
                raise AsrError(
                    f"missing Gipformer ({self.quantization}) model files in {search_dir}: "
                    f"{', '.join(problems)}{opposite_hint}. "
                    + download_hint(self.spec, search_dir, self.quantization)
                )
            self._resolved_paths = {role: search_dir / name for role, name in target_files.items()}
        else:
            try:
                import huggingface_hub  # noqa: PLC0415
            except ImportError as exc:
                raise AsrError(
                    "online model resolution (offline=False) requires 'huggingface_hub'. "
                    "Install with: pip install \"huggingface-hub>=0.20.0\""
                ) from exc
            if not self.revision:
                raise AsrError(
                    f"spec {self.spec.key!r} has no pinned revision; pass revision=... "
                    "explicitly for an online download instead of downloading a moving branch."
                )
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

        try:
            import sherpa_onnx  # noqa: PLC0415
        except ImportError as exc:
            missing_pkg = getattr(exc, "name", "sherpa_onnx")
            raise AsrError(
                f"GipformerBackend requires 'sherpa-onnx' and 'sherpa-onnx-core' (missing: {missing_pkg}). "
                f"Model files resolved fine in {self._resolved_paths['tokens'].parent}. "
                "Install with: pip install \"sherpa-onnx>=1.10\" \"sherpa-onnx-core>=1.10\" soundfile "
                "or pip install -e \".[asr]\"."
            ) from exc

        # Build recognizer
        sig = inspect.signature(sherpa_onnx.OfflineRecognizer.from_transducer)
        kwargs: dict[str, Any] = {
            "encoder": str(self._resolved_paths["encoder"]),
            "decoder": str(self._resolved_paths["decoder"]),
            "joiner": str(self._resolved_paths["joiner"]),
            "tokens": str(self._resolved_paths["tokens"]),
            "num_threads": self.num_threads,
            "sample_rate": self.spec.sample_rate,
            "feature_dim": self.spec.feature_dim,
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
            "spec_key": self.spec.key,
            "spec_verified": self.spec.verified,
            "sample_rate": self.spec.sample_rate,
            "feature_dim": self.spec.feature_dim,
            "nbest_supported": self.nbest_supported,
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

    def _decode(
        self, audio_path: str | Path
    ) -> tuple[str, float | None, int | None, float, float]:
        """Decode one file.

        Returns ``(raw_text, acoustic_logprob, n_tokens, duration_s, latency_s)``.
        ``acoustic_logprob`` is the sum of the result's ``ys_log_probs`` — the
        only per-token score the pinned runtime exposes. It is comparable only
        against other decodes of *this* model, never across models.
        """
        t0 = time.perf_counter()
        samples, rate, duration_s = self._load_audio(audio_path)

        try:
            import numpy as np  # noqa: PLC0415

            wave_input = np.asarray(samples, dtype=np.float32)
        except ImportError:
            wave_input = samples

        stream = self._recognizer.create_stream()
        stream.accept_waveform(rate, wave_input)
        self._recognizer.decode_streams([stream])
        result = getattr(stream, "result", None)
        raw_text = str(getattr(result, "text", "") or "").strip()

        log_probs = getattr(result, "ys_log_probs", None)
        acoustic_logprob: float | None = None
        n_tokens: int | None = None
        try:
            if log_probs is not None and len(log_probs) > 0:
                acoustic_logprob = float(sum(float(value) for value in log_probs))
                n_tokens = int(len(log_probs))
        except (TypeError, ValueError):
            # A result that carries something unsummable must not invent a score.
            acoustic_logprob, n_tokens = None, None

        return raw_text, acoustic_logprob, n_tokens, duration_s, time.perf_counter() - t0

    #: The pinned runtime exposes no N-best, so this is False by contract and
    #: :meth:`transcribe_nbest` always raises (see ``lifemate.asr.nbest``).
    nbest_supported = False

    def transcribe_nbest(
        self,
        audio_path: str | Path,
        *,
        nbest_size: int = 3,
        sample_id: str | None = None,
    ) -> NBestResult:
        """Always raises :class:`NBestUnsupportedError` on this runtime.

        A 1-element "N-best" would be a fabricated result: sherpa-onnx 2.0.1's
        offline API fills exactly one ``text`` per stream.
        """
        validate_nbest_size(nbest_size)
        require_nbest_support(self, nbest_size=nbest_size)
        raise NBestUnsupportedError(
            "unreachable: GipformerBackend.nbest_supported is False by contract"
        )

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        sample_id: str | None = None,
        lenient: bool = False,
    ) -> TranscriptResult:
        duration_s: float | None = None
        t0 = time.perf_counter()
        effective_lenient = lenient or self.lenient

        try:
            raw_text, acoustic_logprob, n_tokens, duration_s, latency_s = self._decode(audio_path)

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
                latency_s=latency_s,
                backend=self.name,
                model_name=self.model_id,
                acoustic_logprob=acoustic_logprob,
                n_tokens=n_tokens,
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

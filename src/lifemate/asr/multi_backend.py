"""Composed ASR backend: a primary recognizer plus an optional secondary one.

The idea is coverage, not fusion: when a second model is available, its top-1
transcript joins the primary's candidates, so the agent sees more than one
reading of the same audio. Nothing is averaged and no score is ever compared
across models — the ordering is **(source priority, rank)** and the result says
so via ``scores_calibrated=False``.

Contracts that keep this honest:

* ``enabled=False`` → the primary is called and **its own result object is
  returned unchanged** (identity preserved, asserted in the tests). The
  secondary is not constructed, not called, not touched.
* Exactly one backend contributing → ``NBestResult.multi_asr`` stays False and
  the "multi-ASR" claim is never made in the report.
* ``failure_policy="primary_only"`` → a secondary exception is recorded in
  ``secondary_error`` and the run continues; ``"strict"`` → re-raised.
* ``budget_seconds`` is wall clock: an overrun after the primary stage drops the
  secondary stage and sets ``budget_exceeded=True``.

Real second models are absent on this machine (chunkformer-rnnt-large-vie and
gipformer1.5 have no weights), so the composed path is exercised with fake
backends and the real command is reported as BLOCKED.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from collections.abc import Mapping
from dataclasses import replace

from .interface import ASRBackend, AsrError, TranscriptResult
from .nbest import AsrHypothesis, NBestResult, dedupe_hypotheses, normalize_hypothesis_text
from lifemate.errors import ProhibitedAccessError

logger = logging.getLogger(__name__)

__all__ = [
    "MultiAsrBackend",
    "MULTI_ASR_POLICIES",
    "MULTI_ASR_FAILURE_POLICIES",
    "REAL_SECONDARY_BLOCKED",
]

#: Supported combination policies. Only one is implemented because only one
#: makes sense without cross-model score calibration: keep the primary's
#: candidates and append the secondary's best.
MULTI_ASR_POLICIES = ("primary_topk_secondary_top1",)

#: What to do when the secondary backend fails.
MULTI_ASR_FAILURE_POLICIES = ("primary_only", "strict")

#: Operator-facing status of the real-secondary path on this machine.
REAL_SECONDARY_BLOCKED = (
    "BLOCKED: no second ASR checkpoint exists locally. khanhld/chunkformer-rnnt-large-vie "
    "and g-group-ai-lab/gipformer1.5-68M-rnnt have no weights and no Hugging Face cache "
    "entries here, so the composed path is only exercised with fake backends. Once a second "
    "model is downloadable, run: python3 -c \"from lifemate.asr import create_asr_backend; "
    "from lifemate.asr.multi_backend import MultiAsrBackend; p=create_asr_backend(backend='gipformer'); "
    "s=create_asr_backend({'model_id':'<secondary-model-id>'}, backend='whisper'); "
    "m=MultiAsrBackend(p, s, enabled=True); "
    "print(m.transcribe_multi('data/public_test/audio/public_0656/public_0656.wav'))\""
)


class MultiAsrBackend:
    """Primary + optional secondary backend, composed without score fusion."""

    name = "multi"
    is_oracle = False

    def __init__(
        self,
        primary: ASRBackend,
        secondary: ASRBackend | None = None,
        *,
        enabled: bool = False,
        policy: str = "primary_topk_secondary_top1",
        max_hypotheses: int = 3,
        failure_policy: str = "primary_only",
        primary_topk: int = 3,
        budget_seconds: float | None = None,
        backends: Mapping[str, ASRBackend] | None = None,
    ) -> None:
        if primary is None or not callable(getattr(primary, "transcribe", None)):
            raise AsrError("MultiAsrBackend requires a primary ASR backend with transcribe()")
        if secondary is not None:
            if not callable(getattr(secondary, "transcribe", None)):
                raise AsrError("secondary must be an ASR backend with transcribe()")
            if secondary is primary:
                raise AsrError(
                    "secondary must be a DIFFERENT backend instance; composing a backend with "
                    "itself would duplicate its own hypotheses and fake multi-ASR coverage"
                )
        if policy not in MULTI_ASR_POLICIES:
            raise AsrError(f"unknown multi-ASR policy {policy!r}; choose from {MULTI_ASR_POLICIES}")
        if failure_policy not in MULTI_ASR_FAILURE_POLICIES:
            raise AsrError(
                f"unknown failure_policy {failure_policy!r}; choose from {MULTI_ASR_FAILURE_POLICIES}"
            )
        if int(max_hypotheses) < 1:
            raise AsrError(f"max_hypotheses must be >= 1, got {max_hypotheses!r}")
        if int(primary_topk) < 1:
            raise AsrError(f"primary_topk must be >= 1, got {primary_topk!r}")
        if budget_seconds is not None and float(budget_seconds) <= 0:
            raise AsrError(f"budget_seconds must be > 0 when set, got {budget_seconds!r}")

        self.primary = primary
        self.secondary = secondary
        self.enabled = bool(enabled)
        self.policy = policy
        self.max_hypotheses = int(max_hypotheses)
        self.failure_policy = failure_policy
        self.primary_topk = int(primary_topk)
        self.budget_seconds = float(budget_seconds) if budget_seconds is not None else None
        self.backends = dict(backends) if backends is not None else None
        if self.backends is not None:
            if not self.backends or any(not k or not callable(getattr(v, 'transcribe', None)) for k, v in self.backends.items()):
                raise AsrError('backends must contain uniquely named transcribe-capable backends')
            if len({id(v) for v in self.backends.values()}) != len(self.backends):
                raise AsrError('each model must use a distinct backend instance')

    # ------------------------------------------------------------------ public
    @property
    def model_name(self) -> str:
        """Primary's model name; a composed run does not have one model."""
        return str(getattr(self.primary, "model_id", "") or getattr(self.primary, "name", ""))

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        sample_id: str | None = None,
        lenient: bool = False,
    ) -> TranscriptResult:
        """One transcript: the primary's own result when disabled.

        Enabled, this is the rank-0 hypothesis of :meth:`transcribe_multi` — the
        primary's top-1 unless the secondary produced something and the primary
        produced nothing (then the secondary text is the only text there is).
        """
        if not self.enabled or (self.secondary is None and self.backends is None):
            # Identity, not a copy: the caller's result object comes back as-is.
            return self.primary.transcribe(audio_path, sample_id=sample_id, lenient=lenient)

        nbest = self.transcribe_multi(audio_path, sample_id=sample_id)
        if nbest.hypotheses:
            return nbest.to_transcript_result(backend=nbest.hypotheses[0].backend or "multi")
        return TranscriptResult(
            text="",
            error="multi-ASR produced no hypothesis",
            backend=self.name,
            model_name=self.model_name,
        )

    def transcribe_multi(
        self,
        audio_path: str | Path,
        *,
        sample_id: str | None = None,
    ) -> NBestResult:
        """Every candidate text from the enabled sources, in priority order."""
        started = time.monotonic()
        result = NBestResult(scores_calibrated=False)

        if self.enabled and self.backends is not None:
            return self._transcribe_top1_sources(audio_path, sample_id, started)

        if not self.enabled or self.secondary is None:
            # A single-ASR run: degraded single-hypothesis result, multi_asr False.
            single = self.primary.transcribe_multi(audio_path, sample_id=sample_id) if callable(
                getattr(self.primary, "transcribe_multi", None)
            ) else None
            if single is not None and single.hypotheses:
                tagged = [
                    AsrHypothesis(
                        text=hypothesis.text,
                        rank=index,
                        score=hypothesis.score,
                        source=str(getattr(self.primary, "name", "primary")),
                        backend=str(getattr(self.primary, "name", "primary")),
                        score_kind=hypothesis.score_kind,
                    )
                    for index, hypothesis in enumerate(single.hypotheses[: self.primary_topk])
                ]
                result.hypotheses = dedupe_hypotheses(tagged, limit=self.max_hypotheses)
                result.sources = [str(getattr(self.primary, "name", "primary"))]
                result.reasons = list(single.reasons)
                return result
            if single is not None:
                result.reasons = list(single.reasons)
            primary_result = self.primary.transcribe(audio_path, sample_id=sample_id, lenient=True)
            return self._single_from_transcript(primary_result, reason="multi-ASR disabled; primary only")

        primary_name = str(getattr(self.primary, "name", "primary"))
        secondary_name = str(getattr(self.secondary, "name", "secondary"))
        if secondary_name == primary_name:
            primary_name, secondary_name = "primary", "secondary"

        primary_hypotheses = self._primary_hypotheses(audio_path, sample_id=sample_id)
        primary_hypotheses = primary_hypotheses[: self.primary_topk]
        if primary_name == "primary":
            from dataclasses import replace
            primary_hypotheses = [replace(h, source=primary_name) for h in primary_hypotheses]
        if not primary_hypotheses:
            result.reasons.append("primary backend produced no transcript")

        if self.budget_seconds is not None and (time.monotonic() - started) > self.budget_seconds:
            result.hypotheses = dedupe_hypotheses(primary_hypotheses, limit=self.max_hypotheses)
            result.sources = [primary_name]
            result.budget_exceeded = True
            result.degraded = True
            result.reasons.append(
                f"budget_seconds={self.budget_seconds:g} exceeded after the primary stage; "
                "secondary hypotheses were dropped"
            )
            result.requested_size = self.max_hypotheses
            return result

        secondary_hypotheses: list[AsrHypothesis] = []
        try:
            secondary_hypotheses = [
                AsrHypothesis(
                    text=item.text,
                    rank=item.rank,
                    score=item.score,
                    source=secondary_name,
                    backend=secondary_name,
                    score_kind=item.score_kind,
                )
                for item in self._backend_hypotheses(self.secondary, audio_path, sample_id=sample_id)[:1]
            ]
        except Exception as exc:  # noqa: BLE001 - policy decides whether it escapes
            message = f"{type(exc).__name__}: {exc}"
            if self.failure_policy == "strict":
                raise
            # Recorded, never swallowed silently, and never reported as success.
            logger.warning("multi-ASR secondary %r failed (primary_only): %s", secondary_name, message)
            result.secondary_error = message
            result.reasons.append(f"secondary backend failed: {message}")
            result.degraded = True

        # Reserve room for a secondary candidate when the cap permits it.
        primary_hypotheses = dedupe_hypotheses(primary_hypotheses)
        if (secondary_hypotheses and self.max_hypotheses > 1
                and secondary_hypotheses[0].normalized not in {h.normalized for h in primary_hypotheses}):
            primary_hypotheses = primary_hypotheses[:self.max_hypotheses - 1]
        ordered = list(primary_hypotheses) + list(secondary_hypotheses)
        result.hypotheses = dedupe_hypotheses(ordered, limit=self.max_hypotheses)
        result.sources = sorted({h.source for h in result.hypotheses})
        result.multi_asr = len({h.source for h in result.hypotheses}) > 1
        result.requested_size = self.max_hypotheses
        if not result.multi_asr and not result.degraded:
            result.degraded = True
            result.reasons.append(
                "only one backend contributed a transcript; this run is NOT a multi-ASR result"
            )
        if self.budget_seconds is not None and (time.monotonic() - started) > self.budget_seconds:
            result.budget_exceeded = True
            result.degraded = True
            result.reasons.append(
                f"budget_seconds={self.budget_seconds:g} exceeded; ordering kept but the run is over budget"
            )
        return result

    def _transcribe_top1_sources(self, audio_path, sample_id, started):
        """One independent Top-1 per model; deduplicate only after recording all results."""
        result = NBestResult(scores_calibrated=False, requested_size=self.max_hypotheses)
        distinct = {}
        for label, backend in self.backends.items():
            if self.budget_seconds is not None and time.monotonic() - started >= self.budget_seconds:
                result.budget_exceeded = result.degraded = True
                result.reasons.append(f'{label}: skipped because the shared ASR budget expired')
                result.raw_results.append({'model': label, 'status': 'blocked', 'error': 'budget expired'})
                if self.failure_policy == 'strict': raise AsrError(result.reasons[-1])
                continue
            t0 = time.monotonic()
            record = {'model': label, 'rank': 1, 'model_id': getattr(backend, 'model_id', None) or getattr(backend, 'model_name', None),
                      'revision': getattr(backend, 'revision', None), 'backend': getattr(backend, 'name', label),
                      'configuration': {k: getattr(backend, k) for k in ('quantization','decoding_method','language') if hasattr(backend,k)}}
            try:
                item = backend.transcribe(audio_path, sample_id=sample_id, lenient=True)
                record['cache_hit'] = bool(getattr(backend, 'last_cache_hit', False))
                record['recognition_latency_ms'] = round(float(getattr(item, 'latency_s', 0.0) or 0.0) * 1000, 3)
                text = str(item.text or '').strip()
                if item.error or not text: raise AsrError(item.error or 'empty transcript')
                record.update(status='executed', raw_transcript=getattr(item, 'raw_text', None) or text,
                              transcript=text, error=None)
                result.sources.append(label)
                key = normalize_hypothesis_text(text)
                if key in distinct:
                    old = distinct[key]; distinct[key] = replace(old, provenance=(*old.provenance,label))
                else:
                    distinct[key] = AsrHypothesis(text=text, rank=0, source=label, backend=record['backend'], provenance=(label,))
            except ProhibitedAccessError:
                raise
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}', raw_transcript=None)
                result.degraded = True; result.reasons.append(f"{label}: {record['error']}")
                if self.failure_policy == 'strict': raise AsrError(result.reasons[-1]) from exc
            finally:
                record['latency_ms'] = round((time.monotonic()-t0)*1000, 3)
                result.raw_results.append(record)
            if self.budget_seconds is not None and time.monotonic() - started > self.budget_seconds:
                result.budget_exceeded = result.degraded = True
                result.reasons.append(f'{label}: ASR budget exceeded during decoding')
                if self.failure_policy == 'strict': raise AsrError(result.reasons[-1])
        result.hypotheses = list(distinct.values())[:self.max_hypotheses]
        result.multi_asr = len(result.sources) >= 2
        return result

    # ----------------------------------------------------------------- helpers
    def _single_from_transcript(self, transcript: TranscriptResult, *, reason: str) -> NBestResult:
        hypothesis = AsrHypothesis(
            text=transcript.text,
            rank=0,
            score=transcript.acoustic_logprob,
            source=str(getattr(self.primary, "name", "primary")),
            backend=transcript.backend,
            score_kind="sum_ys_log_probs" if transcript.acoustic_logprob is not None else None,
        )
        nbest = NBestResult(
            hypotheses=dedupe_hypotheses([hypothesis]) if hypothesis.normalized else [],
            requested_size=1,
            scores_calibrated=False,
            degraded=True,
            reasons=[reason],
            sources=[hypothesis.source] if hypothesis.normalized else [],
            multi_asr=False,
        )
        if transcript.error:
            nbest.reasons.append(f"primary error: {transcript.error}")
        return nbest

    def _backend_hypotheses(
        self, backend: ASRBackend, audio_path: str | Path, *, sample_id: str | None
    ) -> list[AsrHypothesis]:
        """Backend's hypotheses, degrading explicitly when N-best is unavailable."""
        name = str(getattr(backend, "name", "backend"))
        multi = getattr(backend, "transcribe_multi", None)
        if callable(multi):
            produced = multi(audio_path, sample_id=sample_id)
            if produced.hypotheses:
                return [
                    AsrHypothesis(
                        text=hypothesis.text,
                        rank=index,
                        score=hypothesis.score,
                        source=str(hypothesis.source or name),
                        backend=str(hypothesis.backend or name),
                        score_kind=hypothesis.score_kind,
                    )
                    for index, hypothesis in enumerate(produced.hypotheses)
                ]
            return []
        nbest = getattr(backend, "transcribe_nbest", None)
        if callable(nbest) and self.primary_topk > 1 and getattr(backend, "nbest_supported", False) is True:
            produced_nbest = nbest(audio_path, nbest_size=max(2, self.primary_topk), sample_id=sample_id)
            return list(produced_nbest.hypotheses)
        transcript = backend.transcribe(audio_path, sample_id=sample_id, lenient=False)
        if not transcript.ok or not transcript.text:
            return []
        return [
            AsrHypothesis(
                text=transcript.text,
                rank=0,
                score=transcript.acoustic_logprob,
                source=name,
                backend=transcript.backend,
                score_kind="sum_ys_log_probs" if transcript.acoustic_logprob is not None else None,
            )
        ]

    def _primary_hypotheses(self, audio_path: str | Path, *, sample_id: str | None) -> list[AsrHypothesis]:
        try:
            return self._backend_hypotheses(self.primary, audio_path, sample_id=sample_id)
        except Exception as exc:  # noqa: BLE001 - primary failure is a hard error
            raise AsrError(
                f"multi-ASR primary backend {getattr(self.primary, 'name', 'primary')!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MultiAsrBackend(primary={getattr(self.primary, 'name', '?')!r}, "
            f"secondary={getattr(self.secondary, 'name', None)!r}, enabled={self.enabled!r}, "
            f"policy={self.policy!r}, failure_policy={self.failure_policy!r})"
        )


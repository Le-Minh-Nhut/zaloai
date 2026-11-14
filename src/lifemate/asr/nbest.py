"""N-best plumbing that refuses to fake N-best, plus real within-model rescoring.

Verdict for the pinned runtime (sherpa-onnx 2.0.1) — N-best is **NOT EXPOSED**.
Measured, not assumed:

* ``OfflineRecognizer`` has exactly ``decode_stream``/``decode_streams``; both
  return ``None`` and fill ``stream.result`` with ONE text.
* ``OfflineRecognitionResult`` fields: ``text``, ``tokens``, ``timestamps``,
  ``ys_log_probs``, ``durations``, ``words``, ``segment_texts``,
  ``segment_durations``, ``segment_timestamps``, ``lang``, ``emotion``,
  ``event``. There is no ``hypotheses``/``nbest``/``confidence``/``lm_probs``.
* ``OfflineRecognizer.from_transducer`` has no ``nbest``/``num_hypotheses``/
  ``return_nbest``/``best_of`` parameter. ``max_active_paths`` is the
  modified-beam width, not a request for multiple outputs.

So :meth:`GipformerBackend.transcribe_nbest` raises
:class:`NBestUnsupportedError` with all of the above and the caller's options.
A one-element list is never returned as if it were an N-best result, and no
alternative text is ever fabricated by perturbing a transcript.

What the runtime *does* support, and what this module therefore implements:
**multi-configuration rescoring**. ``ys_log_probs`` carries one log probability
per emitted token (verified: 17 tokens → 17 values, sum −3.6049 on a real
public_test wav), so the SAME audio can be decoded under several decoding
configurations and the resulting texts ranked by summed acoustic score. That is
not N-best: it costs one full decode per configuration, and the scores are only
comparable inside one model. It is named ``asr_rescore`` and never reported as
N-best.

Pure stdlib import (sherpa/torch stay lazily imported inside the capability probe).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .interface import AsrError, TranscriptResult

__all__ = [
    "AsrHypothesis",
    "NBestResult",
    "NBestUnsupportedError",
    "NBEST_MAX",
    "NBEST_MIN",
    "OFFLINE_RESULT_FIELDS",
    "REQUIRED_FROM_TRANSDUCER_ARGS",
    "ScenarioRanking",
    "dedupe_hypotheses",
    "normalize_hypothesis_text",
    "require_nbest_support",
    "rescore_configurations",
    "runtime_nbest_capability",
    "validate_nbest_size",
]

#: Accepted range for ``nbest_size``; below 2 there is nothing to rank.
NBEST_MIN = 2
NBEST_MAX = 10

#: ``dir()`` of a decoded ``OfflineRecognitionResult`` on sherpa-onnx 2.0.1.
#: Recorded so the unsupported-N-best message can name what IS available without
#: importing the package.
OFFLINE_RESULT_FIELDS: tuple[str, ...] = (
    "text",
    "tokens",
    "timestamps",
    "ys_log_probs",
    "durations",
    "words",
    "segment_texts",
    "segment_durations",
    "segment_timestamps",
    "lang",
    "emotion",
    "event",
)

#: ``from_transducer`` parameters on sherpa-onnx 2.0.1 (verbatim).
REQUIRED_FROM_TRANSDUCER_ARGS: tuple[str, ...] = (
    "encoder",
    "decoder",
    "joiner",
    "tokens",
    "num_threads",
    "sample_rate",
    "feature_dim",
    "dither",
    "decoding_method",
    "max_active_paths",
    "hotwords_file",
    "hotwords_score",
    "blank_penalty",
    "modeling_unit",
    "bpe_vocab",
    "debug",
    "provider",
    "model_type",
    "rule_fsts",
    "rule_fars",
    "lm",
    "lm_scale",
    "hr_dict_dir",
    "hr_rule_fsts",
    "hr_lexicon",
    "lodr_fst",
    "lodr_scale",
)


class NBestUnsupportedError(AsrError):
    """The configured runtime cannot return more than one hypothesis."""


def validate_nbest_size(nbest_size: object) -> int:
    """``nbest_size`` as an int in ``[2, 10]``; anything else is an error."""
    try:
        value = int(nbest_size)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise AsrError(f"nbest_size must be an integer in [{NBEST_MIN}, {NBEST_MAX}], got {nbest_size!r}") from exc
    if isinstance(nbest_size, bool) or not (NBEST_MIN <= value <= NBEST_MAX):
        raise AsrError(f"nbest_size must be in [{NBEST_MIN}, {NBEST_MAX}], got {nbest_size!r}")
    return value


def normalize_hypothesis_text(text: object) -> str:
    """Comparison key for dedupe: NFC + collapsed whitespace + case-fold."""
    folded = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(folded.split()).casefold()


def runtime_nbest_capability() -> dict[str, Any]:
    """What the *installed* sherpa-onnx actually exposes (probe, never a guess)."""
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "decode_methods": (),
        "has_hypotheses_field": False,
        "nbest_kwargs": (),
        "result_fields": OFFLINE_RESULT_FIELDS,
    }
    try:
        import sherpa_onnx  # noqa: PLC0415 - probe only
    except ImportError:
        return info
    info["installed"] = True
    info["version"] = getattr(sherpa_onnx, "__version__", None)
    recognizer = getattr(sherpa_onnx, "OfflineRecognizer", None)
    if recognizer is None:
        return info
    info["decode_methods"] = tuple(name for name in dir(recognizer) if name.startswith("decode"))
    try:
        import inspect  # noqa: PLC0415 - probe only

        params = tuple(inspect.signature(recognizer.from_transducer).parameters)
        info["from_transducer_params"] = params
        info["nbest_kwargs"] = tuple(
            name for name in params if any(tag in name.lower() for tag in ("nbest", "num_hypo", "best_of", "return_n"))
        )
    except (TypeError, ValueError, AttributeError):
        pass
    result_type = getattr(sherpa_onnx, "OfflineRecognitionResult", None)
    info["has_hypotheses_field"] = bool(
        result_type is not None
        and any("hypo" in name or "nbest" in name for name in dir(result_type))
    )
    return info


def _nbest_unsupported_message(*, what: str, nbest_size: int | None) -> str:
    capability = runtime_nbest_capability()
    installed = (
        f"sherpa-onnx {capability['version']} is installed"
        if capability["installed"]
        else "sherpa-onnx is not importable here"
    )
    asked = f"nbest_size={nbest_size}" if nbest_size is not None else "N-best"
    decode = ", ".join(capability["decode_methods"]) or "<none found>"
    return (
        f"N-best decoding is not available: {what} asked for {asked}, but {installed} and its "
        f"offline API returns exactly ONE hypothesis. "
        f"Evidence: decode methods are {decode} and both fill a single stream result; "
        f"OfflineRecognitionResult exposes {', '.join(capability['result_fields'])} "
        f"and no hypotheses/nbest/confidence field; "
        f"OfflineRecognizer.from_transducer takes no n-best argument "
        f"(params: {', '.join(REQUIRED_FROM_TRANSDUCER_ARGS)}) — 'max_active_paths' is only the "
        f"modified-beam width. Returning a 1-element list as 'N-best' would be a fabricated result, "
        "so this call fails instead. Options: (1) install a sherpa-onnx build that exposes N-best "
        "hypotheses for offline transducers, (2) use a decoder that can emit N-best "
        "(e.g. a beam-search wrapper over the same transducer, or a CTC/attention model with "
        "beam search), or (3) keep the feature off and use asr_rescore "
        "(rescore_configurations) for multiple candidate texts via repeated decodes, or the "
        "single top-1 transcript."
    )


def require_nbest_support(backend: Any = None, *, nbest_size: int | None = None) -> None:
    """Raise :class:`NBestUnsupportedError` unless ``backend`` really has N-best.

    "Really" means: it exposes ``transcribe_nbest`` *and* advertises
    ``nbest_supported = True``. A method that exists only to raise does not count.
    """
    name = getattr(backend, "name", type(backend).__name__ if backend is not None else "the backend")
    if getattr(backend, "nbest_supported", False) is True and callable(
        getattr(backend, "transcribe_nbest", None)
    ):
        return
    raise NBestUnsupportedError(
        _nbest_unsupported_message(what=f"backend {name!r}", nbest_size=nbest_size)
    )


@dataclass(frozen=True)
class AsrHypothesis:
    """One candidate transcript with explicit provenance.

    ``score`` is the producing recognizer's own score (for the Gipformer
    backend: the sum of ``ys_log_probs``). It is comparable only against scores
    from the same model — hence :attr:`NBestResult.scores_calibrated`.
    """

    text: str
    rank: int = 0
    score: float | None = None
    source: str = ""
    backend: str | None = None
    score_kind: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text or ""))
        object.__setattr__(self, "rank", int(self.rank))

    @property
    def normalized(self) -> str:
        """Dedupe key (NFC + whitespace collapse + case-fold)."""
        return normalize_hypothesis_text(self.text)

    def to_transcript_result(self, **overrides: Any) -> TranscriptResult:
        """Degrade explicitly to a :class:`TranscriptResult` (``backend`` kept)."""
        fields = {
            "text": self.text,
            "raw_text": None,
            "backend": self.backend or "unknown",
            "acoustic_logprob": self.score,
        }
        fields.update({key: value for key, value in overrides.items() if key in _TRANSCRIPT_FIELDS})
        return TranscriptResult(**fields)


#: Field names of :class:`TranscriptResult`, used to filter degrade overrides.
_TRANSCRIPT_FIELDS = frozenset(TranscriptResult.__dataclass_fields__)


@dataclass
class NBestResult:
    """Ranked hypotheses from ONE model. Never a cross-model score mix.

    ``degraded`` says the run did not deliver what was asked (e.g. N-best was
    unavailable and only the top-1 transcript exists); ``reasons`` says why.
    """

    hypotheses: list[AsrHypothesis] = field(default_factory=list)
    requested_size: int | None = None
    scores_calibrated: bool = False
    degraded: bool = False
    reasons: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    #: True only when 2+ DIFFERENT backends actually contributed (multi-ASR).
    multi_asr: bool = False
    #: Secondary-backend failure text when ``failure_policy="primary_only"``.
    secondary_error: str | None = None
    #: Set when the wall-clock budget cut the run short.
    budget_exceeded: bool = False

    @property
    def text(self) -> str:
        """Rank-0 text (``""`` when empty)."""
        return self.hypotheses[0].text if self.hypotheses else ""

    @property
    def best(self) -> AsrHypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None

    @property
    def is_nbest(self) -> bool:
        """True only with 2+ distinct alternatives and no degradation."""
        return not self.degraded and len(self.hypotheses) > 1

    def to_transcript_result(self, *, source: str | None = None, **overrides: Any) -> TranscriptResult:
        """Rank-0 as a :class:`TranscriptResult`; errors when there is none."""
        if not self.hypotheses:
            raise AsrError("cannot degrade an empty NBestResult to a TranscriptResult")
        chosen = self.hypotheses[0]
        if source is not None:
            chosen = AsrHypothesis(
                text=chosen.text,
                rank=0,
                score=chosen.score,
                source=source,
                backend=chosen.backend,
                score_kind=chosen.score_kind,
            )
        return chosen.to_transcript_result(**overrides)

    @classmethod
    def single(
        cls,
        result: TranscriptResult,
        *,
        source: str = "",
        backend: str | None = None,
        scored: bool = False,
        reason: str = "",
    ) -> NBestResult:
        """Wrap one existing transcript as a clearly-degraded 'best of one'."""
        producer = backend or result.backend
        hypothesis = AsrHypothesis(
            text=result.text,
            rank=0,
            score=result.acoustic_logprob,
            source=source or producer,
            backend=producer,
            score_kind="sum_ys_log_probs" if result.acoustic_logprob is not None else None,
        )
        return cls(
            hypotheses=[hypothesis],
            requested_size=1,
            scores_calibrated=False,
            degraded=True,
            reasons=[reason or "only one hypothesis available"],
            sources=[hypothesis.source],
        )


def dedupe_hypotheses(
    hypotheses: Iterable[AsrHypothesis], *, limit: int | None = None
) -> list[AsrHypothesis]:
    """Drop repeated texts, keeping the first (best-ranked) original.

    Order is the input order, so callers control ranking; ``limit`` truncates
    afterwards. Empty texts are dropped: a recognizer that emitted nothing is
    not a candidate transcript.
    """
    seen: set[str] = set()
    kept: list[AsrHypothesis] = []
    for hypothesis in hypotheses:
        key = hypothesis.normalized
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(
            AsrHypothesis(
                text=hypothesis.text,
                rank=len(kept),
                score=hypothesis.score,
                source=hypothesis.source,
                backend=hypothesis.backend,
                score_kind=hypothesis.score_kind,
            )
        )
        if limit is not None and len(kept) >= int(limit):
            break
    return kept


@dataclass
class ScenarioRanking:
    """Result of :func:`rescore_configurations` — named ``asr_rescore``, not N-best.

    ``scores_comparable`` is True only when every decode came from one model id
    (within-model scores are legitimately comparable; across models they are
    not, so cross-model ordering would be meaningless).
    """

    name: str = "asr_rescore"
    hypotheses: list[AsrHypothesis] = field(default_factory=list)
    configs: list[dict[str, Any]] = field(default_factory=list)
    results: list[TranscriptResult] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)
    scores_comparable: bool = False
    skipped_reason: str | None = None
    reasons: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def text(self) -> str:
        return self.hypotheses[0].text if self.hypotheses else ""

    @property
    def n_decodes(self) -> int:
        return len(self.results)


def rescore_configurations(
    backend_factory: Callable[..., Any],
    audio_path: str | Path,
    configs: Sequence[Mapping[str, Any]],
    *,
    sample_id: str | None = None,
    enabled: bool = True,
    max_hypotheses: int = 3,
) -> ScenarioRanking:
    """Decode ``audio_path`` once per configuration and rank the candidate texts.

    Costs ``len(configs)`` full decodes — that is the price of getting more than
    one text out of a runtime with no N-best. ``backend_factory(**config)`` must
    return an object with ``transcribe``; build one recognizer per configuration
    (that is what the configurations are for; reusing one recognizer would
    silently ignore them).

    Ranking: summed acoustic score descending (``ys_log_probs`` when the backend
    provides it), then configuration order. Identical normalized texts are
    merged keeping the best-scoring original. **Scores are only comparable within
    one model**, so a ranking whose decodes report different ``model_name``s is
    flagged ``scores_comparable=False`` and its order is configuration order only.

    ``enabled=False`` constructs ZERO backends and returns an empty ranking.
    """
    import time  # noqa: PLC0415 - keep module import cheap

    started = time.perf_counter()
    ranking = ScenarioRanking(configs=[dict(config) for config in configs])
    if not enabled:
        ranking.skipped_reason = "asr_rescore disabled (enabled=False); nothing was decoded"
        return ranking
    if not ranking.configs:
        ranking.skipped_reason = "no configurations given; nothing was decoded"
        return ranking

    scored: list[tuple[int, AsrHypothesis]] = []
    for index, config in enumerate(ranking.configs):
        backend = backend_factory(**dict(config))
        try:
            result = backend.transcribe(audio_path, sample_id=sample_id, lenient=True)
        except Exception as exc:  # noqa: BLE001 - one bad config must not kill the ranking
            # Recorded as a failed decode, never dropped: the ranking reports how
            # many decodes it paid for, and why one produced nothing.
            message = f"{type(exc).__name__}: {exc}"
            result = TranscriptResult(text="", error=message, backend=str(getattr(backend, "name", "")))
            ranking.reasons.append(f"configuration {index} failed: {message}")
        ranking.results.append(result)
        model_id = str(result.model_name or getattr(backend, "model_id", "") or getattr(backend, "name", ""))
        ranking.model_ids.append(model_id)
        if not result.ok or not result.text:
            continue
        scored.append(
            (
                index,
                AsrHypothesis(
                    text=result.text,
                    rank=index,
                    score=result.acoustic_logprob,
                    source=_config_label(config),
                    backend=result.backend,
                    score_kind="sum_ys_log_probs" if result.acoustic_logprob is not None else None,
                ),
            )
        )

    ranking.scores_comparable = len(set(ranking.model_ids)) <= 1 and bool(scored)
    if ranking.scores_comparable:
        # Descending score; configurations without a score keep configuration order last.
        scored.sort(key=lambda item: (-(item[1].score if item[1].score is not None else float("-inf")), item[0]))
    else:
        scored.sort(key=lambda item: item[0])

    ranking.hypotheses = dedupe_hypotheses(
        (hypothesis for _, hypothesis in scored), limit=max_hypotheses
    )
    ranking.elapsed_seconds = time.perf_counter() - started
    if not ranking.hypotheses:
        ranking.skipped_reason = "every configuration returned an empty or errored transcript"
    return ranking


def _config_label(config: Mapping[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in sorted(config.items())]
    return "config:" + ",".join(parts) if parts else "config:default"

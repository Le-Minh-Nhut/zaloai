"""N-best tests: the pinned runtime cannot do it, so the code says so and offers rescoring.

Two behaviours are covered, both real:

* :func:`require_nbest_support` / ``GipformerBackend.transcribe_nbest`` fail with a
  message that names the evidence (no hypotheses field, no n-best kwarg);
* :func:`rescore_configurations` gets genuinely different candidate texts by
  decoding once per configuration and ranking summed ``ys_log_probs``.

Stdlib-runnable: the sherpa-dependent test skips itself when the package is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None  # stdlib runner: assertions only

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.asr.interface import AsrError, TranscriptResult
from lifemate.asr.nbest import (
    NBEST_MAX,
    NBEST_MIN,
    NBestResult,
    NBestUnsupportedError,
    OFFLINE_RESULT_FIELDS,
    REQUIRED_FROM_TRANSDUCER_ARGS,
    AsrHypothesis,
    dedupe_hypotheses,
    normalize_hypothesis_text,
    require_nbest_support,
    rescore_configurations,
    runtime_nbest_capability,
    validate_nbest_size,
)


class _FakeSingleBackend:
    """A backend whose recognizer returns exactly one text (the real shape)."""

    name = "fake"
    is_oracle = False
    nbest_supported = False

    def __init__(self, text: str, score: float | None = None, model_id: str = "fake/model") -> None:
        self.text = text
        self.score = score
        self.model_id = model_id
        self.calls = 0

    def transcribe(self, audio_path, *, sample_id=None, lenient=False) -> TranscriptResult:
        self.calls += 1
        return TranscriptResult(
            text=self.text,
            backend=self.name,
            model_name=self.model_id,
            acoustic_logprob=self.score,
            n_tokens=3 if self.score is not None else None,
        )


class _RaisingBackend(_FakeSingleBackend):
    def transcribe(self, audio_path, *, sample_id=None, lenient=False) -> TranscriptResult:
        self.calls += 1
        raise RuntimeError("decoder exploded")


def test_nbest_size_is_validated():
    assert validate_nbest_size(2) == 2
    assert validate_nbest_size(NBEST_MAX) == NBEST_MAX
    for bad in (1, 0, -3, NBEST_MAX + 1, "many", None, True):
        try:
            validate_nbest_size(bad)
        except AsrError:
            continue
        raise AssertionError(f"nbest_size={bad!r} must be rejected (range {NBEST_MIN}..{NBEST_MAX})")


def test_require_nbest_support_refuses_a_backend_without_it():
    backend = _FakeSingleBackend("xin chào")
    try:
        require_nbest_support(backend, nbest_size=3)
    except NBestUnsupportedError as exc:
        message = str(exc)
        # What was asked...
        assert "nbest_size=3" in message
        assert "'fake'" in message
        # ...what the runtime actually returns...
        assert "exactly ONE hypothesis" in message
        assert "ys_log_probs" in message
        # ...and the concrete options.
        assert "install a sherpa-onnx build that exposes N-best" in message
        assert "asr_rescore" in message
        assert "keep the feature off" in message
    else:
        raise AssertionError("a backend without N-best must raise NBestUnsupportedError")


def test_require_nbest_support_accepts_a_truly_supporting_backend():
    class _NBest:
        name = "nbest-capable"
        nbest_supported = True

        def transcribe_nbest(self, audio_path, *, nbest_size=3, sample_id=None):
            return NBestResult(hypotheses=[], requested_size=nbest_size)

    assert require_nbest_support(_NBest(), nbest_size=4) is None
    # A method that exists but does not advertise support still counts as unsupported.
    class _Lying:
        name = "lying"

        def transcribe_nbest(self, audio_path, **kwargs):
            return NBestResult()

    try:
        require_nbest_support(_Lying(), nbest_size=3)
    except NBestUnsupportedError:
        pass
    else:
        raise AssertionError("a transcribe_nbest that does not advertise support must not pass")


def test_gipformer_backend_transcribe_nbest_raises_and_never_fakes_a_list():
    try:
        from lifemate.asr.gipformer_backend import GipformerBackend
    except Exception:  # pragma: no cover - module import is stdlib-safe
        return
    backend = object.__new__(GipformerBackend)  # no recognizer needed: the method raises first
    backend.nbest_size = 3
    try:
        backend.transcribe_nbest("whatever.wav", nbest_size=3)
    except NBestUnsupportedError as exc:
        assert "exactly ONE hypothesis" in str(exc)
    else:
        raise AssertionError("transcribe_nbest must raise on the pinned runtime")
    # And the flag is honest, so callers can branch without try/except.
    assert GipformerBackend.nbest_supported is False


def test_one_element_result_is_never_reported_as_nbest():
    single = TranscriptResult(text="xin chào", backend="fake")
    result = NBestResult.single(single, reason="only one hypothesis available")
    assert len(result.hypotheses) == 1
    assert result.is_nbest is False
    assert result.degraded is True
    assert "only one hypothesis available" in result.reasons
    # Degrading is explicit and keeps the producer's backend name.
    transcript = result.to_transcript_result()
    assert transcript.text == "xin chào"
    assert transcript.backend == "fake"


def test_two_element_result_is_nbest():
    result = NBestResult(
        hypotheses=[AsrHypothesis(text="a"), AsrHypothesis(text="b", rank=1)]
    )
    assert result.is_nbest is True
    assert result.scores_calibrated is False


def test_dedupe_keeps_best_ranked_original_and_is_stable():
    hypotheses = [
        AsrHypothesis(text="Xin chào", rank=0),
        AsrHypothesis(text="  xin   CHÀO ", rank=1),
        AsrHypothesis(text="Alô", rank=2),
        AsrHypothesis(text="", rank=3),
        AsrHypothesis(text="ALÔ", rank=4),
    ]
    kept = dedupe_hypotheses(hypotheses)
    assert [item.text for item in kept] == ["Xin chào", "Alô"]
    assert [item.rank for item in kept] == [0, 1]
    assert dedupe_hypotheses(hypotheses, limit=1)[0].text == "Xin chào"
    # Reordering the input changes which original survives, but never the count.
    assert len(dedupe_hypotheses(list(reversed(hypotheses)))) == 2


def test_normalize_is_nfc_whitespace_and_case_fold():
    assert normalize_hypothesis_text("  Xin\tCHÀO ") == normalize_hypothesis_text("xin chào")
    decomposed = "cha\u0300o"  # 'chào' as base + combining grave
    assert normalize_hypothesis_text(decomposed) == normalize_hypothesis_text("chào")


def test_rescore_configurations_disabled_constructs_no_backend():
    built: list[dict] = []

    def factory(**kwargs):
        built.append(kwargs)
        return _FakeSingleBackend("never")

    ranking = rescore_configurations(
        factory, "audio.wav", [{"decoding_method": "greedy_search"}], enabled=False
    )
    assert built == []
    assert ranking.hypotheses == []
    assert ranking.skipped_reason and "disabled" in ranking.skipped_reason


def test_rescore_configurations_ranks_within_model_and_dedupes():
    """Two configurations, one shared text and one better-scoring alternative."""
    scores = {"greedy_search": -5.0, "modified_beam_search": -2.0}
    texts = {"greedy_search": "Xin chào", "modified_beam_search": "Xin chào"}

    def factory(**kwargs):
        method = kwargs["decoding_method"]
        return _FakeSingleBackend(texts[method], score=scores[method])

    ranking = rescore_configurations(
        factory,
        "audio.wav",
        [{"decoding_method": "greedy_search"}, {"decoding_method": "modified_beam_search"}],
    )
    assert ranking.name == "asr_rescore"
    assert ranking.n_decodes == 2
    assert ranking.scores_comparable is True
    # Same text twice → one hypothesis, and it carries the better score.
    assert len(ranking.hypotheses) == 1
    assert ranking.hypotheses[0].score == -2.0
    assert ranking.hypotheses[0].source.startswith("config:")
    assert ranking.text == "Xin chào"

    # Different texts: the higher summed logprob wins, deterministically.
    texts["modified_beam_search"] = "Xin chao"
    ranking2 = rescore_configurations(
        factory,
        "audio.wav",
        [{"decoding_method": "greedy_search"}, {"decoding_method": "modified_beam_search"}],
    )
    assert [h.text for h in ranking2.hypotheses] == ["Xin chao", "Xin chào"]
    assert [h.score for h in ranking2.hypotheses] == [-2.0, -5.0]


def test_rescore_configurations_flags_cross_model_scores_as_incomparable():
    def factory(**kwargs):
        return _FakeSingleBackend(kwargs["text"], score=kwargs["score"], model_id=kwargs["model"])

    ranking = rescore_configurations(
        factory,
        "audio.wav",
        [
            {"text": "một", "score": -1.0, "model": "model-a"},
            {"text": "hai", "score": -9.0, "model": "model-b"},
        ],
    )
    assert ranking.scores_comparable is False
    # Ordering falls back to configuration order, never to a cross-model score.
    assert [h.text for h in ranking.hypotheses] == ["một", "hai"]


def test_rescore_configurations_keeps_the_run_alive_when_one_config_fails():
    calls = {"n": 0}

    def factory(**kwargs):
        calls["n"] += 1
        return _RaisingBackend("") if kwargs.get("boom") else _FakeSingleBackend("ok", score=-1.0)

    ranking = rescore_configurations(
        factory,
        "audio.wav",
        [{"boom": True}, {"decoding_method": "modified_beam_search"}],
    )
    assert calls["n"] == 2
    assert ranking.text == "ok"
    assert ranking.n_decodes == 2  # the failed decode is still recorded


def test_runtime_probe_reports_the_pinned_facts_when_sherpa_is_installed():
    """When sherpa_onnx is importable, the probe must agree with the recorded facts."""
    capability = runtime_nbest_capability()
    assert capability["result_fields"] == OFFLINE_RESULT_FIELDS
    assert "nbest" not in " ".join(OFFLINE_RESULT_FIELDS)
    if not capability["installed"]:
        assert capability["decode_methods"] == ()
        return
    assert set(capability["decode_methods"]) == {"decode_stream", "decode_streams"}
    assert capability["nbest_kwargs"] == ()
    assert capability["has_hypotheses_field"] is False
    params = capability.get("from_transducer_params") or ()
    for name in ("encoder", "tokens", "max_active_paths", "decoding_method"):
        assert name in params
    for missing in ("nbest", "num_hypotheses", "return_nbest", "best_of"):
        assert missing not in params


def test_transcript_result_carries_acoustic_score_fields():
    result = TranscriptResult(text="a", acoustic_logprob=-1.5, n_tokens=2)
    payload = result.as_dict()
    assert payload["acoustic_logprob"] == -1.5
    assert payload["n_tokens"] == 2
    # Backwards compatible: existing fields keep their defaults.
    plain = TranscriptResult(text="a")
    assert plain.acoustic_logprob is None and plain.n_tokens is None
    assert REQUIRED_FROM_TRANSDUCER_ARGS[0] == "encoder"

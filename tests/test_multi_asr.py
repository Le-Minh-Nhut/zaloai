"""Multi-ASR composition tests, with fake backends (no weights needed).

The real second-model path is BLOCKED on this machine (no chunkformer / 1.5
weights), so every behaviour asserted here is the composition logic itself:
identity when disabled, dedupe, ordering stability, failure policies, wall-clock
budget, and "a single-ASR run is never reported as multi-ASR".
"""

from __future__ import annotations

import sys
import time
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
from lifemate.asr.multi_backend import REAL_SECONDARY_BLOCKED, MultiAsrBackend
from lifemate.asr.nbest import NBestResult, AsrHypothesis


class FakeBackend:
    """Stand-in recognizer: canned texts, optional failure/latency, call log."""

    is_oracle = False

    def __init__(
        self,
        name: str,
        texts: list[str],
        *,
        scores: list[float] | None = None,
        error: BaseException | None = None,
        delay: float = 0.0,
        nbest: bool = False,
    ) -> None:
        self.name = name
        self.texts = list(texts)
        self.scores = list(scores or [])
        self.error = error
        self.delay = delay
        self.nbest_supported = nbest
        self.transcribe_calls = 0
        self.nbest_calls = 0

    def transcribe(self, audio_path, *, sample_id=None, lenient=False) -> TranscriptResult:
        self.transcribe_calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        score = self.scores[0] if self.scores else None
        return TranscriptResult(
            text=self.texts[0] if self.texts else "",
            backend=self.name,
            model_name=f"{self.name}-model",
            acoustic_logprob=score,
            n_tokens=4 if score is not None else None,
        )

    def transcribe_nbest(self, audio_path, *, nbest_size=3, sample_id=None) -> NBestResult:
        self.nbest_calls += 1
        if self.error is not None:
            raise self.error
        return NBestResult(
            hypotheses=[
                AsrHypothesis(
                    text=text,
                    rank=index,
                    score=self.scores[index] if index < len(self.scores) else None,
                    source=self.name,
                    backend=self.name,
                    score_kind="sum_ys_log_probs" if self.scores else None,
                )
                for index, text in enumerate(self.texts)
            ],
            requested_size=nbest_size,
            sources=[self.name],
        )


def _identity_disabled():
    primary = FakeBackend("primary", ["Xin chào"])
    secondary = FakeBackend("secondary", ["Alô"])
    backend = MultiAsrBackend(primary, secondary, enabled=False)
    sentinel = object()
    original = primary.transcribe

    def _tracked(*args, **kwargs):
        primary.tracked = sentinel
        return original(*args, **kwargs)

    primary.transcribe = _tracked  # type: ignore[method-assign]
    result = backend.transcribe("audio.wav")
    assert getattr(primary, "tracked", None) is sentinel
    # The primary's own object comes back unchanged — identity, not a copy.
    assert result is primary.transcribe("audio.wav") or isinstance(result, TranscriptResult)
    assert result.text == "Xin chào"
    assert result.backend == "primary"
    # The secondary was never constructed into the run nor called.
    assert secondary.transcribe_calls == 0
    assert secondary.nbest_calls == 0
    return backend, result


def test_disabled_returns_primary_result_and_never_touches_secondary():
    primary = FakeBackend("primary", ["Xin chào"])
    secondary = FakeBackend("secondary", ["Alô"])
    backend = MultiAsrBackend(primary, secondary, enabled=False)

    result = backend.transcribe("audio.wav")
    assert result.text == "Xin chào"
    assert result.backend == "primary"
    assert secondary.transcribe_calls == 0 and secondary.nbest_calls == 0

    # Identity: the very object the primary returns is handed back unchanged.
    stored: list[TranscriptResult] = []
    real = primary.transcribe

    def _capture(*args, **kwargs):
        out = real(*args, **kwargs)
        stored.append(out)
        return out

    primary.transcribe = _capture  # type: ignore[method-assign]
    assert backend.transcribe("audio.wav") is stored[0]


def test_disabled_transcribe_multi_is_not_reported_as_multi_asr():
    primary = FakeBackend("primary", ["Xin chào"])
    secondary = FakeBackend("secondary", ["Alô"])
    backend = MultiAsrBackend(primary, secondary, enabled=False)
    result = backend.transcribe_multi("audio.wav")
    assert result.multi_asr is False
    assert len(result.hypotheses) == 1
    assert result.sources == ["primary"]
    assert result.secondary_error is None
    assert secondary.transcribe_calls == 0


def test_enabled_without_a_secondary_is_single_asr():
    primary = FakeBackend("primary", ["Xin chào"])
    backend = MultiAsrBackend(primary, None, enabled=True)
    result = backend.transcribe_multi("audio.wav")
    assert result.multi_asr is False
    assert len(result.hypotheses) == 1
    assert result.degraded is True
    assert "multi-ASR disabled" in " ".join(result.reasons)


def test_enabled_dedupes_primary_topk_against_the_secondary():
    """Primary top-k is used when the backend exposes it; duplicates collapse."""
    primary = FakeBackend(
        "primary", ["Xin chào", "  xin   CHÀO", "Chào bạn"], scores=[-1.0, -1.2, -2.0], nbest=True
    )
    secondary = FakeBackend("secondary", ["chào BẠN", "xin chào"], scores=[-3.0, -4.0], nbest=True)
    backend = MultiAsrBackend(primary, secondary, enabled=True, primary_topk=3, max_hypotheses=3)
    result = backend.transcribe_multi("audio.wav")

    # Primary keeps its (source priority, rank) order; the second candidate is the
    # same text after NFC + whitespace collapse + case-fold, so it is dropped —
    # diacritics are NOT folded, so "Xin chao" would stay a distinct candidate.
    assert [h.text for h in result.hypotheses] == ["Xin chào", "Chào bạn"]
    assert [h.source for h in result.hypotheses] == ["primary", "primary"]
    assert [h.rank for h in result.hypotheses] == [0, 1]
    # Both of the secondary's texts were already covered: it contributed nothing,
    # so the run is honestly reported as single-source, not as multi-ASR.
    assert secondary.nbest_calls == 1
    assert result.multi_asr is False
    assert result.sources == ["primary"]
    assert result.scores_calibrated is False
    assert any("NOT a multi-ASR result" in reason for reason in result.reasons)


def test_secondary_novel_text_is_kept_and_tagged():
    primary = FakeBackend("primary", ["Xin chào"], scores=[-1.0])
    secondary = FakeBackend("secondary", ["Xin chào bạn"], scores=[-2.0])
    backend = MultiAsrBackend(primary, secondary, enabled=True)
    result = backend.transcribe_multi("audio.wav")
    assert [h.text for h in result.hypotheses] == ["Xin chào", "Xin chào bạn"]
    assert [h.source for h in result.hypotheses] == ["primary", "secondary"]
    assert result.multi_asr is True
    assert result.sources == ["primary", "secondary"]
    assert result.degraded is False


def test_ordering_is_stable_under_input_reordering():
    texts = ["một hai", "ba bốn"]
    first = MultiAsrBackend(
        FakeBackend("primary", texts), FakeBackend("secondary", ["năm sáu"]), enabled=True
    ).transcribe_multi("a.wav")
    second = MultiAsrBackend(
        FakeBackend("primary", list(reversed(texts))),
        FakeBackend("secondary", ["năm sáu"]),
        enabled=True,
    ).transcribe_multi("a.wav")
    # Each run is internally ordered by (source priority, rank) — reordering a
    # backend's own candidate list changes its ranks, never the source priority.
    assert first.hypotheses[0].source == "primary"
    assert second.hypotheses[0].source == "primary"
    assert first.hypotheses[-1].source == "secondary"
    assert second.hypotheses[-1].source == "secondary"
    # Running the SAME inputs twice is byte-identical.
    again = MultiAsrBackend(
        FakeBackend("primary", texts), FakeBackend("secondary", ["năm sáu"]), enabled=True
    ).transcribe_multi("a.wav")
    assert [(h.text, h.rank, h.source) for h in again.hypotheses] == [
        (h.text, h.rank, h.source) for h in first.hypotheses
    ]


def test_max_hypotheses_caps_the_output():
    primary = FakeBackend("primary", [f"text {i}" for i in range(5)])
    secondary = FakeBackend("secondary", ["secondary text"])
    backend = MultiAsrBackend(primary, secondary, enabled=True, max_hypotheses=2)
    result = backend.transcribe_multi("audio.wav")
    assert len(result.hypotheses) == 2
    assert [h.rank for h in result.hypotheses] == [0, 1]


def test_failure_policy_primary_only_records_the_error_and_continues():
    primary = FakeBackend("primary", ["Xin chào"])
    secondary = FakeBackend("secondary", ["Alô"], error=RuntimeError("secondary crashed"))
    backend = MultiAsrBackend(primary, secondary, enabled=True, failure_policy="primary_only")
    result = backend.transcribe_multi("audio.wav")

    assert result.text == "Xin chào"
    assert result.secondary_error is not None and "secondary crashed" in result.secondary_error
    assert result.degraded is True
    assert result.multi_asr is False  # only one backend contributed
    assert any("secondary backend failed" in reason for reason in result.reasons)


def test_failure_policy_strict_reraises():
    primary = FakeBackend("primary", ["Xin chào"])
    secondary = FakeBackend("secondary", ["Alô"], error=RuntimeError("secondary crashed"))
    backend = MultiAsrBackend(primary, secondary, enabled=True, failure_policy="strict")
    try:
        backend.transcribe_multi("audio.wav")
    except RuntimeError as exc:
        assert "secondary crashed" in str(exc)
    else:
        raise AssertionError('failure_policy="strict" must re-raise the secondary failure')


def test_primary_failure_is_always_an_error():
    primary = FakeBackend("primary", ["Xin chào"], error=RuntimeError("primary down"))
    secondary = FakeBackend("secondary", ["Alô"])
    backend = MultiAsrBackend(primary, secondary, enabled=True, failure_policy="primary_only")
    try:
        backend.transcribe_multi("audio.wav")
    except AsrError as exc:
        assert "primary backend" in str(exc) and "primary down" in str(exc)
    else:
        raise AssertionError("a primary failure must never be silently downgraded")


def test_budget_seconds_drops_secondary_hypotheses():
    primary = FakeBackend("primary", ["Xin chào"])
    # The probe decode is fast; the secondary never gets a chance because the
    # primary stage already consumed the budget.
    primary.delay = 0.05
    secondary = FakeBackend("secondary", ["Alô"])
    backend = MultiAsrBackend(primary, secondary, enabled=True, budget_seconds=0.01)
    result = backend.transcribe_multi("audio.wav")

    assert result.budget_exceeded is True
    assert secondary.transcribe_calls == 0 and secondary.nbest_calls == 0
    assert [h.text for h in result.hypotheses] == ["Xin chào"]
    assert result.degraded is True
    assert any("budget_seconds" in reason for reason in result.reasons)


def test_budget_seconds_not_exceeded_keeps_the_secondary():
    primary = FakeBackend("primary", ["Xin chào"])
    secondary = FakeBackend("secondary", ["Xin chào bạn"])
    backend = MultiAsrBackend(primary, secondary, enabled=True, budget_seconds=30.0)
    result = backend.transcribe_multi("audio.wav")
    assert result.budget_exceeded is False
    assert result.multi_asr is True


def test_composition_parameters_are_validated():
    primary = FakeBackend("primary", ["x"])
    for kwargs in (
        {"policy": "score_fusion"},
        {"failure_policy": "ignore"},
        {"max_hypotheses": 0},
        {"primary_topk": 0},
        {"budget_seconds": 0},
    ):
        try:
            MultiAsrBackend(primary, FakeBackend("secondary", ["y"]), enabled=True, **kwargs)
        except AsrError:
            continue
        raise AssertionError(f"invalid parameter accepted: {kwargs}")


def test_secondary_must_be_a_different_backend():
    primary = FakeBackend("primary", ["x"])
    try:
        MultiAsrBackend(primary, primary, enabled=True)
    except AsrError as exc:
        assert "DIFFERENT backend instance" in str(exc)
    else:
        raise AssertionError("composing a backend with itself must be rejected")


def test_backend_without_nbest_degrades_to_its_single_transcript():
    primary = FakeBackend("primary", ["Xin chào"])
    secondary = FakeBackend("secondary", ["Alô"])
    backend = MultiAsrBackend(primary, secondary, enabled=True)
    result = backend.transcribe_multi("audio.wav")
    # Neither fake advertises nbest_supported, so both contribute exactly one text.
    assert [h.text for h in result.hypotheses] == ["Xin chào", "Alô"]


def test_real_secondary_is_reported_blocked():
    assert REAL_SECONDARY_BLOCKED.startswith("BLOCKED")
    assert "python3 -c" in REAL_SECONDARY_BLOCKED
    assert "gipformer1.5-68M-rnnt" in REAL_SECONDARY_BLOCKED

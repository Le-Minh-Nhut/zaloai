#!/usr/bin/env python3
"""Competition entrypoint: dataset -> ASR -> agent -> ``submission.jsonl``.

Usage::

    python scripts/infer.py --split public_test --data-root ./data \\
        --output submission.jsonl --model-mode full --checkpoint outputs/checkpoints/full

Every row is produced from the official environment: ``tool_calls`` comes from
``env.get_tool_calls()`` and ``state_hash`` from ``env.get_state_hash()``. Failed
calls are kept, because the official scorer replays the whole call history.

Exit codes: 0 ok, 2 dataset/environment missing, 3 model load failure,
4 submission validation failure.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.agent.agent import ToolAgent  # noqa: E402
from lifemate.agent.tool_registry import ToolRegistry  # noqa: E402
from lifemate.asr import AsrError, create_asr_backend  # noqa: E402
from lifemate.compliance import assert_path_allowed  # noqa: E402
from lifemate.config import load_config  # noqa: E402
from lifemate.data.discovery import available_splits, require_split  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples  # noqa: E402
from lifemate.env.factory import create_environment  # noqa: E402
from lifemate.errors import DatasetNotFoundError, LifeMateError, ProhibitedAccessError  # noqa: E402
from lifemate.evaluation.submission import SubmissionError, SubmissionWriter, assert_env_sourced  # noqa: E402
from lifemate.evaluation.submission_validator import validate_submission  # noqa: E402

MOCK_BANNER = "MOCK ENVIRONMENT — NOT VALID FOR SUBMISSION"
OFFICIAL_OUTPUT = "outputs/submission.jsonl"

#: Splits whose rows carry gold labels; only these may use oracle transcripts.
LABELED_SPLITS = ("train", "validation", "val", "dev")

_PII_KEYS = frozenset(
    {
        "user_profile",
        "messages",
        "message",
        "text",
        "audio_text",
        "transcript",
        "content",
        "history",
        "phone",
        "email",
        "address",
    }
)

EXIT_OK = 0
EXIT_DATASET = 2
EXIT_MODEL = 3
EXIT_SUBMISSION = 4

#: Refuse to write a submission when more than this share of samples failed ASR.
ASR_FAILURE_RATIO = 0.5


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Run LifeMate inference and write a competition submission.",
    )
    parser.add_argument("--split", default=None, help="split name (default: config inference.split)")
    parser.add_argument("--data-root", default=None, help="dataset root (default: config data.root)")
    parser.add_argument("--output", default=None, help="submission path (default: config inference.output)")
    parser.add_argument(
        "--model-mode",
        choices=("base", "full", "lora"),
        default=None,
        help="base model, full fine-tune checkpoint, or LoRA adapter",
    )
    parser.add_argument("--checkpoint", default=None, help="checkpoint directory (full/lora)")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE", help="dotted config override")
    _add_feature_arguments(parser)
    parser.add_argument("--limit", type=int, default=None, help="max samples")
    parser.add_argument("--asr", choices=("gipformer", "oracle", "whisper", "none"), default=None, help="ASR backend")
    parser.add_argument("--transcripts", default=None, help="JSONL/JSON mapping sample_id -> transcript (oracle)")
    parser.add_argument("--allow-oracle", action="store_true", help="permit oracle ASR on a LABELED split only")
    parser.add_argument("--use-mock-env", action="store_true", help="synthetic MockToolEnv (never valid for submission)")
    parser.add_argument("--max-steps", type=int, default=None, help="agent step budget")
    parser.add_argument("--time-budget", type=float, default=None, help="seconds per sample")
    parser.add_argument("--batch-size", type=int, default=None, help="reserved: generation batch size")
    parser.add_argument("--seed", type=int, default=None, help="reserved: RNG seed")
    parser.add_argument("--device", default=None, help="torch device for the model")
    parser.add_argument("--model", default=None, help="override model.name (base model id or path)")
    parser.add_argument("--debug-dump", default=None, metavar="DIR", help="write sanitized per-sample prompts")
    parser.add_argument("--resume", action="store_true", help="skip ids already present in --output")
    parser.add_argument(
        "--allow-empty-transcripts",
        action="store_true",
        help="diagnostic only; never for a competition submission",
    )
    parser.add_argument(
        "--generator",
        default=None,
        metavar="dumpscript:PATH",
        help="TEST HOOK: JSON file mapping sample_id -> list of raw model outputs (mock only, "
        "or the official env with --allow-scripted-official and a non-official output path)",
    )
    parser.add_argument(
        "--allow-scripted-official",
        action="store_true",
        help="PIPELINE TEST ONLY: permit --generator with the REAL env. The rows are a scripted "
        "artefact, never a model prediction, and the official output path stays refused.",
    )
    parser.add_argument(
        "--history-transcripts",
        default=None,
        metavar="FILE",
        help="ASR transcripts of history user turns (JSONL keyed by split/id/turn_index). "
        "A test split's history carries audio only, so without this its prior turns stay blank.",
    )
    parser.add_argument(
        "--allow-missing-history",
        action="store_true",
        help="DIAGNOSTIC ONLY: accept history user turns that stay blank (no --history-transcripts file, "
        "or turns the file does not cover). A submission run must never take this flag",
    )
    parser.add_argument("--quiet", action="store_true", help="only print errors and the summary line")
    return parser.parse_args(argv)


def _apply_overrides(cfg: Any, overrides: Sequence[str]) -> Any:
    from lifemate.config import _set_dotted

    for item in overrides:
        if "=" not in item:
            raise LifeMateError(f"--override expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        _set_dotted(cfg, key.strip(), value.strip())
    cfg.validate()
    return cfg

# --------------------------------------------------------------------------- #
# Feature flags
# --------------------------------------------------------------------------- #
def _add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    """``--enable-feature/--disable-feature``; a checkout without the slice gets none."""
    try:
        from lifemate.features.overrides import add_feature_arguments  # noqa: PLC0415
    except ImportError:  # pragma: no cover - optional slice absent
        return
    add_feature_arguments(parser)

def _effective_features(cfg: Any, args: argparse.Namespace) -> Any:
    """Resolve the feature chain onto ``cfg``; ``None`` when the slice is absent.

    Every flag is resolved ONCE here, so the rest of the script can only see the
    effective state. An unknown id raises (``LifeMateError``) with a suggestion
    instead of being silently ignored.
    """
    try:
        from lifemate.features.config import FeatureConfig  # noqa: PLC0415
        from lifemate.features.overrides import build_feature_config  # noqa: PLC0415
    except ImportError:  # pragma: no cover - optional slice absent
        return None
    if getattr(cfg, "features", None) is None:  # wiring not landed yet: defaults
        cfg.features = FeatureConfig()
    features, _provenance = build_feature_config(cfg, args)
    cfg.features = features
    return features

def _feature_report(cfg: Any) -> dict[str, Any]:
    """JSON-ready feature report (``{}`` when the slice is absent)."""
    from lifemate.features.report import feature_report_dict

    features = getattr(cfg, "features", None)
    if features is None:  # pragma: no cover - slice absent
        return {}
    return feature_report_dict(features, model_id=cfg.model.name, asr_id=cfg.asr.model_id)

def _print_feature_report(cfg: Any) -> dict[str, Any]:
    """Print the effective configuration once, before any model is loaded."""
    from lifemate.features.report import format_feature_report

    features = getattr(cfg, "features", None)
    if features is None:  # pragma: no cover - slice absent
        return {}
    print(
        format_feature_report(features, model_id=cfg.model.name, asr_id=cfg.asr.model_id),
        file=sys.stderr,
    )
    return _feature_report(cfg)

def _flag(features: Any, section: str, name: str) -> bool:
    """One feature flag; ``False`` when the object (or the section) is absent."""
    return bool(getattr(getattr(features, section, None), name, False))

def _param(features: Any, section: str, name: str, default: Any) -> Any:
    """One feature parameter; ``default`` when absent."""
    value = getattr(getattr(features, section, None), name, default)
    return default if value is None else value

def _build_config(args: argparse.Namespace) -> Any:
    cfg = load_config(args.config)
    if args.split:
        cfg.inference.split = args.split
    if args.data_root:
        cfg.data.root = args.data_root
    if args.output:
        cfg.inference.output = args.output
    if args.model_mode:
        cfg.inference.model_mode = args.model_mode
    if args.checkpoint:
        cfg.inference.checkpoint = args.checkpoint
    if args.limit is not None:
        cfg.inference.limit = args.limit
    if args.batch_size is not None:
        cfg.inference.batch_size = args.batch_size
    if args.use_mock_env:
        cfg.inference.use_mock_env = True
    if args.asr:
        cfg.asr.backend = args.asr
    if args.max_steps is not None:
        cfg.agent.max_steps = args.max_steps
    if args.time_budget is not None:
        cfg.agent.time_budget_seconds = args.time_budget
    if args.device:
        cfg.model.device = args.device
    if getattr(args, "model", None):
        cfg.model.name = args.model
    if args.seed is not None:
        cfg.training.seed = args.seed
    if getattr(args, "history_transcripts", None):
        cfg.inference.history_transcripts = args.history_transcripts
    cfg = _apply_overrides(cfg, args.override)
    # Feature chain LAST: --override may set features.* paths, and CLI flags win.
    _effective_features(cfg, args)
    return cfg


# --------------------------------------------------------------------------- #
# Test hook: scripted generator
# --------------------------------------------------------------------------- #
class ScriptedGenerator:
    """Deterministic generator replaying raw model outputs per sample id.

    Used by the mock-environment end-to-end test. It never consults the model,
    and it refuses to answer ids it was not given, so a missing script entry is a
    visible error rather than a fabricated action.
    """

    def __init__(self, script: Mapping[str, Any], *, current_id: Callable[[], str] | None = None) -> None:
        self.script = {str(key): list(value) if isinstance(value, list) else [value] for key, value in script.items()}
        self.current_id = current_id
        self.calls: list[str] = []
        self.max_new_tokens = 512

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        """Return the next scripted output for the current sample."""
        sample_id = self.current_id() if self.current_id else ""
        if not sample_id and len(self.script) == 1:
            sample_id = next(iter(self.script))
        queue = self.script.get(sample_id)
        if not queue:
            raise LifeMateError(
                f"ScriptedGenerator has no output left for sample {sample_id!r}; "
                "add another action (or the finish marker) to the dump script"
            )
        self.calls.append(sample_id)
        return queue.pop(0)

    def remaining(self) -> dict[str, int]:
        return {key: len(value) for key, value in self.script.items()}


def load_dump_script(path: str | Path) -> dict[str, Any]:
    """Read the ``--generator`` JSON file (``{"sample_id": ["<action>..."]}``)."""
    script_path = Path(path)
    payload = json.loads(script_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise LifeMateError(f"generator script {script_path} must be a JSON object of sample_id -> outputs")
    return {str(key): value for key, value in payload.items()}


def _validate_generator_flag(args: argparse.Namespace, mock: bool = False) -> None:
    """Refuse the test hook unless a mock run, or an explicit pipeline-test opt-in."""
    if not args.generator:
        return
    if not args.generator.startswith("dumpscript:"):
        raise LifeMateError(
            f"--generator only supports 'dumpscript:PATH' (got {args.generator!r})"
        )
    if getattr(args, "allow_scripted_official", False):
        # Opt-in exists solely to prove the env+submission plumbing on the real
        # split without weights. Rows produced this way are scripted artefacts.
        return
    if not args.use_mock_env and not mock:
        raise LifeMateError(
            "--generator is a TEST HOOK and requires --use-mock-env (or the explicit "
            "pipeline-test flag --allow-scripted-official with a non-official output); "
            "never use a scripted generator for an official submission"
        )

# --------------------------------------------------------------------------- #
# ASR
# --------------------------------------------------------------------------- #
def _load_transcripts(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        raise LifeMateError(f"transcripts file not found: {target}")
    text = target.read_text(encoding="utf-8")
    if target.suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise LifeMateError(f"transcripts JSON must be an object of sample_id -> text: {target}")
        return {str(key): str(value) for key, value in payload.items()}
    mapping: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise LifeMateError(f"{target}:{lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, Mapping) or "id" not in row:
            raise LifeMateError(f"{target}:{lineno}: transcript rows need an 'id' field")
        value = row.get("text", row.get("transcript", row.get("audio_text", "")))
        mapping[str(row["id"])] = str(value or "")
    return mapping


def _build_asr(args: argparse.Namespace, cfg: Any, samples: Sequence[Any]) -> Any:
    """Pick the ASR backend, refusing oracle transcripts on unlabeled splits.

    ``public_test``/``private_test`` carry no gold transcript, so oracle there
    would either fail or (worse) leak labels. It is therefore an error unless the
    split is labeled AND ``--allow-oracle`` is passed; the warning is loud.
    """
    backend = (args.asr or cfg.asr.backend or "none").lower()
    split = cfg.inference.split
    labeled = split in LABELED_SPLITS

    if backend == "oracle":
        if not labeled:
            raise LifeMateError(
                f"--asr oracle is forbidden for split {split!r}: test splits have no gold transcript. "
                "Use --asr gipformer for inference, or evaluate on a labeled split "
                f"({', '.join(LABELED_SPLITS)}) with --allow-oracle."
            )
        if not args.allow_oracle:
            raise LifeMateError(
                "oracle ASR requires --allow-oracle and is for training/evaluation only; "
                "it must never produce a competition submission"
            )
        transcripts = _load_transcripts(getattr(args, "transcripts", None))
        for sample in samples:
            transcripts.setdefault(str(sample.id), str(getattr(sample, "audio_text", "") or ""))
        if not any(transcripts.values()):
            raise LifeMateError(
                f"oracle ASR found no transcripts for split {split!r}; pass --transcripts PATH"
            )
        print(
            f"WARNING: oracle ASR is active on labeled split {split!r}. "
            "These predictions are only meaningful for pipeline checks, not as ASR performance.",
            file=sys.stderr,
        )
        primary = create_asr_backend(cfg, transcripts=transcripts, backend="oracle")
        return _compose_multi_asr(cfg, primary)

    if backend == "whisper":
        primary = create_asr_backend(cfg, backend="whisper")
        return _compose_multi_asr(cfg, primary)

    if backend == "gipformer":
        overrides = {}
        if _flag(getattr(cfg, "features", cfg), "asr", "gipformer_v15_enabled"):
            overrides = {"model_id": "g-group-ai-lab/gipformer1.5-68M-rnnt", "model_dir": None, "revision": ""}
        primary = create_asr_backend(cfg, backend="gipformer", **overrides)
        return _compose_multi_asr(cfg, primary)

    if backend == "none":
        return None

    # Unknown backend names must fail loudly; the factory raises AsrError and
    # lists the valid choices, so a typo can never yield an empty submission.
    return _compose_multi_asr(cfg, create_asr_backend(cfg, backend=backend))

def _compose_multi_asr(cfg: Any, primary: Any) -> Any:
    """Wrap ``primary`` in :class:`MultiAsrBackend` only when ``asr.multi_asr`` is on.

    Flag off returns ``primary`` itself — the exact object ``create_asr_backend``
    built, so the baseline path is byte-identical and no secondary is constructed.
    """
    features = getattr(cfg, "features", cfg)   # a FeatureConfig works directly
    if not _flag(features, "asr", "multi_asr_enabled"):
        return primary
    if getattr(primary, "is_oracle", False):
        # Oracle text is a gold label, not a recognizer output: composing it would
        # also fail (it takes no ``lenient`` kwarg). Say so instead of breaking a
        # pipeline that works.
        print(
            "WARNING: asr.multi_asr is ON but the primary backend is the ORACLE "
            "transcript source; multi-ASR is not applied to gold transcripts.",
            file=sys.stderr,
        )
        return primary
    from lifemate.asr.multi_backend import MultiAsrBackend

    secondary = None
    secondary_id = str(_param(features, "asr", "secondary_model_id", "") or "").strip()
    sec_backend = str(_param(features, "asr", "secondary_backend", "whisper") or "whisper").strip().lower()
    if "chunkformer" in secondary_id.lower() or "chunkformer" in sec_backend:
        sec_backend = "chunkformer"
    if secondary_id:
        try:
            if sec_backend == "chunkformer":
                raise AsrError(
                    f"secondary_backend='chunkformer' ({secondary_id!r}) is unsupported: "
                    "no chunkformer weights or runtime available locally"
                )
            model_arg = "model_id" if sec_backend == "gipformer" else "model_name"
            secondary = create_asr_backend(cfg, backend=sec_backend, **{model_arg: secondary_id})
        except Exception as exc:  # noqa: BLE001 - degrade loudly, never silently
            failure_pol = str(_param(features, "asr", "multi_asr_failure_policy", "primary_only"))
            if failure_pol == "strict":
                raise
            print(
                f"WARNING: asr.multi_asr is ON but the secondary backend {secondary_id!r} could not "
                f"be built ({type(exc).__name__}: {exc}); running PRIMARY ONLY. This run is not a "
                "multi-ASR result.",
                file=sys.stderr,
            )
    else:
        if str(_param(features, "asr", "multi_asr_failure_policy", "primary_only")) == "strict":
            raise AsrError("multi-ASR secondary_model_id is empty in strict mode")
        print(
            "WARNING: asr.multi_asr is ON but features.asr.secondary_model_id is empty; "
            "running PRIMARY ONLY. This run is not a multi-ASR result.",
            file=sys.stderr,
        )
    return MultiAsrBackend(
        primary,
        secondary,
        enabled=True,
        policy=str(_param(features, "asr", "multi_asr_policy", "primary_topk_secondary_top1")),
        max_hypotheses=int(_param(features, "asr", "multi_asr_max_hypotheses", 3)),
        primary_topk=int(_param(features, "asr", "nbest_size", 3)) if _flag(features, "asr", "nbest_enabled") else 1,
        failure_policy=str(_param(features, "asr", "multi_asr_failure_policy", "primary_only")),
    )

def _write_wav(path: Path, samples: Sequence[float], rate: int) -> None:
    """Write 16-bit PCM mono; the conditioned samples need a path for the backend."""
    import wave

    frames = bytearray()
    for value in samples:
        clamped = -1.0 if value < -1.0 else (1.0 if value > 1.0 else float(value))
        frames += int(round(clamped * 32767.0)).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(bytes(frames))

def _condition_audio(
    audio_path: Path, features: Any, directory: Path
) -> tuple[Path, dict[str, Any]]:
    """VAD trim then RMS levelling, each gated on its own flag; returns a temp WAV.

    ``GipformerBackend.transcribe`` accepts a path, not samples, so conditioned
    audio is written once to ``directory`` (a caller-owned temp dir, deleted right
    after transcription). Cost: one extra decode of the input plus one WAV write
    per sample; nothing else is re-measured here.
    """
    from lifemate.asr.audio_ops import normalize_rms, trim_silence
    from lifemate.asr.interface import TARGET_SAMPLE_RATE, load_audio_16k_mono

    samples, rate = load_audio_16k_mono(audio_path, target_rate=TARGET_SAMPLE_RATE)
    stats: dict[str, Any] = {}
    if _flag(features, "audio", "vad_enabled"):
        trimmed = trim_silence(
            samples,
            sample_rate=rate,
            enabled=True,
            min_silence_ms=int(_param(features, "audio", "vad_min_silence_ms", 300)),
            padding_ms=int(_param(features, "audio", "vad_padding_ms", 200)),
            channels=1,
        )
        samples = trimmed.samples
        stats["vad"] = {**trimmed.stats, "changed": trimmed.changed}
    if _flag(features, "audio", "rms_normalization_enabled"):
        levelled = normalize_rms(
            samples,
            enabled=True,
            target_dbfs=float(_param(features, "audio", "rms_target_dbfs", -23.0)),
        )
        samples = levelled.samples
        stats["rms"] = {**levelled.stats, "changed": levelled.changed}
    target = Path(directory) / "conditioned.wav"
    _write_wav(target, samples, rate)
    return target, stats

def _nbest_text(asr: Any, audio_path: Path, sample_id: str, features: Any) -> tuple[Any, str | None]:
    """``transcribe_nbest`` result, or ``None`` plus the reason it is unavailable.

    A 1-element "N-best" would fabricate a result, so an unsupported runtime
    falls back to top-1 with a printed line naming the flag and the reason.

    The CAPABILITY GATE comes first. ``transcribe_nbest`` exists on
    ``GipformerBackend`` and ``MultiAsrBackend`` only: ``WhisperBackend`` and
    ``OracleTranscriptBackend`` have neither the method nor ``nbest_supported``,
    so calling them raised ``AttributeError`` here, which the enclosing handler
    turned into an EMPTY transcript with no warning — silent data loss on a
    submission path. The broad catch that follows is defence in depth for a
    backend that advertises the capability but has a different signature; it
    must never be the primary mechanism.
    """
    from lifemate.asr.nbest import NBestUnsupportedError, require_nbest_support

    size = int(_param(features, "asr", "nbest_size", 3))
    try:
        require_nbest_support(asr, nbest_size=size)
        return asr.transcribe_nbest(audio_path, nbest_size=size, sample_id=sample_id), None
    except (NBestUnsupportedError, AttributeError, NotImplementedError, TypeError) as exc:
        reason = str(exc).strip() or (
            f"{type(exc).__name__}: backend {getattr(asr, 'name', type(asr).__name__)!r} "
            f"cannot decode N-best (nbest_size={size})"
        )
        return None, reason

def _transcribe(
    asr: Any,
    sample: Any,
    *,
    features: Any = None,
) -> tuple[str, dict[str, Any]]:
    """Transcribe one sample; errors produce an empty text plus a reason.

    The LLM-facing transcript is normalized here, exactly once, for EVERY backend
    (including oracle): a gold transcript and a recognized one must reach the agent
    through the same text pipe, or the ASR-vs-agent comparison is confounded.
    """
    if asr is None:
        return "", {"backend": "none", "error": None}
    from lifemate.data.loader import read_sample_audio

    try:
        audio_path = read_sample_audio(sample)   # guards *_environment; None when the file is absent
    except ProhibitedAccessError:
        raise                                    # compliance violations are never downgraded
    if audio_path is None:
        return "", {"backend": getattr(asr, "name", "unknown"), "error": "audio file missing for sample"}
    sample_id = str(getattr(sample, "id", ""))
    backend_label = str(getattr(asr, "name", "unknown"))
    condition = _flag(features, "audio", "vad_enabled") or _flag(
        features, "audio", "rms_normalization_enabled"
    )
    conditioning: dict[str, Any] = {}
    nbest_reason: str | None = None
    nbest_requested = False
    hypotheses: list[Any] | None = None
    try:
        with contextlib.ExitStack() as stack:
            transcription_path = audio_path
            if condition:
                # One extra WAV write per sample: the recognizer takes a path.
                import tempfile

                directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="lifemate_cond_"))
                transcription_path, conditioning = _condition_audio(
                    audio_path, features, Path(directory)
                )
            if _flag(features, "asr", "multi_asr_enabled") and hasattr(asr, "transcribe_multi"):
                nbest_requested = _flag(features, "asr", "nbest_enabled")
                multi_res = asr.transcribe_multi(transcription_path, sample_id=sample_id)
                if not multi_res.hypotheses:
                    raise AsrError("multi-ASR produced no hypotheses")
                result = multi_res.to_transcript_result(backend=multi_res.hypotheses[0].backend or backend_label)
                hypotheses = list(multi_res.hypotheses)
                conditioning["multi_asr"] = {"executed": bool(multi_res.multi_asr), "degraded": bool(multi_res.degraded), "reasons": list(multi_res.reasons)}
                if nbest_requested:
                    if not getattr(asr.primary, "nbest_supported", False):
                        nbest_reason = "primary backend does not support decoder N-best"
                    else:
                        source = hypotheses[0].source
                        conditioning["nbest"] = {"requested": int(_param(features, "asr", "nbest_size", 3)), "returned": sum(h.source == source for h in hypotheses)}
            elif _flag(features, "asr", "nbest_enabled"):
                nbest_requested = True
                nbest, nbest_reason = _nbest_text(asr, transcription_path, sample_id, features)
                if nbest is None:
                    print(
                        f"WARNING: asr.nbest is ON but backend {backend_label!r} cannot decode "
                        f"N-best; falling back to its top-1 transcript. Reason: {nbest_reason}",
                        file=sys.stderr,
                    )
                    result = asr.transcribe(transcription_path, sample_id=sample_id)
                else:
                    result = nbest.to_transcript_result(source=backend_label)
                    hypotheses = list(getattr(nbest, "hypotheses", []) or [])
                    conditioning["nbest"] = {
                        "requested": int(_param(features, "asr", "nbest_size", 3)),
                        "returned": len(hypotheses),
                        "degraded": bool(getattr(nbest, "degraded", False)),
                    }
            else:
                result = asr.transcribe(transcription_path, sample_id=sample_id)
    except ProhibitedAccessError:
        raise                                    # a backend that reached a banned path must stop the run
    except Exception as exc:
        if _flag(features, "asr", "multi_asr_enabled") and _param(features, "asr", "multi_asr_failure_policy", "primary_only") == "strict":
            raise
        info = {
            "backend": backend_label,
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "nbest_requested": nbest_requested,
        }
        if nbest_reason is not None:
            info["nbest_fallback"] = nbest_reason[:300]
        return "", info
    from lifemate.text.itn import apply_itn_pipeline

    raw_text = str(getattr(result, "text", "") or "")
    # ``apply_itn_pipeline(enabled=False)`` IS ``to_llm_text``: one pipe either way.
    text = apply_itn_pipeline(raw_text, enabled=_flag(features, "prompt", "itn_enabled")) if raw_text else raw_text
    reported_backend = str(getattr(result, "backend", "") or "")
    info = {
        # A backend that reports nothing usable must not hide which one ran.
        "backend": backend_label if reported_backend in {"", "unknown"} else reported_backend,
        "error": getattr(result, "error", None),
        "latency_s": round(float(getattr(result, "latency_s", 0.0) or 0.0), 3),
        "raw_transcript": raw_text,
        "model_id": getattr(getattr(asr, "primary", asr), "model_id", None),
    }
    if conditioning:
        info["conditioning"] = conditioning
    if hypotheses is not None:
        from dataclasses import replace
        info["raw_hypotheses"] = hypotheses
        info["hypotheses"] = [replace(h, text=apply_itn_pipeline(h.text, enabled=_flag(features, "prompt", "itn_enabled"))) for h in hypotheses]
    if nbest_requested:
        info["nbest_requested"] = True
    if nbest_reason is not None:
        info["nbest_fallback"] = nbest_reason[:300]
    # Audio was fine and the recognizer returned nothing: that is an ASR failure,
    # never a silent blank prompt. The run counts it (and refuses a submission of
    # nothing but empties) because the caller reads ``error``.
    if not text and not info["error"]:
        info["error"] = (
            f"empty transcript from backend {info['backend']!r} for a readable audio file"
            + ("" if nbest_reason is None else " (after the asr.nbest fallback)")
        )
    return text, info


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def _load_generator(cfg: Any, args: argparse.Namespace, current_id: Callable[[], str]) -> Any:
    """Build the model generator (base/full/lora) or the scripted test double."""
    if args.generator:
        return ScriptedGenerator(load_dump_script(args.generator.split(":", 1)[1]), current_id=current_id)

    mode = cfg.inference.model_mode
    checkpoint = cfg.inference.checkpoint
    if mode in {"full", "lora"} and not checkpoint:
        raise LifeMateError(
            f"--model-mode {mode} requires --checkpoint (a directory written by scripts/train.py)"
        )
    if mode in {"full", "lora"}:
        ckpt = Path(str(checkpoint))
        if not ckpt.is_dir():
            raise LifeMateError(f"{mode} checkpoint directory does not exist: {ckpt}")
        adapter = (ckpt / "adapter_config.json").is_file()
        if mode == "lora" and not adapter:
            raise LifeMateError(f"LoRA checkpoint {ckpt} lacks adapter_config.json; select a trained adapter")
        full_weights = any(ckpt.glob("model*.safetensors")) or any(ckpt.glob("pytorch_model*.bin"))
        if mode == "full" and adapter and not full_weights:
            raise LifeMateError(f"{ckpt} is a LoRA adapter, not a full model; use --model-mode lora")
    from lifemate.models.generation import HFGenerator
    from lifemate.models.loader import load_model_and_tokenizer

    model_source = checkpoint if (mode == "base" and checkpoint) else cfg.model.name
    bundle = load_model_and_tokenizer(
        model_source,
        torch_dtype=cfg.model.torch_dtype,
        device=cfg.model.device,
        tuning_mode="lora" if mode == "lora" else "full",
        checkpoint=checkpoint,
        attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=cfg.model.trust_remote_code,
        temperature=cfg.agent.temperature,
        max_new_tokens=cfg.agent.max_new_tokens,
    )
    # Base identity comes from the config, not the checkpoint directory: a saved
    # tokenizer loses the family name, and the profile decides the prompt
    # template, the terminator and the decode stop ids.
    return HFGenerator(
        bundle.model,
        bundle.tokenizer,
        model_name=cfg.model.name,
        max_new_tokens=cfg.agent.max_new_tokens,
        temperature=cfg.agent.temperature,
        device=getattr(bundle, "device", None),
        **_constrained_kwargs(cfg, HFGenerator),
    )

def _constrained_kwargs(cfg: Any, generator_cls: Any) -> dict[str, Any]:
    """``constrained=`` for ``HFGenerator`` when the flag is on AND the hook exists.

    Feature-detected: a generator build without the ``constrained`` parameter
    cannot honour the flag, so it is reported as PENDING and the run continues
    UNCONSTRAINED — never claiming the feature ran.
    """
    features = getattr(cfg, "features", cfg)   # a FeatureConfig works directly
    if not _flag(features, "decoding", "constrained_decoding_enabled"):
        return {}
    import inspect

    try:
        accepted = set(inspect.signature(generator_cls.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic generator
        accepted = set()
    if "constrained" not in accepted:
        print(
            "constrained decoding: PENDING (generator has no hook yet) — this run is "
            "UNCONSTRAINED even though decoding.constrained is enabled",
            file=sys.stderr,
        )
        return {}
    # ``allowed_tools`` is per sample: the setter replaces it before each run.
    return {"constrained": True, "allowed_tools": ()}


# --------------------------------------------------------------------------- #
# Debug dumps
# --------------------------------------------------------------------------- #
#: Keys ``--debug-dump`` records on purpose (the ASR-vs-agent comparison needs them).
_DEBUG_KEEP = frozenset({"transcript", "raw_transcript"})


def _sanitize(value: Any, depth: int = 0) -> Any:
    """Produce bounded JSON-safe diagnostics while removing user-content keys."""
    import dataclasses
    if depth >= 20:
        return "<depth limit>"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v, depth + 1) for k, v in value.items() if str(k) not in _PII_KEYS}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth + 1) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


def _dump_debug(directory: str | Path, sample_id: str, payload: Mapping[str, Any]) -> None:
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in sample_id) or "sample"
    # The top-level transcript keys are requested explicitly by --debug-dump, so
    # they bypass the PII filter; nested payloads are still fully scrubbed.
    safe_payload = {
        str(key): (value if str(key) in _DEBUG_KEEP else _sanitize(value))
        for key, value in payload.items()
    }
    (base / f"{safe_name}.json").write_text(
        json.dumps(safe_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def _mock_state_for(sample_id: str, data_root: Path) -> dict[str, Any] | None:
    """Synthetic initial state from ``tests/fixtures`` — never from an env dir."""
    for candidate in (
        data_root / "mock_state" / f"{sample_id}.json",
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_lifemate" / "mock_state" / f"{sample_id}.json",
    ):
        allowed = assert_path_allowed(candidate)
        if allowed.is_file():
            payload = json.loads(allowed.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                return dict(payload)
    return None


def _refuse_mock_on_official_output(
    mock_or_args: Any,
    output: Path,
    explicit_output: bool | None = None,
) -> None:
    """The mock must not silently produce a file at the official submission path."""
    if isinstance(mock_or_args, argparse.Namespace):
        mock = bool(getattr(mock_or_args, "use_mock_env", False))
        if explicit_output is None:
            explicit_output = getattr(mock_or_args, "output", None) is not None
    else:
        mock = bool(mock_or_args)
        if explicit_output is None:
            explicit_output = True

    if not mock:
        return
    official = Path(OFFICIAL_OUTPUT).resolve()
    try:
        same = output.resolve() == official
    except OSError:
        same = str(output) == OFFICIAL_OUTPUT
    if same:
        raise LifeMateError(
            f"--use-mock-env refuses to write the default official output path ({OFFICIAL_OUTPUT}). "
            "Pass an explicit non-official path (e.g. --output /tmp/mock_submission.jsonl) for a self-test."
        )

def run_inference(argv: Sequence[str] | None = None) -> int:
    """Execute the inference pipeline; returns the process exit code."""
    args = parse_args(argv)
    try:
        cfg = _build_config(args)
        mock = bool(cfg.inference.use_mock_env)
        _validate_generator_flag(args, mock=mock)
        # Printed once, before any model/ASR weight is loaded.
        feature_report = _print_feature_report(cfg)
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_DATASET
    features = getattr(cfg, "features", None)

    split = cfg.inference.split
    data_root = Path(cfg.data.root)
    output = Path(cfg.inference.output)
    mock = bool(cfg.inference.use_mock_env)

    try:
        _refuse_mock_on_official_output(mock, output, explicit_output=(args.output is not None))
        if args.generator and output.resolve() == Path(OFFICIAL_OUTPUT).resolve():
            raise LifeMateError(
                f"a scripted generator must not write the official submission path ({OFFICIAL_OUTPUT}); "
                "pass a diagnostic --output path"
            )
    except (LifeMateError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_SUBMISSION

    if mock:
        print("=" * 72, file=sys.stderr)
        print(MOCK_BANNER, file=sys.stderr)
        print("Rows come from MockToolEnv; they are pipeline artefacts, not a submission.", file=sys.stderr)
        print("=" * 72, file=sys.stderr)

    # -- dataset ----------------------------------------------------------- #
    try:
        paths = require_split(data_root, split)
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_DATASET
    try:
        samples = load_samples(data_root, split, limit=cfg.inference.limit, with_labels=False)
        catalog = load_catalog(data_root, split)
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_DATASET
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_DATASET
    if not samples:
        print(
            f"ERROR: split {split!r} has no usable samples under {paths.root}. "
            f"Discovered splits: {', '.join(available_splits(data_root)) or '(none)'}",
            file=sys.stderr,
        )
        return EXIT_DATASET

    # -- history transcripts ------------------------------------------------ #
    # A test split ships history AUDIO only, so its user turns would reach the
    # agent blank. Fill them from ASR output; existing text (train's gold
    # audio_text) always wins. Never auto-discovered inside the dataset tree.
    history_path = getattr(cfg.inference, "history_transcripts", None)
    if history_path:
        from lifemate.data.history import HistoryTranscriptError, apply_history_texts, load_history_texts

        try:
            history_texts = load_history_texts(history_path, data_root=data_root)
        except (HistoryTranscriptError, ProhibitedAccessError) as exc:
            print(f"ERROR: cannot use --history-transcripts {history_path}: {exc}", file=sys.stderr)
            return EXIT_DATASET
        stats = apply_history_texts(samples, history_texts, split)
        print(
            f"history transcripts: split={split} file={history_path} "
            f"filled={stats['filled']} existing={stats['existing']} missing={stats['missing']} "
            f"(samples with history={stats['samples_with_history']}, "
            f"samples receiving text={stats['samples_touched']})",
            file=sys.stderr,
        )
        blank = stats["missing"]
        reason = (
            f"{blank} history user turn(s) are not covered by {history_path}"
        )
    else:
        blank = sum(
            1
            for sample in samples
            for turn in (getattr(sample, "history", None) or [])
            if str(getattr(turn, "role", "")).lower() == "user" and not str(getattr(turn, "content", "") or "").strip()
        )
        print(
            f"history transcripts: NONE configured (--history-transcripts / inference.history_transcripts); "
            f"split={split} user history turns without text={blank} — these render blank",
            file=sys.stderr,
        )
        reason = "no --history-transcripts file was configured"
    if blank and (args.asr or cfg.asr.backend or "none").lower() in {"none", "oracle"} and not getattr(args, "allow_missing_history", False):
        print(f"ERROR: {reason}; no audio ASR backend is available for history", file=sys.stderr)
        return EXIT_DATASET
    # Remaining turns are transcribed per sample after the backend is loaded.

    # -- environment preflight --------------------------------------------- #
    # Import the split's SDK before loading weights: a missing simulator must fail
    # in seconds (exit 2), not after a multi-GB model load. This must NOT build a
    # session: ToolEnv may be created exactly once per sample, and the loop below
    # creates the real one.
    try:
        if mock:
            create_environment(
                split,
                str(samples[0].id),
                data_root,
                use_mock=True,
                catalog=catalog,
                mock_state=_mock_state_for(str(samples[0].id), paths.root),
                apps=list(getattr(samples[0].context, "installed_apps", []) or []),
            )
        else:
            from lifemate.env.official import check_official_sdk

            check_official_sdk(split, data_root)
    except LifeMateError as exc:
        print(f"ERROR: cannot create the environment: {exc}", file=sys.stderr)
        print(
            "The official simulator (zaic_tools) is required for competition inference. "
            "Use --use-mock-env for a synthetic self-test.",
            file=sys.stderr,
        )
        return EXIT_DATASET

    # -- ASR --------------------------------------------------------------- #
    try:
        asr = _build_asr(args, cfg, samples)
    except (LifeMateError, AsrError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_DATASET
    # -- generator --------------------------------------------------------- #
    current_id = {"value": ""}
    try:
        generator = _load_generator(cfg, args, lambda: current_id["value"])
    except ImportError as exc:
        print(f"ERROR: model dependencies missing: {exc}", file=sys.stderr)
        return EXIT_MODEL
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_MODEL
    except Exception as exc:  # transformers/OOM/network failures land here
        print(f"ERROR: failed to load the model: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_MODEL

    # -- resume ------------------------------------------------------------ #
    writer = SubmissionWriter(output)
    skip_ids: set[str] = set()
    expected_ids = [str(sample.id) for sample in samples]
    if args.resume and output.exists():
        from lifemate.evaluation.submission import load_submission

        try:
            for idx, row in enumerate(load_submission(output)):
                row_id = str(row["id"])
                if idx < len(expected_ids) and row_id != expected_ids[idx]:
                    raise SubmissionError(
                        f"row {idx} has id {row_id!r}, expected dataset id {expected_ids[idx]!r}"
                    )
                writer.add(row_id, row["tool_calls"], row["state_hash"])
                skip_ids.add(row_id)
        except (SubmissionError, LifeMateError) as exc:
            print(
                f"ERROR: --resume cannot reuse {output}: {exc}. Delete it or drop --resume.",
                file=sys.stderr,
            )
            return EXIT_SUBMISSION
        if skip_ids:
            print(f"resume: {len(skip_ids)} row(s) already present in {output}", file=sys.stderr)
    started = time.monotonic()
    failures = 0
    asr_failures = 0
    non_empty_transcripts = 0
    asr_errors: list[str] = []
    backend_name = (args.asr or cfg.asr.backend or "none").lower()
    asr_checked = backend_name not in {"none", "oracle"}

    for sample in samples:
        sample_id = str(sample.id)
        if sample_id in skip_ids:
            continue
        current_id["value"] = sample_id
        deadline = time.monotonic() + float(cfg.agent.time_budget_seconds)

        from lifemate.data.history import transcribe_missing_history
        history_errors = transcribe_missing_history(
            sample, split=split, data_root=data_root, deadline=deadline,
            transcribe=lambda item: _transcribe(asr, item, features=features),
        )
        history_incomplete = bool(history_errors) and not getattr(args, "allow_missing_history", False)
        if history_errors:
            failures += 1
            print(f"WARNING: {sample_id}: incomplete history: {'; '.join(history_errors)}", file=sys.stderr)
        if time.monotonic() >= deadline:
            transcript, asr_info = "", {"backend": backend_name, "error": "time budget exhausted before current-turn ASR"}
        else:
            transcript, asr_info = _transcribe(asr, sample, features=features)
        asr_error = asr_info.get("error")
        if asr_error or (asr_checked and not transcript):
            asr_failures += 1
            failures += 1
            reason = str(asr_error or "empty transcript")
            asr_errors.append(f"{sample_id}: {reason}")
            print(f"WARNING: {sample_id}: asr failed: {reason[:160]}", file=sys.stderr)
        if transcript:
            non_empty_transcripts += 1
        try:
            mock_state = _mock_state_for(sample_id, paths.root) if mock else None
            env = create_environment(
                split,
                sample_id,
                data_root,
                use_mock=mock,
                catalog=catalog,
                mock_state=mock_state,
                apps=list(getattr(sample.context, "installed_apps", []) or []) if mock else None,
            )
        except LifeMateError as exc:
            print(f"ERROR: cannot create the environment for {sample_id}: {exc}", file=sys.stderr)
            print(
                "The official simulator (zaic_tools) is required for competition inference. "
                "Use --use-mock-env for a synthetic self-test.",
                file=sys.stderr,
            )
            return EXIT_DATASET

        registry = ToolRegistry(catalog).for_sample(sample) if catalog is not None else ToolRegistry(_EmptyCatalog())
        # Constraints are per sample: the generator is built once, the tool set is
        # not. Only touched when the flag is on, so the off path is untouched.
        if _flag(features, "decoding", "constrained_decoding_enabled"):
            setter = getattr(generator, "set_allowed_tools", None)
            if callable(setter):
                setter(registry.tool_names())
            else:  # pragma: no cover - generator without the hook
                print(
                    "constrained decoding: PENDING (generator has no set_allowed_tools hook)",
                    file=sys.stderr,
                )
        agent = ToolAgent(
            generator=generator,
            registry=registry,
            env=env,
            config=cfg,
            guardrail=_flag(features, "postprocessing", "enhanced_schema_guardrail_enabled"),
        )
        try:
            asr_hyp = asr_info.get("hypotheses")
            # Do not guess actions when a required history turn is unavailable.
            result = None if history_incomplete else agent.run(
                sample, transcript=transcript, deadline=deadline, asr_hypotheses=asr_hyp)
        except ProhibitedAccessError:
            raise  # compliance violation: never downgrade to a warning + empty row
        except Exception as exc:  # keep the run alive; record the failure
            failures += 1
            print(f"WARNING: {sample_id}: agent failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            result = None

        # Rows come from the ENV, never from the agent's own bookkeeping, so a
        # crashed run still reports whatever actually reached the simulator.
        tool_calls = env.get_tool_calls()
        state_hash = env.get_state_hash()
        if result is not None and getattr(result, "status", "") not in {"finished", "ok"}:
            failures += 1

        try:
            writer.add(sample_id, tool_calls, state_hash)
            assert_env_sourced(writer.rows[-1], env)
        except (SubmissionError, LifeMateError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_SUBMISSION
        if args.debug_dump:
            _dump_debug(
                args.debug_dump,
                sample_id,
                {
                    "id": sample_id,
                    "split": split,
                    "asr": asr_info,
                    "transcript": transcript,
                    "raw_transcript": asr_info.get("raw_transcript", ""),
                    "history_errors": history_errors,
                    "status": "history_incomplete" if history_incomplete else (None if result is None else result.status),
                    "finish_reason": None if result is None else result.finish_reason,
                    "tool_calls": tool_calls,
                    "state_hash": state_hash,
                    "n_llm_calls": None if result is None else result.n_llm_calls,
                    "elapsed_seconds": None if result is None else round(result.elapsed_seconds, 4),
                    "errors": [] if result is None else list(result.errors),
                },
            )
        if not args.quiet:
            status = "error" if result is None else result.status
            print(f"[{len(writer)}/{len(samples)}] {sample_id}: {status} ({len(tool_calls)} calls)")

    # -- write + validate -------------------------------------------------- #
    # A backend that never produced text would otherwise "succeed" with an empty
    # submission. Refuse BEFORE finalize so no output file is created.
    if asr_checked and asr_failures > 0 and non_empty_transcripts == 0:
        message = (
            f"ASR produced no usable transcript for any of {len(samples)} sample(s) "
            f"with backend {backend_name!r}; refusing to write a submission."
        )
        if args.allow_empty_transcripts:
            print(f"WARNING: {message} --allow-empty-transcripts is set: submission is diagnostic only.",
                  file=sys.stderr)
        else:
            print(f"ERROR: {message} First errors: {' | '.join(asr_errors[:3])}", file=sys.stderr)
            return EXIT_MODEL
    if asr_failures > max(1, math.ceil(ASR_FAILURE_RATIO * len(samples))):
        print(
            f"ERROR: ASR failed for {asr_failures} of {len(samples)} sample(s) "
            f"(> {ASR_FAILURE_RATIO:.0%}) with backend {backend_name!r}; refusing to write a submission. "
            f"First errors: {' | '.join(asr_errors[:3])}",
            file=sys.stderr,
        )
        return EXIT_MODEL

    try:
        writer.finalize()
    except SubmissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_SUBMISSION

    report = validate_submission(
        output, expected_ids, require_state_hash=True, calls_source="env", require_order=True
    )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if not report["ok"]:
        for error in report["errors"][:20]:
            print(f"ERROR: submission validation: {error}", file=sys.stderr)
        return EXIT_SUBMISSION

    elapsed = time.monotonic() - started
    _write_run_provenance(output, cfg, feature_report, backend_name, report["n"])
    print(
        f"wrote {report['n']} row(s) to {output} in {elapsed:.1f}s "
        f"({failures} sample(s) with failures, {asr_failures} ASR failure(s))"
    )
    if mock:
        print(MOCK_BANNER, file=sys.stderr)
    return EXIT_OK

def _write_run_provenance(
    output: Path,
    cfg: Any,
    feature_report: Mapping[str, Any],
    backend_name: str,
    n_rows: int,
) -> None:
    """Sidecar ``<output>.run.json``: which features produced these rows.

    Written only for an EXPERIMENTAL run (at least one flag on), so a baseline
    run leaves the directory exactly as before. No transcript, audio or user
    content — ids, flags and fingerprints only.
    ponytail: sidecar non-experimental too when a run report exists; upgrade path
    is to fold this into that report instead of a second file.
    """
    if not feature_report or feature_report.get("mode") == "BASELINE":
        return
    payload = {
        "output": str(output),
        "rows": int(n_rows),
        "split": str(getattr(cfg.inference, "split", "")),
        "asr_backend": backend_name,
        "features": feature_report,
    }
    sidecar = output.with_name(output.name + ".run.json")
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"feature provenance: {sidecar}", file=sys.stderr)


class _EmptyCatalog:
    """Catalog stand-in for splits without a tool catalog (registry stays empty)."""

    apps: list[str] = []
    tools: dict[str, Any] = {}

    def tools_for_apps(self, apps: Any, include_meta: bool = True) -> list[Any]:
        return []

    def to_schema_prompt(self, apps: Any = None, include_meta: bool = True) -> str:
        return "(no tool catalog found for this split)"

    def validate_call(self, name: str, args: Any = None) -> dict[str, Any]:
        return {"errors": [], "warnings": []}

    def __contains__(self, name: object) -> bool:
        return False

    def __len__(self) -> int:
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``python -m lifemate infer``."""
    try:
        return run_inference(argv)
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_DATASET
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_DATASET


if __name__ == "__main__":
    sys.exit(main())

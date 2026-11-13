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
    return _apply_overrides(cfg, args.override)


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
        transcripts = _load_transcripts(args.transcripts)
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
        return create_asr_backend(cfg, transcripts=transcripts, backend="oracle")

    if backend == "whisper":
        return create_asr_backend(cfg, backend="whisper")

    if backend == "gipformer":
        return create_asr_backend(cfg, backend="gipformer")

    if backend == "none":
        return None

    # Unknown backend names must fail loudly; the factory raises AsrError and
    # lists the valid choices, so a typo can never yield an empty submission.
    return create_asr_backend(cfg, backend=backend)

def _transcribe(asr: Any, sample: Any) -> tuple[str, dict[str, Any]]:
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
    try:
        result = asr.transcribe(audio_path, sample_id=str(getattr(sample, "id", "")))
    except ProhibitedAccessError:
        raise                                    # a backend that reached a banned path must stop the run
    except Exception as exc:
        return "", {"backend": getattr(asr, "name", "unknown"), "error": f"{type(exc).__name__}: {exc}"[:300]}
    from lifemate.asr.normalization import to_llm_text

    raw_text = str(getattr(result, "text", "") or "")
    text = to_llm_text(raw_text) if raw_text else raw_text
    info = {
        "backend": getattr(result, "backend", getattr(asr, "name", "unknown")),
        "error": getattr(result, "error", None),
        "latency_s": round(float(getattr(result, "latency_s", 0.0) or 0.0), 3),
        "raw_transcript": raw_text,
    }
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
    return HFGenerator(
        bundle.model,
        bundle.tokenizer,
        max_new_tokens=cfg.agent.max_new_tokens,
        temperature=cfg.agent.temperature,
        device=getattr(bundle, "device", None),
    )


# --------------------------------------------------------------------------- #
# Debug dumps
# --------------------------------------------------------------------------- #
#: Keys ``--debug-dump`` records on purpose (the ASR-vs-agent comparison needs them).
_DEBUG_KEEP = frozenset({"transcript", "raw_transcript"})


def _sanitize(value: Any) -> Any:
    """Drop user-content keys from debug dumps (no profile/message bodies)."""
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v) for k, v in value.items() if str(k) not in _PII_KEYS}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


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
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_DATASET

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
    # Blank history is silent prompt degradation: the model sees an empty prior
    # turn instead of the user's words and nothing in the submission says so.
    # Refuse before any weights are loaded; a diagnostic run opts out explicitly.
    if blank and not getattr(args, "allow_missing_history", False):
        print(
            f"ERROR: {reason}; split={split} would render {blank} history user turn(s) blank. "
            "Pass --history-transcripts FILE, or --allow-missing-history for a diagnostic run.",
            file=sys.stderr,
        )
        return EXIT_DATASET

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

        transcript, asr_info = _transcribe(asr, sample)
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
        agent = ToolAgent(generator=generator, registry=registry, env=env, config=cfg)
        deadline = time.monotonic() + float(cfg.agent.time_budget_seconds)
        try:
            result = agent.run(sample, transcript=transcript, deadline=deadline)
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
                    "status": None if result is None else result.status,
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
    print(
        f"wrote {report['n']} row(s) to {output} in {elapsed:.1f}s "
        f"({failures} sample(s) with failures, {asr_failures} ASR failure(s))"
    )
    if mock:
        print(MOCK_BANNER, file=sys.stderr)
    return EXIT_OK


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

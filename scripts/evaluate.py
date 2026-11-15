#!/usr/bin/env python3
"""Evaluate a checkpoint: teacher-forced decisions and/or closed-loop episodes.

Usage::

    # teacher-forced over the SFT validation split (needs worker D's export)
    python scripts/evaluate.py --validation-only --checkpoint outputs/checkpoints/full \\
        --model-mode full --predictions outputs/datasets/train/val.jsonl

    # closed loop against the official simulator (needs zaic_tools)
    python scripts/evaluate.py --closed-loop --split train --limit 20

    # pipeline self-test only (never a competition number)
    python scripts/evaluate.py --closed-loop --use-mock-env --split train --limit 5

    # append an ASR block (WER/CER, critical tokens) to the same artifacts
    python scripts/evaluate.py --validation-only --asr-predictions outputs/asr/train_predictions.jsonl

Exit codes: 0 ok, 2 data/environment missing, 3 model load failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.config import load_config  # noqa: E402
from lifemate.errors import DatasetNotFoundError, LifeMateError  # noqa: E402
from lifemate.evaluation.evaluator import (  # noqa: E402
    evaluate_closed_loop,
    evaluate_split,
    evaluate_teacher_forced,
    load_decision_records,
)

DEFAULT_VAL_RECORDS = "outputs/datasets/train/val.jsonl"
MOCK_NOTE = "MOCK ENV / SYNTHETIC"

EXIT_OK = 0
EXIT_DATA = 2
EXIT_MODEL = 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Evaluate a LifeMate checkpoint offline.")
    parser.add_argument("--split", default="train", help="split to evaluate (closed loop)")
    parser.add_argument("--data-root", default=None, help="dataset root (default: config data.root)")
    parser.add_argument("--validation-only", action="store_true", help="teacher-forced scoring only")
    parser.add_argument("--closed-loop", action="store_true", help="run full episodes in an environment")
    parser.add_argument("--checkpoint", default=None, help="checkpoint directory")
    parser.add_argument("--heldout-dataset", default=None, help="prepared train dataset directory used to select verified validation source IDs")
    parser.add_argument("--allow-training-samples", action="store_true", help="explicitly permit non-held-out train diagnostics")
    parser.add_argument("--model-mode", choices=("base", "full", "lora"), default=None)
    parser.add_argument("--config", default=None, help="YAML config file")
    try:  # optional slice: absent -> no feature flags at all
        from lifemate.features.overrides import add_feature_arguments  # noqa: PLC0415

        add_feature_arguments(parser)
    except ImportError:  # pragma: no cover - features slice absent
        pass
    parser.add_argument("--limit", type=int, default=None, help="max samples/records")
    parser.add_argument("--output", default=None, help="report directory (default: config evaluation.output_dir)")
    parser.add_argument("--predictions", default=None, help=f"decision-record JSONL (default: {DEFAULT_VAL_RECORDS})")
    parser.add_argument(
        "--asr-predictions",
        default=None,
        metavar="FILE",
        help="ASR prediction JSONL {id, reference, hypothesis, latency_s}; appends an `asr` block",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="generation cap for teacher forcing")
    parser.add_argument("--use-mock-env", action="store_true", help="MockToolEnv self-test (marks every artifact)")
    parser.add_argument(
        "--generator",
        default=None,
        metavar="dumpscript:PATH",
        help="TEST HOOK: JSON file mapping sample_id -> raw model outputs (requires --use-mock-env)",
    )
    parser.add_argument(
        "--allow-scripted-official",
        action="store_true",
        help="PIPELINE TEST ONLY: permit --generator against the REAL env. Rows are a scripted "
        "artefact, never a model prediction.",
    )
    parser.add_argument("--device", default=None, help="torch device")
    parser.add_argument("--model", default=None, help="override model.name (base model id or path)")
    parser.add_argument("--quiet", action="store_true", help="suppress per-sample progress lines")
    parser.add_argument(
        "--evaluation-mode",
        "--eval-mode",
        choices=("text_to_action", "audio_to_action"),
        default="text_to_action",
        help="evaluation mode: text_to_action (gold/fixed transcript) or audio_to_action (full audio->ASR->agent)",
    )
    parser.add_argument("--replay-multi-asr", help="verified train multi-ASR JSONL artifact; excludes ASR decode latency")
    parser.add_argument("--replay-k", type=int, choices=(1,2,3,4), default=3)
    parser.add_argument("--transcript-cache", default=None, help="transcript cache JSON file")
    parser.add_argument('--force-asr-history', action='store_true', help='remove train user-history gold text and transcribe its supplied audio')
    parser.add_argument("--end-to-end-runtime", action="store_true", help="run uncached current-turn ASR within each sample deadline")
    parser.add_argument(
        "--asr",
        choices=("gipformer", "oracle", "whisper", "none"),
        default=None,
        help="ASR backend for audio_to_action mode",
    )
    parser.add_argument("--allow-oracle", action="store_true", help="permit oracle ASR on labeled split")
    return parser.parse_args(argv)


def _feature_config(cfg: Any, args: argparse.Namespace) -> Any:
    """Resolve the CLI/env/YAML feature chain onto ``cfg`` (``None`` if absent)."""
    try:
        from lifemate.features.config import FeatureConfig  # noqa: PLC0415
        from lifemate.features.overrides import build_feature_config  # noqa: PLC0415
    except ImportError:  # pragma: no cover - features slice absent
        return None
    if getattr(cfg, "features", None) is None:  # wiring not landed: defaults
        cfg.features = FeatureConfig()
    features, _provenance = build_feature_config(cfg, args)
    cfg.features = features
    return features

def _flag(features: Any, section: str, name: str) -> bool:
    """One feature flag; ``False`` when the object (or its section) is absent."""
    return bool(getattr(getattr(features, section, None), name, False))

def _print_feature_report(cfg: Any) -> dict[str, Any]:
    """Print the effective configuration once; return its JSON half."""
    try:  # noqa: PLC0415
        from lifemate.features.report import (  # noqa: PLC0415
            feature_report_dict,
            format_feature_report,
        )
    except ImportError:  # pragma: no cover - features slice absent
        return {}
    features = getattr(cfg, "features", None)
    if features is None:  # pragma: no cover - slice present but off
        return {}
    print(
        format_feature_report(features, model_id=cfg.model.name, asr_id=cfg.asr.model_id),
        file=sys.stderr,
    )
    return feature_report_dict(features, model_id=cfg.model.name, asr_id=cfg.asr.model_id)

def _build_config(args: argparse.Namespace) -> Any:
    cfg = load_config(args.config)
    if args.checkpoint:
        cfg.inference.checkpoint = args.checkpoint
    if args.model_mode:
        cfg.inference.model_mode = args.model_mode
    if args.limit is not None:
        cfg.inference.limit = args.limit
    if args.output:
        cfg.evaluation.output_dir = args.output
    if args.data_root:
        cfg.data.root = args.data_root
    if args.device:
        cfg.model.device = args.device
    if getattr(args, "model", None):
        cfg.model.name = args.model
    if args.use_mock_env:
        cfg.inference.use_mock_env = True
    cfg.inference.split = args.split
    if getattr(args, "asr", None):
        cfg.asr.backend = args.asr
    _feature_config(cfg, args)
    return cfg


def _infer_module() -> Any:
    """Import ``scripts/infer.py`` by path (scripts/ is not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("lifemate_infer_script", Path(__file__).with_name("infer.py"))
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise LifeMateError("cannot load scripts/infer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scripted_generator(path: Path) -> Any:
    """Load ``ScriptedGenerator`` from scripts/infer.py (test hook only)."""
    module = _infer_module()
    return module.ScriptedGenerator(module.load_dump_script(path))


def _generator(cfg: Any, args: argparse.Namespace | None = None, sample_id: list[str] | None = None) -> Any:
    """Load the model generator (torch imported only here)."""
    scripted = getattr(args, "generator", None)
    if scripted:
        if not str(scripted).startswith("dumpscript:"):
            raise LifeMateError(f"--generator only supports 'dumpscript:PATH' (got {scripted!r})")
        if not getattr(args, "use_mock_env", False) and not getattr(args, "allow_scripted_official", False):
            raise LifeMateError(
                "--generator is a TEST HOOK and requires --use-mock-env (or the explicit "
                "pipeline-test flag --allow-scripted-official); never for a real score"
            )
        holder = sample_id if sample_id is not None else [""]
        generator = _scripted_generator(Path(str(scripted).split(":", 1)[1]))
        generator.current_id = lambda: holder[0]
        generator.set_sample = lambda sid: holder.__setitem__(0, str(sid))
        return generator
    mode = cfg.inference.model_mode
    checkpoint = cfg.inference.checkpoint
    if mode in {"full", "lora"} and not checkpoint:
        raise LifeMateError(f"--model-mode {mode} requires --checkpoint")
    if mode == "lora" and (not Path(str(checkpoint)).is_dir() or not (Path(str(checkpoint)) / "adapter_config.json").is_file()):
        raise LifeMateError(f"LoRA adapter checkpoint missing or invalid: {checkpoint}")
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
        temperature=0.0,
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
        temperature=0.0,
        device=getattr(bundle, "device", None),
        **_constrained_kwargs(cfg, HFGenerator),
    )

def _constrained_kwargs(cfg: Any, generator_cls: Any) -> dict[str, Any]:
    """``constrained=True`` when the flag is on AND the generator has the hook.

    Feature-detected: without the ``constrained`` parameter the flag cannot be
    honoured, so the run continues UNCONSTRAINED and says so.
    """
    if not _flag(getattr(cfg, "features", None), "decoding", "constrained_decoding_enabled"):
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
    return {"constrained": True, "allowed_tools": ()}


def _teacher_forced(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    records_path = Path(args.predictions or DEFAULT_VAL_RECORDS)
    if not records_path.exists():
        print(
            f"ERROR: decision records not found: {records_path}\n"
            "Build them first with scripts/prepare_dataset.py (worker D writes "
            f"{DEFAULT_VAL_RECORDS}), or pass --predictions PATH.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_DATA)
    records = load_decision_records(records_path, limit=args.limit)
    if not records:
        print(f"ERROR: {records_path} contains no usable decision records", file=sys.stderr)
        raise SystemExit(EXIT_DATA)
    try:
        generator = _generator(cfg, args)  # scripted double allowed only with --use-mock-env
    except ImportError as exc:
        print(f"ERROR: torch/transformers missing: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_MODEL)
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_MODEL)
    payload = evaluate_teacher_forced(
        generator,
        records,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    if not payload.get("n_records"):
        print(
            f"ERROR: {records_path} has no gradeable record: {payload.get('n_ungradeable', 0)} "
            "row(s) lack a prompt or a gold target (a decision record needs `messages`/`prompt` "
            "plus `target` or `gold_tool_calls`; a submission/predictions file cannot be scored).",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_DATA)
    return payload


def _feature_execution_counts(samples: Any, features: Any, info: Mapping[str, Any]) -> dict[str, Any]:
    execution = {}
    enabled_features = (
        ("asr.nbest", _flag(features, "asr", "nbest_enabled")),
        ("asr.multi_asr", _flag(features, "asr", "multi_asr_enabled")),
        ("asr.gipformer_v15", _flag(features, "asr", "gipformer_v15_enabled")),
        ("audio.vad", _flag(features, "audio", "vad_enabled")),
        ("audio.rms_normalization", _flag(features, "audio", "rms_normalization_enabled")),
        ("prompt.itn", _flag(features, "prompt", "itn_enabled")),
    )
    for feature, enabled in enabled_features:
        if not enabled:
            continue
        counts = {"n_requested": len(samples), "n_executed": 0, "n_degraded": 0,
                  "n_blocked": 0, "n_not_observed": 0}
        for sample in samples:
            item = info.get(str(sample.id))
            if item is None:
                counts["n_not_observed"] += 1
                continue
            if item.get("error"):
                counts["n_blocked"] += 1
                continue
            conditioning = item.get("conditioning") or {}
            if feature == "asr.nbest":
                observed = "nbest" in conditioning or bool(item.get("nbest_fallback"))
                success = not item.get("nbest_fallback") and conditioning.get("nbest", {}).get("returned", 0) > 1
            elif feature == "asr.multi_asr":
                observed = "multi_asr" in conditioning
                success = conditioning.get("multi_asr", {}).get("executed", False) and not conditioning.get("multi_asr", {}).get("degraded", False)
            elif feature == "asr.gipformer_v15":
                observed = bool(item.get("model_id"))
                success = item.get("model_id") == "g-group-ai-lab/gipformer1.5-68M-rnnt"
            elif feature == "prompt.itn":
                observed = "itn_applied" in item
                success = bool(item.get("itn_applied"))
            else:
                key = "vad" if feature == "audio.vad" else "rms"
                observed = key in conditioning
                success = observed
            counts["n_executed" if success else "n_degraded" if observed else "n_not_observed"] += 1
        execution[feature] = counts
    return execution


def _closed_loop(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    """Run full episodes. Fails loudly when the official env cannot be built."""
    features = getattr(cfg, "features", None)
    from lifemate.agent.agent import ToolAgent
    from lifemate.agent.tool_registry import ToolRegistry
    from lifemate.data.discovery import require_split
    from lifemate.data.loader import load_catalog, load_samples
    from lifemate.env.factory import create_environment

    data_root = Path(cfg.data.root)
    try:
        paths = require_split(data_root, args.split)
        heldout = getattr(args, "heldout_dataset", None)
        if args.split == "train" and not heldout and not getattr(args, "use_mock_env", False) and not getattr(args, "allow_training_samples", False) and not getattr(args, "allow_scripted_official", False):
            raise LifeMateError("train closed-loop requires --heldout-dataset; use --allow-training-samples for explicitly non-held-out diagnostics")
        samples = load_samples(data_root, args.split, limit=None if heldout else args.limit, with_labels=True)
        if heldout:
            from lifemate.evaluation.heldout import select_heldout
            samples, heldout_info = select_heldout(samples, heldout, split=args.split, limit=args.limit)
        else:
            heldout_info = None
        catalog = load_catalog(data_root, args.split)
    except (DatasetNotFoundError, LifeMateError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(EXIT_DATA)
    if not samples:
        print(f"ERROR: split {args.split!r} has no usable samples", file=sys.stderr)
        raise SystemExit(EXIT_DATA)

    mock = bool(args.use_mock_env)
    try:
        # Verify the SDK early, but do NOT build a session: ToolEnv may be created
        # exactly once per sample and the factory below builds the real one.
        if mock:
            create_environment(args.split, str(samples[0].id), data_root, use_mock=True, catalog=catalog)
        else:
            from lifemate.env.official import check_official_sdk

            check_official_sdk(args.split, data_root)
    except LifeMateError as exc:
        print(f"ERROR: cannot build the environment for evaluation: {exc}", file=sys.stderr)
        print(
            "Real evaluation needs the official simulator (zaic_tools). "
            "Use --use-mock-env only for a pipeline self-test.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_DATA)
    current_id = [""]
    try:
        generator = _generator(cfg, args, current_id)
    except ImportError as exc:
        print(f"ERROR: torch/transformers missing: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_MODEL)
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_MODEL)

    def factory(sample: Any) -> Any:
        current_id[0] = str(sample.id)
        mock_state = None
        if mock:
            mock_state = _infer_module()._mock_state_for(str(sample.id), paths.root)
        env = create_environment(
            args.split,
            str(sample.id),
            data_root,
            use_mock=mock,
            catalog=catalog,
            mock_state=mock_state,
            apps=list(getattr(sample.context, "installed_apps", []) or []) if mock else None,
        )
        registry = ToolRegistry(catalog).for_sample(sample) if catalog is not None else ToolRegistry(catalog)
        if _flag(features, "decoding", "constrained_decoding_enabled"):
            setter = getattr(generator, "set_allowed_tools", None)
            if callable(setter):
                setter(registry.tool_names())
        return ToolAgent(
            generator=generator,
            registry=registry,
            env=env,
            config=cfg,
            guardrail=_flag(features, "postprocessing", "enhanced_schema_guardrail_enabled"),
        )

    eval_mode = getattr(args, "evaluation_mode", "text_to_action")
    if getattr(args, 'force_asr_history', False) and eval_mode != 'audio_to_action':
        raise LifeMateError('--force-asr-history requires audio_to_action evaluation')
    transcripts = None
    cache_stats = {}
    prepare_history = None
    transcribe_sample = None
    history_path = getattr(cfg.inference, "history_transcripts", None)
    if history_path:
        from lifemate.data.history import apply_history_texts, load_history_texts
        apply_history_texts(samples, load_history_texts(history_path, data_root=data_root), args.split)
    replay_path = getattr(args, 'replay_multi_asr', None)
    if replay_path:
        if eval_mode != 'audio_to_action' or getattr(args, 'end_to_end_runtime', False):
            raise LifeMateError('artifact replay requires audio_to_action quality evaluation; decoding latency is excluded')
        if not _flag(features, 'asr', 'conditioned_multi_sft_enabled') or not _flag(features, 'asr', 'multi_asr_enabled'):
            raise LifeMateError('artifact replay requires conditioned and multi-ASR prompt configuration')
        from lifemate.evaluation.artifact_replay import ArtifactReplay
        replay = ArtifactReplay(replay_path, samples, split=args.split, data_root=data_root,
                                k=args.replay_k, itn=_flag(features, 'prompt', 'itn_enabled'))
        # The prompt renderer reads this setting from the feature configuration.
        features.asr.multi_asr_max_hypotheses = args.replay_k
        prepare_history = replay.history
        transcripts, cache_stats['asr_info'] = {}, {}
        for sample in samples:
            transcripts[str(sample.id)], cache_stats['asr_info'][str(sample.id)] = replay.current(sample)
        cache_stats['replay_identity'] = replay.manifest['run_identity']
    elif eval_mode == "audio_to_action":
        from lifemate.asr.transcription import transcribe_samples_cached

        infer = _infer_module()
        asr_backend = infer._build_asr(args, cfg, samples)
        from lifemate.data.history import transcribe_missing_history, clear_user_history_transcripts
        def prepare_history(sample, deadline):
            if getattr(args, 'force_asr_history', False):
                clear_user_history_transcripts(sample)
            errors = transcribe_missing_history(sample, split=args.split, data_root=data_root, deadline=deadline,
                transcribe=lambda item: infer._transcribe(asr_backend, item, features=features))
            if errors:
                raise LifeMateError("; ".join(errors))
        if getattr(args, "end_to_end_runtime", False):
            cache_stats["asr_info"] = {}
            def transcribe_sample(sample):
                text, info = infer._transcribe(asr_backend, sample, features=features)
                cache_stats["asr_info"][str(sample.id)] = info
                return text, info
        else:
            cache_path = getattr(args, "transcript_cache", None) or (Path(cfg.evaluation.output_dir) / "transcripts_cache.json")
            transcripts, cache_stats = transcribe_samples_cached(
                samples,
                cfg,
                features=features,
                asr=asr_backend,
                split=args.split,
                cache_path=cache_path,
                args=args,
            )
            if not getattr(args, "quiet", False):
                print(
                    f"audio_to_action: {len(transcripts)} transcripts ready "
                    f"(cache hits={cache_stats.get('cache_hit')}, misses={cache_stats.get('cache_miss')})"
                )

    summary = evaluate_closed_loop(
        factory,
        samples,
        split=args.split,
        data_root=data_root,
        max_samples=args.limit,
        deadline_seconds=float(cfg.agent.time_budget_seconds),
        split_paths=paths,
        evaluation_mode=eval_mode,
        transcripts=transcripts,
        asr_info=cache_stats.get("asr_info"),
        prepare_history=prepare_history,
        transcribe_sample=transcribe_sample,
    )
    if heldout_info is not None:
        summary["heldout_selection"] = heldout_info
    summary['replay_identity'] = cache_stats.get('replay_identity')
    if replay_path:
        summary['latency_accounting'] = {'mode':'artifact_replay_quality', 'asr_decode_included':False, 'history_decode_included':False, 'note':'historical decode latency is not current execution latency; unsuitable for runtime compliance'}
    summary['history_protocol'] = 'forced_audio_asr' if replay_path else ('forced_audio_asr' if getattr(args, 'force_asr_history', False) else 'existing_or_missing_audio_asr')
    if eval_mode == "audio_to_action":
        info = cache_stats.get("asr_info", {})
        execution = _feature_execution_counts(samples, features, info)
        summary["feature_execution"] = execution
        summary["degraded_features"] = [name for name, counts in execution.items() if counts["n_degraded"] or counts["n_blocked"]]
    if mock:
        summary["mock_env"] = True
        summary["note"] = f"{MOCK_NOTE}: MockToolEnv, not the official simulator"
    return summary


def _eda_module() -> Any:
    """Import ``scripts/run_eda.py`` by path, reusing its prediction loader."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("lifemate_eda_script", Path(__file__).with_name("run_eda.py"))
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise LifeMateError("cannot load scripts/run_eda.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _asr_block(path: str) -> dict[str, Any]:
    """Build the `asr` block from a prediction JSONL (never zero-filled)."""
    from lifemate.asr.metrics import critical_token_diagnostics, evaluate_asr

    predictions_path = Path(path).expanduser()
    if not predictions_path.is_file():
        return {
            "available": False,
            "reason": f"ASR predictions file not found: {predictions_path}",
            "backend": None,
            "n": 0,
        }
    loaded = _eda_module()._load_asr_predictions(predictions_path)
    for message in loaded["skipped"]:
        print(f"warning: skipped malformed ASR prediction row ({message})", file=sys.stderr)
    if not loaded["pairs"]:
        return {
            "available": False,
            "reason": f"no usable rows in {predictions_path}",
            "backend": loaded["backend"],
            "n": 0,
        }
    print(
        f"warning: ASR references were read from {predictions_path}; "
        "file-supplied predictions are not a benchmark result.",
        file=sys.stderr,
    )
    block = dict(evaluate_asr(loaded["pairs"], loaded["backend"] or "file-supplied", latencies=loaded["latencies"]))
    block["critical_tokens"] = critical_token_diagnostics(loaded["pairs"])
    block["source"] = "file"
    block["predictions_file"] = str(predictions_path)
    block["n_skipped_rows"] = len(loaded["skipped"])
    return block

def _asr_section_md(asr: dict[str, Any]) -> str:
    """Metric-independent ASR section appended to ``metrics.md``."""
    lines = ["", "## ASR evaluation", ""]
    lines.append(
        "ASR WER/CER scores the transcript alone. It is independent of final-state correctness, "
        "which stays `available: false` here: a perfect transcript says nothing about the tool calls."
    )
    if not asr.get("available"):
        lines += ["", f"Unavailable: {asr.get('reason')}"]
        return "\n".join(lines) + "\n"
    norm = asr.get("normalized") or {}
    raw = asr.get("raw") or {}
    lines += [
        "",
        f"Backend: `{asr.get('backend')}`, n={asr.get('n')} (predictions supplied from a file, "
        "not a benchmark result).",
        "",
        "| metric | raw | normalized (`for_wer`) |",
        "| --- | --- | --- |",
        f"| WER (mean) | {_num(raw.get('wer_mean'))} | {_num(norm.get('wer_mean'))} |",
        f"| CER (mean) | {_num(raw.get('cer_mean'))} | {_num(norm.get('cer_mean'))} |",
        f"| sentence exact match | - | {_num(asr.get('sentence_exact_match'))} |",
    ]
    latency = asr.get("latency_s") or {}
    if latency.get("count"):
        lines += [
            "",
            f"Latency: p50={_num(latency.get('p50'))} s, p95={_num(latency.get('p95'))} s "
            f"(n={latency.get('count')}).",
        ]
    classes = (asr.get("critical_tokens") or {}).get("classes") or {}
    if classes:
        lines += ["", "| class | ref | hyp | exact set match | magnitude errors |", "| --- | --- | --- | --- | --- |"]
        for name, stats in classes.items():
            lines.append(
                f"| {name} | {stats.get('n_ref', 0)} | {stats.get('n_hyp', 0)} | "
                f"{_num(stats.get('exact_set_match_rate'))} | {stats.get('magnitude_errors', 0)} |"
            )
    return "\n".join(lines) + "\n"

def _num(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4f}"

def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        cfg = _build_config(args)
        # Printed once, before any weights load; the same dict goes into report.json.
        feature_report = _print_feature_report(cfg)
    except LifeMateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_DATA

    do_teacher = args.validation_only or (not args.closed_loop and getattr(args, "evaluation_mode", "text_to_action") != "audio_to_action")
    do_closed = args.closed_loop or not args.validation_only
    if not do_teacher and not do_closed:
        print("ERROR: nothing to evaluate (pass --validation-only or --closed-loop)", file=sys.stderr)
        return EXIT_DATA

    if args.use_mock_env:
        print(
            "======================================================================\n"
            f"{MOCK_NOTE} — every artifact from this run is marked MOCK ENV / SYNTHETIC.\n"
            "======================================================================",
            file=sys.stderr,
        )

    teacher_payload: dict[str, Any] | None = None
    closed_payload: dict[str, Any] | None = None
    predictions: list[dict[str, Any]] = []

    if do_teacher:
        try:
            teacher_payload = _teacher_forced(args, cfg)
        except SystemExit as exc:
            return int(exc.code or EXIT_DATA)
        predictions.extend(teacher_payload.get("predictions") or [])
        metrics = teacher_payload.get("metrics", {})
        print(
            f"teacher-forced: n={teacher_payload.get('n_records', 0)} "
            f"exact_match={_fmt(metrics.get('tool_call_exact_match'))} "
            f"f1={_fmt(metrics.get('sequence_f1'))}"
        )

    if do_closed:
        try:
            closed_payload = _closed_loop(args, cfg)
        except SystemExit as exc:
            return int(exc.code or EXIT_DATA)
        print(
            f"closed loop: n={closed_payload.get('n_run', 0)} "
            f"status={closed_payload.get('status_counts')}"
        )

    asr_payload = _asr_block(args.asr_predictions) if args.asr_predictions else None
    if asr_payload is not None:
        print(
            f"ASR: n={asr_payload.get('n', 0)} available={asr_payload.get('available')} "
            f"WER(norm)={_num((asr_payload.get('normalized') or {}).get('wer_mean'))}"
        )

    extra: dict[str, Any] = {"features": feature_report} if feature_report else {}
    if asr_payload is not None:
        extra["asr"] = asr_payload
    eval_mode = getattr(args, "evaluation_mode", "text_to_action")
    if do_teacher and not do_closed:
        eval_mode = "text_to_action"
    extra["evaluation_mode"] = eval_mode
    if closed_payload:
        extra["feature_execution"] = closed_payload.get("feature_execution", {})
        extra["degraded_features"] = closed_payload.get("degraded_features", [])
    payload = evaluate_split(
        cfg=cfg,
        split=args.split,
        output_dir=cfg.evaluation.output_dir,
        teacher_forced=teacher_payload,
        closed_loop=closed_payload,
        predictions=predictions or None,
        data_root=cfg.data.root,
        checkpoint=cfg.inference.checkpoint,
        model_mode=cfg.inference.model_mode,
        mock_env=bool(args.use_mock_env),
        extra=extra or None,
        evaluation_mode=eval_mode,
    )
    if asr_payload is not None:
        metrics_md = Path(cfg.evaluation.output_dir) / "metrics.md"
        metrics_md.write_text(
            metrics_md.read_text(encoding="utf-8") + _asr_section_md(asr_payload), encoding="utf-8"
        )
    print(f"wrote evaluation artifacts to {Path(cfg.evaluation.output_dir).resolve()}")
    for key, path in payload.get("artifacts", {}).items():
        print(f"  {key}: {path}")
    return EXIT_OK


def _fmt(block: Any) -> str:
    if isinstance(block, dict):
        if block.get("available") is False:
            return "UNAVAILABLE"
        value = block.get("value")
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return "UNAVAILABLE"


if __name__ == "__main__":
    sys.exit(main())

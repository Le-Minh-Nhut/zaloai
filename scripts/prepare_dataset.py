#!/usr/bin/env python3
"""Build causal SFT datasets (train/val JSONL of decision points).

Usage:
    python3 scripts/prepare_dataset.py --data-root ./data --split train
    python3 scripts/prepare_dataset.py --data-root ./data --split train \
        --tokenizer Qwen/Qwen3.5-4B --max-seq-length 8192
    python3 scripts/prepare_dataset.py --data-root tests/fixtures/synthetic_lifemate \
        --split train --output /tmp/sft_synth

What it writes (``<output>/<split>/``):

* ``train.jsonl`` / ``val.jsonl`` — compact SFT records
  ``{sample_id, step_index, messages, target, meta}``; prompts are already
  rendered, so training never rebuilds them (deterministic by construction);
* ``manifest.json`` — split fingerprint (counts, seed, ratio, id/group hashes);
* ``stats.json`` — decision-point accounting, drop reasons, token-size summary.

Optional training-data augmentations (all OFF by default; with every flag absent
the output is byte-identical to a build without them, and ``val.jsonl`` is never
touched):

* ``--noisy-asr-transcripts/--noisy-asr-ratio/--noisy-asr-enabled`` — add a
  variant per sampled record whose request section carries the recogniser's own
  best hypothesis instead of the gold transcript (target kept verbatim);
* ``--nbest-transcripts/--nbest-training-format`` — add a variant whose request
  section lists up to 3 hypotheses as a numbered list;
* ``--slot-masking-enabled`` — add a variant with one required gold argument
  removed from the request and ``sys.ask_missing_info`` as the target;
* ``--tool-dropout-enabled/--tool-dropout-max-tools`` — add a variant whose
  rendered catalog section omits up to N tools the gold trajectory never calls.

The CANONICAL switch is ``--enable-feature <id>`` / ``--disable-feature <id>``
(or a ``features:`` block in ``--config``), resolved by
``lifemate.features.overrides`` as CLI > env > YAML > defaults:

* ``training.noisy_asr`` — ``--noisy-asr-enabled`` is a documented alias;
* ``prompt.nbest_format`` — ``--nbest-training-format`` is an alias (needs
  ``asr.nbest``; the run refuses the combination otherwise, naming both ids);
* ``training.slot_masking`` — ``--slot-masking-enabled`` is an alias;
* ``training.tool_dropout`` — ``--tool-dropout-enabled`` is an alias, and
  ``--tool-dropout-max-tools`` / ``--noisy-asr-ratio`` / ``--augmentation-seed``
  override the matching ``features.training.*`` value.

Features are validated and reported BEFORE any data is read, so an impossible
configuration fails fast and the effective set is always printed. When at least
one augmentation is active the manifest gains ``augmentation`` (flags, numbers,
seed, transcript paths + sha256, variant counts, per-rule skip counts) plus
``features`` (:func:`lifemate.features.report.feature_report_dict`) and
``training_feature_fingerprint``, so ``scripts/train.py`` can compare the run it
launched with the distribution this dataset actually has.

Exit codes: 0 ok, 2 dataset not found (or augmentation input unusable), 3 optional
tokenizer requested but unavailable (records are still written with char/4 token
estimates).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.config import load_config  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples  # noqa: E402
from lifemate.data.split_builder import split_samples  # noqa: E402
from lifemate.data.trajectory_builder import (  # noqa: E402
    build_sft_dataset,
    to_hf_dataset_records,
)
from lifemate.errors import DatasetNotFoundError, LifeMateError  # noqa: E402
from lifemate.features.config import training_feature_fingerprint  # noqa: E402
from lifemate.features.report import feature_report_dict  # noqa: E402

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Build causal SFT decision points from a LifeMate split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="./data", help="dataset root directory")
    parser.add_argument("--split", default="train", help="split to convert")
    parser.add_argument("--output", default="outputs/datasets", help="output root directory")
    parser.add_argument("--config", default=None, help="optional YAML config for defaults")
    parser.add_argument('--conditioned-multi-records', help='versioned train multi-ASR artifact')
    parser.add_argument('--conditioned-ratio', type=float, default=1.0)
    parser.add_argument('--conditioned-unknown', choices=('exclude','include'), default='exclude')
    parser.add_argument('--conditioned-mode', choices=('single','multi','mixed'), default='multi')
    parser.add_argument('--conditioned-single-ratio', type=float, default=1.0)
    parser.add_argument('--conditioned-multi-ratio', type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=None, help="validation fraction")
    parser.add_argument("--seed", type=int, default=None, help="split seed")
    parser.add_argument(
        "--tokenizer",
        default="none",
        help="HF tokenizer id for real token counts; 'none' => char/4 estimate only",
    )
    parser.add_argument("--limit", type=int, default=None, help="max samples to load")
    parser.add_argument("--max-seq-length", type=int, default=None, help="budget for the fit check")
    parser.add_argument("--history-limit", type=int, default=None, help="conversation turns kept in prompts")
    # -- optional training-data augmentations (all OFF by default) ----------- #
    # The CANONICAL switch is ``--enable-feature <id>`` (or a ``features:`` block
    # in --config). The per-feature flags below are documented aliases kept so
    # older command lines keep working; each maps 1:1 onto one feature id.
    parser.add_argument(
        "--nbest-transcripts",
        default=None,
        help="transcript JSONL from scripts/transcribe_split.py --transcript-records (N-best source)",
    )
    parser.add_argument(
        "--nbest-training-format",
        action="store_true",
        help=(
            "render the request section as a numbered hypothesis list "
            "(alias of --enable-feature prompt.nbest_format; also needs asr.nbest)"
        ),
    )
    parser.add_argument(
        "--noisy-asr-transcripts",
        default=None,
        help="transcript JSONL whose non-best hypothesis replaces the gold transcript in a variant",
    )
    parser.add_argument(
        "--noisy-asr-ratio",
        type=float,
        default=None,
        help=(
            "fraction of samples that get a noisy-ASR variant, in (0, 1] "
            "(alias of features.training.noisy_asr_ratio; overrides the config value)"
        ),
    )
    parser.add_argument(
        "--noisy-asr-enabled",
        action="store_true",
        help="enable noisy-ASR variants (alias of --enable-feature training.noisy_asr)",
    )
    parser.add_argument(
        "--noisy-asr-mode",
        choices=("augment", "replace"),
        default=None,
        help=(
            "'augment' keeps every clean record and appends a noisy variant (noisy share "
            "= ratio/(1+ratio)); 'replace' swaps the sampled record for its noisy variant "
            "(record count unchanged, noisy share = ratio). "
            "Alias of features.training.noisy_asr_mode"
        ),
    )
    parser.add_argument(
        "--slot-masking-enabled",
        action="store_true",
        help="enable slot-masking variants (alias of --enable-feature training.slot_masking)",
    )
    parser.add_argument(
        "--tool-dropout-enabled",
        action="store_true",
        help="enable tool-dropout variants (alias of --enable-feature training.tool_dropout)",
    )
    parser.add_argument(
        "--tool-dropout-max-tools",
        type=int,
        default=None,
        help=(
            "tools removed per record by tool dropout "
            "(alias of features.training.tool_dropout_max_tools; overrides the config value)"
        ),
    )
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=None,
        help=(
            "selection seed for augmentations "
            "(alias of features.training.augmentation_seed; overrides the config value)"
        ),
    )
    # --enable-feature / --disable-feature (the canonical feature switch).
    from lifemate.features.overrides import add_feature_arguments  # noqa: PLC0415

    add_feature_arguments(parser)
    return parser.parse_args(argv)

def _load_tokenizer(spec: str) -> tuple[object | None, str]:
    """Best-effort tokenizer load; returns ``(tokenizer, status)``."""
    if not spec or spec.strip().lower() in {"none", "no", "false", "-"}:
        return None, "deferred to training (no tokenizer requested)"
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415 - lazy, optional
    except Exception as exc:
        print(f"warning: --tokenizer {spec} ignored (transformers unavailable: {exc})")
        return None, "deferred to training (transformers unavailable)"
    try:
        return AutoTokenizer.from_pretrained(spec), f"loaded {spec}"
    except Exception as exc:  # pragma: no cover - network/weights dependent
        print(f"warning: --tokenizer {spec} could not be loaded ({exc}); using estimates")
        return None, "deferred to training (tokenizer load failed)"

def _fit_stats(records: list[dict], tokenizer: object | None, max_seq_length: int) -> dict:
    """How the sequence budget treats these examples: truncated vs dropped.

    Two different outcomes must never be conflated (a previous version reported
    only "n_fit", which read as "everything fits" while the collator was
    silently middle-truncating most of the records — measured 11.17 % of train
    prompts at max_seq_length 8192 with the DEFAULT MiniCPM5-2B tokenizer, and
    48.1 % at 5632):

    * **truncated** — `kept=True` but the prompt lost its middle; the example is
      still trained on, with the request prefix intact;
    * **dropped** — `kept=False`, counted by exact reason: `target_too_long`
      (a target that alone exceeds the budget cannot be supervised at all) and
      `target_value_truncated` (a `tool_call` whose reference ID/name lived in
      the middle the cut removed; supervising it teaches an identifier the model
      was never shown). Measured with the DEFAULT model at 8192: 70 dropped
      (1.60 %), 0 losing a tool schema block; the earlier "0 at 8192, 122 at
      4096" figures were Qwen3.5-4B numbers.
    """
    if tokenizer is None:
        # No tokenizer: the char/4 heuristic UNDER-counts Vietnamese by ~2.5x,
        # so its numbers must not be presented as a measured answer.
        estimate = [
            int(record["meta"].get("token_estimate") or 0)
            for record in records
        ]
        over_estimate = sum(1 for value in estimate if value > max_seq_length)
        return {
            "method": "chars4_estimate",
            "caveat": (
                "no tokenizer supplied: counts are chars/4 ESTIMATES, not measurements. "
                "Vietnamese text tokenizes to roughly 2.5x more tokens than chars/4, so "
                "n_truncated_estimate is a LOWER BOUND. Pass --tokenizer for real counts."
            ),
            "max_seq_length": max_seq_length,
            "n_records": len(records),
            "n_truncated_estimate": over_estimate,
            "n_truncated": None,
            "n_dropped": None,
            "n_fit": len(records),  # records written; NOT a claim that none truncates
        }
    from lifemate.training.collator import build_supervised_example  # noqa: PLC0415

    truncated = 0
    dropped: dict[str, int] = {}
    for record in records:
        example = build_supervised_example(
            tokenizer, record["messages"], record["target"], max_seq_length
        )
        if example.get("kept"):
            # build_supervised_example marks a middle cut in `reason`
            if str(example.get("reason") or "").startswith("prompt_middle_truncated_"):
                truncated += 1
        else:
            reason = str(example.get("reason") or "unknown")
            dropped[reason] = dropped.get(reason, 0) + 1
    n_dropped = sum(dropped.values())
    return {
        "method": "tokenizer",
        "caveat": None,
        "max_seq_length": max_seq_length,
        "n_records": len(records),
        "n_truncated": truncated,
        "n_truncated_fraction": round(truncated / len(records), 6) if records else 0.0,
        "n_dropped": n_dropped,
        "n_fit": len(records) - n_dropped,
        "dropped_reasons": dict(sorted(dropped.items())),
    }

def _print_table(stats: dict, n_read: int, n_val: int, n_val_points: int) -> None:
    """Compact human-readable summary of the build (train split, then val)."""
    print("SFT dataset")
    print(f"  samples read        : {n_read} (train {stats['n_samples']}, val {n_val})")
    print(f"  samples kept        : {stats['n_samples_kept']} (train split)")
    print(f"  decision points     : {stats['n_decision_points']}")
    print(f"  dropped             : {stats['n_dropped']}")
    for reason, count in sorted((stats.get("dropped") or {}).items()):
        if count:
            print(f"    - {reason}: {count}")
    if stats.get("n_dropped_decision_points"):
        print(f"  decision points dropped: {stats['n_dropped_decision_points']}")
        for reason, count in sorted((stats.get("dropped_decision_points") or {}).items()):
            if count:
                print(f"    - {reason}: {count}")
    print(f"  visibility limited  : {stats['visibility_limited']}")
    print(f"  unobservable x-ref  : {stats['unobservable_ref']} sample(s) -> affected targets dropped")
    lengths = stats.get("trajectory_length_histogram") or {}
    print("  calls per sample    : " + ", ".join(f"{k}={v}" for k, v in sorted(lengths.items())))
    prompt = stats.get("prompt_tokens_estimate") or {}
    target = stats.get("target_tokens_estimate") or {}
    print(f"  prompt tokens (est) : min {prompt.get('min')} mean {prompt.get('mean')} max {prompt.get('max')}")
    print(f"  target tokens (est) : min {target.get('min')} mean {target.get('mean')} max {target.get('max')}")

def _sha256_file(path: str | Path) -> str:
    """sha256 of a transcript file (provenance kept in the manifest)."""
    import hashlib  # noqa: PLC0415 - only needed when an augmentation is on

    from lifemate.compliance import safe_open  # noqa: PLC0415 - prohibited paths refused

    digest = hashlib.sha256()
    with safe_open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

#: Bespoke boolean flags and the feature id each one aliases.
AUGMENTATION_ALIASES: tuple[tuple[str, str], ...] = (
    ("noisy_asr_enabled", "training.noisy_asr"),
    ("nbest_training_format", "prompt.nbest_format"),
    ("slot_masking_enabled", "training.slot_masking"),
    ("tool_dropout_enabled", "training.tool_dropout"),
)

#: ``kind -> feature id`` for the transforms ``_augment`` runs.
_AUGMENTATION_FEATURES: dict[str, str] = {
    "noisy_asr": "training.noisy_asr",
    "nbest_format": "prompt.nbest_format",
    "slot_masking": "training.slot_masking",
    "tool_dropout": "training.tool_dropout",
}

def _resolve_augmentation_config(
    cfg: object, args: argparse.Namespace
) -> tuple[object, list[str], dict[str, object]]:
    """Merge ``cfg.features`` with the deprecated per-feature CLI aliases.

    Precedence, applied in this order:

    1. ``cfg.features`` is authoritative: YAML ``features:`` block, then
       ``LIFEMATE_FEATURE_*`` env, then ``--enable-feature`` / ``--disable-feature``
       (all resolved by :func:`lifemate.features.overrides.build_feature_config`);
    2. a bespoke boolean alias that is *passed* while its feature is off turns the
       feature ON (old command lines keep working) and is added to the returned
       deprecation notes;
    3. an explicit numeric alias (``--noisy-asr-ratio``, ``--tool-dropout-max-tools``,
       ``--augmentation-seed``) overrides the config value and is noted too;
    4. a feature forced OFF on the CLI (``--disable-feature``) still wins unless its
       own alias was passed — in that case the alias wins and note 2 is printed, so
       the run never silently contradicts a visible flag.

    Returns ``(cfg, notes, effective)`` where ``effective`` holds the resolved
    booleans and numbers that ``_augment`` actually uses.
    """
    from lifemate.features.config import FeatureConfig  # noqa: PLC0415 - optional slice
    from lifemate.features.overrides import build_feature_config  # noqa: PLC0415

    features = getattr(cfg, "features", None)
    if not isinstance(features, FeatureConfig):  # pragma: no cover - slice absent
        features = FeatureConfig()
    requested_cli = set()
    for raw in list(getattr(args, "enable_feature", None) or []) + list(
        getattr(args, "disable_feature", None) or []
    ):
        requested_cli.add(str(raw).split("=", 1)[0].strip())
    resolved, _provenance = build_feature_config(cfg, args)
    cfg.features = resolved  # type: ignore[attr-defined]

    notes: list[str] = []
    for flag, feature_id in AUGMENTATION_ALIASES:
        if not getattr(args, flag, False):
            continue
        if not resolved_value(resolved, feature_id):
            set_feature_value(resolved, feature_id, True)
            # The alias IS a command-line switch, so provenance reads "cli" for
            # both spellings: the dataset's fingerprint must not depend on which
            # spelling enabled the feature. The deprecation note below is the
            # visible record of which one was used.
            provenance = dict(getattr(resolved, "provenance", {}) or {})
            provenance[feature_id] = "cli"
            resolved.provenance = provenance
            notes.append(f"note: --{flag.replace('_', '-')} is a deprecated alias for --enable-feature {feature_id}")
        elif feature_id in requested_cli:
            notes.append(
                f"note: --{flag.replace('_', '-')} agrees with --enable-feature {feature_id}"
            )

    training = resolved.training
    effective: dict[str, object] = {}
    for numeric, section, name, cast in (
        ("noisy_asr_ratio", training, "noisy_asr_ratio", float),
        ("tool_dropout_max_tools", training, "tool_dropout_max_tools", int),
        ("augmentation_seed", training, "augmentation_seed", int),
    ):
        value = getattr(args, numeric, None)
        if value is None:
            effective[numeric] = getattr(section, name)
            continue
        setattr(section, name, cast(value))
        effective[numeric] = getattr(section, name)
        notes.append(
            f"note: --{numeric.replace('_', '-')} overrides features.training.{name} "
            f"(now {effective[numeric]})"
        )
    for kind, feature_id in _AUGMENTATION_FEATURES.items():
        effective[kind] = bool(resolved_value(resolved, feature_id))
    mode = getattr(args, "noisy_asr_mode", None)
    if mode:
        training.noisy_asr_mode = str(mode)
        notes.append(
            f"note: --noisy-asr-mode overrides features.training.noisy_asr_mode "
            f"(now {training.noisy_asr_mode})"
        )
    effective["noisy_asr_mode"] = str(training.noisy_asr_mode)
    return cfg, notes, effective

def resolved_value(cfg: object, feature_id: str) -> object:
    """Value of one feature id without importing the registry's private maps."""
    from lifemate.features.registry import REGISTRY  # noqa: PLC0415

    return REGISTRY.value(feature_id, cfg)

def set_feature_value(cfg: object, feature_id: str, value: bool) -> object:
    """Set one feature id in place (the registry owns the config path)."""
    from lifemate.features.registry import REGISTRY  # noqa: PLC0415

    return REGISTRY.set_enabled(feature_id, cfg, bool(value))

def _validate_features(cfg: object) -> None:
    """Apply the feature registry's rules (dependencies, conflicts, value ranges)."""
    from lifemate.features.registry import REGISTRY  # noqa: PLC0415 - optional slice

    REGISTRY.validate_or_raise(cfg.features)

def _print_feature_report(cfg: object) -> None:
    """Print the effective feature configuration (labels are contractual)."""
    from lifemate.features.report import format_feature_report  # noqa: PLC0415

    print(
        format_feature_report(
            cfg.features, model_id=cfg.model.name, asr_id=cfg.asr.model_id
        )
    )

def _effective_augmentation_line(effective: dict[str, object]) -> str:
    """One line naming the resolved augmentation set (after alias merging)."""
    active = sorted(kind for kind in _AUGMENTATION_FEATURES if effective[kind])
    return (
        "augmentation set    : "
        + (", ".join(active) if active else "NONE (no augmentation runs)")
        + f"  ratio={effective['noisy_asr_ratio']} "
        f"mode={effective.get('noisy_asr_mode', 'augment')} "
        f"max_tools={effective['tool_dropout_max_tools']} seed={effective['augmentation_seed']}"
    )

def _augment(
    train_records: list[dict],
    val_records: list[dict],
    *,
    args: argparse.Namespace,
    catalog: object,
    effective: dict[str, object],
) -> tuple[dict, list[dict]]:
    """Apply the augmentations the EFFECTIVE feature config enables, train only.

    ``effective`` comes from :func:`_resolve_augmentation_config`: ``cfg.features``
    (already merged with the deprecated CLI aliases) is the single source of
    truth, so a ``features:`` YAML block and ``--enable-feature`` change the built
    dataset exactly like the old bespoke flags do.

    Every transform receives the *original* train records and its variants are
    appended after them, so each variant carries exactly one provenance label and
    a variant is never fed into another transform. Val is asserted untouched.
    """
    from lifemate.data.asr_variants import (  # noqa: PLC0415 - optional path
        apply_nbest_format,
        apply_noisy_asr,
        load_transcripts,
    )
    from lifemate.data.augmentation import (  # noqa: PLC0415 - optional path
        apply_slot_masking,
        apply_tool_dropout,
    )

    seed = int(effective["augmentation_seed"])
    active = {kind: bool(effective[kind]) for kind in _AUGMENTATION_FEATURES}
    ratio = float(effective["noisy_asr_ratio"])
    max_tools = int(effective["tool_dropout_max_tools"])
    if not any(active.values()):
        return {}, train_records
    if not train_records:
        return {"enabled": active, "n_records_in": 0, "n_records_out": 0}, train_records

    base = list(train_records)
    variants: list[dict] = []
    replaced: list[dict] | None = None
    reports: dict[str, object] = {}
    sources: dict[str, object] = {}
    mode = str(effective.get("noisy_asr_mode", "augment"))

    for name, path in (
        ("noisy_asr", args.noisy_asr_transcripts),
        ("nbest_format", args.nbest_transcripts),
    ):
        if active[name]:
            if not path:
                raise LifeMateError(f"{name} is enabled but no transcript file was given")
            sources[name] = {"path": str(path), "sha256": _sha256_file(path)}

    if active["noisy_asr"]:
        transcripts = load_transcripts(args.noisy_asr_transcripts)
        produced, report = apply_noisy_asr(
            base, transcripts, enabled=True, ratio=ratio, seed=seed, mode=mode
        )
        reports["noisy_asr"] = report
        if mode == "replace":
            # Replace mode swaps the sampled clean records in place, so the
            # record count stays the same and `ratio` is the real noisy share.
            replaced = produced
        else:
            variants.extend(_variant_records(produced, base))
    if active["nbest_format"]:
        transcripts = load_transcripts(args.nbest_transcripts, require_multi=True)
        produced, report = apply_nbest_format(base, transcripts, enabled=True, k=3, seed=seed)
        reports["nbest_format"] = report
        variants.extend(_variant_records(produced, base))
    if active["slot_masking"]:
        produced, report = apply_slot_masking(base, enabled=True, seed=seed, catalog=catalog)
        reports["slot_masking"] = report
        variants.extend(_variant_records(produced, base))
    if active["tool_dropout"]:
        produced, report = apply_tool_dropout(
            base,
            enabled=True,
            max_tools=max_tools,
            seed=seed,
            catalog=catalog,
        )
        reports["tool_dropout"] = report
        variants.extend(_variant_records(produced, base))

    train_ids = {str(record.get("sample_id")) for record in base}
    val_ids = {str(record.get("sample_id")) for record in val_records}
    leaked = train_ids & val_ids
    if leaked:
        raise LifeMateError(
            f"augmentation would leak {len(leaked)} sample id(s) into val (e.g. {sorted(leaked)[0]})"
        )
    outside = sorted({str(record.get("sample_id")) for record in variants} - train_ids)
    if outside:
        raise LifeMateError(
            f"augmented variant(s) outside the train group (e.g. {outside[0]!r}); "
            "augment only after the group split"
        )

    out_records = replaced if replaced is not None else base
    augmented = [
        record for record in out_records + variants if (record.get("meta") or {}).get("augmentation")
    ]
    report = {
        "enabled": active,
        "seed": seed,
        "noisy_asr_ratio": ratio,
        "noisy_asr_mode": mode,
        "tool_dropout_max_tools": max_tools,
        "transcript_sources": sources,
        "n_records_in": len(base),
        "n_records_out": len(out_records) + len(variants),
        # A replaced record is an augmented record too: the count follows the
        # dataset, so `augment` and `replace` are reported the same way.
        "n_variants": len(augmented),
        "n_appended_variants": len(variants),
        "n_variants_by_kind": _count_variants(augmented),
        "reports": reports,
        "val_augmented": False,
    }
    if replaced is not None:
        report["n_replaced"] = sum(
            1 for record in out_records if (record.get("meta") or {}).get("augmentation") == "noisy_asr"
        )
    return report, out_records + variants

def _variant_records(produced: list[dict], base: list[dict]) -> list[dict]:
    """Variants a transform appended next to its originals.

    Transforms keep the original object and emit the variant right after it, so
    the variants are the produced records that are neither an original object nor
    untagged.
    """
    original_ids = {id(record) for record in base}
    return [
        record
        for record in produced
        if id(record) not in original_ids and (record.get("meta") or {}).get("augmentation")
    ]

def _count_variants(variants: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in variants:
        kind = str((record.get("meta") or {}).get("augmentation"))
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))

def _write_records(records: list[dict], path: Path) -> None:
    """Write SFT records exactly as ``trajectory_builder.to_jsonl`` serialises them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    tmp.replace(path)

def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = parse_args(argv)
    try:
        cfg = load_config(args.config) if args.config else load_config()
    except LifeMateError as exc:
        print(f"invalid config: {exc}")
        return 1

    # Features are resolved and VALIDATED before any data is read: an impossible
    # combination (e.g. noisy_asr_ratio out of range) must fail fast, and the
    # effective set is printed once so a run can never silently build a
    # distribution the caller did not ask for.
    try:
        cfg, alias_notes, effective = _resolve_augmentation_config(cfg, args)
    except LifeMateError as exc:
        print(f"configuration error: {exc}")
        return 1
    for note in alias_notes:
        print(note)
    try:
        _validate_features(cfg)
    except LifeMateError as exc:
        print(f"configuration error: {exc}")
        return 1
    _print_feature_report(cfg)
    print(_effective_augmentation_line(effective))

    if args.conditioned_multi_records:
        if args.split != 'train' or not cfg.features.asr.conditioned_multi_sft_enabled or not cfg.features.asr.multi_asr_enabled:
            print('configuration error: conditioned data requires train split and both conditioned_multi_sft_enabled and multi_asr_enabled')
            return 1
        if any(effective[kind] for kind in _AUGMENTATION_FEATURES):
            print('configuration error: combining conditioned data with other training augmentation is not verified')
            return 1
        dest=Path(args.output)/args.split
        if any((dest/name).exists() for name in ('train.jsonl','val.jsonl','manifest.json','stats.json')):
            print('configuration error: conditioned dataset destination already exists')
            return 1

    val_ratio = args.val_ratio if args.val_ratio is not None else float(cfg.training.validation_ratio)
    seed = args.seed if args.seed is not None else int(cfg.training.seed)
    history_limit = args.history_limit if args.history_limit is not None else 10
    max_seq_length = args.max_seq_length or int(cfg.training.max_seq_length)
    limit = args.limit if args.limit is not None else cfg.data.max_samples

    try:
        samples = load_samples(args.data_root, args.split, limit=limit)
        catalog = load_catalog(args.data_root, args.split)
    except DatasetNotFoundError as exc:
        print(str(exc))
        return 2
    except LifeMateError as exc:
        print(f"failed to read split {args.split!r}: {exc}")
        return 1

    if catalog is None:
        print(
            f"warning: split {args.split!r} has no tool catalog; every gold call will be "
            "flagged catalog_missing_tool and nothing will be written"
        )
        catalog = _EmptyCatalog()
    if not samples:
        print(DatasetNotFoundError(args.split, str(Path(args.data_root) / args.split)))
        return 2

    split = split_samples(samples, ratio=val_ratio, seed=seed)
    train_points, train_stats = build_sft_dataset(split.train, catalog, history_limit=history_limit)
    val_points, val_stats = build_sft_dataset(split.val, catalog, history_limit=history_limit)

    tokenizer, tokenizer_status = _load_tokenizer(args.tokenizer)
    train_records = to_hf_dataset_records(train_points)
    val_records = to_hf_dataset_records(val_points)

    # Augmentation runs AFTER the group split and only on train: a variant keeps
    # its sample_id, so it stays inside the group split_builder assigned.
    try:
        augmentation, train_records = _augment(
            train_records,
            val_records,
            args=args,
            catalog=catalog,
            effective=effective,
        )
    except LifeMateError as exc:
        print(f"augmentation failed: {exc}")
        return 2
    conditioned_report = None
    if args.conditioned_multi_records:
        try:
            from lifemate.data.conditioned_sft import load_records, augment_conditioned
            index, artifact = load_records(args.conditioned_multi_records)
            artifact_itn=bool(((artifact.get('identity',{}).get('features',{}).get('prompt',{})) or {}).get('itn_enabled',False))
            if artifact_itn != bool(cfg.features.prompt.itn_enabled):
                raise LifeMateError('history ASR and training ITN configurations differ')
            train_records, counts = augment_conditioned(train_records, split.train, index,
                k=cfg.features.asr.multi_asr_max_hypotheses, ratio=args.conditioned_ratio, seed=seed,
                unknown_policy=args.conditioned_unknown, itn=cfg.features.prompt.itn_enabled,
                split=args.split, data_root=args.data_root, history_limit=history_limit, mode=args.conditioned_mode,
                single_ratio=args.conditioned_single_ratio, multi_ratio=args.conditioned_multi_ratio)
            if not counts['created']:
                raise LifeMateError(f"no conditioned variants retained; audit counts={ {k:v for k,v in counts.items() if k != 'decisions'} }")
            conditioned_report={'schema':'lifemate.conditioned_sft.v1','artifact_sha256':artifact['sha256'],
                'artifact_run_identity':artifact['run_identity'],'candidate_k':cfg.features.asr.multi_asr_max_hypotheses,
                'ratio':args.conditioned_ratio,'single_ratio':args.conditioned_single_ratio,'multi_ratio':args.conditioned_multi_ratio,
                'seed':seed,'unknown_policy':args.conditioned_unknown,'mode':args.conditioned_mode,
                'itn':cfg.features.prompt.itn_enabled,'prompt_version':'conditioned_multi_v1','coverage':counts,
                'asr_models':artifact['identity']['models'],'candidate_selection':'configured_priority_after_itn_dedup',
                'history_protocol':'replace_visible_user_history_with_verified_asr',
                'loss_mask_validation':'not_per_record_verified'}
        except (LifeMateError, OSError, ValueError, KeyError) as exc:
            print(f'conditioned augmentation failed: {exc}')
            return 2

    if augmentation:
        print(
            f"  augmentation        : {augmentation['n_records_in']} -> {augmentation['n_records_out']} "
            f"train record(s), +{augmentation['n_variants']} variant(s) "
            f"({', '.join(f'{k}={v}' for k, v in augmentation['n_variants_by_kind'].items()) or 'none'})"
        )
    fit = _fit_stats(train_records, tokenizer, max_seq_length)
    fit_val = _fit_stats(val_records, tokenizer, max_seq_length)

    out_dir = Path(args.output) / args.split
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Same serialisation as ``to_jsonl`` (compact JSON, sorted keys, one
        # record per line); records are written instead of points so the
        # augmented train list can carry its variants.
        _write_records(train_records, out_dir / "train.jsonl")
        _write_records(val_records, out_dir / "val.jsonl")
    except OSError as exc:
        print(f"failed to write {out_dir}: {exc}")
        return 1

    manifest = {
        **split.manifest(),
        "split": args.split,
        "data_root": str(Path(args.data_root).resolve()),
        "n_train_decision_points": len(train_records),
        "n_val_decision_points": len(val_points),
        # Points the builder refused to supervise (a gold argument the prompt
        # cannot show). Without these keys the manifest's point counts silently
        # shrink versus ``n_train_decision_points`` of an older build.
        "n_train_dropped_decision_points": train_stats["n_dropped_decision_points"],
        "n_val_dropped_decision_points": val_stats["n_dropped_decision_points"],
        "dropped_decision_points": {
            "train": train_stats["dropped_decision_points"],
            "val": val_stats["dropped_decision_points"],
        },
        "history_limit": history_limit,
        "tokenization": tokenizer_status,
        "fit": fit,
        "fit_val": fit_val,
    }
    stats = {
        "split": args.split,
        "train": train_stats,
        "val": val_stats,
        "tokenization": tokenizer_status,
        "fit": fit,
        "fit_val": fit_val,
    }
    if conditioned_report is not None:
        conditioned_report['train_sha256'] = _sha256_file(out_dir / 'train.jsonl')
        conditioned_report['validation_sha256'] = _sha256_file(out_dir / 'val.jsonl')
        manifest['conditioned_multi_asr'] = conditioned_report
        stats['conditioned_multi_asr'] = conditioned_report
    if augmentation:
        # Only present when an augmentation is active: a no-flag run must stay
        # byte-identical to the pre-augmentation build.
        manifest["augmentation"] = augmentation
        stats["augmentation"] = augmentation
        # The dataset's own record of WHICH feature configuration produced it, so
        # ``scripts/train.py`` can compare its run fingerprint with this build's
        # instead of trusting a flag it merely recorded.
        manifest["features"] = feature_report_dict(
            cfg.features, model_id=cfg.model.name, asr_id=cfg.asr.model_id
        )
        manifest["training_feature_fingerprint"] = training_feature_fingerprint(cfg.features)
        stats["features"] = manifest["features"]

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    _print_table(train_stats, len(samples), len(split.val), len(val_points))
    print(f"  tokenization        : {tokenizer_status}")
    print(f"  max_seq_length      : {max_seq_length} (effective, from config unless --max-seq-length)")
    if fit["n_truncated"] is None:
        print(
            f"  truncation          : UNMEASURED — {fit['n_truncated_estimate']}/{fit['n_records']} "
            "over budget by the chars/4 estimate (lower bound); pass --tokenizer"
        )
    else:
        print(
            f"  truncation (train)  : {fit['n_truncated']}/{fit['n_records']} "
            f"({fit['n_truncated_fraction']*100:.2f}%) middle-truncated, "
            f"{fit['n_dropped']} dropped"
        )
        print(
            f"  truncation (val)    : {fit_val['n_truncated']}/{fit_val['n_records']} "
            f"({fit_val['n_truncated_fraction']*100:.2f}%) middle-truncated, "
            f"{fit_val['n_dropped']} dropped"
        )
    if train_stats["visibility_limited"]:
        print(
            "warning: "
            f"{train_stats['visibility_limited']} sample(s) are visibility_limited "
            "(missing final tool results) and got no terminal finish example"
        )
    if train_stats["unobservable_ref"]:
        print(
            "warning: "
            f"{train_stats['unobservable_ref']} sample(s) had reference arguments not observable from "
            f"the prompt; {train_stats['n_dropped_decision_points']} affected decision point(s) were "
            "dropped (the rest of each trajectory is kept) — see stats.json"
        )
    if train_stats["n_dropped"]:
        print(f"warning: {train_stats['n_dropped']} sample(s) dropped (see stats.json)")
    print(f"wrote {out_dir}/train.jsonl, val.jsonl, manifest.json, stats.json")
    return 0

class _EmptyCatalog:
    """Placeholder catalog so the pipeline reports rather than crashes."""

    tools: dict = {}
    apps: list = []
    specs: list = []

    def tools_for_apps(self, apps=None, include_meta: bool = True) -> list:
        return []

    def validate_call(self, name: str, args=None) -> dict:
        return {"errors": [f"unknown tool: {name}"], "warnings": []}

    def get(self, name: str):
        return None

    def to_schema_prompt(self, apps=None, include_meta: bool = True) -> str:
        return "Available tools: none"

if __name__ == "__main__":
    sys.exit(main())

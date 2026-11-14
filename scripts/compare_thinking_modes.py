#!/usr/bin/env python3
"""A/B the two chat-template thinking branches on REAL weights, one prompt.

The repo ships ``enable_thinking=False`` for MiniCPM5-2B. That is a prompt-format
fact (an empty closed think block, byte-identical to the training serialization);
this script measures the *behavioural* question: which branch produces decisions
:func:`lifemate.agent.tool_parser.parse_model_output` can actually execute.

For every sampled real record both arms see the SAME record, the SAME rendered
messages, the SAME decode budget and a greedy (therefore deterministic) decode.
Only ``enable_thinking`` differs. Nothing else is allowed to drift: a difference
in the table is a difference caused by that one template kwarg.

Honest accounting:

* a case is a "valid action" only when the repo's parser + registry validation
  say ``tool_call``; ``finish`` and ``invalid`` are counted separately;
* a decode cut off by the token budget is reported as truncated and never
  retried with a bigger budget (that would make the comparison meaningless);
  its text is re-captured at the SAME budget with the early-stop criterion off,
  purely so the raw shape can be shown;
* a crashed run reports nothing but the crash.

Usage::

    WANDB_BASE_URL=http://127.0.0.1:9 PYTHONPATH=src \\
        python3 scripts/compare_thinking_modes.py --n 20 --out outputs/reports/thinking_ab.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lifemate.agent.tool_parser import parse_model_output  # noqa: E402
from lifemate.agent.tool_registry import ToolRegistry  # noqa: E402
from lifemate.compliance import is_prohibited_path  # noqa: E402
from lifemate.data.loader import load_catalog, load_samples  # noqa: E402
from lifemate.data.trajectory_builder import build_decision_points  # noqa: E402
from lifemate.errors import LifeMateError  # noqa: E402
from lifemate.models.generation import HFGenerator  # noqa: E402
from lifemate.models.profiles import resolve_profile  # noqa: E402
from lifemate.training.collator import render_prompt_text  # noqa: E402

#: Labeled splits only. The unlabeled test splits carry no gold trajectory, so no
#: A/B over them could exist; and nothing here may ever point at an env dir.
ALLOWED_SPLITS = ("train",)

MODES = {"no_think": False, "think": True}

#: Card ceiling for the runs behind ``docs/INFERENCE_GUIDE.md``: 14.37/15.17 GB
#: peak at 8k context on a 16.57 GB RTX 5070 Ti leaves no room for longer ones.
MEASURED_BUDGET_CEILING = 512


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def check_paths(data_root: str | Path, split: str, out: str | Path) -> None:
    """Refuse a split that is unlabeled, unknown, or inside a ``*_environment``."""
    if split not in ALLOWED_SPLITS:
        raise SystemExit(
            f"refusing --split {split!r}: this harness runs on {list(ALLOWED_SPLITS)} only "
            "(public_test/private_test carry no gold trajectory, so no A/B exists there)"
        )
    for label, path in (("--data-root", data_root), ("--out", out)):
        if is_prohibited_path(str(path)):
            raise SystemExit(f"refusing {label} {str(path)!r}: a *_environment path is never readable")


# --------------------------------------------------------------------------- #
# Decode plumbing
# --------------------------------------------------------------------------- #
class _IdCapturingModel:
    """Forward to the real model, keeping the last ``generate`` output ids.

    ``HFGenerator`` returns decoded TEXT and drops the ids, but the terminator
    (``<|im_end|>`` = 130073) is a special token, so ``skip_special_tokens=True``
    erases exactly the evidence one of the reported columns needs. This proxy is
    the smallest hook that keeps it; every other attribute is delegated.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self.last_output_ids: list[int] = []
        self.last_prompt_ids: list[int] = []

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        prompt = kwargs.get("input_ids")
        if prompt is None and args:
            prompt = args[0]
        self.last_prompt_ids = [int(i) for i in prompt[0].tolist()] if prompt is not None else []
        output = self._model.generate(*args, **kwargs)
        self.last_output_ids = [int(i) for i in output[0].tolist()]
        return output

    def raw_decode(self, tokenizer: Any, ids: Sequence[int]) -> str:
        """Decode WITH special tokens kept.

        ``HFGenerator`` decodes with ``skip_special_tokens=True``, which erases
        the template's own tool syntax: ``<function`` (id 18) and ``<param``
        (id 20) are added special tokens, so a turn that emits
        ``<function name="x">`` instead of ``<action>`` would look like plain
        prose. Every text-based column here counts ids, not greppable text.
        """
        return tokenizer.decode(ids, skip_special_tokens=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def run_case(
    proxy: _IdCapturingModel,
    tokenizer: Any,
    profile: Any,
    messages: Sequence[dict[str, str]],
    registry: ToolRegistry,
    max_new_tokens: int,
) -> dict[str, Any]:
    """One greedy decode in one mode; never raises (truncation is a value)."""
    generator = HFGenerator(
        proxy,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_on_action=True,
        profile=profile,
    )
    proxy.last_output_ids = []
    started = time.monotonic()
    truncated = False
    try:
        text = generator.generate(messages)
    except LifeMateError as exc:
        # A budget cut is data, not a crash. The proxy already holds the ids of
        # the failed decode, so the raw shape is re-decoded from THOSE: no retry,
        # no bigger budget, no second sampling.
        if "without emitting" not in str(exc):
            raise
        truncated = True
        text = ""
    elapsed = time.monotonic() - started
    new_ids = proxy.last_output_ids[len(proxy.last_prompt_ids) :]
    # Specials KEPT: the only view in which the template's own <function/<param
    # syntax survives. Parsing still uses the text the AGENT would see, which is
    # HFGenerator's skip_special_tokens=True decode.
    raw_text = proxy.raw_decode(tokenizer, new_ids)
    if truncated:
        text = raw_text
    return _score(text, raw_text, new_ids, len(proxy.last_prompt_ids), truncated, elapsed, generator, registry)


def _score(
    text: str,
    raw_text: str,
    new_ids: list[int],
    n_prompt: int,
    truncated: bool,
    elapsed: float,
    generator: HFGenerator,
    registry: ToolRegistry,
) -> dict[str, Any]:
    action = parse_model_output(text, registry)
    words = text.split()
    return {
        "prompt_tokens": n_prompt,
        "generated_tokens": len(new_ids),
        # ``<|im_end|>`` is a special token: only the id list can witness it.
        "im_end_emitted": bool(new_ids) and new_ids[-1] in set(generator.stop_ids),
        "has_action_open": "<action>" in text,
        "has_action_close": "</action>" in text,
        # The template's native tool syntax, counted on a specials-kept decode.
        "function_blocks": raw_text.count("<function"),
        "looped": _repeats(words),
        "truncated": bool(truncated),
        "action_type": action.type,
        "action": action.to_dict() if action.type != "invalid" else None,
        "parse_error": action.error,
        "seconds": round(elapsed, 3),
        "text": text,
        "raw_text": raw_text,
    }


def _repeats(words: Sequence[str], run: int = 8) -> bool:
    """True when some 8-word window repeats: the loop/rambling witness."""
    for start in range(len(words) - 2 * run):
        if words[start : start + run] == words[start + run : start + 2 * run]:
            return True
    return False


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-mode summary. Rates are over cases that actually decoded."""
    n = len(cases)
    if not n:
        return {"n": 0}
    tokens = [c["generated_tokens"] for c in cases]
    whole = [c for c in cases if not c["truncated"]]
    return {
        "n": n,
        "valid_action_rate": round(sum(c["action_type"] == "tool_call" for c in cases) / n, 4),
        "finish_rate": round(sum(c["action_type"] == "finish" for c in cases) / n, 4),
        "parse_failure_rate": round(sum(c["action_type"] == "invalid" for c in cases) / n, 4),
        "truncation_rate": round(sum(c["truncated"] for c in cases) / n, 4),
        "im_end_rate": round(sum(c["im_end_emitted"] for c in cases) / n, 4),
        "function_block_rate": round(sum(c["function_blocks"] > 0 for c in cases) / n, 4),
        "looped_rate": round(sum(c["looped"] for c in cases) / n, 4),
        # Truncation is its own outcome; these two keep the parse verdict
        # interpretable instead of folding "ran out of budget" into "garbage".
        "n_untruncated": len(whole),
        "valid_action_rate_untruncated": (
            round(sum(c["action_type"] == "tool_call" for c in whole) / len(whole), 4) if whole else None
        ),
        "mean_tokens": round(statistics.fmean(tokens), 1),
        "median_tokens": int(statistics.median(tokens)),
        "max_tokens": max(tokens),
        "mean_seconds": round(statistics.fmean([c["seconds"] for c in cases]), 3),
        "total_seconds": round(sum(c["seconds"] for c in cases), 1),
    }


def print_table(summary: dict[str, Any]) -> None:
    rows = [
        ("n", "n"),
        ("valid action", "valid_action_rate"),
        ("valid (untrunc)", "valid_action_rate_untruncated"),
        ("finish", "finish_rate"),
        ("parse failure", "parse_failure_rate"),
        ("truncated", "truncation_rate"),
        ("looped", "looped_rate"),
        ("<|im_end|>", "im_end_rate"),
        ("<function> blocks", "function_block_rate"),
        ("mean tokens", "mean_tokens"),
        ("median tokens", "median_tokens"),
        ("max tokens", "max_tokens"),
        ("mean seconds", "mean_seconds"),
    ]
    modes = list(summary)
    width = max(len(label) for label, _ in rows) + 2
    print("\n" + "metric".ljust(width) + "".join(m.rjust(14) for m in modes))
    print("-" * (width + 14 * len(modes)))
    for label, key in rows:
        print(label.ljust(width) + "".join(str(summary[m].get(key, "-")).rjust(14) for m in modes))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="openbmb/MiniCPM5-2B")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=20, help="records to sample (first decision point each)")
    parser.add_argument("--seed", type=int, default=0, help="record sampling seed (decode is greedy)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--modes", default="think,no_think")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="outputs/reports/thinking_ab.json")
    return parser.parse_args(argv)


def sample_decisions(data_root: str, split: str, n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` real (record, first decision point) pairs, deterministic for a seed."""
    catalog = load_catalog(data_root, split)
    if catalog is None:
        raise SystemExit(f"no tool catalog for split {split!r} under {data_root!r}")
    samples = load_samples(data_root, split)
    picked: list[dict[str, Any]] = []
    for index in sorted(random.Random(seed).sample(range(len(samples)), min(n, len(samples)))):
        sample = samples[index]
        registry = ToolRegistry(catalog).for_sample(sample)
        points = build_decision_points(sample, catalog, registry=registry)
        if not points:
            continue
        picked.append({"sample": sample, "registry": registry, "point": points[0]})
    return picked


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    check_paths(args.data_root, args.split, args.out)
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"unknown --modes entries {unknown}; expected a subset of {list(MODES)}")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")

    decisions = sample_decisions(args.data_root, args.split, args.n, args.seed)
    if not decisions:
        raise SystemExit(f"no decision points in split {args.split!r}; nothing to compare")

    import torch

    from lifemate.models.loader import load_model_and_tokenizer

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU free {free / 2**30:.2f} GiB / {total / 2**30:.2f} GiB")
    bundle = load_model_and_tokenizer(args.model, torch_dtype="bfloat16", device=args.device)
    tokenizer, model = bundle.tokenizer, bundle.model
    profile = resolve_profile(args.model)
    proxy = _IdCapturingModel(model)

    results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for arm in modes:
        arm_profile = replace(profile, chat_template_kwargs={**profile.chat_template_kwargs, "enable_thinking": MODES[arm]})
        print(f"\n[{arm}] enable_thinking={MODES[arm]}")
        for decision in decisions:
            sample, point = decision["sample"], decision["point"]
            messages = point.messages()
            # Both arms see the same messages; only the kwarg may differ.
            rendered = render_prompt_text(messages, tokenizer, arm_profile)
            case = run_case(proxy, tokenizer, arm_profile, messages, decision["registry"], args.max_new_tokens)
            case.update({"sample_id": sample.id, "step_index": point.step_index, "mode": arm,
                         "prompt_tail": rendered[-40:]})
            results[arm].append(case)
            print(
                f"  {sample.id}#{point.step_index} {case['action_type']:>9} "
                f"tok={case['generated_tokens']:>3} im_end={int(case['im_end_emitted'])} "
                f"trunc={int(case['truncated'])} fn={case['function_blocks']} {case['seconds']:.2f}s"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {mode: aggregate(cases) for mode, cases in results.items()}
    payload = {
        "config": {
            "model": args.model,
            "model_meta": bundle.meta,
            "split": args.split,
            "data_root": str(args.data_root),
            "n_requested": args.n,
            "n_used": len(decisions),
            "seed": args.seed,
            "decode": "greedy (temperature=0, do_sample=False)",
            "max_new_tokens": args.max_new_tokens,
            "modes": {m: {"enable_thinking": MODES[m]} for m in modes},
            "default_in_profile": dict(profile.chat_template_kwargs),
            "budget_ceiling_note": (
                "512 is the card ceiling for these runs "
                f"(measured peak 14.37/15.17 GB at seq 8192 on a 16.57 GB card); "
                f"{MEASURED_BUDGET_CEILING}+ tokens was NOT tried"
            ),
            # Measured: the think-mode prompt TEXT is a strict prefix of the no-think
            # text (it stops at "assistant\nthink\n"); the ID lists are NOT prefix
            # related (18 vs 20 ids), because BPE merges across the newline boundary.
            "prompt_note": "think text = no_think text minus the trailing closed-think tail; ids are not prefix related",
        },
        "summary": summary,
        "cases": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_table(summary)
    print(f"\nwrote {out}" if out.is_file() else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())

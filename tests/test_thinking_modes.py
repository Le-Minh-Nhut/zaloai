"""The two template branches, and the guards that keep the A/B honest.

Cheap on purpose: no weights, no GPU. The one test that needs the real
tokenizer skips loudly when `models/MiniCPM5-2B` is absent.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

LOCAL_MINICPM = ROOT / "models" / "MiniCPM5-2B"


def test_harness_refuses_unlabeled_env_and_unknown_paths() -> None:
    """Only ``train`` is readable, and never through a ``*_environment`` path."""
    import compare_thinking_modes as harness

    refused = [
        ("data", "public_test", "/tmp/ab.json"),
        ("data", "private_test", "/tmp/ab.json"),
        ("data", "not_a_split", "/tmp/ab.json"),
        ("data/train_environment", "train", "/tmp/ab.json"),  # compliance-scan: allow
        ("data", "train", "data/train_environment/ab.json"),  # compliance-scan: allow
    ]
    for args in refused:
        try:
            harness.check_paths(*args)
        except SystemExit as exc:
            assert "refusing" in str(exc), exc
        else:
            raise AssertionError(f"harness accepted a path it must refuse: {args}")
    harness.check_paths("data", "train", "/tmp/ab.json")
    print("  harness -> train-only, no *_environment path")


def test_aggregate_never_reports_a_rate_from_nothing() -> None:
    """A crashed/empty arm reports ``{"n": 0}`` instead of a 0% action rate."""
    import compare_thinking_modes as harness

    assert harness.aggregate([]) == {"n": 0}
    print("  aggregate -> no rate without cases")


def test_thinking_branches_differ_only_in_the_think_tail() -> None:
    """think=True opens the block; the delta vs the default is that tail alone.

    Needs the real tokenizer (CPU, no weights). Measured: the think rendering is
    a strict TEXT prefix of the default one, the remainder is exactly
    ``\\n</think>\\n\\n``, and the two ID lists are NOT prefix related (18 vs 20
    ids) because BPE merges across the newline.
    """
    if not (LOCAL_MINICPM / "tokenizer_config.json").is_file():
        print("SKIP: models/MiniCPM5-2B tokenizer absent")
        return
    try:
        from transformers import AutoTokenizer
    except Exception:  # pragma: no cover - environment dependent
        print("SKIP: transformers unavailable")
        return

    from lifemate.models.profiles import MINICPM5_PROFILE
    from lifemate.training.collator import render_prompt_text

    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MINICPM), local_files_only=True)
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    plain = render_prompt_text(messages, tokenizer, MINICPM5_PROFILE)
    thinking = render_prompt_text(
        messages, tokenizer, replace(MINICPM5_PROFILE, chat_template_kwargs={"enable_thinking": True})
    )
    assert plain.startswith(thinking), "the two branches must agree up to the think tail"
    assert plain[len(thinking) :] == "\n</think>\n\n", repr(plain[len(thinking) :])
    # The open branch is the LAST thing in the prompt: the model must close it.
    assert thinking.endswith("<|im_start|>assistant\n\x3cthink\x3e\n"), repr(thinking[-40:])
    print(f"  branches -> same prompt up to the {len(plain) - len(thinking)}-char closed-think tail")

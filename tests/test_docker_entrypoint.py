"""Static + behavioural checks for ``docker/entrypoint.sh`` and its optional features.

No pytest, no docker: every check is a regex over the shell text or a real
``bash`` run of the script with a stub ``python`` on ``PATH`` (so the generated
argv is printed instead of inference starting). The image itself cannot be built
or run here (no daemon), so nothing below claims the image works.

The interesting invariant is DEFAULT-OFF: an unset ``LIFEMATE_FEATURE_*`` must
mean "off", so the printed summary line has to read ``features=NONE``. That is
asserted three ways -- statically (no truthy default and no assignment in the
script), behaviourally (a real run with the prefix stripped from the
environment), and against a mutated copy of the script as a negative control, so
the check itself is known to fail when the invariant breaks.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

ENTRYPOINT = ROOT / "docker" / "entrypoint.sh"
DOCKERFILE = ROOT / "docker" / "Dockerfile"
README = ROOT / "docker" / "README.md"

#: ``LIFEMATE_FEATURE_ASR_NBEST`` and friends.
FEATURE_ENV_RE = re.compile(r"LIFEMATE_FEATURE_[A-Z0-9_]+")
#: ``${LIFEMATE_FEATURE_X:-1}`` / ``${LIFEMATE_FEATURE_X:=true}``.
FEATURE_DEFAULT_RE = re.compile(r"\$\{(LIFEMATE_FEATURE_[A-Z0-9_]+):?[-=]([^}]*)\}")
#: ``LIFEMATE_FEATURE_X=1`` at the start of a line (an assignment the script makes).
FEATURE_ASSIGN_RE = re.compile(r"^[ \t]*(?:export[ \t]+)?(LIFEMATE_FEATURE_[A-Z0-9_]+)=", re.M)
FALSY = frozenset({"0", "false", "no", "off"})

HAVE_BASH = shutil.which("bash") is not None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _text(path: Path) -> str:
    assert path.is_file(), f"missing {path}"
    return path.read_text(encoding="utf-8")


def _code_lines(text: str) -> list[str]:
    """Lines that are executed: comments and blank lines dropped."""
    return [
        line for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _entrypoint_feature_vars(text: str) -> set[str]:
    """The names in the script's ``FEATURE_VARS=( ... )`` declaration."""
    match = re.search(r"^FEATURE_VARS=\((.*?)^\)", text, re.S | re.M)
    assert match is not None, "entrypoint.sh has no FEATURE_VARS=( ... ) block"
    return set(FEATURE_ENV_RE.findall(match.group(1)))


def _readme_feature_vars(text: str) -> set[str]:
    return set(FEATURE_ENV_RE.findall(text))


def _default_off_violations(text: str) -> list[str]:
    """Every way ``text`` could make a feature default to ON (empty = clean)."""
    violations = [
        f"{name}=${{{name}:-{value}}}"
        for name, value in FEATURE_DEFAULT_RE.findall(text)
        if value.strip().lower() not in FALSY
    ]
    violations += [f"assignment to {m.group(1)}" for m in FEATURE_ASSIGN_RE.finditer(text)]
    return violations


def _stub_path(tmp: Path) -> str:
    """``PATH`` with a fake ``python`` first, printing its argv instead of running."""
    shim = tmp / "shim"
    shim.mkdir(exist_ok=True)
    fake = shim / "python"
    fake.write_text('#!/usr/bin/env bash\nprintf "ARG %s\\n" "$@"\n', encoding="utf-8")
    fake.chmod(0o755)
    return f"{shim}:{os.environ.get('PATH', '')}"


def _run(
    script: Path,
    tmp: Path,
    extra_env: dict[str, str] | None = None,
    args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Run ``script`` with no ``LIFEMATE_FEATURE_*`` in the environment plus extras.

    ``MODEL_ROOT`` points at a directory that does not exist, so the weights
    preflight takes its documented "bare host checkout" branch and the run
    reaches ``exec`` instead of exiting 2 on missing weights.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("LIFEMATE_FEATURE_")}
    env["PATH"] = _stub_path(tmp)
    env["MODEL_ROOT"] = str(tmp / "absent-models")
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(script), *args], capture_output=True, text=True, env=env, cwd=ROOT
    )


def _summary_line(stderr: str) -> str:
    lines = [line for line in stderr.splitlines() if "features=" in line]
    assert lines, f"no feature summary on stderr: {stderr!r}"
    return lines[-1]


# --------------------------------------------------------------------------- #
# static checks
# --------------------------------------------------------------------------- #
def test_entrypoint_passes_bash_syntax_check() -> None:
    """``bash -n`` must accept the script (shellcheck is not installed here)."""
    if not HAVE_BASH:
        print("  SKIP: no bash on this machine; shell syntax unverified")
        return
    run = subprocess.run(["bash", "-n", str(ENTRYPOINT)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    print("  bash -n docker/entrypoint.sh -> 0")


def test_entrypoint_never_installs_or_downloads() -> None:
    """No executable line may install a package or fetch anything."""
    forbidden = {
        "pip install": re.compile(r"\bpip3?\s+install\b"),
        "curl": re.compile(r"\bcurl\b"),
        "wget": re.compile(r"\bwget\b"),
        "git clone": re.compile(r"\bgit\s+clone\b"),
        "hf download": re.compile(r"\b(?:snapshot_download|huggingface-cli)\b"),
    }
    code = "\n".join(_code_lines(_text(ENTRYPOINT)))
    hits = [label for label, pattern in forbidden.items() if pattern.search(code)]
    assert not hits, f"entrypoint.sh runs network/install commands: {hits}"
    print(f"  entrypoint code lines -> none of {sorted(forbidden)}")


def test_dockerfile_extras_default_to_none() -> None:
    """``LIFEMATE_EXTRAS`` must default to empty and gate the extras install."""
    text = _text(DOCKERFILE)
    assert re.search(r'^ARG\s+LIFEMATE_EXTRAS=""', text, re.M), text[:400]
    lines = text.splitlines()
    guard = next(
        (i for i, line in enumerate(lines) if '-n "${LIFEMATE_EXTRAS}"' in line), None
    )
    assert guard is not None, "the extras install is not guarded by a non-empty check"
    install = next(
        (i for i, line in enumerate(lines) if 'pip install ".[${LIFEMATE_EXTRAS}]"' in line),
        None,
    )
    assert install is not None, 'no `pip install ".[${LIFEMATE_EXTRAS}]"` line'
    assert install > guard, "the extras install runs before its non-empty guard"
    # Baseline (the else branch) must stay dependency-free and installed as-is.
    assert any("pip install --no-deps ." in line for line in lines), "baseline install gone"
    print('  Dockerfile -> ARG LIFEMATE_EXTRAS="" and a guarded `pip install ".[...]"`')


def test_feature_vars_are_the_same_in_entrypoint_and_readme() -> None:
    """The documented set and the validated set cannot drift apart."""
    entry = _entrypoint_feature_vars(_text(ENTRYPOINT))
    readme = _readme_feature_vars(_text(README))
    assert entry, "entrypoint advertises no feature variables"
    assert readme == entry, (
        f"only in README: {sorted(readme - entry)}; only in entrypoint: {sorted(entry - readme)}"
    )
    print(f"  {len(entry)} LIFEMATE_FEATURE_* names match in both files")


def test_entrypoint_advertises_exactly_the_registry_ids() -> None:
    """Adding a registry flag without documenting it in the entrypoint must fail."""
    from lifemate.features.overrides import env_name_for
    from lifemate.features.registry import FEATURES

    registry = {env_name_for(spec.id) for spec in FEATURES}
    entry = _entrypoint_feature_vars(_text(ENTRYPOINT))
    assert entry == registry, (
        f"undocumented in entrypoint: {sorted(registry - entry)}; "
        f"not in the registry: {sorted(entry - registry)}"
    )
    assert not any(spec.default for spec in FEATURES), "a registry flag defaults to True"
    # The header doc block may not name a variable the validator does not know.
    mentioned = set(FEATURE_ENV_RE.findall(_text(ENTRYPOINT)))
    assert mentioned == entry, f"stale names in the header docs: {sorted(mentioned - entry)}"
    print(f"  entrypoint names == registry ids ({len(registry)} features, all default False)")


# --------------------------------------------------------------------------- #
# default-off: static + behavioural + negative control
# --------------------------------------------------------------------------- #
def test_feature_defaults_are_all_off_statically() -> None:
    """No truthy ``:-default`` and no assignment of a ``LIFEMATE_FEATURE_*``."""
    violations = _default_off_violations(_text(ENTRYPOINT))
    assert not violations, f"entrypoint defaults a feature ON: {violations}"
    print("  static: no LIFEMATE_FEATURE_* default or assignment in entrypoint.sh")


def test_feature_summary_is_none_with_nothing_set() -> None:
    """A real run with the prefix stripped must print ``features=NONE``."""
    if not HAVE_BASH:
        print("  SKIP: no bash; summary line unverified")
        return
    with tempfile.TemporaryDirectory() as tmp:
        run = _run(ENTRYPOINT, Path(tmp))
    assert run.returncode == 0, run.stderr
    line = _summary_line(run.stderr)
    assert line.endswith("features=NONE"), line
    print(f"  behavioural: {line}")


def test_enabled_feature_is_named_in_the_summary() -> None:
    """The summary is not stuck on NONE: an enabled flag must appear, sorted."""
    if not HAVE_BASH:
        print("  SKIP: no bash; summary line unverified")
        return
    with tempfile.TemporaryDirectory() as tmp:
        run = _run(
            ENTRYPOINT,
            Path(tmp),
            {"LIFEMATE_FEATURE_PROMPT_ITN": "true", "LIFEMATE_FEATURE_ASR_NBEST": "1"},
        )
    assert run.returncode == 0, run.stderr
    line = _summary_line(run.stderr)
    assert line.endswith(
        "features=LIFEMATE_FEATURE_ASR_NBEST,LIFEMATE_FEATURE_PROMPT_ITN"
    ), line
    # The variables are passed through untouched, never rewritten or case-folded.
    assert "LIFEMATE_FEATURE_PROMPT_ITN=TRUE" not in run.stderr, run.stderr
    print(f"  behavioural: {line}")


def test_default_off_check_rejects_a_hardcoded_enable() -> None:
    """Negative control: the default-off check must fail on a mutated copy."""
    original = _text(ENTRYPOINT)
    mutated = original.replace(
        "set -euo pipefail\n",
        'set -euo pipefail\nLIFEMATE_FEATURE_ASR_NBEST="${LIFEMATE_FEATURE_ASR_NBEST:-1}"\n',
        1,
    )
    assert mutated != original, "could not inject the mutation"
    violations = _default_off_violations(mutated)
    assert any("LIFEMATE_FEATURE_ASR_NBEST" in item for item in violations), violations

    if not HAVE_BASH:
        print(f"  static negative control caught it: {violations[0]}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        copy = tmp_path / "entrypoint_mutated.sh"
        copy.write_text(mutated, encoding="utf-8")
        run = _run(copy, tmp_path)
    assert run.returncode == 0, run.stderr
    line = _summary_line(run.stderr)
    assert "features=NONE" not in line, f"mutation went undetected: {line}"
    print(f"  negative control: static {violations[0]!r}, behavioural {line!r}")


# --------------------------------------------------------------------------- #
# early validation (exit 2, before any weight is touched)
# --------------------------------------------------------------------------- #
def test_bad_feature_value_exits_two_and_names_the_variable() -> None:
    if not HAVE_BASH:
        print("  SKIP: no bash; validation unverified")
        return
    with tempfile.TemporaryDirectory() as tmp:
        run = _run(ENTRYPOINT, Path(tmp), {"LIFEMATE_FEATURE_ASR_NBEST": "maybe"})
    assert run.returncode == 2, (run.returncode, run.stdout, run.stderr)
    assert "LIFEMATE_FEATURE_ASR_NBEST" in run.stderr, run.stderr
    assert "maybe" in run.stderr, run.stderr
    assert "1|0|true|false|yes|no" in run.stderr, run.stderr
    assert "ARG " not in run.stdout, "inference started despite an invalid flag"
    print("  LIFEMATE_FEATURE_ASR_NBEST=maybe -> exit 2 naming var, value, accepted set")


def test_unknown_feature_name_exits_two() -> None:
    """``parse_env_overrides`` raises on an unknown name; fail before the weights."""
    if not HAVE_BASH:
        print("  SKIP: no bash; validation unverified")
        return
    with tempfile.TemporaryDirectory() as tmp:
        run = _run(ENTRYPOINT, Path(tmp), {"LIFEMATE_FEATURE_ASR_NBES": "1"})
    assert run.returncode == 2, (run.returncode, run.stdout, run.stderr)
    assert "LIFEMATE_FEATURE_ASR_NBES" in run.stderr, run.stderr
    assert "ARG " not in run.stdout, "inference started despite an unknown flag"
    print("  LIFEMATE_FEATURE_ASR_NBES=1 -> exit 2 naming the unknown variable")


def test_non_default_asr_model_id_requires_the_v15_flag() -> None:
    """Selecting another ASR checkpoint must not silently move the baseline."""
    if not HAVE_BASH:
        print("  SKIP: no bash; validation unverified")
        return
    other = "g-group-ai-lab/gipformer1.5-68M-rnnt"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        refused = _run(ENTRYPOINT, tmp_path, {"ASR_MODEL_ID": other})
        allowed = _run(
            ENTRYPOINT,
            tmp_path,
            {"ASR_MODEL_ID": other, "LIFEMATE_FEATURE_ASR_GIPFORMER_V15": "1"},
        )
        baseline = _run(ENTRYPOINT, tmp_path)
    assert refused.returncode == 2, (refused.returncode, refused.stdout, refused.stderr)
    assert "ASR_MODEL_ID" in refused.stderr and other in refused.stderr, refused.stderr
    assert "LIFEMATE_FEATURE_ASR_GIPFORMER_V15" in refused.stderr, refused.stderr
    assert "ARG " not in refused.stdout, "inference started with an unguarded ASR model id"

    assert allowed.returncode == 0, allowed.stderr
    argv = allowed.stdout.splitlines()
    assert f"ARG asr.model_id={other}" in argv, argv
    # The override must be a separate argv word after `--override` (argparse order).
    assert argv[argv.index(f"ARG asr.model_id={other}") - 1] == "ARG --override", argv

    assert baseline.returncode == 0, baseline.stderr
    assert "ARG asr.model_id=" not in baseline.stdout, baseline.stdout
    print("  non-default ASR_MODEL_ID -> exit 2 alone, --override with the flag, absent by default")

"""Path guard: nothing under a ``*_environment`` directory may be opened.

The competition rules forbid using the organizer-provided simulator
environment when producing predictions. :func:`assert_path_allowed` is used by
every code path in this package that touches dataset files.

Limits of the static scanner
----------------------------
:func:`scan_source_for_prohibited_access` works on **string literals in path
position**: a literal is reported when it contains a separator or is an argument
of ``/``, ``joinpath``, ``os.path.join``, ``Path(...)``/``open(...)`` and one of
its components is a ``*_environment`` name. A path assembled dynamically
(``"_env" + "ironment"``, a name read from config, argv or the environment) is
invisible, and ``docs/`` is not scanned at all. It is a review aid, not a proof.
Runtime enforcement is :func:`assert_path_allowed`.
"""

from __future__ import annotations

import ast
import contextlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import ProhibitedAccessError

__all__ = [
    "ENV_DIR_SUFFIX",
    "is_prohibited_path",
    "assert_path_allowed",
    "safe_open",
    "read_json",
    "read_jsonl",
    "assert_no_environment_access",
    "scan_source_for_prohibited_access",
    "SCAN_DIRS",
]

ENV_DIR_SUFFIX = "_environment"

# Directory names scanned by the static checker, relative to the repo root.
SCAN_DIRS = ("src", "scripts", "cloud", "tests")


# A source line may opt out with this trailing marker when it deliberately
# constructs a prohibited path (guard tests). Keep it rare and obvious.
SCAN_ALLOW_MARKER = "compliance-scan: allow"

_STRING_LITERAL_RE = re.compile(r"""(['"])((?:\\.|(?!\1).)*)\1""")

# Call targets whose string arguments are treated as path components.
PATH_CALL_NAMES = frozenset(
    {"joinpath", "join", "Path", "PurePath", "PosixPath", "WindowsPath", "open", "read_text", "read_bytes"}
)


def _literal_path_components(text: str, whole: bool = False) -> list[str]:
    """Path components of a string literal.

    ``whole=True`` appends the literal itself as a single component, which is
    what makes joined-argument patterns such as
    ``base.joinpath("public_test_environment")`` visible (no separator there).
    """
    parts = text.replace("\\", "/").split("/")
    components = [part.strip() for part in parts if part.strip() not in ("", ".", "..")]
    if whole:
        components.append(text.strip())
    return components


def _is_env_component(component: str) -> bool:
    """True when a component is a ``*_environment`` name (glob stars ignored)."""
    stripped = component.strip().strip("*")
    return stripped == ENV_DIR_SUFFIX or stripped.endswith(ENV_DIR_SUFFIX)


def _path_context_lines(text: str) -> set[int]:
    """Line numbers where a string literal is used as a path component.

    Covers ``a / "x"`` (both operands) and ``joinpath``/``os.path.join``/
    ``Path(...)``/``open(...)`` call arguments. Gating the whole-literal match on
    this keeps ordinary identifiers such as ``"create_environment"`` — a
    function name, not a directory — out of the report.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    lines: set[int] = set()

    def _arg_lines(call: ast.Call, from_index: int = 0) -> None:
        for node in list(call.args)[from_index:] + [kw.value for kw in call.keywords]:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant):
                    lines.add(getattr(sub, "lineno", 0))

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant):
                    lines.add(getattr(sub, "lineno", 0))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in PATH_CALL_NAMES:
                _arg_lines(node, from_index=1 if name == "join" else 0)
    return lines

# Files whose own text defines the rule and therefore mention it.
_SELF_ALLOWLIST = {"compliance.py", "INTERFACES.md", "README.md", "test_compliance.py"}


def _components(path: object) -> list[str]:
    # Path.parts is platform aware but also keeps drive/root anchors; walking
    # the raw string is more forgiving for glob-ish inputs such as
    # "data/*_environment/sample.jsonl".
    text = str(path).replace("\\", "/")
    out: list[str] = []
    for part in text.split("/"):
        if part in ("", ".", ".."):
            continue
        out.append(part)
    return out


def is_prohibited_path(path: object) -> bool:
    """Return True if any component is a ``*_environment`` directory.

    Matches ``train_environment``, ``public_test_environment``,
    ``private_test_environment`` and any ``*_environment`` component at any
    depth, including glob text such as ``data/*_environment``.
    """
    for comp in _components(path):
        # Strip a trailing/leading glob star so `*_environment` and
        # `_environment*` are both recognised.
        normalized = comp.strip("*")
        if normalized == ENV_DIR_SUFFIX or normalized.endswith(ENV_DIR_SUFFIX):
            return True
    return False


def assert_path_allowed(path: object) -> Path:
    """Return ``path`` as :class:`Path`, raising if it is prohibited."""
    if is_prohibited_path(path):
        raise ProhibitedAccessError(path, "reading a *_environment path is prohibited")
    return Path(path)


@contextlib.contextmanager
def safe_open(path: str | Path, mode: str = "r", **kwargs: Any) -> Iterator[Any]:
    """``open()`` wrapper that refuses prohibited paths first."""
    allowed = assert_path_allowed(path)
    with open(allowed, mode, **kwargs) as handle:  # noqa: PTH123
        yield handle


def read_json(path: str | Path) -> Any:
    """Read JSON through the compliance guard."""
    with safe_open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[Any]:
    """Read a JSONL file through the compliance guard (blank lines skipped)."""
    rows: list[Any] = []
    with safe_open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def assert_no_environment_access(paths: Iterable[object]) -> None:
    """Raise on the first prohibited path in ``paths``."""
    for candidate in paths:
        assert_path_allowed(candidate)


def scan_source_for_prohibited_access(
    root: str | Path,
    scan_dirs: Iterable[str] = SCAN_DIRS,
    allowlist: Iterable[str] | None = None,
) -> list[dict]:
    """Statically flag literal ``*_environment`` path usages in source files.

    A string literal is reported when it is used as a **path component** —
    either it contains a separator (``"data/train_environment/x.json"``,
    ``"data/*_environment"``) *or* it sits in path position
    (``base / "train_environment"``, ``base.joinpath("public_test_environment")``,
    ``os.path.join(base, "private_test_environment")``, ``Path("train_environment")``)
    — and one of its components is a ``_environment`` name.

    The path-position gate is what keeps ordinary identifiers such as
    ``"create_environment"`` (a function name) out of the report while still
    catching the bare-literal join that an accidental violation would use.

    Returns a list of ``{"file", "line", "text", "reason"}`` findings.
    Comments, ``# compliance-scan: allow`` lines, and allowlisted file names
    (the guard module, docs, the compliance test) are skipped.

    Limits: detection is literal-based. A path assembled dynamically
    (``"_env" + "ironment"``, a name read from config or argv) is invisible
    here. ``docs/`` is not scanned. This is a review aid; runtime enforcement is
    :func:`assert_path_allowed`.
    """
    root_path = Path(root)
    allowed_names = set(_SELF_ALLOWLIST) | {Path(a).name for a in (allowlist or ())}
    findings: list[dict] = []
    for sub in scan_dirs:
        base = root_path / sub
        if not base.is_dir():
            continue
        for file_path in sorted(base.rglob("*")):
            if not file_path.is_file() or file_path.suffix not in {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".toml"}:
                continue
            if file_path.name in allowed_names:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            path_lines = _path_context_lines(text) if file_path.suffix == ".py" else set()
            for lineno, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or SCAN_ALLOW_MARKER in line:
                    continue
                for match in _STRING_LITERAL_RE.finditer(line):
                    literal = match.group(2)
                    joined = "/" in literal or "\\" in literal
                    if not joined and lineno not in path_lines:
                        continue  # bare identifier, not a path position
                    # The whole literal counts as one component so
                    # joinpath("public_test_environment") is caught too.
                    if not any(_is_env_component(comp) for comp in _literal_path_components(literal, whole=True)):
                        continue
                    findings.append(
                        {
                            "file": str(file_path.relative_to(root_path)),
                            "line": lineno,
                            "text": stripped[:200],
                            "reason": "literal *_environment path in source",
                        }
                    )
                    break
    return findings

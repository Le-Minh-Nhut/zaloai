"""Dynamic split discovery under a data root.

Nothing here hard-codes split names: whatever ``<root>/<name>/<name>.jsonl``
(or ``<root>/<name>.jsonl``) exists is offered.

``*_environment`` directories are NEVER listed or opened. The only permitted
touching is a boolean ``Path.is_dir()`` existence check, recorded as
``SplitPaths.has_environment_dir`` for reporting ("is the organizer env
folder present?"). That boolean must never be used to open files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..compliance import ENV_DIR_SUFFIX
from ..errors import DatasetNotFoundError

__all__ = [
    "SplitPaths",
    "discover_splits",
    "require_split",
    "available_splits",
    "catalog_candidates",
    "sample_submission_candidates",
    "is_environment_dir",
]

CATALOG_SUFFIX = "_tools_catalog.json"
GENERIC_CATALOG = "tools_catalog.json"
SAMPLE_SUBMISSION = "sample_submission.jsonl"
AUDIO_DIR = "audio"


@dataclass
class SplitPaths:
    """Filesystem layout of one split, with optional members as ``None``."""

    split: str
    root: Path
    jsonl: Path | None = None
    audio_dir: Path | None = None
    catalog: Path | None = None
    sample_submission: Path | None = None
    has_environment_dir: bool = False

    def as_dict(self) -> dict:
        """JSON-friendly snapshot (paths as strings)."""
        return {
            "split": self.split,
            "root": str(self.root),
            "jsonl": None if self.jsonl is None else str(self.jsonl),
            "audio_dir": None if self.audio_dir is None else str(self.audio_dir),
            "catalog": None if self.catalog is None else str(self.catalog),
            "sample_submission": None if self.sample_submission is None else str(self.sample_submission),
            "has_environment_dir": self.has_environment_dir,
        }


def is_environment_dir(path: Path) -> bool:
    """True when ``path``'s name is a ``*_environment`` directory name."""
    return path.name.endswith(ENV_DIR_SUFFIX)


def catalog_candidates(split_dir: Path, split: str) -> list[Path]:
    """Conventional catalog paths for a split, most specific first."""
    return [
        split_dir / f"{split}{CATALOG_SUFFIX}",
        split_dir / GENERIC_CATALOG,
    ]


def sample_submission_candidates(split_dir: Path) -> list[Path]:
    """Conventional sample-submission paths."""
    return [split_dir / SAMPLE_SUBMISSION, split_dir.parent / SAMPLE_SUBMISSION]


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _build_split(name: str, root: Path, jsonl: Path | None) -> SplitPaths:
    candidates = catalog_candidates(root, name)
    submission = _first_existing(sample_submission_candidates(root))
    audio_dir = root / AUDIO_DIR
    return SplitPaths(
        split=name,
        root=root,
        jsonl=jsonl,
        audio_dir=audio_dir if audio_dir.is_dir() else None,
        catalog=_first_existing(candidates),
        sample_submission=submission,
        # Existence check only. Opening it is prohibited and guarded elsewhere.
        # Official layout nests it as ``<root>/<name>_environment``; the flat
        # fixture layout keeps ``<name>_environment`` beside the sample files.
        has_environment_dir=(
            (root / f"{name}{ENV_DIR_SUFFIX}").is_dir()
            or (root.parent / f"{name}{ENV_DIR_SUFFIX}").is_dir()
        ),
    )


def discover_splits(data_root: str | Path) -> dict[str, SplitPaths]:
    """Scan ``data_root`` for splits; returns ``{split: SplitPaths}``.

    Split directories are only considered when they contain either
    ``<name>/<name>.jsonl`` or a catalog. ``*_environment`` directories are
    skipped before any listing.
    """
    root = Path(data_root)
    found: dict[str, SplitPaths] = {}
    if not root.is_dir():
        return found
    for child in sorted(root.iterdir()):
        if is_environment_dir(child):
            continue
        if not child.is_dir():
            continue
        name = child.name
        nested = child / f"{name}.jsonl"
        jsonl = nested if nested.is_file() else None
        if jsonl is None and _first_existing(catalog_candidates(child, name)) is None:
            continue
        found[name] = _build_split(name, child, jsonl)
    for child in sorted(root.iterdir()):
        if is_environment_dir(child) or child.suffix != ".jsonl":
            continue
        name = child.stem
        if name in found or name == sample_submission_candidates(root)[0].stem:
            continue
        found.setdefault(name, _build_split(name, root, child))
    return found


def available_splits(data_root: str | Path) -> list[str]:
    """Sorted split names discovered under ``data_root``."""
    return sorted(discover_splits(data_root))


def require_split(data_root: str | Path, split: str) -> SplitPaths:
    """Return the split layout or raise :class:`DatasetNotFoundError`."""
    splits = discover_splits(data_root)
    if split in splits:
        return splits[split]
    raise DatasetNotFoundError(split, str(Path(data_root) / split))

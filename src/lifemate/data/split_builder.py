"""Deterministic train/val splitting with a near-duplicate leakage guard.

Order of operations matters and is the whole point of this module:

1. **Group first.** Samples whose normalised request/history hash collides are
   placed in one group (``preprocessing.sample_hash``). A group is atomic, so a
   near-duplicate request can never appear on both sides of the split.
2. **Stratify groups.** Groups are bucketed by feature keys (default
   ``n_calls_bucket`` + ``categories``); each bucket with at least two groups
   contributes ``round(ratio * bucket)`` groups (clamped to keep ≥1 train, and
   ≥1 val when the bucket is big enough). Buckets too small to stratify fall
   back to a seeded random draw over the leftover groups.
3. **Assert disjointness.** ``TrainVal.__post_init__`` re-derives group hashes
   and hard-fails on any overlap.

Deterministic for a fixed (samples, ratio, seed): iteration is always over
sorted keys/hashes, so no dict- or set-ordering leaks into the result.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from lifemate.errors import LifeMateError

from .preprocessing import hash_ids, sample_features, sample_hash

__all__ = ["TrainVal", "split_samples"]

DEFAULT_STRATIFY_KEYS = ("n_calls_bucket", "categories")

def _balanced_val_count(n_groups: int, ratio: float) -> int:
    """Groups to send to val from a stratum of ``n_groups`` (0 keeps everything)."""
    if ratio <= 0 or n_groups < 2:
        return 0
    target = int(round(ratio * n_groups))
    return max(1, min(n_groups - 1, target))

@dataclass
class TrainVal:
    """Split result. ``train``/``val`` are sample lists; groups never overlap."""

    train: list[Any]
    val: list[Any]
    seed: int = 42
    ratio: float = 0.1
    groups: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.assert_no_overlap()

    def _hashes(self, split: str) -> set[str]:
        samples = self.train if split == "train" else self.val
        return {sample_hash(sample) for sample in samples}

    def assert_no_overlap(self) -> None:
        """Raise when a duplicate group appears in both splits."""
        shared = self._hashes("train") & self._hashes("val")
        if shared:
            raise LifeMateError(
                f"split leakage: {len(shared)} duplicate group(s) present in both train and val "
                f"(e.g. {sorted(shared)[0]}); group before splitting"
            )

    def manifest(self) -> dict[str, Any]:
        """Counts + seed + ratio + id/group fingerprints (the split fingerprint)."""
        train_hashes = self._hashes("train")
        val_hashes = self._hashes("val")
        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_groups": len(train_hashes | val_hashes),
            "seed": self.seed,
            "ratio": self.ratio,
            "train_ids_hash": hash_ids(s.id for s in self.train),
            "val_ids_hash": hash_ids(s.id for s in self.val),
            "group_hash": hash_ids(train_hashes | val_hashes),
        }

def split_samples(
    samples: Sequence[Any],
    ratio: float = 0.1,
    seed: int = 42,
    stratify_keys: Iterable[str] = DEFAULT_STRATIFY_KEYS,
) -> TrainVal:
    """Split ``samples`` into train/val deterministically (see module docstring)."""
    if not 0.0 <= float(ratio) < 1.0:
        raise LifeMateError(f"validation ratio must be in [0, 1), got {ratio!r}")
    keys = tuple(stratify_keys)

    # 1. group by duplicate hash (first occurrence order is irrelevant: sorted below)
    groups: dict[str, list[Any]] = {}
    for sample in samples:
        groups.setdefault(sample_hash(sample), []).append(sample)

    val_hashes: set[str] = set()
    small_strata: list[str] = []

    # 2. stratified draw per feature bucket
    strata: dict[str, list[str]] = {}
    for group_hash in sorted(groups):
        features = sample_features(groups[group_hash][0], keys)
        stratum = "|".join(f"{k}={features[k]}" for k in keys)
        strata.setdefault(stratum, []).append(group_hash)

    rng = random.Random(seed)
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        count = _balanced_val_count(len(members), float(ratio))
        if count == 0:
            if len(members) < 2 and ratio > 0 and len(strata) > 1:
                small_strata.append(stratum)
            continue
        picked = rng.sample(members, count)  # deterministic given the seeded rng + sorted members
        val_hashes.update(picked)

    # 3. buckets too small to stratify: seeded random draw over what is left
    if small_strata:
        leftover = sorted(strata[s][0] for s in small_strata)
        budget = max(0, int(round(ratio * len(groups))) - len(val_hashes))
        budget = min(budget, max(0, len(leftover) - 1))
        if budget:
            val_hashes.update(rng.sample(leftover, budget))

    train: list[Any] = []
    val: list[Any] = []
    for group_hash in sorted(groups):
        target = val if group_hash in val_hashes else train
        target.extend(groups[group_hash])

    if not val and ratio > 0 and len(groups) >= 2:
        # guarantee a non-empty val set on tiny datasets (still group-atomic)
        moved = sorted(groups)[-1]
        train = [s for s in train if sample_hash(s) != moved]
        val = list(groups[moved])

    return TrainVal(
        train=train,
        val=val,
        seed=seed,
        ratio=float(ratio),
        groups={g: [s.id for s in members] for g, members in groups.items()},
    )

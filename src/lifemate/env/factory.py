"""Environment selection.

Mock is opt-in only: ``use_mock`` must be passed explicitly. Competition
inference MUST use ``use_mock=False`` so predictions come from the official
simulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..data.catalog import ToolCatalog
from .base import ToolEnvironment
from .mock import MockToolEnv
from .official import OfficialToolEnv

__all__ = ["create_environment"]


def create_environment(
    split: str,
    sample_id: str,
    data_root: str | Path,
    use_mock: bool = False,
    catalog: ToolCatalog | None = None,
    mock_state: dict | None = None,
    apps: list[str] | None = None,
) -> ToolEnvironment:
    """Return an official or mock environment.

    Competition inference MUST use ``use_mock=False`` (the default); the mock
    path exists only for synthetic dry runs and requires ``use_mock=True``.
    ``mock_state`` is the synthetic initial state from ``tests/fixtures`` —
    never state read from a ``*_environment`` directory.
    """
    if use_mock:
        return MockToolEnv(
            catalog=catalog,
            initial_state=mock_state,
            apps=apps,
            sample_id=sample_id,
        )
    return OfficialToolEnv(split=split, sample_id=sample_id, data_root=data_root, catalog=catalog)

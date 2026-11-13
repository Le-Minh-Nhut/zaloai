"""Shared test setup: make ``src/`` importable without installation.

No fixtures are defined here on purpose — every test takes zero arguments so
``python3 tests/run_tests.py`` works with the stdlib alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

FIXTURES = ROOT / "tests" / "fixtures" / "synthetic_lifemate"

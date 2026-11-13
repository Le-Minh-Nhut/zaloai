"""``python -m lifemate``: thin dispatcher over the scripts in ``scripts/``."""

from __future__ import annotations

import sys

from .cli import COMMANDS, main, script_main

if __name__ == "__main__":
    sys.exit(main())

__all__ = ["COMMANDS", "main", "script_main"]

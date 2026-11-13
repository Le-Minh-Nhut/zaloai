"""``python -m lifemate <command>`` dispatcher.

Each command maps to the ``main(argv)`` of the matching script under
``scripts/``; imports are lazy so ``python -m lifemate --help`` stays instant
and works without torch. Scripts live outside the package (they are thin
entrypoints), so they are loaded by path.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Sequence

__all__ = ["COMMANDS", "main", "script_main"]

#: ``command -> script file`` (resolved relative to the repo root).
COMMANDS: dict[str, str] = {
    "inspect-data": "inspect_data.py",
    "eda": "run_eda.py",
    "prepare": "prepare_dataset.py",
    "train": "train.py",
    "infer": "infer.py",
    "evaluate": "evaluate.py",
    "export": "export_model.py",
    "download-checkpoint": "download_checkpoint.py",
    "verify": "verify_all.py",
}


def _scripts_dir() -> Path:
    """``<repo>/scripts`` (works from an installed or a source checkout)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "scripts"
        if candidate.is_dir() and (candidate / "infer.py").is_file():
            return candidate
    return here.parents[2] / "scripts"


def script_main(script: str) -> Callable[[Sequence[str] | None], int]:
    """Import ``scripts/<script>`` by path and return its ``main``."""

    def _load() -> Callable[[Sequence[str] | None], int]:
        path = _scripts_dir() / script
        if not path.is_file():
            raise SystemExit(f"ERROR: script not found: {path}")
        module_name = f"lifemate_script_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
            raise SystemExit(f"ERROR: cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        # Make the file importable (dataclasses/pickling need it registered).
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        main_fn = getattr(module, "main", None)
        if not callable(main_fn):
            raise SystemExit(f"ERROR: {path} has no main(argv)")
        return main_fn

    def runner(argv: Sequence[str] | None = None) -> int:
        return int(_load()(argv) or 0)

    return runner


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command and dispatch; returns the script's exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m lifemate",
        description="LifeMate Track 1 tooling. Each command forwards to scripts/<name>.py.",
    )
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS), help="command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the command")
    parsed = parser.parse_args(argv)

    if not parsed.command:
        parser.print_help()
        print("\ncommands:")
        for name, script in COMMANDS.items():
            print(f"  {name:20s} scripts/{script}")
        return 0

    return script_main(COMMANDS[parsed.command])(parsed.args)


__all__ = ["COMMANDS", "main", "script_main"]

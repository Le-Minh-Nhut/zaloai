#!/usr/bin/env python3
"""Minimal stdlib test runner (no pytest needed).

Collects ``tests/test_*.py``, imports each module, and runs every
``def test_*()`` with no required parameters. Prints a PASS/FAIL summary and
exits 1 when anything fails.

Usage: python3 tests/run_tests.py [pattern ...]
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def _import(path: Path):
    name = f"lifemate_tests_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _needs_args(func) -> bool:
    for parameter in inspect.signature(func).parameters.values():
        if parameter.default is inspect.Parameter.empty and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def main(argv: list[str]) -> int:
    """Run all collected tests; return the process exit code."""
    for path in (str(ROOT / "src"), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    patterns = argv or ["test_*.py"]
    files = sorted({path for pattern in patterns for path in TESTS.glob(pattern)})
    if not files:
        print("no test files found")
        return 1
    passed = failed = skipped = 0
    failures: list[tuple[str, str]] = []
    skipped_modules: list[str] = []
    for file_path in files:
        try:
            module = _import(file_path)
        except (KeyboardInterrupt, SystemExit):
            raise
        except ModuleNotFoundError as exc:
            if "pytest" in str(exc):
                print(f"SKIP {file_path.name}: needs pytest (stdlib-only runner)")
                skipped += 1
                skipped_modules.append(file_path.name)
                continue
            failed += 1
            failures.append((f"{file_path.name} (import)", traceback.format_exc()))
            continue
        except AssertionError:
            failed += 1
            failures.append((f"{file_path.name} (import)", traceback.format_exc()))
            continue
        except Exception:
            failed += 1
            failures.append((f"{file_path.name} (import)", traceback.format_exc()))
            continue
        except BaseException as exc:
            print(f"SKIP {file_path.name}: {exc}")
            skipped_modules.append(file_path.name)
            continue
        if getattr(module, "HAVE_DEPS", True) is False:
            test_funcs = [attr for attr in sorted(vars(module)) if attr.startswith("test_")]
            count = len(test_funcs) or 1
            skipped += count
            skipped_modules.append(file_path.name)
            continue
        for attr in sorted(vars(module)):
            if not attr.startswith("test_"):
                continue
            func = getattr(module, attr)
            if not callable(func) or _needs_args(func):
                skipped += 1
                continue
            label = f"{file_path.name}::{attr}"
            try:
                with redirect_stdout(io.StringIO()):
                    func()
            except (KeyboardInterrupt, SystemExit):
                raise
            except AssertionError:
                failed += 1
                failures.append((label, traceback.format_exc()))
            except Exception:
                failed += 1
                failures.append((label, traceback.format_exc()))
            except BaseException as exc:
                print(f"SKIP {label}: {exc}")
                skipped += 1
                continue
            else:
                passed += 1
    for label, tb in failures:
        print(f"\nFAIL {label}\n{tb}")
    print(f"\npassed={passed} failed={failed} skipped(needs fixtures)={skipped}")
    print(f"skipped modules: {skipped_modules}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

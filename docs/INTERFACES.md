# LifeMate interfaces

Contract for every module in `src/lifemate/`. Written by the foundation worker;
other workers code against it.

Status of this repo: **the official dataset, model weights, GPU and the
`zaic_tools` SDK are not present.** Nothing in this document describes an
official schema — the real record format is unknown and the parsers are
deliberately tolerant. Synthetic fixtures live in
`tests/fixtures/synthetic_lifemate/` and are labelled as fake everywhere.

## Environment probe (recorded 2026-11-07)

| item | result |
|---|---|
| `python3 -V` | `Python 3.12.3` |
| `python3 -m pip` | **absent** (`No module named pip`; system Python is externally managed) |
| network | reachable (`https://pypi.org/simple/numpy/` → `200`) |
| `uv` | 0.11.7 available (`~/.local/bin/uv`) |
| importable on system `python3` | `yaml` 6.0.1 only |
| missing on system `python3` | `numpy`, `matplotlib`, `pytest`, `torch`, `transformers`, `peft`, `soundfile`, `datasets` |
| installed in `.venv` (via `uv pip install`) | `pytest` 9.2.0, `pyyaml` 6.0.3, `numpy` 2.5.4, `matplotlib` 3.11.2 |
| still missing everywhere | `torch`, `transformers`, `peft`, `soundfile`, `datasets`, `zaic_tools` |

Consequence: **all foundation code imports and tests on bare `python3`** — no
`numpy`, `yaml` or `pytest` needed. Heavy dependencies are imported lazily inside
functions, and `import lifemate.data` pulls in none of them. Two ways to run the
suite:

```
.venv/bin/python -m pytest -q          # pytest 9.2.0, full suite
python3 tests/run_tests.py            # stdlib-only runner, no pytest required
```

`pyyaml` is optional. Without it, `lifemate.config` falls back to a built-in
YAML-subset parser and dumps JSON text for `dump_config`; `load_config` accepts
that JSON-in-`.yaml`.

---

## `lifemate.errors`

```python
class LifeMateError(Exception)
class DatasetNotFoundError(LifeMateError)      # ctor (split: str, expected_path: str)
class ProhibitedAccessError(LifeMateError)     # ctor (path, reason=...)
class ConfigError(LifeMateError)
class SchemaError(LifeMateError)
```

`DatasetNotFoundError.__str__` is fixed text (blank lines significant):

```
Dataset not found.

Expected directory:
<expected_path>

Please extract the official LifeMate dataset
into the configured data directory.

Implementation is available.
Real-data validation has not yet been performed.
```

CLIs catch it and exit `2` **without** a traceback.

## `lifemate.config`

```python
load_config(path: str | Path | None = None, overrides: dict | None = None) -> AppConfig
dump_config(cfg: AppConfig, path: str | Path) -> Path
```

- Defaults are complete: `load_config()` with no file and no arguments succeeds.
- `overrides` uses dotted keys (`"training.tuning_mode"`, `"model.name"`,
  `"training.num_train_epochs"`); values are coerced to the dataclass field type.
  Unknown section/key → `ConfigError`.
- Sections: `data, model, lora, training, asr, agent, inference, evaluation,
  wandb, cloud` (fields exactly as in `src/lifemate/config.py`).
- `training.tuning_mode` ∈ `{full, lora}`, **defaults to `full`**; anything else
  raises `ConfigError`. Same for `validation_ratio ∈ [0,1)`,
  `asr.sample_rate > 0`, `agent.max_steps >= 1`.
- `config.to_dict()` → nested plain dict. `config.validate()` is called on load.
- `configs/*.yaml` (owned elsewhere) load through the same path; both YAML and
  JSON-in-`.yaml` are accepted.

## `lifemate.compliance`

```python
ENV_DIR_SUFFIX = "_environment"
is_prohibited_path(path: object) -> bool
assert_path_allowed(path: object) -> Path
safe_open(path, mode="r", **kw) -> ContextManager[IO]
read_json(path) -> Any
read_jsonl(path) -> list[Any]
assert_no_environment_access(paths: Iterable[object]) -> None
scan_source_for_prohibited_access(root, scan_dirs=("src","scripts","cloud","tests"),
                                  allowlist=None) -> list[dict]
```

- Rule: any path **component** equal to or ending in `_environment`
  (`train_environment`, `public_test_environment`,
  `private_test_environment`, `data/*_environment/...`, nested at any depth) is
  prohibited. `environmental_science/` is *not* matched.
- Everything that opens dataset files goes through `safe_open`/`assert_path_allowed`.
  Violations raise `ProhibitedAccessError` (attributes `.path`, `.reason`).
- Static scan: works on **string literals in path position**. A literal is
  reported when (a) it contains a separator — `"data/train_environment/x.json"`,
  `"data/*_environment"` — or (b) it sits where a path component is built:
  `base / "train_environment"`, `base.joinpath("public_test_environment")`,
  `os.path.join(base, "private_test_environment")`, `Path("train_environment")`,
  `open("train_environment/...")`; and one of its components — including the
  whole literal — equals or ends with `_environment` after stripping glob stars.
  Bare identifiers such as `"create_environment"` (a function name) or
  `"environmental_science"` are therefore **not** flagged. Findings are
  `[{"file", "line", "text", "reason"}]`; a trailing `# compliance-scan: allow`
  marker suppresses one line (used by the guard tests that must build a
  prohibited path on purpose), and comments / allowlisted file names are skipped.
  **Limits:** detection is literal-based, so a path assembled dynamically
  (`"_env" + "ironment"`, a name read from config, argv or the environment) is
  invisible, and `docs/` is not scanned. This cannot prove the absence of
  dynamic access — runtime enforcement is `assert_path_allowed`.

## `lifemate.data.discovery`

```python
@dataclass SplitPaths:
    split: str; root: Path; jsonl: Path|None; audio_dir: Path|None
    catalog: Path|None; sample_submission: Path|None; has_environment_dir: bool
    def as_dict(self) -> dict

discover_splits(data_root) -> dict[str, SplitPaths]
available_splits(data_root) -> list[str]
require_split(data_root, split) -> SplitPaths     # raises DatasetNotFoundError
```

Dynamic: a directory is a split when it contains `<name>/<name>.jsonl` or a
catalog; `<root>/<name>.jsonl` is also accepted. Catalog lookup is
`<split>_tools_catalog.json` then `tools_catalog.json`. Nothing is hard-coded,
so `private_test` (absent) is simply not discovered.

`*_environment` directories are **never listed or opened**. The single
permitted touch is `Path.is_dir()` recorded as `has_environment_dir`; that
boolean must not be used to open anything.

## `lifemate.data.schemas`

Dataclasses, all tolerant, all keeping the original record in `.raw`:

```python
Context(current_time, timezone, current_location, user_profile: dict, installed_apps: list[str])
HistoryTurn(role, content, tool_call: dict|None, tool_result: dict|None, raw: dict)
ToolCall(name, arguments: dict)         # .to_dict() -> {"name", "arguments"}
ToolResult(name, ok, result, error)     # .from_raw(), .to_dict()
Label(tool_calls, tool_results, final_state, is_model_input=False, raw)
Sample(id, split, audio_path: Path|None, audio_text, context: Context,
       history: list[HistoryTurn], label: Label|None, raw: dict)

Sample.from_raw(raw, split, data_root)  # data_root = the SPLIT directory
Sample.installed_apps -> list[str]
Sample.to_model_input() -> dict          # no label / final_state / tool_calls
parse_context / parse_history / parse_tool_calls / parse_tool_results
```

Accepted shapes include `role|speaker` + `content|utterance`, `tool_calls`,
`tool_call`/`arguments`, `tool_result|result|observation`, `history|messages|
conversation|turns|dialog`, `audio|audio_path|wav|path`, `audio_text|
transcript|text|asr_text`. Unknown fields are kept, never rejected.

`Label.final_state` is parsed for scoring only and is flagged
`is_model_input = False`. `Sample.to_model_input()` is the single source of
model-visible content.

## `lifemate.data.loader`

```python
iter_raw_records(path, skip_malformed=False) -> Iterator[dict]
load_raw_split(data_root, split, limit=None, skip_malformed=False) -> list[dict]
load_samples(data_root, split, limit=None, with_labels=True) -> list[Sample]
load_catalog(data_root, split) -> ToolCatalog|None
read_sample_audio(sample) -> Path|None
```

Malformed JSONL lines become `{"__line__": n, "__error__": msg, "__raw__": ...}`
unless `skip_malformed=True`. Missing split → `DatasetNotFoundError`
(no traceback). `read_sample_audio` returns `None` for a missing file and raises
`ProhibitedAccessError` for a prohibited path.

## `lifemate.data.catalog`

```python
ToolCatalog.from_dict(data) / .from_file(path)
    .apps: list[str]
    .tools: dict[str, ToolSpec]
    .tools_for_apps(apps, include_meta=True) -> list[ToolSpec]
    .to_schema_prompt(apps=None, include_meta=True) -> str
    .validate_call(name, args) -> {"errors": [...], "warnings": [...]}
    .dependency_edges() -> list[(tool, ref, field)]
ToolSpec(name, app_name, description, parameters: dict, kind, annotations)
    .required -> list[str]; .properties -> dict
```

- `kind` is inferred: `sys.*`/`ask_missing*` → `meta`;
  `readOnlyHint` or a read verb leaf (`get|list|search|read|find|query|status|
  check|lookup|retrieve`) → `read`; `destructiveHint` or anything else → `write`.
- Accepted containers: `apps:[{app_name, tools:[...]}]`, `tools:[...]`,
  `tools:{name: {...}}`, OpenAI `tools:[{type:"function", function:{...}}]`.
- Extensions: `x-ref`, `x-match`, `x-one-of` (and `oneOf`, both on a property and
  on `parameters`).
- `validate_call` returns human-readable problems: unknown tool, missing
  required, wrong type, enum violation, `x-one-of` violation, unknown argument
  (warning), unresolvable `x-ref` (**warning, not error**).
- `tools_for_apps` always includes `sys.ask_missing_info` when the catalog has it
  and `include_meta=True`.

## `lifemate.data.validator`

```python
validate_split(paths: SplitPaths, samples=None, catalog=None) -> dict
parse_record_safely(record, split, split_dir=None) -> Sample|None
```

Report keys: `split, root, jsonl, catalog, has_environment_dir, counts,
categories, duplicate_ids, missing_audio, schema_issues, malformed_rows,
unknown_top_level_fields, field_presence, errors, warnings,
gold_dependency_edges`. `errors` break the benchmark contract (duplicate ids,
missing audio, unknown/uninstalled/invalid gold calls); `warnings` describe
unrecognised shape — expected while real data is absent.

Privacy: only counts, field names and problem ids are reported. `user_profile`
values, contacts and message bodies never enter a report.

## `lifemate.env`

```python
class ValidationOutcome: ok: bool; errors: list[str]
class ToolEnvironment(Protocol):      # runtime_checkable
    name: str
    validate(tool_name, arguments) -> ValidationOutcome
    call(tool_name, arguments) -> dict      # {"ok": True, "result": ...} | {"ok": False, "error": ...}
    get_tool_calls() -> list[dict]
    get_state_hash() -> str
    available_tools() -> list[str]

MockToolEnv(catalog=None, initial_state=None, apps=None, sample_id="mock", fail_tools=None)
OfficialToolEnv(split, sample_id, data_root, catalog=None)
create_environment(split, sample_id, data_root, use_mock=False, catalog=None,
                   mock_state=None, apps=None) -> ToolEnvironment
```

`MockToolEnv` is deterministic: state is seeded only from `initial_state`
(synthetic fixture dicts, never a `*_environment` directory), every call is
appended to the log and returned verbatim by `get_tool_calls()`, and
`get_state_hash()` is sha256 over canonical JSON of state excluding the write
transcript — it changes only on successful writes. Validation precedes
execution; unknown tool → `{"ok": False, "error": "unknown tool: <name>"}`.

`OfficialToolEnv` imports `zaic_tools` lazily in `__init__` and raises
`LifeMateError("zaic_tools not installed ...")` when absent. It asserts the
backend class comes from a module starting with `zaic_tools`. No mock fallback.

`create_environment(..., use_mock=False)` is the competition path. Mock is
opt-in only; never implicit.

## `scripts/inspect_data.py`

```
python3 scripts/inspect_data.py --data-root ./data [--split train ...] [--output-dir outputs] [--limit N]
```

Writes `outputs/data_inspection.json` and `outputs/data_inspection.md`. Exit
codes: `0` ok, `1` unexpected failure (short message, no traceback), `2` dataset
not found (prints the exact `DatasetNotFoundError` text). Importable as
`inspect_data.main([...]) -> int`.

## How other workers must integrate

1. **Data access**: use `lifemate.data.load_samples` / `load_catalog` /
   `discover_splits`. Do not `open()` dataset files directly — go through
   `lifemate.compliance.safe_open` so the `*_environment` guard applies.
2. **Model input**: build prompts from `Sample.to_model_input()` only. Never
   read `sample.label` outside scoring code, and never render `final_state`.
3. **Tool schemas**: filter with `catalog.tools_for_apps(sample.installed_apps)`
   and validate with `catalog.validate_call(...)`; treat `warnings` as advisory.
4. **Environments**: pass a `ToolEnvironment`. Use
   `create_environment(..., use_mock=False)` for anything that produces a
   submission; `use_mock=True` only for synthetic dry runs.
5. **Config**: take an `AppConfig` (never re-read env vars in your module).
   New settings require a field in `config.py` plus a `configs/*.yaml` entry.
6. **Missing data**: catch `DatasetNotFoundError` and exit 2 with the given
   message. Do not invent data or download anything.
7. **Fixtures**: `tests/fixtures/synthetic_lifemate/` is fake, generated by
   `tests/fixtures/synthetic_lifemate/_generate.py`. Regenerate with
   `python3 tests/fixtures/synthetic_lifemate/_generate.py`. Never present its
   numbers as competition results.

## Known divergences from the original brief

- `validate_call` returns `{"errors": [...], "warnings": [...]}` (the brief
  allowed either that or a tuple).
- `create_environment` accepts an extra optional `apps` argument, forwarded to
  `MockToolEnv`.
- `ToolEnvironment.validate` for the official adapter falls back to
  catalog-based validation when the SDK has no `validate` method, because the
  real SDK surface is unknown.
- `Sample.from_raw(..., data_root)` expects the **split directory** (e.g.
  `data/train`), not the dataset root, since audio is resolved relative to it.
- The static scanner works on string literals in path position (separator in the
  literal, or a `/` operand / `joinpath` / `os.path.join` / `Path(...)` /
  `open(...)` argument) and supports a `# compliance-scan: allow` marker;
  otherwise guard tests could not name prohibited paths. Bare-literal joins such
  as `base.joinpath("public_test_environment")` **are** detected; dynamically
  constructed names are not.

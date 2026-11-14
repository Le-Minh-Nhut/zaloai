"""Optional training-data augmentations (all OFF by default, all identity when off).

Two transforms, both strictly training-only, both *conservative*: an example
that cannot be proven consistent is skipped and counted with its rule number,
never silently rewritten.

* :func:`apply_slot_masking` — drop a required argument of the gold call and
  supervise ``sys.ask_missing_info`` instead, only when the dropped value is a
  literal in the request text and nowhere else in the prompt.
* :func:`apply_tool_dropout` — remove an unused tool from the rendered catalog
  section so the model practises choosing among fewer tools.

Both delegate rendering to :mod:`lifemate.agent.prompt_builder` (the single
source of prompt bytes) and return a report dict. A record's ``target`` is only
ever replaced by :func:`apply_slot_masking`, and only with a schema-valid
``sys.ask_missing_info`` call.

``apply_tool_dropout`` MUST never be used at inference time: the inference tool
registry is derived from the catalog by ``ToolRegistry``, and filtering it there
would hide callable tools from the agent. This transform only edits the prompt
bytes stored inside a training record.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from lifemate.agent.prompt_builder import _render_tools
from lifemate.agent.protocol import ACTION_OPEN, make_tool_call, parse_action, render_action
from lifemate.agent.tool_registry import META_TOOL_NAME, ToolRegistry
from lifemate.data.catalog import META_TOOL_NAMES
from lifemate.errors import LifeMateError

__all__ = [
    "HISTORY_HEADER",
    "META_TOOL_CANDIDATES",
    "NBEST_HEADER",
    "REQUEST_HEADER",
    "apply_slot_masking",
    "apply_tool_dropout",
    "prompt_section",
    "prompt_sections",
    "request_section",
    "replace_first_section",
    "replace_section_body",
]

#: Names the meta tool may carry in a catalog (``sys`` app or a bare alias).
META_TOOL_CANDIDATES = tuple(sorted(META_TOOL_NAMES))

REQUEST_HEADER = "# Yêu cầu"
#: Alternative request header used by the N-best training format.
NBEST_HEADER = "# Các giả thuyết ASR"
TOOLS_HEADER = "# Công cụ khả dụng"
STEPS_HEADER = "# Các bước đã thực hiện"
HISTORY_HEADER = "# Lịch sử hội thoại"
CONTEXT_HEADER = "# Ngữ cảnh"
RESPONSE_HEADER = "# Định dạng trả lời"

#: Section order of ``build_decision_prompt`` (steps section is optional).
_REQUEST_HEADERS = (REQUEST_HEADER, NBEST_HEADER)
_SECTION_ORDER = (REQUEST_HEADER, CONTEXT_HEADER, HISTORY_HEADER, TOOLS_HEADER, STEPS_HEADER, RESPONSE_HEADER)
_OPTIONAL_SECTIONS = frozenset({STEPS_HEADER})
_SEP = "\n\n"

_ACTION_BLOCK = re.compile(re.escape(ACTION_OPEN) + r"(.*?)</action>", re.S)
#: Clause boundaries used when deleting a value: sentence end, comma, or newline.
#: A comma-delimited clause is the unit Vietnamese requests actually use to add
#: one extra requirement ("... chiều nay, nhắc tôi mang laptop trước giờ hẹn").
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*,\s*|\n+")
_CLAUSE_JOIN = " "

#: Shorter literals are too ambiguous to locate safely in prose.
_MIN_LITERAL_CHARS = 2

# --------------------------------------------------------------------------- #
# Prompt sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Section:
    header: str
    start: int  # index of the header line
    body_start: int  # index just after "<header>\n"
    end: int  # index where the next section (or the prompt) ends

def _sections(prompt_body: str) -> list[_Section] | None:
    """Header spans of a rendered prompt, or ``None`` when it is not one.

    The prompt is anchored on the exact headers ``build_decision_prompt`` emits,
    in their fixed order: section bodies legitimately contain blank lines (the
    catalog section separates apps with one), so splitting on ``"\\n\\n"`` would
    fragment them.
    """
    if not isinstance(prompt_body, str):
        return None
    first = next((name for name in _REQUEST_HEADERS if prompt_body.startswith(name)), None)
    if first is None:
        return None
    found: list[tuple[str, int, int]] = []
    cursor = 0
    for position, header in enumerate(_SECTION_ORDER):
        token = first if position == 0 else _SEP + header
        index = prompt_body.find(token, cursor)
        if index < 0:
            if header in _OPTIONAL_SECTIONS:
                continue
            return None
        name = first if position == 0 else header
        start = index if position == 0 else index + len(_SEP)
        body_start = start + len(name)
        if prompt_body[body_start : body_start + 1] == "\n":
            body_start += 1
        found.append((name, start, body_start))
        cursor = body_start
    sections: list[_Section] = []
    for position, (header, start, body_start) in enumerate(found):
        end = found[position + 1][1] - len(_SEP) if position + 1 < len(found) else len(prompt_body)
        sections.append(_Section(header, start, body_start, end))
    return sections

def prompt_sections(prompt_body: str) -> dict[str, str] | None:
    """``{header: body}`` of a rendered prompt (``None`` when unrecognised)."""
    sections = _sections(prompt_body)
    if sections is None:
        return None
    return {section.header: prompt_body[section.body_start : section.end] for section in sections}

def prompt_section(prompt_body: str, header: str) -> str | None:
    """Body of one section (``None`` when the prompt is unrecognised/absent)."""
    included = prompt_sections(prompt_body)
    return None if included is None else included.get(header)

def request_section(prompt_body: str) -> str | None:
    """Body of the first (request) section, whatever header it carries."""
    sections = _sections(prompt_body)
    if not sections:
        return None
    first = sections[0]
    return prompt_body[first.body_start : first.end]

def replace_first_section(prompt_body: str, section_text: str) -> str | None:
    """Swap the whole request section (header + body) of a rendered prompt."""
    if not isinstance(section_text, str) or "\n" not in section_text:
        return None
    sections = _sections(prompt_body)
    if not sections:
        return None
    first = sections[0]
    return prompt_body[: first.start] + section_text + prompt_body[first.end :]

def replace_section_body(prompt_body: str, header: str, body: str) -> str | None:
    """Swap one section body, keeping every other byte of the prompt."""
    sections = _sections(prompt_body)
    if sections is None:
        return None
    for section in sections:
        if section.header == header:
            return prompt_body[: section.body_start] + body + prompt_body[section.end :]
    return None

# --------------------------------------------------------------------------- #
# Record plumbing
# --------------------------------------------------------------------------- #
def _field(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)

def _messages_of(record: Any) -> list[dict[str, Any]]:
    return [dict(message) for message in (_field(record, "messages") or [])]

def _prompt_of(record: Any) -> str:
    messages = _field(record, "messages") or []
    if not messages:
        return ""
    last = messages[-1]
    return str(last.get("content", "")) if isinstance(last, Mapping) else str(getattr(last, "content", ""))

def _target_of(record: Any) -> str:
    return str(_field(record, "target") or "")

def _copy_with(
    record: Any,
    *,
    prompt_body: str | None = None,
    target: str | None = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Shallow record copy with a replaced prompt/target and merged meta."""
    out = dict(record) if isinstance(record, Mapping) else dict(vars(record))
    if prompt_body is not None:
        messages = _messages_of(record)
        if not messages:
            raise LifeMateError("record has no messages to rewrite")
        messages[-1]["content"] = prompt_body
        out["messages"] = messages
    if target is not None:
        out["target"] = target
    merged = dict(out.get("meta") or {})
    merged.update(extra_meta or {})
    out["meta"] = merged
    return out

def _record_key(record: Any, seed: int, extra: str = "") -> str:
    """Stable per-record ordering key (independent of input order)."""
    sample_id = str(_field(record, "sample_id") or "")
    step_index = _field(record, "step_index") or 0
    return hashlib.sha256(f"{seed}:{sample_id}:{step_index}:{extra}".encode("utf-8")).hexdigest()

def _action_blocks(text: str) -> list[dict[str, Any]]:
    """Parsed actions written inside ``text`` (prompt steps or a target)."""
    actions: list[dict[str, Any]] = []
    for match in _ACTION_BLOCK.finditer(text or ""):
        action = parse_action(match.group(0))
        if action.type != "invalid":
            actions.append(action.to_dict())
    return actions

def _installed_apps(context_body: str | None) -> tuple[list[str], bool]:
    """``(apps, readable)`` from the rendered context section."""
    if context_body is None:
        return [], False
    for line in context_body.splitlines():
        if line.startswith("installed_apps: "):
            value = line[len("installed_apps: ") :].strip()
            if value in {"", "(không có)"}:
                return [], True
            return [item.strip() for item in value.split(",") if item.strip()], True
    return [], False

def _tools_in_section(tools_body: str | None) -> tuple[dict[str, str], dict[str, list[str]]]:
    """``(tool -> app, app -> tools)`` as the catalog section renders them."""
    tool_app: dict[str, str] = {}
    app_tools: dict[str, list[str]] = {}
    if not tools_body:
        return tool_app, app_tools
    app = ""
    for line in tools_body.splitlines():
        if line.startswith("## "):
            app = line[3:].strip()
            app_tools.setdefault(app, [])
        elif line.startswith("### "):
            name = line[4:].split(" (kind=")[0].strip()
            if name:
                tool_app[name] = app
                app_tools.setdefault(app, []).append(name)
    return tool_app, app_tools

def _meta_spec(catalog: Any) -> Any | None:
    """The catalog's meta-tool spec (``None`` when the catalog has none)."""
    get = getattr(catalog, "get", None)
    if callable(get):
        for name in META_TOOL_CANDIDATES:
            spec = get(name)
            if spec is not None:
                return spec
    for spec in (getattr(catalog, "tools", {}) or {}).values():
        if str(getattr(spec, "kind", "") or "") == "meta":
            return spec
    return None

def _meta_arguments(spec: Any, tool_name: str, missing: str) -> dict[str, Any] | None:
    """Meta-call arguments derived from ITS schema (no field name is invented).

    A ``string`` property receives the tool name; an array-of-strings property
    receives the missing argument name. Any other shape cannot be filled from
    data we hold, so the example is skipped rather than guessed.
    """
    properties = getattr(spec, "properties", None)
    if not isinstance(properties, Mapping) or not properties:
        return None
    arguments: dict[str, Any] = {}
    for key in sorted(properties, key=str):
        schema = properties[key]
        if not isinstance(schema, Mapping):
            return None
        kind = schema.get("type")
        items = schema.get("items") if isinstance(schema.get("items"), Mapping) else {}
        if kind == "string":
            arguments[str(key)] = tool_name
        elif kind == "array" and str(items.get("type") or "") == "string":
            arguments[str(key)] = [missing]
        else:
            return None
    if any(str(key) not in arguments for key in (getattr(spec, "required", None) or [])):
        return None
    return arguments

def _literal(value: Any) -> str | None:
    """Verbatim literal a scalar value appears as in prose, or ``None``."""
    if value is None or isinstance(value, bool) or isinstance(value, (list, tuple, dict)):
        return None
    text = str(value)
    return text if len(text.strip()) >= _MIN_LITERAL_CHARS else None

def _remove_clause(text: str, literal: str) -> str | None:
    """Delete the whole clause carrying ``literal`` (``None`` when unsafe)."""
    if text.count(literal) != 1:
        return None
    clauses = _CLAUSE_SPLIT.split(text)
    hits = [index for index, clause in enumerate(clauses) if literal in clause]
    if len(hits) != 1:
        return None
    survivor = [clause.strip() for index, clause in enumerate(clauses) if index != hits[0]]
    joined = _CLAUSE_JOIN.join(part for part in survivor if part).strip()
    if len(joined) < 4:
        return None
    # The result must still be a plausible Vietnamese request: it may not start
    # with the join punctuation of the removed clause or end mid-clause.
    return joined if re.match(r"[\w\d]", joined) else None

# --------------------------------------------------------------------------- #
# Slot masking
# --------------------------------------------------------------------------- #
def apply_slot_masking(
    records: Sequence[Any],
    *,
    enabled: bool = False,
    seed: int = 42,
    catalog: Any = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Add a variant whose gold required argument was removed and is asked for.

    Skip rules, each counted in ``skips_by_rule`` / named in ``per_record``:

    1. the target is not exactly one ``tool_call`` (a ``finish``, several calls,
       or a target that is not one action block);
    2. the tool's schema declares no required argument, or the tool is not in
       ``catalog`` at all;
    3. no required argument value is a literal appearing exactly once in the
       request section, or its clause cannot be deleted cleanly (or the value
       survives the deletion);
    4. the conversation history already shows the same value, so the model could
       legitimately reuse it;
    5. ``sys.ask_missing_info`` is absent from the catalog, not shown in this
       prompt's tool section, its schema cannot fill the call without inventing
       a field, or the resulting arguments fail schema validation.

    The original record is kept unchanged; a created variant replaces the target
    with a schema-valid ``sys.ask_missing_info`` call and drops the value from the
    request text.
    """
    if not enabled:
        return records, {}
    if catalog is None:
        raise LifeMateError("slot masking needs the tool catalog to prove the target is ask_missing_info")

    meta_spec = _meta_spec(catalog)
    out: list[Any] = []
    skips: list[dict[str, Any]] = []
    n_created = 0
    for record in records:
        out.append(record)
        variant, reason = _slot_mask_one(record, seed=seed, catalog=catalog, meta_spec=meta_spec)
        if variant is None:
            skips.append(
                {
                    "sample_id": str(_field(record, "sample_id") or ""),
                    "step_index": _field(record, "step_index"),
                    "rule": int(reason.split(":", 1)[0]),
                    "reason": reason,
                }
            )
            continue
        out.append(variant)
        n_created += 1
    return out, {
        "enabled": True,
        "seed": int(seed),
        "n_records_in": len(records),
        "n_variants_created": n_created,
        "n_skipped": len(skips),
        "skips_by_rule": _count_by(skips, "rule"),
        "skips_by_reason": _count_by(skips, "reason"),
        "per_record": skips,
    }

def _count_by(entries: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        name = str(entry[key])
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))

def _slot_mask_one(
    record: Any, *, seed: int, catalog: Any, meta_spec: Any
) -> tuple[dict[str, Any] | None, str]:
    """``(variant_or_None, skip_reason)`` for one record."""
    prompt = _prompt_of(record)
    target_text = _target_of(record)
    if len(_ACTION_BLOCK.findall(target_text)) != 1:
        return None, "1:target_is_not_exactly_one_action_block"
    action = parse_action(target_text)
    if action.type != "tool_call" or not action.name:
        return None, "1:target_is_not_a_tool_call"
    tool_name = str(action.name)

    getter = getattr(catalog, "get", None)
    spec = getter(tool_name) if callable(getter) else None
    required = [str(key) for key in (getattr(spec, "required", None) or [])] if spec is not None else []
    if spec is None:
        return None, "2:tool_absent_from_catalog"
    if not required:
        return None, "2:tool_declares_no_required_argument"

    included = prompt_sections(prompt)
    if included is None:
        return None, "3:prompt_is_not_a_rendered_decision_prompt"
    request = included.get(REQUEST_HEADER) or ""
    history = included.get(HISTORY_HEADER) or ""
    arguments = dict(action.arguments or {})

    visible: list[tuple[str, str]] = []
    for key in sorted(required):
        literal = _literal(arguments.get(key))
        if literal is not None and literal in request:
            visible.append((key, literal))
    if not visible:
        return None, "3:no_required_value_is_a_literal_in_request"

    if not _meta_tool_in_section(included.get(TOOLS_HEADER) or ""):
        return None, "5:meta_tool_not_shown_in_prompt"
    if meta_spec is None:
        return None, "5:meta_tool_absent_from_catalog"

    # Every visible required argument is tried in sorted order: one whose value
    # the history already shows is skipped, and a clause that cannot be deleted
    # cleanly falls through to the next candidate rather than dropping the record.
    removal_skipped = False
    for dropped_key, literal in visible:
        if literal in history:
            continue
        new_request = _remove_clause(request, literal)
        if new_request is None or literal in new_request:
            removal_skipped = True
            continue
        new_prompt = replace_section_body(prompt, REQUEST_HEADER, new_request)
        if new_prompt is None:
            removal_skipped = True
            continue
        meta_name = str(getattr(meta_spec, "name", "") or META_TOOL_NAME)
        meta_arguments = _meta_arguments(meta_spec, tool_name, dropped_key)
        if meta_arguments is None:
            return None, "5:meta_schema_unfillable_without_invention"
        validate = getattr(catalog, "validate_call", None)
        if callable(validate) and validate(meta_name, meta_arguments).get("errors"):
            return None, "5:meta_arguments_fail_schema_validation"
        return (
            _copy_with(
                record,
                prompt_body=new_prompt,
                target=render_action(make_tool_call(meta_name, meta_arguments)),
                extra_meta={
                    "augmentation": "slot_masking",
                    "slot_masked_tool": tool_name,
                    "slot_masked_argument": dropped_key,
                    "slot_masked_value": literal,
                    "gold_target_replaced": True,
                    "augmentation_seed": int(seed),
                    "augmentation_key_hash": _record_key(record, seed, "slot_masking"),
                },
            ),
            "",
        )

    # No candidate produced a coherent example; name the rule that blocked most.
    if removal_skipped:
        return None, "3:clause_not_cleanly_removable"
    return None, "4:value_already_in_history"

def _meta_tool_in_section(tools_body: str) -> bool:
    return any(f"### {name}" in tools_body for name in META_TOOL_CANDIDATES)

# --------------------------------------------------------------------------- #
# Tool dropout
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _SpecView:
    """Minimal catalog view: only ``specs``, the attribute the renderer reads."""

    specs: tuple[Any, ...]

def apply_tool_dropout(
    records: Sequence[Any],
    *,
    enabled: bool = False,
    max_tools: int = 1,
    seed: int = 42,
    catalog: Any = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Add a variant whose rendered catalog section omits up to ``max_tools`` tools.

    TRAINING ONLY. The inference registry (``ToolRegistry``) is never filtered by
    this function: hiding a callable tool from the agent at inference loses
    points, while hiding it in a *training* prompt teaches the model to work with
    a smaller toolbox. ``catalog`` is read, never mutated.

    Rules, each skip counted per reason:

    1. candidates are the tools listed in the prompt's catalog section;
    2. a tool called by the gold target, by any earlier step, or by any history
       turn is never removed;
    3. the meta tool is never removed (asking for missing information is a
       behaviour taught elsewhere in the data);
    4. a tool is skipped when its app stays in ``installed_apps`` and the tool is
       that app's only one — the rendered context and tool list would then
       contradict each other (``installed_apps`` is left untouched);
    5. the filtered prompt must be reproducible: the record's own tool section is
       rebuilt through ``ToolRegistry`` + ``_render_tools`` and compared with the
       stored bytes first. A mismatch means the record was rendered from another
       catalog state, so it is skipped instead of partially rewritten.

    ``max_tools`` caps removals; which tools go is a per-record hash of ``seed``,
    so input order cannot change the result.
    """
    if not enabled:
        return records, {}
    if int(max_tools) < 1:
        raise LifeMateError(f"max_tools must be >= 1, got {max_tools!r}")
    if catalog is None:
        raise LifeMateError("tool dropout needs the tool catalog to re-render the tool section")

    out: list[Any] = []
    skips: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    n_created = 0
    for record in records:
        out.append(record)
        variant, dropped, reason = _dropout_one(
            record, seed=seed, max_tools=int(max_tools), catalog=catalog
        )
        if variant is None:
            skips.append(
                {
                    "sample_id": str(_field(record, "sample_id") or ""),
                    "step_index": _field(record, "step_index"),
                    "reason": reason,
                }
            )
            continue
        out.append(variant)
        n_created += 1
        for name in dropped:
            counts[name] = counts.get(name, 0) + 1
    return out, {
        "enabled": True,
        "max_tools": int(max_tools),
        "seed": int(seed),
        "n_records_in": len(records),
        "n_variants_created": n_created,
        "n_tools_dropped": sum(counts.values()),
        "tools_dropped": dict(sorted(counts.items())),
        "n_records_skipped": len(skips),
        "skips_by_reason": _count_by(skips, "reason"),
        "per_record": skips,
    }

def _dropout_one(
    record: Any, *, seed: int, max_tools: int, catalog: Any
) -> tuple[dict[str, Any] | None, list[str], str]:
    """``(variant_or_None, dropped_tools, skip_reason)`` for one record."""
    prompt = _prompt_of(record)
    included = prompt_sections(prompt)
    if included is None:
        return None, [], "not_a_rendered_decision_prompt"
    tools_body = included.get(TOOLS_HEADER)
    if tools_body is None:
        return None, [], "no_tools_section"
    tool_app, app_tools = _tools_in_section(tools_body)
    if not tool_app:
        return None, [], "empty_tools_section"
    apps, readable = _installed_apps(included.get(CONTEXT_HEADER))
    if not readable:
        return None, [], "unreadable_installed_apps"

    try:
        specs = list(ToolRegistry(catalog, apps=apps).specs)
    except Exception as exc:  # noqa: BLE001 - a malformed catalog must not crash the build
        return None, [], f"registry_rebuild_failed:{type(exc).__name__}"
    if _render_tools(_SpecView(tuple(specs)), None) != tools_body:
        return None, [], "tools_section_not_reproducible_from_catalog"

    protected = {str(action.get("name") or "") for action in _action_blocks(_target_of(record))}
    for header in (STEPS_HEADER, HISTORY_HEADER):
        protected |= {
            str(action.get("name") or "")
            for action in _action_blocks(included.get(header) or "")
        }
    candidates: list[str] = []
    for name in sorted(tool_app):
        if name in protected or name in META_TOOL_CANDIDATES:
            continue
        app = tool_app[name]
        if app in apps and len(app_tools.get(app, [])) <= 1:
            continue  # app stays installed with no tool left: inconsistent context
        candidates.append(name)
    if not candidates:
        return None, [], "no_removable_tool"

    order = sorted(candidates, key=lambda name: _record_key(record, seed, f"drop:{name}"))
    dropped = order[:max_tools]
    remaining = [spec for spec in specs if str(getattr(spec, "name", "")) not in dropped]
    new_tools = _render_tools(_SpecView(tuple(remaining)), None) if remaining else ""
    if not new_tools or new_tools == "(không có)":
        return None, [], "catalog_section_would_be_empty"
    new_prompt = replace_section_body(prompt, TOOLS_HEADER, new_tools)
    if new_prompt is None or new_prompt == prompt:
        return None, [], "prompt_unchanged"
    return (
        _copy_with(
            record,
            prompt_body=new_prompt,
            extra_meta={
                "augmentation": "tool_dropout",
                "tools_dropped": list(dropped),
                "n_tools_dropped": len(dropped),
                "augmentation_seed": int(seed),
            },
        ),
        dropped,
        "",
    )

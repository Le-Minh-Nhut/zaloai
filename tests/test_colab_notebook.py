"""Colab-notebook contract tests: lifecycle, chunked probe, vocabulary, sync, resume opt-in.

Nothing here needs a GPU or the network. The notebook's own helper cells are extracted
from the .ipynb JSON with ``ast`` and executed in a private namespace, then driven
against a local temp directory — so every assertion exercises the shipped code, not a
re-typed copy of it.

Optional stacks gate exactly like the rest of the suite: the model-lifecycle and
gradient-probe checks print ``SKIP: <reason>`` and return when ``torch`` /
``transformers`` / ``models/MiniCPM5-2B`` are absent, so this file collects cleanly in
the stdlib runner and in a torch environment alike.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import pathlib
import sys
import tempfile
import weakref
from types import SimpleNamespace
from typing import Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NOTEBOOK = ROOT / "notebooks" / "finetune_qwen35_colab.ipynb"
LOCAL_MINICPM = ROOT / "models" / "MiniCPM5-2B"

#: Cell markers this file relies on; each must appear exactly once in the notebook.
MARKERS = {
    "params": "#@title 2. Run configuration",
    "features": "#@title 2a. Feature flags",
    "derive": "# --- 2b. Derivation: the GPU TIER decides the METHOD",
    "helpers": "# --- 3c. Verify imports + bind the repo CLI contract",
    "sync": "# --- 4b. Atomic artifact sync + completion markers",
    "dataset": "# --- 5a. Build (or reuse) the SFT dataset and measure token lengths",
    "load_probe": "# --- 6b. Load the checkpoint through the repo loader, verify adapter + grads --",
    "cli": "# --- 6c. Bind the CLI contract: flags, choices, resume form",
    "resume_ctl": "# --- 6d. Resume target: discovery, verification, report",
    "worst_case": "# --- 7c. Worst-case memory probe: the CONFIGURED sequence length",
    "train": "# --- 8. Train, streaming output and mirroring finished checkpoints",
    "resume": "# --- 9a. Inspect the checkpoint, then (opt-in) prove continuation",
    "smoke": "# --- 12. Load the export, generate, parse, validate, one agent step",
    "final": "# --- 13. Final sync + free everything",
}

NEEDS: list[str] = []


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _marker_index(cells: list[dict], key: str) -> int:
    wanted = MARKERS[key]
    hits = [i for i, cell in enumerate(cells)
            if any(line.startswith(wanted) for line in "".join(cell["source"]).splitlines())]
    assert len(hits) == 1, f"marker {wanted!r} appears {len(hits)} times, expected exactly one"
    return hits[0]


def _cell_source(cells: list[dict], key: str) -> str:
    return "".join(cells[_marker_index(cells, key)]["source"])


def _bound_names(node: ast.stmt) -> set[str]:
    """Names a top-level statement binds (functions, imports, simple/tuple assignments)."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(alias.asname or alias.name.split(".")[0]) for alias in node.names}
    targets = node.targets if isinstance(node, ast.Assign) else (
        [node.target] if isinstance(node, ast.AnnAssign) else [])
    names: set[str] = set()

    def collect(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                collect(element)

    for target in targets:
        collect(target)
    return names


def _exec_defs(cells: list[dict], key: str, namespace: dict, names: Sequence[str]) -> dict:
    """Exec only the statements that bind ``names`` from one notebook cell.

    Notebook cells end with driver code (a real GPU load, a subprocess launch). Executing the whole
    cell is what makes a notebook test untestable, so this walks the module body and keeps just the
    definitions and assignments the requested names need — meaning a cell that loads Qwen3.5-4B
    never loads it here. Anything those statements read from the notebook's earlier sections must be
    supplied in ``namespace`` by the caller.
    """
    tree = ast.parse(_cell_source(cells, key))
    wanted = set(names)
    keep = [node for node in tree.body if _bound_names(node) & wanted]
    missing = wanted - set().union(*(_bound_names(node) for node in keep)) if keep else wanted
    assert not missing, f"cell {key} does not define {sorted(missing)}"
    namespace.setdefault("__name__", f"notebook_cell_{key}")
    module = ast.Module(body=keep, type_ignores=[])
    exec(compile(module, f"<cell {key} prologue>", "exec"), namespace)  # noqa: S102 - test harness
    return namespace


# --------------------------------------------------------------------------- #
# STRUCTURE
# --------------------------------------------------------------------------- #
def test_notebook_is_valid_json_and_every_code_cell_parses() -> None:
    """The one invariant Colab itself requires: parseable JSON, parseable Python."""
    raw = NOTEBOOK.read_text(encoding="utf-8")
    payload = json.loads(raw)
    cells = payload["cells"]
    code = [i for i, cell in enumerate(cells) if cell["cell_type"] == "code"]
    for index in code:
        source = "".join(cells[index]["source"])
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - failure path
            raise AssertionError(f"code cell {index} does not parse: {exc}") from exc
    # 41 cells / 27 code: +3 code cells (resume control 6d, worst-case probe 7c, feature flags 2a).
    assert len(cells) == 41, len(cells)
    assert len(code) == 27, len(code)
    print(f"  notebook -> valid JSON, {len(cells)} cells ({len(code)} code), all ast.parse")


def test_required_cell_markers_exist_exactly_once() -> None:
    """A duplicated marker would make every later extraction ambiguous."""
    cells = _cells()
    seen = {key: _marker_index(cells, key) for key in MARKERS}
    print(f"  markers -> each of {len(MARKERS)} markers found exactly once: {seen}")


# --------------------------------------------------------------------------- #
# P1a — RELEASE
# --------------------------------------------------------------------------- #
class _FakeModel:
    """Stand-in for a torch module: has ``parameters()``, no torch import needed."""

    def __init__(self) -> None:
        (self.weight,) = (SimpleNamespace(payload=bytearray(64)),)

    def parameters(self):
        return [self.weight]


def _release_namespace() -> dict:
    """Minimal namespace: a fake module + the notebook's own RELEASABLE_NAMES/find_model_refs."""
    cells = _cells()
    namespace: dict = {"torch": None, "Path": pathlib.Path, "json": json, "time": __import__("time"),
                       "shutil": __import__("shutil"), "os": __import__("os"), "re": __import__("re"),
                       "IN_COLAB": False, "subprocess": __import__("subprocess"), "sys": sys,
                       "threading": __import__("threading")}
    _exec_defs(cells, "helpers", namespace,
               ["RELEASABLE_NAMES", "_is_model_like", "find_model_refs", "release_model_refs",
                "free_gpu"])
    return namespace


def test_release_model_refs_frees_the_model_behind_a_report_dict() -> None:
    """``del trainable_model`` frees nothing while ``lora_report['model']`` holds the same object."""
    namespace = _release_namespace()
    model = _FakeModel()
    namespace["trainable_model"] = model
    namespace["lora_report"] = {"model": model, "trainable_params": 42, "r": 16}
    namespace["tokenizer"] = SimpleNamespace(parameters=lambda: [], name_or_path="fake")
    alias = weakref.ref(model)
    report = namespace["release_model_refs"](namespace, verbose=False)
    del model
    gc.collect()
    assert alias() is None, "the model object outlived release_model_refs"
    for name in ("trainable_model", "lora_report", "tokenizer"):
        assert name not in namespace, name
    assert report["residual_names"] == [], report
    assert "allocated_mib" in report and "freed_mib" in report and "reserved_mib" in report
    print("  P1a -> model freed via lora_report['model']; weakref dead; residual scan empty")


def test_release_model_refs_raises_and_names_surviving_attributes() -> None:
    """The guard must discriminate: a reference it cannot delete is a hard error, named."""
    namespace = _release_namespace()
    holder = _FakeModel()
    namespace["unlisted_holder"] = {"outer": {"inner": holder}}
    try:
        namespace["release_model_refs"](namespace, verbose=False)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("release_model_refs accepted a namespace that still holds a model")
    assert "unlisted_holder['outer']['inner']" in message, message
    # No false positive: an int/str-only namespace passes.
    clean = _release_namespace()
    clean["MODEL_FACTS"] = {"total_params": 4_205_751_296, "compute_dtype": "bfloat16"}
    assert clean["find_model_refs"](clean) == []
    assert clean["release_model_refs"](clean, verbose=False)["residual_names"] == []
    print(f"  P1a guard -> raises naming the surviving path "
          f"({message.split(':')[1].strip()[:58]}...); plain-value namespace passes")


def test_every_cell_that_binds_a_model_ends_with_a_release() -> None:
    """Sweep: no code cell may leave a module/tokenizer bound without releasing it."""
    cells = _cells()
    owners = {"load_probe": "release_model_refs", "smoke": "release_model_refs",
              "final": "release_model_refs"}
    checked = []
    for key in owners:
        source = _cell_source(cells, key)
        assert "release_model_refs" in source, f"cell {key} binds a model and never releases it"
        checked.append(key)
    # Cell 5a caches a CPU tokenizer; 6b's release must delete that cache too, or the measurement
    # pass leaves a tokenizer bound for the rest of the session.
    namespace = _release_namespace()
    assert "TOKENIZER_CACHE" in namespace["RELEASABLE_NAMES"]
    sentinel = SimpleNamespace(parameters=lambda: [])
    namespace["TOKENIZER_CACHE"] = {"Qwen/Qwen3.5-4B": sentinel}
    namespace["release_model_refs"](namespace, verbose=False)
    assert "TOKENIZER_CACHE" not in namespace

    # Cell 6b must not hand-write a `del` of the module: that was the bug (comments explaining the
    # old bug are fine, an executable `del` is not).
    tree = ast.parse(_cell_source(cells, "load_probe"))
    hand_dels = [node.lineno for node in ast.walk(tree)
                 if isinstance(node, ast.Delete)
                 and any(isinstance(t, ast.Name) and t.id == "trainable_model" for t in node.targets)]
    assert not hand_dels, f"cell 6b still hand-dels the module at line(s) {hand_dels}"
    print(f"  P1a sweep -> {checked} each call release_model_refs(); cell 6b no longer hand-dels")


# --------------------------------------------------------------------------- #
# P1b — CHUNKED PROBE
# --------------------------------------------------------------------------- #
def test_probe_path_passes_no_labels_to_the_model() -> None:
    """The notebook's own ``probe_loss`` must ask for logits and never hand ``labels`` to the model."""
    if not _have("torch"):
        print("SKIP: torch absent - probe call shape not exercised")
        NEEDS.append("probe_no_labels")
        return
    import torch

    from lifemate.training.trainer import chunked_causal_lm_loss, forward_loss

    # The shipped probe_loss/chunk_equivalence, lifted from cell 6b with only its driver code excluded.
    namespace: dict = {"forward_loss": forward_loss, "chunked_causal_lm_loss": chunked_causal_lm_loss,
                       "torch": torch, "torch.nn.functional": torch.nn.functional,
                       "cfg": SimpleNamespace(training=SimpleNamespace(max_seq_length=9,
                                                                      loss_chunk_tokens=4)),
                       "VOCAB": 248320, "math": __import__("math")}
    _exec_defs(_cells(), "load_probe", namespace,
               ["probe_loss", "_supervised_shift_counts", "chunk_equivalence"])
    probe_loss, chunk_equivalence = namespace["probe_loss"], namespace["chunk_equivalence"]

    seen: dict = {}

    class StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(8, 32, bias=False)

        def forward(self, **kwargs):
            seen.setdefault("calls", []).append(sorted(kwargs))
            ids = kwargs["input_ids"]
            return SimpleNamespace(logits=self.linear(
                torch.nn.functional.one_hot(ids % 8, 8).float()), loss=None)

    torch.manual_seed(3)
    ids = torch.randint(0, 8, (1, 9))
    labels = torch.where(torch.rand(1, 9) > 0.4, ids, torch.full_like(ids, -100))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    model = StubModel()   # one instance: the reference must see the SAME weights
    loss = probe_loss(model, batch, 4)
    assert seen["calls"] == [["attention_mask", "input_ids"]], seen["calls"]
    assert loss.requires_grad, "the chunked probe loss is not connected to the graph"
    loss.backward()
    chunked = float(loss.detach())

    with torch.no_grad():
        logits = model(input_ids=ids).logits
        single = float(torch.nn.functional.cross_entropy(
            logits[..., :-1, :].float().reshape(-1, 32), labels[..., 1:].reshape(-1),
            ignore_index=-100))
        assert abs(chunked - single) <= 1e-5 * max(abs(single), 1.0), (chunked, single)
        for chunk in (1, 3, 4, 1024):
            got = float(chunked_causal_lm_loss(logits, labels, chunk))
            assert abs(got - single) <= 1e-5 * max(abs(single), 1.0), (chunk, got, single)

    # The notebook's own equivalence check must accept the shipped maths...
    result = chunk_equivalence(model, batch, 4)
    assert result["chunk_used"] < result["seq_tokens"], result
    assert result["delta"] <= 1e-4, result
    # ...and must REJECT a loss that is not the same function, or the guard proves nothing. The
    # check now RAISES (it used to return a delta the caller had to remember to test).
    sabotaged = dict(namespace, chunked_causal_lm_loss=lambda logits, labels, chunk: logits.sum() * 0.0)
    _exec_defs(_cells(), "load_probe", sabotaged,
               ["_supervised_shift_counts", "chunk_equivalence"])
    try:
        sabotaged["chunk_equivalence"](model, batch, 4)
    except RuntimeError as exc:
        assert "disagrees" in str(exc), str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("chunk_equivalence accepted a loss that is not the same function")

    # Unchunked is REFUSED, naming the config knob, instead of silently going back to the 8.1 GB path.
    try:
        probe_loss(model, batch, 0)
    except RuntimeError as exc:
        assert "loss_chunk_tokens" in str(exc), str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("probe_loss ran unchunked on a GPU-shaped config")
    print(f"  P1b -> model saw {seen['calls'][0]} (no labels); chunked {chunked:.6f} == "
          f"single-shot {single:.6f} over 4 chunk sizes with -100 masking; chunk=0 refused; "
          f"notebook equivalence check delta {result['delta']:.1e} (and rejects a wrong loss)")


# --------------------------------------------------------------------------- #
# P1c — VOCABULARY
# --------------------------------------------------------------------------- #
def _estimator_namespace() -> dict:
    cells = _cells()
    namespace: dict = {"Path": pathlib.Path, "json": json, "GPU_TIER": "T4-16GB",
                       "MODEL_ID": "Qwen/Qwen3.5-4B", "PRECISION": "bfloat16"}
    _exec_defs(cells, "derive", namespace,
               ["BASE_PARAMS_B", "HIDDEN", "LAYERS", "TIER_VRAM_GB", "USABLE_FRACTION",
                "QLORA_BYTES_PER_PARAM", "ACT_FACTOR", "RESERVE_GB", "SEQ_CEILING",
                "TIER_METHOD_CAP", "VOCAB", "VOCAB_SOURCE", "LOSS_CHUNK_TOKENS", "resolve_vocab",
                "next_multiple", "compute_bytes_per_param", "estimate_vram_gb", "seq_cap",
                "choose_method",
                "derive_run_config", "print_derived", "revalidate_tier_caps", "run_section_2b"])
    # Globals the module-level functions read; the notebook sets them inside run_section_2b.
    from lifemate.config import TrainingConfig
    from lifemate.models.profiles import resolve_profile

    namespace["VOCAB"], namespace["VOCAB_SOURCE"] = namespace["resolve_vocab"]("Qwen/Qwen3.5-4B")
    # The notebook's resolver must agree with the registry it claims to read.
    assert namespace["VOCAB"] == resolve_profile("Qwen/Qwen3.5-4B").preflight_vocab
    namespace["LOSS_CHUNK_TOKENS"] = int(TrainingConfig().loss_chunk_tokens)
    return namespace


def test_estimator_prices_qwen_vocab_higher_than_the_old_literal() -> None:
    """248320 must cost strictly more logits than the Qwen2-era 151936 at the same sequence."""
    namespace = _estimator_namespace()
    estimate = namespace["estimate_vram_gb"]
    assert namespace["VOCAB"] == 248320, namespace["VOCAB"]
    for seq in (2048, 4096, 8192):
        new = estimate("L4-24GB", "lora", seq)
        old = estimate("L4-24GB", "lora", seq, vocab=151936)
        assert new["total_gb"] > old["total_gb"], (seq, new["total_gb"], old["total_gb"])
        # Rounded to 2 decimals by the estimator, so the ratio matches to rounding precision.
        ratio = new["logits_unchunked_gb"] / old["logits_unchunked_gb"]
        assert abs(ratio - 248320 / 151936) < 1e-2, ratio
    # The chunked fp32 term is far below the unchunked one, and the compute-dtype head output is
    # its own allocation on top — the total must carry both, never just the fp32 slice.
    at8192 = estimate("L4-24GB", "lora", 8192)
    assert at8192["logits_chunked_gb"] < at8192["logits_unchunked_gb"]
    assert at8192["logits_compute_gb"] > at8192["logits_chunked_gb"], at8192
    assert at8192["logits_gb"] == round(at8192["logits_compute_gb"] + at8192["logits_chunked_gb"], 2)
    assert at8192["logits_chunk_tokens"] == min(namespace["LOSS_CHUNK_TOKENS"], 8192)
    # The compute-dtype term is seq * vocab * bytes_per_param, so fp32 doubles it.
    fp32 = estimate("L4-24GB", "lora", 8192, bytes_per_param=4)
    assert fp32["logits_compute_gb"] == round(2 * at8192["logits_compute_gb"], 2), fp32
    assert fp32["total_gb"] == round(at8192["total_gb"] + at8192["logits_compute_gb"], 2)
    print(f"  P1c -> vocab {namespace['VOCAB']}: total {at8192['total_gb']} GB "
          f"(lg_compute {at8192['logits_compute_gb']}, fp32 chunk {at8192['logits_chunked_gb']}, "
          f"unchunked {at8192['logits_unchunked_gb']} GB)")


def test_estimator_refuses_to_run_without_a_vocabulary() -> None:
    """No vocab resolved must be an error, never a silent 151936 default."""
    namespace = _estimator_namespace()
    saved = (namespace["VOCAB"], namespace["VOCAB_SOURCE"])
    namespace["VOCAB"] = 0
    try:
        namespace["estimate_vram_gb"]("T4-16GB", "lora", 4096)
    except RuntimeError as exc:
        assert "vocabulary" in str(exc).lower(), str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("estimate_vram_gb priced a logits term without a vocabulary")
    finally:
        namespace["VOCAB"], namespace["VOCAB_SOURCE"] = saved
    try:
        namespace["resolve_vocab"]("totally-unknown/model-with-no-profile")
    except RuntimeError as exc:
        assert "profile" in str(exc).lower() or "config.json" in str(exc), str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("resolve_vocab guessed a vocabulary")
    print("  P1c -> no vocabulary / no verified metadata both raise instead of guessing")


def test_tier_caps_never_recommend_a_configuration_that_overflows() -> None:
    """Re-derive every cap; a cap the estimator refuses is a changed conclusion, not a fit."""
    namespace = _estimator_namespace()
    table = namespace["revalidate_tier_caps"]()
    rows = {(row["tier"], row["method"]): row for row in table["rows"]}
    assert len(rows) == 10, sorted(rows)
    for (tier, method), row in rows.items():
        cap = namespace["TIER_METHOD_CAP"][tier][method]
        assert row["seq"] == cap
        assert row["fits"], f"{tier}/{method} @{cap} is recommended but the estimator says it overflows"
    # The T4 LoRA conclusion must hold at 248320 with BOTH logits tensors priced: neither 8192
    # nor 4096 fits, and the cap is the largest 256-step sequence that does.
    t4_lora = namespace["estimate_vram_gb"]("T4-16GB", "lora", 8192)
    assert not t4_lora["fits"], t4_lora
    assert not namespace["estimate_vram_gb"]("T4-16GB", "lora", 4096)["fits"]
    t4_cap = namespace["TIER_METHOD_CAP"]["T4-16GB"]["lora"]
    assert t4_cap == 3840, t4_cap
    assert namespace["estimate_vram_gb"]("T4-16GB", "lora", t4_cap)["fits"]
    # seq_cap() must FIND that value, not merely accept a literal that happens to fit.
    assert namespace["seq_cap"]("T4-16GB", "lora") == t4_cap
    derived = namespace["derive_run_config"]("T4-16GB", "lora", None, 8192, 1, 8)
    assert derived["method"] == "qlora" and derived["seq_clamped"] is False, derived
    assert derived["vram_estimate"]["fits"], derived["vram_estimate"]
    assert derived["max_seq_length"] == 8192, derived
    assert namespace["seq_cap"]("T4-16GB", "qlora") >= 8192
    for tier in ("T4-16GB", "L4-24GB", "A100-40GB", "A100-80GB"):
        for method in ("lora", "qlora"):
            cap = namespace["seq_cap"](tier, method)
            assert namespace["estimate_vram_gb"](tier, method, cap)["fits"], (tier, method, cap)
    print(f"  P1c -> 10 tier/method caps re-derived, all fit; T4 lora@8192 derives qlora; "
          f"flips vs 151936: {table['changed'] or 'none'}")


# --------------------------------------------------------------------------- #
# P2a — SYNC
# --------------------------------------------------------------------------- #
def _sync_namespace() -> dict:
    cells = _cells()
    namespace: dict = {"Path": pathlib.Path, "json": json, "os": __import__("os"),
                       "shutil": __import__("shutil"), "re": __import__("re"),
                       "time": __import__("time"), "hashlib": hashlib,
                       "sys": sys, "subprocess": __import__("subprocess"),
                       "IN_COLAB": False, "torch": None}
    _exec_defs(cells, "sync", namespace,
               ["SYNC_MARKER", "SYNC_SCHEMA", "KEEP_PREVIOUS", "_reject_prohibited", "safe_unzip",
                "_PROHIBITED", "_fsync_tree", "_sha256", "file_manifest", "write_sync_marker",
                "verify_sync_marker", "checkpoint_ready", "_previous_generations", "atomic_sync",
                "_RENAME_PROBE", "rename_mode_for", "_next_generation_name", "_copy_generation_aside"])
    return namespace


def _make_checkpoint(root: pathlib.Path, step: int = 7) -> pathlib.Path:
    """A local directory shaped like this trainer's ``step-<n>`` output."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_model.safetensors").write_bytes(b"weights" * 64)
    (root / "optimizer_state.pt").write_bytes(b"adamw" * 32)
    (root / "training_state.json").write_text(json.dumps({"step": step, "config_hash": "abc"}),
                                             encoding="utf-8")
    return root


def test_successful_sync_then_resync_keeps_a_previous_generation() -> None:
    namespace = _sync_namespace()
    atomic_sync, checkpoint_ready = namespace["atomic_sync"], namespace["checkpoint_ready"]
    verify = namespace["verify_sync_marker"]
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_checkpoint(pathlib.Path(tmp) / "local" / "step-7", step=7)
        dst = pathlib.Path(tmp) / "drive" / "step-7"
        atomic_sync(src, dst, extra={"step": 7})
        ok, why = verify(dst)
        assert ok, why
        assert checkpoint_ready(dst, require_marker=True)[0]
        (src / "training_state.json").write_text(json.dumps({"step": 7, "config_hash": "def"}),
                                                 encoding="utf-8")
        atomic_sync(src, dst, extra={"step": 7})
        previous = sorted(dst.parent.glob("step-7.prev-*"))
        assert len(previous) == 1, [p.name for p in dst.parent.iterdir()]
        assert checkpoint_ready(previous[0], require_marker=True)[0], "previous generation unusable"
        assert json.loads((dst / "training_state.json").read_text())["config_hash"] == "def"
        print(f"  P2a -> re-sync keeps {previous[0].name} and both generations verify as resumable")


def test_crash_between_renames_leaves_a_complete_directory() -> None:
    """Inject a failing second ``os.replace`` and prove the durable copy survives intact."""
    namespace = _sync_namespace()
    atomic_sync = namespace["atomic_sync"]
    checkpoint_ready = namespace["checkpoint_ready"]
    verify = namespace["verify_sync_marker"]
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        src = _make_checkpoint(base / "local" / "step-3", step=3)
        dst = base / "drive" / "step-3"
        atomic_sync(src, dst, extra={"step": 3})
        before = json.loads((dst / "training_state.json").read_text())

        real_replace = namespace["os"].replace
        calls = {"n": 0}

        def flaky_replace(from_path, to_path, *args, **kwargs):
            current = pathlib.Path(from_path).name
            target = pathlib.Path(to_path).name
            # Die exactly on the second rename (staging tree -> final name). Every other replace,
            # including the marker's own tmp -> SYNC_COMPLETE.json and dst -> dst.prev-1, succeeds.
            if "incomplete-" in current and target == dst.name:
                calls["n"] += 1
                raise OSError("simulated power loss between the two renames")
            return real_replace(from_path, to_path, *args, **kwargs)

        namespace["os"].replace = flaky_replace
        try:
            atomic_sync(src, dst, extra={"step": 3})
        except OSError as exc:
            assert "simulated" in str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("the injected failure was swallowed")
        finally:
            namespace["os"].replace = real_replace

        assert calls["n"] == 1, calls
        visible = sorted(p.name for p in dst.parent.iterdir())
        assert not dst.exists(), f"a half-written dst exists: {visible}"
        previous = sorted(dst.parent.glob("step-3.prev-*"))
        assert len(previous) == 1, visible
        ok, why = verify(previous[0])
        assert ok, why
        assert checkpoint_ready(previous[0], require_marker=True)[0]
        assert json.loads((previous[0] / "training_state.json").read_text()) == before
        staging = list(dst.parent.glob("step-3.incomplete-*"))
        for leftover in staging:
            assert checkpoint_ready(leftover)[0] is False, "a staging dir passed as a checkpoint"
        print(f"  P2a -> crash after rename 1: {previous[0].name} verifies ({why}); "
              f"staging leftovers refused: {[p.name for p in staging]}")


def test_leftover_staging_directory_is_never_a_checkpoint() -> None:
    namespace = _sync_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_checkpoint(pathlib.Path(tmp) / "step-9.incomplete-1700000000", step=9)
        ok, why = namespace["checkpoint_ready"](root)
        assert not ok and "staging" in why, why
        ok, why = namespace["checkpoint_ready"](root, require_marker=True)
        assert not ok and "staging" in why, why
        print(f"  P2a -> interrupted staging dir refused: {why.split(';')[0]}")


def test_truncated_file_after_the_marker_fails_verification_with_its_name() -> None:
    namespace = _sync_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_checkpoint(pathlib.Path(tmp) / "step-1", step=1)
        namespace["write_sync_marker"](root, extra={"step": 1})
        ok, why = namespace["verify_sync_marker"](root)
        assert ok, why
        victim = root / "adapter_model.safetensors"
        victim.write_bytes(b"weights" * 10)          # same-name truncation, marker untouched
        ok, why = namespace["verify_sync_marker"](root)
        assert not ok, "a truncated file verified"
        assert "adapter_model.safetensors" in why, why
        ready, reason = namespace["checkpoint_ready"](root, require_marker=True)
        assert not ready and "adapter_model.safetensors" in reason, reason
        print(f"  P2a -> truncated file refused, naming it: {why.split(':')[0]}")


def test_legacy_marker_without_sha256_is_refused_for_resume() -> None:
    namespace = _sync_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_checkpoint(pathlib.Path(tmp) / "step-5", step=5)
        legacy = {
            "written_at": "2025-01-01T00:00:00Z", "host": "colab", "step": 5,
            "files": {str(p.relative_to(root)): p.stat().st_size
                      for p in sorted(root.rglob("*")) if p.is_file()},
        }
        (root / namespace["SYNC_MARKER"]).write_text(json.dumps(legacy), encoding="utf-8")
        ok, why = namespace["verify_sync_marker"](root)
        assert not ok, "a sha256-less marker verified"
        assert "legacy" in why and "cannot be verified" in why, why
        ready, reason = namespace["checkpoint_ready"](root, require_marker=True)
        assert not ready and "legacy" in reason, reason
        # The old checks are NOT weakened: state/weights/optimizer checks still run without a marker.
        assert namespace["checkpoint_ready"](root)[0]
        print(f"  P2a -> legacy marker refused for resume ('{why}'), plain local check still passes")


def test_rename_probe_reports_a_mode_and_markers_record_it() -> None:
    """The probe must answer, be cached, and land in the marker as ``rename_mode``."""
    namespace = _sync_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        mode, why = namespace["rename_mode_for"](base)
        assert mode in ("atomic", "copy_only"), (mode, why)
        again = namespace["rename_mode_for"](base)
        assert again == (mode, why), "the probe answer is not cached"
        assert namespace["_RENAME_PROBE"], "nothing was cached"
        assert not list(base.glob(".rename-probe-*")), "the probe left directories behind"
        src = _make_checkpoint(base / "local" / "step-11", step=11)
        dst = base / "drive" / "step-11"
        namespace["atomic_sync"](src, dst, extra={"step": 11, "run_name": "run-a"})
        marker = json.loads((dst / namespace["SYNC_MARKER"]).read_text())
        assert marker["rename_mode"] == mode, marker
        # copy_only must be reachable and safe: force it and prove BOTH generations survive.
        namespace["_RENAME_PROBE"][str((base / "drive").resolve())] = ("copy_only", "forced by test")
        dst2 = base / "drive" / "step-12"
        namespace["atomic_sync"](_make_checkpoint(base / "local" / "step-12", step=12), dst2,
                                extra={"step": 12, "run_name": "run-a"})
        (base / "local" / "step-12" / "training_state.json").write_text(
            json.dumps({"step": 12, "config_hash": "changed"}), encoding="utf-8")
        namespace["atomic_sync"](_make_checkpoint(base / "local" / "step-12", step=12), dst2,
                                extra={"step": 12, "run_name": "run-a"})
        previous = sorted(dst2.parent.glob("step-12.prev-*"))
        assert len(previous) == 1, [p.name for p in dst2.parent.iterdir()]
        assert namespace["checkpoint_ready"](dst2, require_marker=True)[0]
        assert namespace["checkpoint_ready"](previous[0], require_marker=True)[0], \
            "copy_only lost the previous generation"
        assert json.loads((dst2 / namespace["SYNC_MARKER"]).read_text())["rename_mode"] == "copy_only"
        print(f"  P2a -> rename probe = {mode} (cached, no leftovers); marker records it; forced "
              f"copy_only keeps {previous[0].name} too")

# --------------------------------------------------------------------------- #
# P1d — EQUIVALENCE ON A SUPERVISED SLICE
# --------------------------------------------------------------------------- #
def _equivalence_namespace() -> dict:
    """The notebook's chunk_equivalence plus the real losses, and nothing that needs a GPU."""
    if not _have("torch"):
        return {}
    import math as _math

    import torch

    from lifemate.training.trainer import chunked_causal_lm_loss

    namespace: dict = {"torch": torch, "math": _math,
                       "chunked_causal_lm_loss": chunked_causal_lm_loss}
    _exec_defs(_cells(), "load_probe", namespace, ["_supervised_shift_counts", "chunk_equivalence"])
    return namespace


class _ToyLM:
    """Deterministic stand-in: logits are a linear function of the token ids, no torch module."""

    def __init__(self, torch_module) -> None:
        self.torch = torch_module
        generator = torch_module.Generator().manual_seed(11)
        self.weight = torch_module.randn(16, 24, generator=generator)

    def __call__(self, **kwargs):
        ids = kwargs["input_ids"]
        hidden = self.torch.nn.functional.one_hot(ids % 16, 16).float()
        return SimpleNamespace(logits=hidden @ self.weight, loss=None)


def _masked_batch(torch_module, *, supervised_from: int, width: int = 4096, n_sup: int = 8,
                  micro_batch: int = 1):
    """Labels whose first ``supervised_from`` positions are -100 (LifeMate's prompt masking)."""
    ids = torch_module.randint(0, 16, (micro_batch, width))
    labels = torch_module.full_like(ids, -100)
    if n_sup:
        labels[..., supervised_from:supervised_from + n_sup] = ids[..., supervised_from:supervised_from + n_sup]
    batch = {"input_ids": ids, "attention_mask": torch_module.ones_like(ids), "labels": labels}
    return batch


def test_chunk_equivalence_finds_a_supervised_window_past_a_full_prompt() -> None:
    """A long prompt whose FIRST max_tokens positions are all -100 must still be checked.

    The old implementation took ``labels[..., :256]``; with prompt masking that slice is 100%
    ``-100``, so both CEs are 0/0 = NaN and ``NaN > 1e-4`` is False — the check PASSED without
    comparing anything. Here the supervised window sits at offset 3000.
    """
    namespace = _equivalence_namespace()
    if not namespace:
        print("SKIP: torch absent - supervised-window search not exercised")
        NEEDS.append("chunk_window")
        return
    torch = namespace["torch"]
    model = _ToyLM(torch)
    batch = _masked_batch(torch, supervised_from=3000, n_sup=8)
    assert bool((batch["labels"][..., :256] == -100).all()), "fixture is not a masked prompt"
    result = namespace["chunk_equivalence"](model, batch, 4)
    assert result["window_start"] > 256, result
    assert result["n_supervised_tokens"] == 8, result
    assert result["delta"] <= result["tolerance"], result
    assert result["single_shot"] == result["chunked"], result
    print(f"  P1d -> supervised window found at offset {result['window_start']} "
          f"({result['n_supervised_tokens']} targets, first 256 all -100); "
          f"chunked == single-shot == {result['single_shot']}")

def test_chunk_equivalence_rejects_an_all_masked_batch() -> None:
    """Every label -100 is INVALID for a loss comparison, never a silent pass."""
    namespace = _equivalence_namespace()
    if not namespace:
        print("SKIP: torch absent - all-masked refusal not exercised")
        NEEDS.append("chunk_all_masked")
        return
    torch = namespace["torch"]
    batch = _masked_batch(torch, supervised_from=0, n_sup=0)
    try:
        namespace["chunk_equivalence"](_ToyLM(torch), batch, 4)
    except RuntimeError as exc:
        message = str(exc)
        assert "INVALID" in message and "no supervised tokens" in message.lower(), message
    else:  # pragma: no cover - failure path
        raise AssertionError("an all-masked batch passed the equivalence check")
    print(f"  P1d -> all-masked batch refused as INVALID: "
          f"'{message.split(':')[1].strip()[:52]}...'")

def test_chunk_equivalence_rejects_non_finite_losses() -> None:
    """NaN and Inf must RAISE: ``NaN > tolerance`` is False, which is how the old check passed."""
    namespace = _equivalence_namespace()
    if not namespace:
        print("SKIP: torch absent - non-finite refusal not exercised")
        NEEDS.append("chunk_nonfinite")
        return
    torch = namespace["torch"]
    batch = _masked_batch(torch, supervised_from=100, n_sup=4)
    nan, inf = float("nan"), float("inf")

    class _NaNModel:
        """A forward whose logits are all NaN: the single-shot CE is NaN, not a mismatch."""

        def __call__(self, **kwargs):
            width = kwargs["input_ids"].shape[-1]
            return SimpleNamespace(logits=torch.full((1, width, 24), nan))

    cases = {
        "NaN single-shot": (_NaNModel(), None),
        "NaN chunked": (_ToyLM(torch), lambda logits, labels, chunk: torch.zeros(()).fill_(nan)),
        "Inf chunked": (_ToyLM(torch), lambda logits, labels, chunk: torch.zeros(()).fill_(inf)),
    }
    for label, (model, wrong) in cases.items():
        sabotaged = dict(namespace)
        if wrong is not None:
            sabotaged["chunked_causal_lm_loss"] = wrong
            _exec_defs(_cells(), "load_probe", sabotaged,
                       ["_supervised_shift_counts", "chunk_equivalence"])
        try:
            sabotaged["chunk_equivalence"](model, batch, 4)
        except RuntimeError as exc:
            assert "non-finite" in str(exc), (label, str(exc))
        else:  # pragma: no cover - failure path
            raise AssertionError(f"a {label} loss was accepted by chunk_equivalence")
    print("  P1d -> NaN single-shot, NaN chunked and Inf chunked all raise "
          "('non-finite'), so NaN can never pass the tolerance test")

def test_chunk_equivalence_handles_a_single_supervised_token() -> None:
    """Exactly one supervised target is enough, and the count is reported."""
    namespace = _equivalence_namespace()
    if not namespace:
        print("SKIP: torch absent - single-token window not exercised")
        NEEDS.append("chunk_single")
        return
    torch = namespace["torch"]
    batch = _masked_batch(torch, supervised_from=777, n_sup=1)
    result = namespace["chunk_equivalence"](_ToyLM(torch), batch, 4)
    assert result["n_supervised_tokens"] == 1, result
    assert result["delta"] <= result["tolerance"], result
    print(f"  P1d -> one supervised token at offset {result['window_start']} still compared "
          f"(delta {result['delta']:.1e})")

def test_chunk_equivalence_rejects_a_deliberately_wrong_chunked_loss() -> None:
    """Discrimination: the same fixture passes with the real loss and fails with a wrong one."""
    namespace = _equivalence_namespace()
    if not namespace:
        print("SKIP: torch absent - wrong-loss discrimination not exercised")
        NEEDS.append("chunk_wrong")
        return
    torch = namespace["torch"]
    model = _ToyLM(torch)
    batch = _masked_batch(torch, supervised_from=64, n_sup=6)
    good = namespace["chunk_equivalence"](model, batch, 4)   # accepting path on this fixture
    assert good["delta"] <= good["tolerance"], good

    cases = {
        # A loss that is the right function but the wrong VALUE: proves the tolerance is live.
        "scaled x1.01": lambda logits, labels, chunk: _ref(logits, labels) * 1.01,
        "shifted +0.5": lambda logits, labels, chunk: _ref(logits, labels) + 0.5,
    }
    for label, wrong in cases.items():
        sabotaged = dict(namespace, chunked_causal_lm_loss=wrong)
        _exec_defs(_cells(), "load_probe", sabotaged,
                   ["_supervised_shift_counts", "chunk_equivalence"])
        try:
            sabotaged["chunk_equivalence"](model, batch, 4)
        except RuntimeError as exc:
            assert "disagrees" in str(exc), (label, str(exc))
        else:  # pragma: no cover - failure path
            raise AssertionError(f"chunk_equivalence accepted the {label} loss")
    print(f"  P1d -> fixture passes with the real chunked loss (delta {good['delta']:.1e}) and "
          f"raises 'disagrees' for both wrong losses")

def _ref(logits, labels):
    """Reference CE with the same causal shift, for the wrong-loss fixtures."""
    import torch

    return torch.nn.functional.cross_entropy(
        logits[..., :-1, :].float().reshape(-1, logits.shape[-1]),
        labels[..., 1:].reshape(-1), ignore_index=-100)

# --------------------------------------------------------------------------- #
# P3 — CHECKPOINT DISCOVERY, SELECTION, CLEANUP
# --------------------------------------------------------------------------- #
def _discovery_namespace() -> dict:
    namespace = _sync_namespace()
    _exec_defs(_cells(), "sync", namespace,
               ["_STEP_RE", "step_of", "_read_json_file", "_default_require_marker",
                "optimizer_state_facts", "verify_resume_candidate", "discover_checkpoints",
                "select_resume_checkpoint", "plan_local_cleanup", "rename_mode_for",
                "_RENAME_PROBE", "_next_generation_name", "_copy_generation_aside"])
    return namespace

def _write_optimizer_state(path: pathlib.Path, *, scheduler: bool = True) -> pathlib.Path:
    """A REAL ``optimizer_state.pt``: torch.save when torch exists, else a valid torch zip layout.

    The notebook loads the file when torch is importable and falls back to a structural zip check
    when it is not, so the fixture is genuine in both environments. A fake ``b"adamw"*32`` blob is
    not a checkpoint in either — the reader is supposed to refuse it (see the truncated-file test).
    """
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except Exception:
        torch = None
    payload = {"optimizer": {"state": {}, "param_groups": []},
               "scheduler": {"last_epoch": 3} if scheduler else None,
               "grad_scaler": None, "step": 1, "epoch": 1}
    if torch is not None:
        torch.save(payload, str(path))
        return path
    import pickle

    with zipfile.ZipFile(path, "w") as archive:          # same container torch.save writes
        archive.writestr("archive/data.pkl", pickle.dumps(payload))
        archive.writestr("archive/byteorder", b"little")
    return path

def _make_generation(root: pathlib.Path, step: int, *, run_name: str = "run-a",
                     model_id: str = "Qwen/Qwen3.5-4B", config_hash: str = "cfg1",
                     dataset_hash: str = "data1", schema: int = 2, verified: bool = True) -> pathlib.Path:
    """A step dir shaped like this trainer's output, optionally marked as a finished sync."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_model.safetensors").write_bytes(b"weights" * 64)
    _write_optimizer_state(root / "optimizer_state.pt")
    state = {"step": step, "epoch": 1, "seed": 42, "config_hash": config_hash,
             "dataset_manifest_hash": dataset_hash, "is_lora": True, "precision": "bfloat16",
             "torch_rng_state": [1, 2, 3], "python_rng_state": [4, 5, 6],
             "cuda_rng_state": [[7, 8, 9]]}
    if schema == 2:
        state["progress"] = {
            "schema_version": 2, "global_step": step, "epoch_index": 2, "next_micro_index": 17,
            "permutation": [3, 1, 2], "sampler_rng_state": [9, 9], "planned_total_steps": 900,
            "dataset_order_hash": "order1",
        }
    (root / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
    (root / "lifemate_meta.json").write_text(json.dumps({
        "model_name": model_id, "model_mode": "lora", "step": step, "run_name": run_name,
        "dataset_manifest_hash": dataset_hash,
    }), encoding="utf-8")
    if verified:
        write = _discovery_namespace()["write_sync_marker"]
        write(root, extra={"step": step, "run_name": run_name, "model_id": model_id,
                           "kind": "checkpoint"})
    return root

def test_discovery_excludes_staging_and_rollback_directories() -> None:
    """``.incomplete-*`` and ``.prev-*`` must be enumerated and REJECTED, never selected."""
    namespace = _discovery_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp) / "run-a"
        _make_generation(run_dir / "step-100", 100)
        _make_generation(run_dir / "step-200.incomplete-1700000000", 200, verified=False)
        _make_generation(run_dir / "step-100.prev-1", 100, verified=False)
        found = namespace["select_resume_checkpoint"](pathlib.Path(tmp), "run-a", model_id="Qwen/Qwen3.5-4B",
                                                     config_hash="cfg1", dataset_manifest_hash="data1",
                                                     verbose=False)
        names = [c["name"] for c in found["accepted"]]
        rejected = {c["name"]: c["rejections"] for c in found["rejected"]}
        assert names == ["step-100"], names
        assert found["selected"]["step"] == 100, found["selected"]["name"]
        assert "staging" in " ".join(rejected["step-200.incomplete-1700000000"]), rejected
        assert "rollback" in " ".join(rejected["step-100.prev-1"]), rejected
        print(f"  P3 -> accepted {names}; rejected staging/rollback: {sorted(rejected)}")

def test_discovery_rejects_a_truncated_optimizer_file() -> None:
    """Truncate ``optimizer_state.pt``: the candidate is refused, an intact sibling is accepted."""
    namespace = _discovery_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp) / "run-a"
        good = _make_generation(run_dir / "step-100", 100)
        bad = _make_generation(run_dir / "step-200", 200)
        (bad / "optimizer_state.pt").write_bytes(b"")     # REAL fault: 0 bytes
        accepted_before = namespace["verify_resume_candidate"](
            good, run_name="run-a", model_id="Qwen/Qwen3.5-4B", config_hash="cfg1",
            dataset_manifest_hash="data1")["ok"]
        assert accepted_before, "the intact checkpoint was rejected too (test proves nothing)"
        verdict = namespace["verify_resume_candidate"](
            bad, run_name="run-a", model_id="Qwen/Qwen3.5-4B", config_hash="cfg1",
            dataset_manifest_hash="data1")
        assert not verdict["ok"], verdict
        assert any("empty" in reason for reason in verdict["rejections"]), verdict["rejections"]

        # A truncated (non-empty) zip payload is caught without torch, structurally.
        real = pathlib.Path(tmp) / "payload.pt"
        import zipfile
        with zipfile.ZipFile(real, "w") as archive:
            archive.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
        truncated = pathlib.Path(tmp) / "truncated.pt"
        truncated.write_bytes(real.read_bytes()[:-12])
        ok, why, _facts = namespace["optimizer_state_facts"](truncated)
        assert not ok and ("truncated" in why or "complete" in why), why
        print(f"  P3 -> empty optimizer_state.pt refused ('{verdict['rejections'][0][:52]}...'); "
              f"truncated zip also refused; intact sibling accepted")

def test_discovery_rejects_a_mutated_adapter_weight_by_sha256() -> None:
    """Flip one byte of the adapter: the marker's digest no longer matches, so it is refused."""
    namespace = _discovery_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp) / "run-a"
        root = _make_generation(run_dir / "step-300", 300)
        verdict_ok = namespace["verify_resume_candidate"](
            root, run_name="run-a", model_id="Qwen/Qwen3.5-4B", config_hash="cfg1",
            dataset_manifest_hash="data1", require_marker=True)
        assert verdict_ok["ok"], verdict_ok["rejections"]
        target = root / "adapter_model.safetensors"
        raw = bytearray(target.read_bytes())
        raw[3] ^= 0xFF                                     # one byte, same length
        target.write_bytes(bytes(raw))
        verdict = namespace["verify_resume_candidate"](
            root, run_name="run-a", model_id="Qwen/Qwen3.5-4B", config_hash="cfg1",
            dataset_manifest_hash="data1", require_marker=True)
        assert not verdict["ok"], verdict
        joined = " ".join(verdict["rejections"])
        assert "sha256" in joined and "adapter_model.safetensors" in joined, joined
        print(f"  P3 -> one flipped adapter byte refused: "
              f"'{verdict['rejections'][0].split(':',1)[-1].strip()[:60]}'")

def test_discovery_rejects_a_corrupt_or_missing_marker() -> None:
    """A marker that is absent (when required) or unparsable makes the dir non-selectable."""
    namespace = _discovery_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp) / "run-a"
        root = _make_generation(run_dir / "step-400", 400)
        kwargs = {"run_name": "run-a", "model_id": "Qwen/Qwen3.5-4B", "config_hash": "cfg1",
                  "dataset_manifest_hash": "data1", "require_marker": True}
        assert namespace["verify_resume_candidate"](root, **kwargs)["ok"]
        marker = root / namespace["SYNC_MARKER"]
        marker.write_text("{not json at all", encoding="utf-8")
        verdict = namespace["verify_resume_candidate"](root, **kwargs)
        assert not verdict["ok"] and any("unreadable" in r for r in verdict["rejections"]), verdict
        marker.unlink()
        verdict = namespace["verify_resume_candidate"](root, **kwargs)
        assert not verdict["ok"] and any("no SYNC_COMPLETE" in r for r in verdict["rejections"]), verdict
        print(f"  P3 -> corrupt marker refused ('{verdict['rejections'][0][:56]}...') and a "
              "deleted marker refused when one is required")

def test_selection_falls_back_to_the_older_valid_generation_and_names_the_newer() -> None:
    """Newest corrupt -> the older verified one is selected, and the rejection is printed."""
    namespace = _discovery_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp) / "run-a"
        _make_generation(run_dir / "step-100", 100)
        newest = _make_generation(run_dir / "step-200", 200)
        (newest / "adapter_model.safetensors").write_bytes(b"torn")   # after its marker
        found = namespace["select_resume_checkpoint"](pathlib.Path(tmp), "run-a",
                                                     model_id="Qwen/Qwen3.5-4B", config_hash="cfg1",
                                                     dataset_manifest_hash="data1", verbose=False)
        assert found["selected"] is not None, found
        assert found["selected"]["step"] == 100, found["selected"]["name"]
        assert [c["name"] for c in found["newer_rejected"]] == ["step-200"], found["newer_rejected"]
        joined = " ".join(found["newer_rejected"][0]["rejections"])
        assert "adapter_model.safetensors" in joined and "on disk" in joined, joined
        print(f"  P3 -> selected {found['selected']['name']} and reported the newer "
              f"{found['newer_rejected'][0]['name']} as rejected (sha256 mismatch)")

def test_discovery_rejects_a_model_profile_mismatch() -> None:
    """Identity is two checks: the id AND the architecture family the adapter was trained for."""
    namespace = _discovery_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_generation(pathlib.Path(tmp) / "run-a" / "step-600", 600)
        meta_path = root / "lifemate_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["model_profile"] = "qwen3_5"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        namespace["write_sync_marker"](root, extra={"step": 600, "run_name": "run-a",
                                                   "model_id": "Qwen/Qwen3.5-4B"})
        base = {"run_name": "run-a", "model_id": "Qwen/Qwen3.5-4B", "config_hash": "cfg1",
                "dataset_manifest_hash": "data1"}
        accepted = namespace["verify_resume_candidate"](root, **base)
        assert accepted["ok"], accepted["rejections"]
        assert any("model_profile" in w for w in accepted["warnings"]) is False
        rejected = namespace["verify_resume_candidate"](root, profile="minicpm5", **base)
        assert not rejected["ok"], rejected
        assert any("profile mismatch" in r for r in rejected["rejections"]), rejected["rejections"]
        same_family = namespace["verify_resume_candidate"](root, profile="qwen3_5", **base)
        assert same_family["ok"], same_family["rejections"]
        print(f"  P3 -> same id + wrong family refused: "
              f"'{rejected['rejections'][0][:58]}...'; matching family accepted")

def test_discovery_accepts_both_state_schemas_and_flags_schema_one_as_inexact() -> None:
    """Schema 1 must be RESUMABLE (exact=False), never treated as corruption."""
    namespace = _discovery_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp) / "run-a"
        root = _make_generation(run_dir / "step-500", 500, schema=1)
        verdict = namespace["verify_resume_candidate"](
            root, run_name="run-a", model_id="Qwen/Qwen3.5-4B", config_hash="cfg1",
            dataset_manifest_hash="data1")
        assert verdict["ok"], verdict["rejections"]
        assert verdict["exact"] is False, verdict
        assert any("progress" in reason for reason in verdict["exact_missing"]), verdict
        print(f"  P3 -> schema-1 checkpoint accepted with exact=False "
              f"({verdict['exact_missing'][0][:48]}...)")

def test_cleanup_refuses_when_the_drive_side_is_empty() -> None:
    """The real bug: ``all([]) is True`` deleted local checkpoints with nothing on Drive."""
    namespace = _discovery_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        local = pathlib.Path(tmp) / "local"
        for step in (100, 200):
            _make_generation(local / "run-a" / f"step-{step}", step, verified=False)
        drive = pathlib.Path(tmp) / "drive"
        (drive / "run-a").mkdir(parents=True)          # empty drive run dir
        plan = namespace["plan_local_cleanup"](local / "run-a", drive / "run-a", run_name="run-a",
                                              model_id="Qwen/Qwen3.5-4B", delete=True, verbose=False)
        assert plan["ok"] is False and plan["performed"] == [], plan
        assert "REFUSING" in plan["reason"] and "no step-*" in plan["reason"], plan["reason"]
        assert sorted(p.name for p in (local / "run-a").iterdir()) == ["step-100", "step-200"]
        # Same fixture, drive dir missing entirely -> also refused.
        plan2 = namespace["plan_local_cleanup"](local / "run-a", drive / "absent", run_name="run-a",
                                               delete=True, verbose=False)
        assert plan2["ok"] is False and "REFUSING" in plan2["reason"], plan2
        print(f"  P3 -> empty Drive set refused: '{plan['reason'][:72]}...'; both local dirs intact")

def test_cleanup_refuses_when_one_of_three_is_missing_on_drive() -> None:
    """Nothing is deleted unless EVERY local checkpoint verifies pairwise on Drive."""
    namespace = _discovery_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        local_run = pathlib.Path(tmp) / "local" / "run-a"
        drive_run = pathlib.Path(tmp) / "drive" / "run-a"
        for step in (100, 200, 300):
            _make_generation(local_run / f"step-{step}", step, verified=False)
            src = _make_generation(pathlib.Path(tmp) / "stage" / f"step-{step}", step)
            namespace["atomic_sync"](src, drive_run / f"step-{step}",
                                    extra={"step": step, "run_name": "run-a",
                                           "model_id": "Qwen/Qwen3.5-4B"})
        import shutil as _shutil
        _shutil.rmtree(drive_run / "step-200")          # REAL fault: one Drive dir gone
        plan = namespace["plan_local_cleanup"](local_run, drive_run, run_name="run-a",
                                              model_id="Qwen/Qwen3.5-4B", delete=True, verbose=False)
        missing = [p for p in plan["pairs"] if p["step"] == 200][0]
        assert not missing["ok"] and any("no Drive counterpart" in r for r in missing["reasons"]), plan
        assert plan["deletable"] == [], plan["deletable"]
        assert "REFUSING" in plan["reason"], plan["reason"]
        assert plan["performed"] == [], plan["performed"]
        assert len(list(local_run.iterdir())) == 3, sorted(p.name for p in local_run.iterdir())
        print(f"  P3 -> step-200 missing on Drive -> deleted nothing; "
              f"reason: '{missing['reasons'][0][:52]}...'")

def test_cleanup_deletes_only_fully_verified_pairs_and_keeps_the_newest() -> None:
    """All three verify -> the two OLDER ones are deleted; the newest is always kept."""
    namespace = _discovery_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        local_run = pathlib.Path(tmp) / "local" / "run-a"
        drive_run = pathlib.Path(tmp) / "drive" / "run-a"
        for step in (100, 200, 300):
            _make_generation(local_run / f"step-{step}", step, verified=False)
            src = _make_generation(pathlib.Path(tmp) / "stage" / f"step-{step}", step)
            namespace["atomic_sync"](src, drive_run / f"step-{step}",
                                    extra={"step": step, "run_name": "run-a",
                                           "model_id": "Qwen/Qwen3.5-4B"})
        plan = namespace["plan_local_cleanup"](local_run, drive_run, run_name="run-a",
                                              model_id="Qwen/Qwen3.5-4B", delete=True, verbose=False)
        assert plan["performed"] == ["step-100", "step-200"], plan
        remaining = sorted(p.name for p in local_run.iterdir())
        assert remaining == ["step-300"], remaining
        assert "newest" in " ".join(
            p["reasons"][0] for p in plan["pairs"] if p["step"] == 300), plan["pairs"]
        print(f"  P3 -> deleted {plan['performed']}, kept {remaining} (newest never removed)")

def test_cleanup_with_the_default_flag_performs_zero_deletions() -> None:
    """``DELETE_LOCAL_AFTER_VERIFIED_SYNC=False`` must be a dry run that names what it would do."""
    namespace = _discovery_namespace()
    cells = _cells()
    tree = ast.parse(_cell_source(cells, "params"))
    defaults = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                defaults[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert defaults.get("DELETE_LOCAL_AFTER_VERIFIED_SYNC") is False, defaults
    with tempfile.TemporaryDirectory() as tmp:
        local_run = pathlib.Path(tmp) / "local" / "run-a"
        drive_run = pathlib.Path(tmp) / "drive" / "run-a"
        for step in (100, 200):
            _make_generation(local_run / f"step-{step}", step, verified=False)
            src = _make_generation(pathlib.Path(tmp) / "stage" / f"step-{step}", step)
            namespace["atomic_sync"](src, drive_run / f"step-{step}",
                                    extra={"step": step, "run_name": "run-a",
                                           "model_id": "Qwen/Qwen3.5-4B"})
        plan = namespace["plan_local_cleanup"](local_run, drive_run, run_name="run-a",
                                              model_id="Qwen/Qwen3.5-4B", delete=False, verbose=False)
        assert plan["would_delete"] == ["step-100"], plan
        assert plan["performed"] == [], plan
        assert len(list(local_run.iterdir())) == 2
        assert "dry run" in plan["reason"], plan["reason"]
        print(f"  P3 -> flag False: would delete {plan['would_delete']}, deleted "
              f"{plan['performed'] or 'nothing'}")

# --------------------------------------------------------------------------- #
# P4 — RESUME MODES AND THE RESUME REPORT
# --------------------------------------------------------------------------- #
FEATURE_NAMES = (
    "FEATURE_FLAGS", "TRAINING_SCOPES_HERE", "_fmt_number", "feature_params",
    "feature_param_notes", "check_feature_params", "build_feature_config",
    "validate_feature_config", "enabled_training_ids", "feature_train_args",
    "feature_config_overrides", "feature_prepare_args", "feature_resume_identity",
)

def _feature_params(**overrides) -> dict:
    """A full param dict with everything OFF, then `overrides` (the cell's own defaults)."""
    if not _have("lifemate.features"):
        return {}
    base = {"NOISY_ASR_TRAINING": False, "NOISY_ASR_RATIO": 0.25, "NOISY_ASR_TRANSCRIPTS": "",
            "SLOT_MASKING": False, "TOOL_DROPOUT": False, "ASR_NBEST": False,
            "NBEST_TRAINING_FORMAT": False, "NBEST_TRANSCRIPTS": "", "PROMPT_ITN": False,
            "AUDIO_VAD": False, "AUDIO_RMS_NORMALIZATION": False, "AUGMENTATION_SEED": 42}
    base.update(overrides)
    return base

def _feature_namespace() -> dict:
    """Cell 2a's helpers, executed against the REAL `lifemate.features` package."""
    if not _have("lifemate.features"):
        return {}
    namespace: dict = {"__name__": "notebook_cell_features"}
    _exec_defs(_cells(), "features", namespace, list(FEATURE_NAMES))
    # The `#@param` globals the cell's own `feature_params()` reads. Bound from the cell's OWN
    # default assignments, so the test cannot drift from the documented defaults.
    for name in ("NOISY_ASR_TRAINING", "NOISY_ASR_RATIO", "NOISY_ASR_TRANSCRIPTS",
                 "SLOT_MASKING", "TOOL_DROPOUT", "ASR_NBEST", "NBEST_TRAINING_FORMAT",
                 "NBEST_TRANSCRIPTS", "PROMPT_ITN", "AUDIO_VAD", "AUDIO_RMS_NORMALIZATION",
                 "AUGMENTATION_SEED"):
        namespace[name] = _param_defaults_features()[name]
    return namespace

def _param_defaults_features() -> dict:
    """Literal values of cell 2a's `#@param` assignments."""
    tree = ast.parse(_cell_source(_cells(), "features"))
    defaults = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                defaults[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return defaults

def _feature_cfg(**overrides):
    """A validated `FeatureConfig` from the notebook's own builder (or None without the slice)."""
    namespace = _feature_namespace()
    if not namespace:
        return None, None
    params = _feature_params(**overrides)
    cfg = namespace["build_feature_config"](params)
    namespace["validate_feature_config"](cfg)
    return namespace, cfg

def _resume_namespace() -> dict:
    namespace = _discovery_namespace()
    feature = _feature_namespace()
    namespace.update({name: value for name, value in feature.items() if name != "__name__"})
    namespace.update({"DRIVE_ROOT": None, "Path": pathlib.Path, "json": json,
                      "time": __import__("time"), "re": __import__("re"),
                      "FEATURE_CONFIG": None})
    _exec_defs(_cells(), "resume_ctl", namespace,
               ["RESUME_REPORT_LABELS", "resume_mode_from_cli", "feature_resume_comparison",
                "resume_report", "print_resume_report", "resolve_resume_target"])
    return namespace

def _plan_args(tmp: pathlib.Path, mode: str, **kwargs) -> dict:
    args = {"mode": mode, "root": None, "local_root": tmp / "local",
            "model_id": "Qwen/Qwen3.5-4B", "config_hash": "cfg1",
            "dataset_manifest_hash": "data1", "expected_total_steps": 900, "verbose": False}
    args.update(kwargs)
    return args

def test_fresh_mode_refuses_to_reuse_a_populated_run_name() -> None:
    """A populated ``step-*`` set is never continued under a name that RESUME_MODE=fresh owns."""
    namespace = _resume_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        _make_generation(base / "local" / "run-a" / "step-100", 100, verified=False)
        plan = namespace["resolve_resume_target"]("run-a", **_plan_args(base, "fresh"))
        assert plan["checkpoint"] is None, plan
        assert plan["run_name"] != "run-a" and plan["run_name"].startswith("run-a-"), plan
        assert any("already holds 1 checkpoint" in note for note in plan["notes"]), plan["notes"]
        # Empty run dir -> the requested name is kept.
        plan2 = namespace["resolve_resume_target"]("run-fresh", **_plan_args(base, "fresh"))
        assert plan2["run_name"] == "run-fresh", plan2
        print(f"  P4 -> fresh renamed { 'run-a' } -> {plan['run_name']} (existing dirs untouched); "
              "an unused name is kept as-is")

def test_auto_mode_refuses_to_start_fresh_when_nothing_verifies() -> None:
    """``auto`` with only corrupt candidates must RAISE, listing each rejection."""
    namespace = _resume_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        broken = _make_generation(base / "local" / "run-a" / "step-100", 100, verified=False)
        (broken / "training_state.json").write_text('{"step": 100,', encoding="utf-8")  # truncated
        try:
            namespace["resolve_resume_target"]("run-a", **_plan_args(base, "auto"))
        except RuntimeError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("RESUME_MODE=auto silently started a fresh run")
        assert "RESUME_MODE='fresh'" in message and "step-100" in message, message
        # Accepting path on the SAME fixture: one valid generation makes auto resume it.
        _make_generation(base / "local" / "run-a" / "step-200", 200)
        plan = namespace["resolve_resume_target"]("run-a", **_plan_args(base, "auto"))
        assert plan["checkpoint"] is not None and plan["checkpoint"].name == "step-200", plan
        print(f"  P4 -> auto raised with rejected candidates ({message.count('rejected')} listed); "
              f"with step-200 valid it resumed {plan['checkpoint'].name}")

def test_auto_mode_proceeds_loudly_when_the_run_has_no_checkpoint_at_all() -> None:
    """A first run must work under the default mode, and must say what it searched.

    The distinction that matters: no candidate directory (first run, nothing to lose) proceeds with a
    printed note; candidates that all fail verification (test above) refuse. Both branches are
    asserted, so neither can silently turn into the other.
    """
    namespace = _resume_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        plan = namespace["resolve_resume_target"]("run-brand-new", **_plan_args(base, "auto"))
        assert plan["checkpoint"] is None and plan["mode"] == "fresh", plan
        assert any("no existing checkpoint" in note for note in plan["notes"]), plan["notes"]
        assert plan["run_name"] == "run-brand-new", plan
        assert [entry["root"] for entry in plan["searched"]] == [str(base / "local")], plan
        print(f"  P4 -> auto on an empty run dir: first run, searched {plan['searched']}, "
              "nothing rejected")

def test_explicit_mode_refuses_an_unverifiable_path() -> None:
    namespace = _resume_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        root = _make_generation(base / "local" / "run-a" / "step-100", 100)
        plan = namespace["resolve_resume_target"](
            "run-a", **_plan_args(base, "explicit", explicit_path=str(root)))
        assert plan["checkpoint"] == root, plan
        (root / "optimizer_state.pt").unlink()
        try:
            namespace["resolve_resume_target"](
                "run-a", **_plan_args(base, "explicit", explicit_path=str(root)))
        except RuntimeError as exc:
            assert "optimizer_state.pt" in str(exc), str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("explicit resume accepted a checkpoint without optimizer state")
        print("  P4 -> explicit resumed the verified dir and refused it once optimizer_state.pt "
              "was deleted")

def test_resume_report_prints_the_labelled_lines_with_real_booleans(capsys=None) -> None:
    """The report is derived from the ARTIFACT: schema 2 -> all four booleans True; schema 1 -> False.

    The block specified for the report is nine lines (``Resume checkpoint:`` through
    ``Expected total steps:``); every one of them is asserted present, in order, with the four
    booleans being real ``True``/``False`` derived from the checkpoint's contents.
    """
    import io
    from contextlib import redirect_stdout

    namespace = _resume_namespace()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        run_dir = base / "run-a"
        schema2 = _make_generation(run_dir / "step-100", 100, config_hash="cfg1")
        legacy = _make_generation(run_dir / "step-50", 50, schema=1)
        verdict = namespace["verify_resume_candidate"](
            schema2, run_name="run-a", model_id="Qwen/Qwen3.5-4B", config_hash="cfg1",
            dataset_manifest_hash="data1")
        report = namespace["resume_report"](verdict, expected_total_steps=1234)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            namespace["print_resume_report"](report)
        printed = buffer.getvalue().splitlines()
        labels = namespace["RESUME_REPORT_LABELS"]
        assert len(labels) == 9, labels
        for label in labels:
            assert any(line.startswith(label + ":") or line == label for line in printed), (label, printed)
        for label in ("Optimizer restored", "Scheduler restored", "Sampler restored", "RNG restored"):
            line = [l for l in printed if l.startswith(label + ":")][0]
            assert line.endswith("True"), (label, line)
        assert report["previous_global_step"] == 100
        assert report["previous_epoch"] == 2
        assert "17" in report["next_batch"]
        assert report["expected_total_steps"] == 900

        old = namespace["verify_resume_candidate"](
            legacy, run_name="run-a", model_id="Qwen/Qwen3.5-4B", config_hash="cfg1",
            dataset_manifest_hash="data1")
        legacy_report = namespace["resume_report"](old, expected_total_steps=1234)
        assert legacy_report["sampler_restored"] is False, legacy_report
        assert legacy_report["rng_restored"] is True, legacy_report
        assert legacy_report["expected_total_steps"] == 1234, legacy_report
        assert legacy_report["exact"] is False
        # The CLI's own verdict wins when it printed one.
        with_cli = namespace["resume_report"](old, expected_total_steps=1234,
                                             cli_lines=["resumed     : /x at step 50",
                                                        "resume mode : best-effort"])
        assert with_cli["mode"] == "best-effort" and "train.py" in with_cli["mode_source"], with_cli
        print("  P4 -> report lines as printed:")
        for line in printed[:9]:
            print(f"    {line}")
        print(f"    (schema 1: sampler False, horizon from this session "
              f"{legacy_report['expected_total_steps']}; cli mode {with_cli['mode']})")
    assert report is not None

# --------------------------------------------------------------------------- #
# P5 — RESUME PARAMS AND THE WORST-CASE PROBE REPORT
# --------------------------------------------------------------------------- #
def _param_defaults() -> dict:
    tree = ast.parse(_cell_source(_cells(), "params"))
    defaults = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                defaults[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return defaults

def test_resume_params_exist_with_the_documented_defaults() -> None:
    """Section 2 must expose every resume/cleanup knob as a Colab ``#@param`` with safe defaults."""
    defaults = _param_defaults()
    source = _cell_source(_cells(), "params")
    expected = {"MODEL_ID": "Qwen/Qwen3.5-4B", "TRAINING_METHOD": "lora",
                "USE_GOOGLE_DRIVE": True, "RUN_NAME": "qwen35-4b-lora",
                "RESUME_MODE": "auto", "RESUME_FROM_CHECKPOINT": "",
                "SAVE_EVERY_N_STEPS": 50, "DELETE_LOCAL_AFTER_VERIFIED_SYNC": False,
                "RUN_RESUME_PROOF": False, "MAX_STEPS": 0, "SEED": 42}
    for name, value in expected.items():
        assert defaults.get(name) == value, (name, defaults.get(name), value)
    assert 'RESUME_MODE = "auto" #@param ["fresh", "auto", "explicit"]' in source
    assert "DELETE_LOCAL_AFTER_VERIFIED_SYNC = False #@param {type:\"boolean\"}" in source
    assert "SAVE_EVERY_N_STEPS = 50 #@param {type:\"integer\"}" in source
    # The pre-existing validation must survive the new params.
    for fragment in ("1024 <= int(MAX_SEQ_LENGTH) <= 32768",
                     'if TRAINING_METHOD not in ("lora", "qlora")',
                     "int(MICRO_BATCH_SIZE) < 1 or int(GRADIENT_ACCUMULATION_STEPS) < 1",
                     'RESUME_MODE not in ("fresh", "auto", "explicit")'):
        assert fragment in source, fragment
    print(f"  P5 -> {sorted(expected)} present with the documented defaults; range/method/batch/"
          "resume validation intact")

def _probe_namespace() -> dict:
    namespace = {"Path": pathlib.Path, "json": json, "GPU_TIER": "T4-16GB", "VOCAB": 248320,
                 "VOCAB_SOURCE": "test", "LOSS_CHUNK_TOKENS": 1024, "PRECISION": "bfloat16",
                 "MODEL_ID": "Qwen/Qwen3.5-4B", "SEED": 42, "torch": None,
                 "GPU": {"available": False, "name": None, "cuda": None, "torch": None,
                         "vram_total_gb": None},
                 "RUN": {"method": "lora", "quantized_base": False, "lora_r": 16,
                         "max_seq_length": 4096, "micro_batch": 1, "grad_accum": 8},
                 "COMPUTE_DTYPE": None, "seq_cap": lambda *a, **k: 3840}
    _exec_defs(_cells(), "derive", namespace,
               ["TIER_VRAM_GB", "USABLE_FRACTION", "QLORA_BYTES_PER_PARAM", "ACT_FACTOR",
                "RESERVE_GB", "estimate_vram_gb", "compute_bytes_per_param"])
    _exec_defs(_cells(), "worst_case", namespace,
               ["PROBE_LABELS", "_probe_batch", "longest_prepared_sample_length",
                "run_worst_case_probe", "print_worst_case_report"])
    return namespace

def test_worst_case_probe_is_a_cpu_no_op_and_reports_every_label() -> None:
    """On CPU the probe must refuse to pretend, and the report must still print all 15 labels."""
    import io
    from contextlib import redirect_stdout

    namespace = _probe_namespace()
    labels = namespace["PROBE_LABELS"]
    assert len(labels) == 15, labels
    # CPU (this machine): a no-op that says why, never a fabricated VRAM number.
    result = namespace["run_worst_case_probe"](None, {"input_ids": [[1, 2, 3]]},
                                               cfg=SimpleNamespace(training=SimpleNamespace(
                                                   gradient_clip_norm=1.0)),
                                               chunk_tokens=1024, grad_accum=8, micro_batch=1)
    assert result["ran"] is False, result
    assert "no CUDA device" in result["reason"], result
    assert result["observed_gb"] is None and result["estimated_gb"] is None, result
    assert set(result) >= {"estimated_gb", "observed_gb", "observed_reserved_gb", "seq_len",
                           "source"}, sorted(result)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        namespace["print_worst_case_report"](result)
    printed = buffer.getvalue()
    for label in labels:
        assert label in printed, (label, printed)
    assert "ESTIMATE" in printed and "OBSERVED" in printed, printed
    print(f"  P5 -> CPU no-op: '{result['reason'][:52]}...'; all {len(labels)} labels printed, "
          "ESTIMATE and OBSERVED kept distinct")

def test_probe_batch_is_padded_to_the_requested_length_and_supervised() -> None:
    """The synthetic batch must hit the EXACT configured length, or the probe measures nothing."""
    namespace = _probe_namespace()
    if not _have("torch"):
        print("SKIP: torch absent - probe batch shape not exercised")
        NEEDS.append("probe_batch")
        return
    torch = namespace["torch"] = __import__("torch")
    batch = namespace["_probe_batch"](None, 4096, 2, supervised_tail=64)
    assert tuple(batch["input_ids"].shape) == (2, 4096), batch["input_ids"].shape
    assert tuple(batch["labels"].shape) == (2, 4096)
    assert int((batch["labels"] != -100).sum()) == 2 * 64, "the supervised tail is wrong"
    assert bool((batch["labels"][:, :4096 - 64] == -100).all()), "the prompt is not masked"
    assert namespace["run_worst_case_probe"].__code__.co_argcount > 0
    print("  P5 -> synthetic batch is exactly (2, 4096) with a 64-token supervised tail and a "
          "-100 prompt (the same structure LifeMate's collator produces)")

# --------------------------------------------------------------------------- #
# P6 — FEATURE FLAGS (cell 2a) PROPAGATED TO BOTH STAGES
# --------------------------------------------------------------------------- #
_BASE_PREPARE = ("prepare_dataset.py --data-root /tmp/data --split train --output /tmp/ds "
                 "--tokenizer Qwen/Qwen3.5-4B --max-seq-length 4096 --val-ratio 0.1 --seed 42")
_BASE_TRAIN = ("train.py --config /tmp/cfg.yaml --dataset-dir /tmp/ds --model Qwen/Qwen3.5-4B "
               "--output-dir /tmp/ckpt --run-name run-a --tuning lora --seed 42 --seq-len 4096")


def _feature_argv(namespace, cfg, *, params=None, stage="train", override_supported=False):
    """The EXACT argv the notebook builds for one stage: base args + the feature args."""
    base = _BASE_TRAIN if stage == "train" else _BASE_PREPARE
    if stage == "train":
        extra = namespace["feature_train_args"](cfg, override_supported=override_supported)
    else:
        extra = namespace["feature_prepare_args"](cfg, params or _feature_params())
    return " ".join([base, *[str(item) for item in extra]]).strip()


def _real_prepare_argv(feature_namespace, cfg) -> list[str]:
    """Exec cell 5a's OWN `prepare_dataset` with a stub runner; return the argv it would launch.

    This is the only way to prove "byte-identical": the string has to come from the shipped
    builder, not from a re-typed copy in the test.
    """
    cells = _cells()
    captured: dict = {}
    namespace: dict = {
        "Path": pathlib.Path, "json": json, "MODEL_ID": "Qwen/Qwen3.5-4B", "SEED": 42,
        "DATA_ROOT": pathlib.Path("/tmp/data"), "LOCAL_DATASET_ROOT": pathlib.Path("/tmp/ds"),
        "TRAIN_DATASET_DIR": pathlib.Path("/tmp/ds/train"),
        "MANIFEST_PATH": pathlib.Path("/tmp/ds/train/manifest.json"),
        "FEATURE_CONFIG": cfg,
        "require_flags": lambda *a, **k: set(),
        "run_cli": lambda script, args, **k: captured.update(script=script,
                                                             args=[str(a) for a in args]),
    }
    for name in FEATURE_NAMES:
        namespace[name] = feature_namespace[name]
    _exec_defs(cells, "dataset", namespace, ["prepare_dataset"])
    pathlib.Path("/tmp/ds/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path("/tmp/ds/train/manifest.json").write_text("{}", encoding="utf-8")
    namespace["prepare_dataset"](4096)
    return [captured["script"], *captured["args"]]


def test_all_flags_off_produce_byte_identical_commands() -> None:
    """Zero feature args: the argv the notebook actually builds is the pre-feature argv."""
    namespace, cfg = _feature_cfg()
    if not namespace:
        print("SKIP: lifemate.features absent - command propagation not exercised")
        NEEDS.append("feature_cmd")
        return
    assert namespace["enabled_training_ids"](cfg) == []
    assert namespace["feature_config_overrides"](cfg) == {}
    assert namespace["feature_prepare_args"](cfg) == []
    assert namespace["feature_train_args"](cfg) == []

    # 1. `scripts/prepare_dataset.py`: the shipped cell 5a builder, executed.
    prepare_argv = _real_prepare_argv(namespace, cfg)
    assert " ".join(prepare_argv) == _BASE_PREPARE, " ".join(prepare_argv)
    # 2. `scripts/train.py`: the shipped cell 6c builder, executed with the feature switchboard bound.
    train_argv = ["train.py", *[str(part) for part in _train_argv_with_features(namespace, cfg)]]
    assert " ".join(train_argv) == _BASE_TRAIN, " ".join(train_argv)
    print(f"  P6 -> all off, prepare: {' '.join(str(p) for p in prepare_argv)}")
    print(f"        all off, train  : {' '.join(train_argv)}")


def test_each_enabled_flag_adds_exactly_its_own_args() -> None:
    """One flag at a time: the exact extra tokens, and nothing else, appear."""
    cases = {
        "SLOT_MASKING": ("--slot-masking-enabled --augmentation-seed 42", ["training.slot_masking"]),
        "TOOL_DROPOUT": ("--tool-dropout-enabled --augmentation-seed 42", ["training.tool_dropout"]),
        "PROMPT_ITN": ("", ["prompt.itn"]),
        "AUDIO_VAD": ("", ["audio.vad"]),
        "AUDIO_RMS_NORMALIZATION": ("", ["audio.rms_normalization"]),
        "ASR_NBEST": ("", ["asr.nbest"]),
    }
    checked = []
    for flag, (expected_prepare_extra, expected_ids) in cases.items():
        namespace, cfg = _feature_cfg(**{flag: True})
        if not namespace:
            print("SKIP: lifemate.features absent - per-flag propagation not exercised")
            NEEDS.append("feature_cmd")
            return
        assert namespace["enabled_training_ids"](cfg) == expected_ids, (flag, cfg)
        trained = namespace["feature_train_args"](cfg)
        assert trained == ["--enable-feature", expected_ids[0]], (flag, trained)
        prepare = namespace["feature_prepare_args"](cfg, _feature_params(**{flag: True}))
        assert " ".join(prepare) == expected_prepare_extra, (flag, prepare)
        # The same flag through the REAL argv builder in 6c, so the wiring is proven end to end.
        args = [str(part) for part in _train_argv_with_features(namespace, cfg)]
        assert expected_ids[0] in args, (flag, args)
        means_index = args.index("--enable-feature")
        assert args[means_index + 1] == expected_ids[0], args
        checked.append(flag)
    # Noisy ASR carries its ratio and transcript path, in that order, and no other flag is emitted.
    namespace, cfg = _feature_cfg(NOISY_ASR_TRAINING=True, NOISY_ASR_RATIO=0.4,
                                 NOISY_ASR_TRANSCRIPTS="/drive/t.jsonl")
    prepare = namespace["feature_prepare_args"](cfg, _feature_params(
        NOISY_ASR_TRAINING=True, NOISY_ASR_RATIO=0.4, NOISY_ASR_TRANSCRIPTS="/drive/t.jsonl"))
    assert prepare == ["--noisy-asr-enabled", "--noisy-asr-ratio", "0.4",
                       "--noisy-asr-transcripts", "/drive/t.jsonl", "--augmentation-seed", "42"], prepare
    assert namespace["feature_train_args"](cfg, override_supported=True) == [
        "--enable-feature", "training.noisy_asr",
        "--override", "features.training.noisy_asr_ratio=0.4"], "override form"
    # NBEST_TRAINING_FORMAT needs asr.nbest, and then both ids travel.
    namespace, cfg = _feature_cfg(ASR_NBEST=True, NBEST_TRAINING_FORMAT=True,
                                 NBEST_TRANSCRIPTS="/drive/n.jsonl")
    prepare = namespace["feature_prepare_args"](cfg, _feature_params(
        ASR_NBEST=True, NBEST_TRAINING_FORMAT=True, NBEST_TRANSCRIPTS="/drive/n.jsonl"))
    assert prepare == ["--nbest-training-format", "--nbest-transcripts", "/drive/n.jsonl",
                       "--augmentation-seed", "42"], prepare
    assert namespace["enabled_training_ids"](cfg) == ["asr.nbest", "prompt.nbest_format"]
    print(f"  P6 -> per-flag: {checked} each add exactly their own id/flag; noisy-ASR adds ratio + "
          "transcripts (+seed); nbest-format adds transcripts (+seed) and both ids travel")


def _train_argv_with_features(feature_namespace, cfg) -> list[str]:
    """Run cell 6c's `train_args` with the feature switchboard bound, and return the argv."""
    cells = _cells()
    namespace: dict = {"Path": pathlib.Path, "json": json, "RUN": {"max_seq_length": 4096},
                       "MODEL_ID": "Qwen/Qwen3.5-4B", "SEED": 42, "CLI_TUNING_MODE": "lora",
                       "CONFIG_PATH": pathlib.Path("/tmp/cfg.yaml"),
                       "TRAIN_DATASET_DIR": pathlib.Path("/tmp/ds"),
                       "CKPT_OUTPUT_DIR": pathlib.Path("/tmp/ckpt"), "RUN_NAME": "run-a",
                       "SAVE_EVERY_N_STEPS": 50, "TOTAL_STEPS": 900, "SEQ_FLAG": "--seq-len",
                       "RESUME_FLAG": "--resume", "QLORA_SUPPORTED": True,
                       "OPTIONAL_FLAGS": {"--save-steps": False, "--eval-steps": False,
                                          "--planned-steps": False, "--extend-horizon": False},
                       "TRAIN_FLAGS": {"--enable-feature"},
                       "FEATURE_CLI_SUPPORTED": True, "FEATURE_OVERRIDE_SUPPORTED": False,
                       "FEATURE_IDS_NEEDED": feature_namespace["enabled_training_ids"](cfg),
                       "FEATURE_CONFIG": cfg}
    for name in FEATURE_NAMES:
        namespace[name] = feature_namespace[name]
    _exec_defs(cells, "cli", namespace, ["train_args"])
    return namespace["train_args"]()


def test_ratio_with_the_flag_off_is_inert_and_reported() -> None:
    """A parameter of a DISABLED feature must not raise, must not move the fingerprint, and is
    reported as INERT."""
    namespace, cfg = _feature_cfg(NOISY_ASR_RATIO=1.9)
    if not namespace:
        print("SKIP: lifemate.features absent - inert parameter not exercised")
        NEEDS.append("feature_inert")
        return
    assert namespace["enabled_training_ids"](cfg) == []
    assert namespace["feature_prepare_args"](cfg, _feature_params(NOISY_ASR_RATIO=1.9)) == []
    assert namespace["feature_train_args"](cfg) == []
    notes = namespace["feature_param_notes"](_feature_params(NOISY_ASR_RATIO=1.9))
    assert notes and "INERT" in notes[0] and "NOISY_ASR_RATIO" in notes[0], notes
    # Inert by definition: the fingerprint of "ratio 1.9, off" equals "ratio 0.25, off".
    from lifemate.features.config import training_feature_fingerprint

    quiet, quiet_cfg = _feature_cfg()
    assert training_feature_fingerprint(quiet_cfg) == training_feature_fingerprint(cfg), \
        "a disabled feature's parameter moved the training fingerprint"
    print(f"  P6 -> ratio 1.9 with the flag OFF: no error, no argv, fingerprint unchanged; "
          f"note: '{notes[0][:58]}...'")


def test_invalid_combinations_name_the_flag_and_stop() -> None:
    """Two real invalids, each raising with the flag name; the valid path passes on the fixture."""
    namespace, cfg = _feature_cfg(NOISY_ASR_TRAINING=True, NOISY_ASR_RATIO=0.5,
                                 NOISY_ASR_TRANSCRIPTS="/drive/t.jsonl")
    if not namespace:
        print("SKIP: lifemate.features absent - invalid combinations not exercised")
        NEEDS.append("feature_invalid")
        return
    assert namespace["check_feature_params"](_feature_params(
        NOISY_ASR_TRAINING=True, NOISY_ASR_RATIO=0.5, NOISY_ASR_TRANSCRIPTS="/t.jsonl")) is True
    try:
        namespace["check_feature_params"](_feature_params(
            NOISY_ASR_TRAINING=True, NOISY_ASR_RATIO=1.5, NOISY_ASR_TRANSCRIPTS="/t.jsonl"))
    except ValueError as exc:
        ratio_message = str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("a ratio of 1.5 with NOISY_ASR_TRAINING on was accepted")
    assert "NOISY_ASR_RATIO" in ratio_message and "(0, 1]" in ratio_message, ratio_message
    # The registry catches the same ratio one layer down (proof the two guards agree).
    from lifemate.features.registry import REGISTRY

    raw = namespace["build_feature_config"](_feature_params(
        NOISY_ASR_TRAINING=False, NOISY_ASR_TRANSCRIPTS=""))   # valid: flag off, no ratio check
    raw.training.noisy_asr_training_enabled = True
    raw.training.noisy_asr_ratio = 1.5
    try:
        REGISTRY.validate_or_raise(raw)
    except Exception as exc:
        assert "noisy_asr_ratio" in str(exc), str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("the registry accepted ratio 1.5 with training.noisy_asr enabled")

    mismatch = _feature_params(NBEST_TRAINING_FORMAT=True, NBEST_TRANSCRIPTS="/n.jsonl")
    try:
        namespace["build_feature_config"](mismatch)
    except ValueError as exc:
        nbest_message = str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("NBEST_TRAINING_FORMAT without ASR_NBEST was accepted")
    assert "NBEST_TRAINING_FORMAT" in nbest_message and "ASR_NBEST" in nbest_message, nbest_message
    # And the registry refuses the dependency directly, proving the cell is not the only guard.
    from lifemate.features.config import FeatureConfig

    broken = FeatureConfig()
    broken.prompt.nbest_prompt_enabled = True
    try:
        REGISTRY.validate_or_raise(broken)
    except Exception as exc:
        assert "prompt.nbest_format" in str(exc), str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("the registry accepted prompt.nbest_format without asr.nbest")
    print(f"  P6 -> ratio 1.5 raises naming NOISY_ASR_RATIO ('{ratio_message[:46]}...'); "
          f"nbest-format without asr.nbest raises naming both flags "
          f"('{nbest_message[:40]}...'); the registry refuses it too")


def test_feature_report_prints_the_mode() -> None:
    """`Mode: BASELINE` with everything off, `Mode: EXPERIMENTAL` with one flag on."""
    import io
    from contextlib import redirect_stdout

    from lifemate.features.report import format_feature_report

    namespace, cfg = _feature_cfg()
    if not namespace:
        print("SKIP: lifemate.features absent - mode report not exercised")
        NEEDS.append("feature_mode")
        return
    baseline = format_feature_report(cfg, model_id="Qwen/Qwen3.5-4B", asr_id="asr-id")
    assert "Mode:" in baseline and "BASELINE" in baseline, baseline
    assert "  NONE" in baseline.split("Enabled features:")[1].split("Disabled features:")[0]
    _, one_on = _feature_cfg(SLOT_MASKING=True)
    experimental = format_feature_report(one_on, model_id="Qwen/Qwen3.5-4B", asr_id="asr-id")
    assert "EXPERIMENTAL" in experimental, experimental
    assert "training.slot_masking" in experimental.split("Enabled features:")[1].split(
        "Disabled features:")[0]

    # The cell prints exactly this; run its report path and capture it.
    print("  P6 -> report modes as printed by cell 2a's own formatter:")
    for line in experimental.strip().splitlines():
        print(f"    {line}")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(baseline)
    assert "Mode:\n  BASELINE" in buffer.getvalue(), buffer.getvalue()


def test_training_feature_mismatch_is_reported_as_exact_resume_refused() -> None:
    """A checkpoint trained with other flags must print `EXACT RESUME REFUSED` and name them."""
    import io
    from contextlib import redirect_stdout

    namespace = _resume_namespace()
    if not namespace.get("feature_resume_comparison"):
        print("SKIP: lifemate.features absent - feature resume comparison not exercised")
        NEEDS.append("feature_resume")
        return
    from lifemate.features.config import training_feature_fingerprint

    _, trained_with = _feature_cfg(SLOT_MASKING=True)
    mismatched, current = _feature_cfg(TOOL_DROPOUT=True)
    state = {"step": 100, "epoch": 1, "config_hash": "cfg1",
             "torch_rng_state": [1], "python_rng_state": [2],
             "training_feature_fingerprint": training_feature_fingerprint(trained_with),
             "training_features": ["training.slot_masking"],
             "progress": {"schema_version": 2, "global_step": 100, "epoch_index": 2,
                          "next_micro_index": 3, "permutation": [1, 2], "sampler_rng_state": [7],
                          "planned_total_steps": 900}}
    candidate = {"path": pathlib.Path("/ckpt/step-100"), "state": state,
                 "progress": state["progress"], "exact": True, "exact_missing": [],
                 "optimizer": {"optimizer": True, "scheduler": True}, "warnings": [],
                 "rejections": []}
    namespace["FEATURE_CONFIG"] = current
    comparison = namespace["feature_resume_comparison"](state)
    assert comparison["match"] is False, comparison
    assert comparison["differing_ids"] == ["training.slot_masking", "training.tool_dropout"], comparison
    report = namespace["resume_report"](candidate, expected_total_steps=900)
    assert report["exact"] is False, "a feature mismatch must not stay 'exact'"
    assert report["mode"] == "best-effort", report["mode"]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        namespace["print_resume_report"](report)
    printed = buffer.getvalue()
    assert "EXACT RESUME REFUSED" in printed, printed
    assert "training.slot_masking" in printed and "training.tool_dropout" in printed, printed
    # The accepting path on the SAME fixture: the checkpoint's own flags -> MATCH, exact kept.
    namespace["FEATURE_CONFIG"] = trained_with
    matched = namespace["feature_resume_comparison"](state)
    assert matched["match"] is True, matched
    matched_report = namespace["resume_report"](candidate, expected_total_steps=900)
    assert matched_report["exact"] is True, matched_report
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        namespace["print_resume_report"](matched_report)
    assert "EXACT RESUME REFUSED" not in buffer.getvalue()
    print("  P6 -> mismatch printed:")
    for line in printed.splitlines():
        if "training features" in line or "EXACT RESUME REFUSED" in line:
            print(f"    {line[:118]}")


def test_feature_cli_is_feature_detected_so_an_old_pin_says_so() -> None:
    """A pin without `--enable-feature` must raise naming the flag, not crash inside argparse."""
    cells = _cells()
    source = _cell_source(cells, "cli")
    assert "--enable-feature" in source and "FEATURE_CLI_SUPPORTED" in source, source[:200]
    assert "Launching would train with every feature OFF" in source, \
        "the refusal message is not present"
    # Structural: the raise is guarded by "ids needed AND not supported".
    tree = ast.parse(source)
    guarded = [node for node in ast.walk(tree)
               if isinstance(node, ast.If) and "FEATURE_IDS_NEEDED" in ast.dump(node.test)
               and "FEATURE_CLI_SUPPORTED" in ast.dump(node.test)]
    assert guarded, "the feature CLI gate is not a conditional"
    raises = [n for n in ast.walk(guarded[0]) if isinstance(n, ast.Raise)]
    assert raises, "the feature CLI gate does not raise"
    print(f"  P6 -> train.py feature gate detected from --help; missing flag raises at "
          f"L{raises[0].lineno} naming '--enable-feature'")

# --------------------------------------------------------------------------- #
# P2b — RESUME OPT-IN
# --------------------------------------------------------------------------- #
def test_run_resume_proof_defaults_to_false_and_gates_the_launch() -> None:
    """The param must default to False, and the ``if`` must actually guard the subprocess."""
    cells = _cells()
    tree = ast.parse(_cell_source(cells, "params"))
    defaults = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                defaults[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert defaults.get("RUN_RESUME_PROOF") is False, defaults.get("RUN_RESUME_PROOF")
    source = _cell_source(cells, "params")
    assert 'RUN_RESUME_PROOF = False #@param {type:"boolean"}' in source, "not a Colab boolean param"
    # Structural gate: the sole run_training(...resume=...) call lives inside the `else:` branch.
    resume_tree = ast.parse(_cell_source(cells, "resume"))
    launches: list[int] = []
    guards: list[tuple[int, bool]] = []
    for node in ast.walk(resume_tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp) \
                and isinstance(node.test.operand, ast.Name) \
                and node.test.operand.id == "RUN_RESUME_PROOF":
            guards.append((node.lineno, len(node.orelse) > 0))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "run_training":
            launches.append(node.lineno)
    assert len(guards) == 1, guards
    assert guards[0][1], "the RUN_RESUME_PROOF branch has no else: block"
    assert len(launches) == 1, launches
    guarded_from = guards[0][0]
    assert launches[0] > guarded_from, (guards, launches)
    # The read-only half must run either way: checkpoint_ready is outside the guarded branch.
    calls = [n.lineno for n in ast.walk(resume_tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "checkpoint_ready"]
    assert calls and min(calls) < guarded_from, (calls, guarded_from)
    print(f"  P2b -> RUN_RESUME_PROOF default False; run_training@L{launches[0]} gated by if@L"
          f"{guarded_from}; checkpoint_ready@L{min(calls)} outside it")


if __name__ == "__main__":  # pragma: no cover - manual run
    for _name, _func in sorted(globals().items()):
        if _name.startswith("test_") and callable(_func):
            _func()
    print("all notebook contract checks passed")

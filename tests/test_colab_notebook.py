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
    "derive": "# --- 2b. Derivation: the GPU TIER decides the METHOD",
    "helpers": "# --- 3c. Verify imports + bind the repo CLI contract",
    "sync": "# --- 4b. Atomic artifact sync + completion markers",
    "load_probe": "# --- 6b. Load the checkpoint through the repo loader, verify adapter + grads --",
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
    assert len(cells) == 38, len(cells)
    assert len(code) == 24, len(code)
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
                       "VOCAB": 248320}
    _exec_defs(_cells(), "load_probe", namespace, ["probe_loss", "chunk_equivalence"])
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
    # ...and must REJECT a loss that is not the same function, or the guard proves nothing.
    sabotaged = dict(namespace, chunked_causal_lm_loss=lambda logits, labels, chunk: logits.sum() * 0.0)
    _exec_defs(_cells(), "load_probe", sabotaged, ["chunk_equivalence"])
    assert sabotaged["chunk_equivalence"](model, batch, 4)["delta"] > 1e-4

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
                "verify_sync_marker", "checkpoint_ready", "_previous_generations", "atomic_sync"])
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

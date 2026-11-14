"""Exact-resume acceptance tests: an interrupted run must replay the SAME run.

Every test here is torch-gated with the same convention as ``tests/test_training.py``:
it prints a ``SKIP`` line and returns when torch is missing, so the file runs on the
stdlib box and on the GPU box without edits.

The core test is the A-vs-B equivalence experiment (fresh run to the horizon vs
stop-mid-epoch-then-resume-in-a-new-trainer), plus a NEGATIVE CONTROL that corrupts
the persisted cursor and proves the comparison can fail. A test that cannot fail is
not evidence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lifemate.errors import LifeMateError  # noqa: E402
from lifemate.training.trainer import (  # noqa: E402
    TRAINING_STATE_SCHEMA,
    TrainingProgress,
    read_training_state,
)

try:
    import pytest
except ImportError:  # stdlib runner without pytest
    pytest = None

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401,PLC0415 - probe only

        return True
    except Exception:
        return False

def _transformers_available() -> bool:
    try:
        import transformers  # noqa: F401,PLC0415 - probe only

        return True
    except Exception:
        return False

def _verify_module():
    """The equivalence harness in ``scripts/verify_training_resume.py``.

    Imported from its real path so the test drives EXACTLY the code that produces
    the CLI report: no second, kinder implementation.
    """
    spec = importlib.util.spec_from_file_location(
        "_verify_training_resume", ROOT / "scripts" / "verify_training_resume.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def _args(*, steps: int, interrupt_at: int, accum: int, records: int = 12, dump=None):
    return SimpleNamespace(
        steps=steps,
        interrupt_at=interrupt_at,
        accum=accum,
        records=records,
        real=False,
        model="tiny-resume",
        dump=dump,
    )

def _experiment(*, steps: int, interrupt_at: int, accum: int, records: int = 12):
    """Run both branches in ONE process and return the comparison report.

    The in-process run does not replace the fresh-PYTHON-PROCESS check that
    ``scripts/verify_training_resume.py`` performs by default; it makes the
    scenarios (epoch boundary, accum=1 vs 3, final step) parametrisable here.
    """
    verify = _verify_module()
    args = _args(steps=steps, interrupt_at=interrupt_at, accum=accum, records=records)
    with tempfile.TemporaryDirectory(prefix="resume_test_") as tmp:
        work = Path(tmp)
        reference = verify.run_reference(args, work)
        resumed = verify.run_resumed(args, work)
    return verify.compare(reference, resumed)

def _report(report: dict, label: str) -> None:
    w, o = report["weights"], report["optimizer"]
    print(
        f"  {label}: weights max_abs={w['max_abs_diff']:.3e} max_rel={w['max_rel_diff']:.3e} "
        f"opt max_abs={o['max_abs_diff']:.3e} "
        f"order_equal={report['sample_order']['identical']} "
        f"lr_equal={report['checks']['lr_list_identical']} "
        f"loss_max_abs={report['loss']['max_abs_diff']:.3e} "
        f"pos={report['position']['global_step']} "
        f"rng=({report['rng']['python']},{report['rng']['torch']})"
    )
    failed = [name for name, ok in report["checks"].items() if not ok]
    assert not failed, f"{label}: failing checks {failed}\n{json.dumps(report, indent=2)[:4000]}"

# --------------------------------------------------------------------------- #
# K: the mandatory equivalence experiment
# --------------------------------------------------------------------------- #
def test_resume_mid_epoch_is_bit_exact_accum3() -> None:
    """K mid-epoch interruption, accum=3: weights/moments/order/LR/RNG all identical."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP exact-resume mid-epoch test: torch/transformers not installed")
        return
    report = _experiment(steps=12, interrupt_at=5, accum=3)
    _report(report, "accum=3 mid-epoch")
    assert report["weights"]["max_abs_diff"] == 0.0
    assert report["optimizer"]["max_abs_diff"] == 0.0
    assert report["sample_order"]["identical"] is True
    assert report["sample_order"]["first_divergence_index"] is None
    print("  exact resume -> sample order, weights, AdamW moments, LR list and RNG all equal")

def test_resume_mid_epoch_is_bit_exact_accum1() -> None:
    """Same, with gradient_accumulation_steps=1 (a step per micro-batch)."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP exact-resume accum=1 test: torch/transformers not installed")
        return
    _report(_experiment(steps=12, interrupt_at=5, accum=1), "accum=1 mid-epoch")

def test_resume_inside_second_epoch() -> None:
    """Interruption lands in epoch 2, not epoch 1."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP exact-resume epoch-2 test: torch/transformers not installed")
        return
    # 12 records, micro 1, accum 3 -> 4 optimizer steps per epoch.
    report = _experiment(steps=12, interrupt_at=6, accum=3)
    _report(report, "interrupt in epoch 2")
    assert report["position"]["epoch_index"] == [3, 3], report["position"]

def test_resume_exactly_at_an_epoch_boundary() -> None:
    """A cursor == end-of-permutation resume advances the epoch ONCE (no skip, no repeat)."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP exact-resume epoch-boundary test: torch/transformers not installed")
        return
    report = _experiment(steps=12, interrupt_at=4, accum=3)
    _report(report, "interrupt at epoch boundary")
    assert report["sample_order"]["identical"] is True
    assert len(report["sample_order"]["reference_head"]) > 0

def test_resume_near_the_planned_final_step() -> None:
    """Resuming one step before the horizon stops cleanly at planned_total_steps."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP exact-resume final-step test: torch/transformers not installed")
        return
    report = _experiment(steps=12, interrupt_at=11, accum=3)
    _report(report, "resume at the last step")
    assert report["position"]["global_step"] == [12, 12], report["position"]

# --------------------------------------------------------------------------- #
# K: the negative control
# --------------------------------------------------------------------------- #
def test_negative_control_corrupt_cursor_fails_the_comparison() -> None:
    """Corrupting the persisted cursor MUST make the comparison fail.

    Without this, a comparison that always passes would look like proof. The test
    tampers with ``progress.next_micro_index`` inside the checkpoint the resume
    reads, re-runs the resumed branch, and asserts the report now reports a
    divergence (different sample order and/or non-zero weight difference).
    """
    if not _torch_available() or not _transformers_available():
        print("  SKIP negative-control test: torch/transformers not installed")
        return
    verify = _verify_module()
    args = _args(steps=12, interrupt_at=5, accum=3)
    with tempfile.TemporaryDirectory(prefix="resume_neg_") as tmp:
        work = Path(tmp)
        reference = verify.run_reference(args, work)
        corrupted = verify.run_resumed(args, work, corrupt_checkpoint=_corrupt_cursor)
        report = verify.compare(reference, corrupted)
    failed = [name for name, ok in report["checks"].items() if not ok]
    print(
        f"  negative control -> weights max_abs={report['weights']['max_abs_diff']:.3e} "
        f"order_equal={report['sample_order']['identical']} "
        f"first_divergence={report['sample_order']['first_divergence_index']} "
        f"failing={failed}"
    )
    assert failed, (
        "corrupting the persisted cursor did not change the outcome: the comparison "
        f"is not discriminating\n{json.dumps(report, indent=2)[:3000]}"
    )
    assert not report["passed"]

def _corrupt_cursor(checkpoint: Path) -> None:
    """Shift the sampler cursor by one micro-batch inside ``training_state.json``."""
    state_path = checkpoint / "training_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    progress = payload["progress"]
    progress["next_micro_index"] = int(progress["next_micro_index"]) + 1
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

# --------------------------------------------------------------------------- #
# J: legacy checkpoints
# --------------------------------------------------------------------------- #
def _legacy_checkpoint(root: Path) -> Path:
    """A schema-1 ``training_state.json`` (step counter only, no sampler position).

    Real weights are written alongside it so the resume reaches the schema check
    rather than failing earlier on a missing weight file.
    """
    verify = _verify_module()
    leg = root / "step-3"
    leg.mkdir(parents=True, exist_ok=True)
    verify.build_tiny_model().save_pretrained(str(leg))
    verify.CharTokenizer().save_pretrained(str(leg))
    (leg / "training_state.json").write_text(
        json.dumps(
            {
                "step": 3,
                "epoch": 1,
                "seed": 0,
                "is_lora": False,
                "precision": "float32",
                "python_rng_state": [0] * 625,
            }
        ),
        encoding="utf-8",
    )
    return leg

def _resume_trainer(root: Path, **kwargs):
    """A fresh trainer wired to ``root``; only the resume path is under test."""
    verify = _verify_module()
    cfg = verify.build_config(accum=1, work=root)
    cfg.model.name = "openbmb/MiniCPM5-2B"  # a real profile, so the world block exists
    from lifemate.training.collator import SupervisedCollator  # noqa: PLC0415
    from lifemate.training.trainer import CheckpointManager, SFTTrainer  # noqa: PLC0415

    tokenizer = verify.CharTokenizer()
    return SFTTrainer(
        model=verify.build_tiny_model(),
        tokenizer=tokenizer,
        collator=SupervisedCollator.from_tokenizer(tokenizer),
        train_records=verify.synthetic_records(4),
        cfg=cfg,
        device="cpu",
        run_name="resume",
        checkpoint_manager=CheckpointManager(root / "ckpts", run_name="resume", save_total_limit=0),
        **kwargs,
    )

def test_schema1_checkpoint_is_refused_for_exact_resume() -> None:
    """A legacy checkpoint raises, naming best_effort as the (approximate) exit."""
    if not _torch_available():
        print("  SKIP legacy-resume refusal test: torch not installed")
        return
    with tempfile.TemporaryDirectory(prefix="resume_legacy_") as tmp:
        path = _legacy_checkpoint(Path(tmp))
        trainer = _resume_trainer(Path(tmp))
        try:
            trainer.resume(path)
        except LifeMateError as exc:
            message = str(exc).lower()
            assert "predates exact-resume" in message, message
            assert "best_effort" in message or "best-effort" in message, message
        else:
            raise AssertionError("a schema-1 checkpoint must be refused for an exact resume")
        print("  legacy checkpoint -> exact resume refused with the best_effort remedy")

def test_schema1_checkpoint_resumes_only_as_best_effort() -> None:
    """``best_effort=True`` is accepted, is labelled, and never claims exactness."""
    if not _torch_available():
        print("  SKIP legacy best-effort test: torch not installed")
        return
    with tempfile.TemporaryDirectory(prefix="resume_legacy_ok_") as tmp:
        root = Path(tmp)
        path = _legacy_checkpoint(root)
        trainer = _resume_trainer(root)
        assert trainer.resume(path, best_effort=True) == 3
        assert trainer.exact_resume is False, "an approximate continuation must not claim exactness"
        assert trainer.start_step == 3
        print("  legacy checkpoint -> best-effort accepted, exact_resume=False")

# --------------------------------------------------------------------------- #
# I/J: dataset order and model world
# --------------------------------------------------------------------------- #
def test_dataset_order_hash_mismatch_is_refused() -> None:
    """Reordered training examples change the sampler position; resume must refuse."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP dataset-order-hash test: torch/transformers not installed")
        return
    verify = _verify_module()
    args = _args(steps=6, interrupt_at=3, accum=1)
    with tempfile.TemporaryDirectory(prefix="resume_order_") as tmp:
        work = Path(tmp)
        interrupted = verify._TracedTrainer(
            cfg=verify.build_config(accum=1, work=work),
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="interrupted",
            planned_total_steps=6,
            stop_after_steps=3,
        )
        interrupted.run()
        checkpoint = interrupted.trainer.checkpoint_manager.latest()
        assert checkpoint is not None

        reordered = list(reversed(verify.synthetic_records(6)))
        resumed = verify._TracedTrainer(
            cfg=verify.build_config(accum=1, work=work),
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=reordered,
            work=work,
            run_name="resumed",
            planned_total_steps=6,
        )
        resumed.prepare_indices()
        try:
            resumed.trainer.resume(checkpoint)
        except LifeMateError as exc:
            message = str(exc).lower()
            assert "dataset order hash" in message, message
        else:
            raise AssertionError("a reordered dataset must not resume an exact checkpoint")
    print("  dataset order -> reordered examples refused (names dataset order hash)")

def test_model_profile_mismatch_is_refused() -> None:
    """A checkpoint from one model family cannot be resumed into another."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP model-profile-mismatch test: torch/transformers not installed")
        return
    verify = _verify_module()
    args = _args(steps=6, interrupt_at=3, accum=1)
    with tempfile.TemporaryDirectory(prefix="resume_profile_") as tmp:
        work = Path(tmp)
        cfg = verify.build_config(accum=1, work=work)
        cfg.model.name = "openbmb/MiniCPM5-2B"
        interrupted = verify._TracedTrainer(
            cfg=cfg,
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="interrupted",
            planned_total_steps=6,
            stop_after_steps=3,
        )
        interrupted.run()
        checkpoint = interrupted.trainer.checkpoint_manager.latest()
        assert checkpoint is not None
        recorded = read_training_state(checkpoint)
        assert recorded["world"]["model_profile"] == "minicpm5", recorded["world"]

        # A different base model name ALSO changes the config fingerprint, which
        # fires first (correctly). To exercise the profile guard itself, keep the
        # config identical and rewrite only the recorded world.
        for target in (checkpoint / "training_state.json",):
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["world"]["model_profile"] = "qwen3_5"
            payload["progress"]["world"]["model_profile"] = "qwen3_5"
            target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        cfg_b = verify.build_config(accum=1, work=work)
        cfg_b.model.name = "openbmb/MiniCPM5-2B"
        resumed = verify._TracedTrainer(
            cfg=cfg_b,
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="resumed",
            planned_total_steps=6,
        )
        resumed.prepare_indices()
        try:
            resumed.trainer.resume(checkpoint)
        except LifeMateError as exc:
            message = str(exc).lower()
            assert "profile" in message, message
            assert "qwen3_5" in message and "minicpm5" in message, message
        else:
            raise AssertionError("a checkpoint recorded for another profile must be refused")
    print("  model world -> MiniCPM checkpoint refused when its world says qwen3_5")

def test_different_base_model_name_is_refused() -> None:
    """Resuming a MiniCPM checkpoint into a Qwen run is refused up front."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP model-name-mismatch test: torch/transformers not installed")
        return
    verify = _verify_module()
    args = _args(steps=6, interrupt_at=3, accum=1)
    with tempfile.TemporaryDirectory(prefix="resume_name_") as tmp:
        work = Path(tmp)
        cfg = verify.build_config(accum=1, work=work)
        cfg.model.name = "openbmb/MiniCPM5-2B"
        interrupted = verify._TracedTrainer(
            cfg=cfg,
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="interrupted",
            planned_total_steps=6,
            stop_after_steps=3,
        )
        interrupted.run()
        checkpoint = interrupted.trainer.checkpoint_manager.latest()
        assert checkpoint is not None
        cfg_b = verify.build_config(accum=1, work=work)
        cfg_b.model.name = "Qwen/Qwen3.5-4B"
        resumed = verify._TracedTrainer(
            cfg=cfg_b,
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="resumed",
            planned_total_steps=6,
        )
        resumed.prepare_indices()
        try:
            resumed.trainer.resume(checkpoint)
        except LifeMateError as exc:
            assert "config differs" in str(exc), str(exc)
        else:
            raise AssertionError("another base model must not resume this checkpoint")

def test_manifest_compat_check_names_the_field_and_fix() -> None:
    """``check_checkpoint_compat`` refuses a profile/quantization/rank mismatch."""
    from lifemate.models.checkpoint import COMPAT_KEYS, check_checkpoint_compat  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="resume_manifest_") as tmp:
        path = Path(tmp) / "step-4"
        path.mkdir(parents=True)
        (path / "lifemate_meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "model_id": "openbmb/MiniCPM5-2B",
                    "model_profile": "minicpm5",
                    "tuning_mode": "lora",
                    "quantization": "none",
                    "precision": "bfloat16",
                    "adapter_rank": 16,
                    "planned_total_steps": 100,
                    "global_step": 4,
                }
            ),
            encoding="utf-8",
        )
        report = check_checkpoint_compat(path, model_profile="minicpm5", tuning_mode="lora")
        assert "model_profile" in report["checked"], report
        for kwargs, needle in (
            ({"model_profile": "qwen3_5"}, "model profile family"),
            ({"tuning_mode": "qlora"}, "tuning mode"),
            ({"quantization": "nf4"}, "quantization"),
            ({"precision": "float32"}, "precision"),
            ({"adapter_rank": 32}, "adapter rank"),
        ):
            try:
                check_checkpoint_compat(path, **kwargs)
            except LifeMateError as exc:
                assert needle in str(exc), (kwargs, str(exc))
            else:
                raise AssertionError(f"{kwargs} must be refused")
        try:
            check_checkpoint_compat(path, planned_total_steps=200)
        except LifeMateError as exc:
            assert "horizon" in str(exc), str(exc)
        else:
            raise AssertionError("a different horizon must be refused")
        assert check_checkpoint_compat(
            path, model_profile="minicpm5", planned_total_steps=100
        )["recorded"]["global_step"] == 4
    print(f"  manifest compat -> every one of {len(COMPAT_KEYS)} recorded keys is comparable")

# --------------------------------------------------------------------------- #
# E: horizon vs early stop
# --------------------------------------------------------------------------- #
def test_horizon_mismatch_is_refused_without_extend_horizon() -> None:
    """A different planned horizon is not a continuation; the error names both numbers."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP horizon-mismatch test: torch/transformers not installed")
        return
    verify = _verify_module()
    with tempfile.TemporaryDirectory(prefix="resume_horizon_") as tmp:
        work = Path(tmp)
        interrupted = verify._TracedTrainer(
            cfg=verify.build_config(accum=1, work=work),
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="interrupted",
            planned_total_steps=6,
            stop_after_steps=3,
        )
        interrupted.run()
        checkpoint = interrupted.trainer.checkpoint_manager.latest()
        assert checkpoint is not None

        other = verify._TracedTrainer(
            cfg=verify.build_config(accum=1, work=work),
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="resumed",
            planned_total_steps=10,
        )
        other.prepare_indices()
        try:
            other.trainer.resume(checkpoint)
        except LifeMateError as exc:
            message = str(exc)
            assert "horizon" in message.lower(), message
            assert "6" in message and "10" in message, message
            assert "extend_horizon" in message or "--extend-horizon" in message, message
        else:
            raise AssertionError("a changed horizon must be refused without extend_horizon")
        print("  horizon -> mismatch refused naming 6 and 10 plus the --extend-horizon remedy")

def test_early_stop_does_not_define_the_schedule() -> None:
    """A 2-step run must not build a 2-step cosine schedule for the real horizon."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP early-stop-horizon test: torch/transformers not installed")
        return
    verify = _verify_module()
    with tempfile.TemporaryDirectory(prefix="resume_stop_") as tmp:
        work = Path(tmp)
        probe = verify._TracedTrainer(
            cfg=verify.build_config(accum=1, work=work),
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="probe",
            planned_total_steps=6,
            stop_after_steps=2,
        )
        result = probe.run()
        assert result.steps == 2, result.steps
        assert result.planned_total_steps == 6, result.planned_total_steps
        assert result.stop_after_steps == 2, result.stop_after_steps
        # The LR at step 2 must come from the 6-step trajectory, not from a schedule
        # that already reached its end at 2.
        lrs = probe.lrs
        assert lrs[1] > 0, f"a 2-step probe must not decay to LR 0 under a 6-step horizon: {lrs}"
        assert probe.trainer.scheduler.last_epoch == 2
        print(f"  horizon -> probe stopped at 2 of 6 planned; LR trace {lrs}")

# --------------------------------------------------------------------------- #
# F: GradScaler-skipped steps
# --------------------------------------------------------------------------- #
class _RecordingScaler:
    """Stand-in ``GradScaler`` with the real call discipline.

    ``step(optimizer)`` is the ONLY place a scaled update is applied, and it
    applies it unless ``skip`` is set — exactly like the real scaler's inf/NaN
    guard. The trainer must therefore count updates by observing
    ``optimizer.step()``, not by assuming every call updated something.
    """

    def __init__(self, *, skip: bool) -> None:
        self.skip = bool(skip)
        self.calls: list[str] = []

    def scale(self, loss):
        self.calls.append("scale")
        return loss

    def unscale_(self, optimizer):
        self.calls.append("unscale")

    def step(self, optimizer):
        self.calls.append("step")
        if not self.skip:
            optimizer.step()

    def update(self):
        self.calls.append("update")

    def state_dict(self):
        return {"skip": self.skip}

    def load_state_dict(self, state):  # pragma: no cover - not exercised
        self.skip = bool(state.get("skip"))

def _skipped_step_experiment(*, skip: bool):
    """Drive the REAL ``optimizer_step()`` path with a scaler that skips or not."""
    import torch  # noqa: PLC0415

    from lifemate.training.collator import SupervisedCollator  # noqa: PLC0415
    from lifemate.training.trainer import CheckpointManager, SFTTrainer  # noqa: PLC0415

    verify = _verify_module()
    with tempfile.TemporaryDirectory(prefix="resume_skip_") as tmp:
        work = Path(tmp)
        records = verify.synthetic_records(4)
        tokenizer = verify.CharTokenizer()
        cfg = verify.build_config(accum=1, work=work)
        cfg.training.save_strategy = "no"
        cfg.training.eval_strategy = "no"
        trainer = SFTTrainer(
            model=verify.build_tiny_model(),
            tokenizer=tokenizer,
            collator=SupervisedCollator.from_tokenizer(tokenizer),
            train_records=records,
            val_records=[],
            cfg=cfg,
            device="cpu",
            run_name="skip",
            planned_total_steps=2,
            checkpoint_manager=CheckpointManager(
                work / "ckpts", run_name="skip", save_total_limit=0
            ),
        )
        trainer.scaler = _RecordingScaler(skip=skip)
        lr_steps: list[int] = []
        real_scheduler = None

        def install_scheduler() -> None:
            nonlocal real_scheduler
            from lifemate.training.full_finetuning import build_scheduler  # noqa: PLC0415

            real_scheduler = build_scheduler(trainer.optimizer, cfg, 2)
            original = real_scheduler.step

            def counted(*a, **kw):
                lr_steps.append(int(real_scheduler.last_epoch) + 1)
                return original(*a, **kw)

            real_scheduler.step = counted  # type: ignore[assignment]
            trainer.scheduler = real_scheduler

        original_ensure = trainer._ensure_optimizer

        def ensure(steps: int) -> None:
            original_ensure(steps)
            install_scheduler()

        trainer._ensure_optimizer = ensure  # type: ignore[assignment]
        result = trainer.run()
        torch.manual_seed(0)
        return result, trainer, lr_steps

def test_skipped_optimizer_step_does_not_advance_step_or_schedule() -> None:
    """An inf/NaN-skipped update must not consume a step number nor a scheduler tick."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP skipped-step test: torch/transformers not installed")
        return
    applied, trainer, ticks = _skipped_step_experiment(skip=False)
    assert applied.steps == 2, applied.steps
    assert applied.skipped_steps == 0, applied.skipped_steps
    assert ticks == [1, 2], ticks
    assert trainer.scheduler.last_epoch == 2, trainer.scheduler.last_epoch
    print(f"  applied scaler -> steps={applied.steps} scheduler ticks={ticks}")

def test_skipped_step_is_counted_but_never_advances() -> None:
    """With every update skipped, the run reports the skips and refuses to loop forever."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP skipped-step counter test: torch/transformers not installed")
        return
    try:
        result, _trainer, ticks = _skipped_step_experiment(skip=True)
    except LifeMateError as exc:
        message = str(exc).lower()
        assert "no optimizer update" in message or "non-finite" in message, message
        print("  skipped steps -> an all-skipped epoch raises instead of spinning silently")
        return
    raise AssertionError(
        "an all-skipped run must not report success as if the parameters moved: "
        f"steps={result.steps} skipped={result.skipped_steps} ticks={ticks}"
    )

# --------------------------------------------------------------------------- #
# A: state shape
# --------------------------------------------------------------------------- #
def test_progress_block_round_trips_and_matches_the_legacy_step() -> None:
    """``TrainingProgress`` persists the next work item and cannot disagree with ``step``."""
    progress = TrainingProgress(
        global_step=7,
        epoch_index=2,
        next_micro_index=3,
        permutation=[4, 1, 0, 3, 2],
        sampler_rng_state=[1, 2, 3],
        planned_total_steps=12,
        stop_after_steps=None,
        dataset_size=5,
        world={"model_profile": "minicpm"},
    )
    restored = TrainingProgress.from_dict(json.loads(json.dumps(progress.to_dict())))
    assert restored == progress, (restored, progress)
    assert restored.schema_version == TRAINING_STATE_SCHEMA == 2
    print("  progress -> dataclass round-trips through JSON unchanged (schema 2)")

def test_checkpoint_written_mid_accumulation_is_refused_for_exact_resume() -> None:
    """A checkpoint with pending_microbatches > 0 is not a consistent boundary."""
    if not _torch_available() or not _transformers_available():
        print("  SKIP pending-microbatch test: torch/transformers not installed")
        return
    verify = _verify_module()
    with tempfile.TemporaryDirectory(prefix="resume_pending_") as tmp:
        work = Path(tmp)
        interrupted = verify._TracedTrainer(
            cfg=verify.build_config(accum=2, work=work),
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="interrupted",
            planned_total_steps=4,
            stop_after_steps=2,
        )
        interrupted.run()
        checkpoint = interrupted.trainer.checkpoint_manager.latest()
        assert checkpoint is not None
        state_path = checkpoint / "training_state.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        payload["progress"]["pending_microbatches"] = 1
        state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        resumed = verify._TracedTrainer(
            cfg=verify.build_config(accum=2, work=work),
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="resumed",
            planned_total_steps=4,
        )
        resumed.prepare_indices()
        try:
            resumed.trainer.resume(checkpoint)
        except LifeMateError as exc:
            assert "uncommitted micro-batch" in str(exc), str(exc)
        else:
            raise AssertionError("a mid-accumulation checkpoint must be refused for exact resume")
    print("  optimization boundary -> pending_microbatches>0 refused for exact resume")

# --------------------------------------------------------------------------- #
# P0: a step dir without optimizer_state.pt must NEVER claim exact resume
# --------------------------------------------------------------------------- #
def test_missing_optimizer_state_is_refused_and_best_effort_downgrades() -> None:
    """Discrimination: file present -> exact; file removed -> refused; effort -> approximate.

    The bug this pins: ``training_state.json`` used to be written before
    ``optimizer_state.pt``, so a crash in that window left a directory that claimed
    ``exact_resume: true`` beside no AdamW moments. ``resume()`` accepted it and
    rebuilt the optimizer from zero while still reporting "exact".
    """
    if not _torch_available() or not _transformers_available():
        print("  SKIP missing-optimizer-state test: torch/transformers not installed")
        return
    verify = _verify_module()

    def _resume_from(checkpoint: Path, work: Path, **kwargs):
        trainer = verify._TracedTrainer(
            cfg=verify.build_config(accum=1, work=work),
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="resumed",
            planned_total_steps=6,
        )
        trainer.prepare_indices()
        step = trainer.trainer.resume(checkpoint, **kwargs)
        return trainer.trainer, step

    with tempfile.TemporaryDirectory(prefix="resume_noopt_") as tmp:
        work = Path(tmp)
        interrupted = verify._TracedTrainer(
            cfg=verify.build_config(accum=1, work=work),
            model=verify.build_tiny_model(),
            tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6),
            work=work,
            run_name="interrupted",
            planned_total_steps=6,
            stop_after_steps=3,
        )
        interrupted.run()
        checkpoint = interrupted.trainer.checkpoint_manager.latest()
        assert checkpoint is not None
        assert (checkpoint / "optimizer_state.pt").is_file(), "the writer must persist it"

        # --- positive half: the intact checkpoint IS exact ---------------------
        trainer, step = _resume_from(checkpoint, work)
        assert step == 3, step
        assert trainer.exact_resume is True, "an intact checkpoint must resume exactly"
        loaded = trainer.optimizer.state_dict().get("state", {})
        assert loaded, "AdamW moments must be restored"

        # --- negative half: delete the tensors, the same call must refuse ------
        (checkpoint / "optimizer_state.pt").unlink()
        try:
            _resume_from(checkpoint, work)
        except LifeMateError as exc:
            message = str(exc)
            assert "optimizer_state.pt" in message, message
            assert "zero moments" in message, message
            assert "best_effort" in message or "best-effort" in message, message
        else:
            raise AssertionError(
                "a step dir with no optimizer_state.pt must be refused for an exact resume"
            )

        # --- explicit opt-in continues approximately, never as exact ----------
        approx, step = _resume_from(checkpoint, work, best_effort=True)
        assert step == 3, step
        assert approx.exact_resume is False, "an approximate continuation must not claim exactness"
        print(
            "  optimizer state -> intact=exact; deleted=refused (names optimizer_state.pt and "
            "best_effort); best_effort=True accepted with exact_resume=False"
        )

def test_training_state_json_is_written_after_optimizer_state() -> None:
    """WRITE ORDER: the state file must not exist when the optimizer tensors are written.

    ``training_state.json`` is the resumability marker (it advertises
    ``exact_resume: true``), so it has to be last. This test fails if anyone
    restores the old order where ``CheckpointManager.save`` wrote the state and the
    trainer persisted the optimizer tensors only afterwards.
    """
    if not _torch_available() or not _transformers_available():
        print("  SKIP write-order test: torch/transformers not installed")
        return
    verify = _verify_module()
    from lifemate.training.collator import SupervisedCollator  # noqa: PLC0415
    from lifemate.training.trainer import CheckpointManager, SFTTrainer  # noqa: PLC0415

    observations: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="resume_order_io_") as tmp:
        work = Path(tmp)
        records = verify.synthetic_records(4)
        tokenizer = verify.CharTokenizer()
        cfg = verify.build_config(accum=1, work=work)
        cfg.training.save_strategy = "steps"
        cfg.training.save_steps = 1
        trainer = SFTTrainer(
            model=verify.build_tiny_model(),
            tokenizer=tokenizer,
            collator=SupervisedCollator.from_tokenizer(tokenizer),
            train_records=records,
            val_records=[],
            cfg=cfg,
            device="cpu",
            run_name="order",
            planned_total_steps=2,
            checkpoint_manager=CheckpointManager(
                work / "ckpts", run_name="order", save_total_limit=0
            ),
        )
        original = trainer._save_optimizer_state

        def spy(target: Path, step: int, epoch: int) -> bool:
            observations.append(
                {
                    "step": int(step),
                    "state_exists_already": (Path(target) / "training_state.json").exists(),
                    "weights_present": bool(
                        list(Path(target).glob("*.safetensors"))
                        + list(Path(target).glob("pytorch_model*.bin"))
                    ),
                }
            )
            return original(target, step, epoch)

        trainer._save_optimizer_state = spy  # type: ignore[assignment]
        trainer.run()

    assert observations, "the trainer wrote no checkpoint at all"
    for entry in observations:
        assert entry["state_exists_already"] is False, (
            "training_state.json existed when the optimizer tensors were written: the "
            f"checkpoint advertises resumability before its state is durable -> {observations}"
        )
        assert entry["weights_present"] is True, (
            f"optimizer state was written before the weights -> {observations}"
        )
    print(
        "  write order -> weights, then optimizer_state.pt, then training_state.json "
        f"({len(observations)} checkpoint(s) observed)"
    )

# --------------------------------------------------------------------------- #
# §28: training features change the supervised distribution -> not a continuation
# --------------------------------------------------------------------------- #
def _feature_trainers(work: Path, *, enabled: bool, run_name: str):
    """Two trainers over the same records; ``enabled`` toggles one TRAINING feature."""
    verify = _verify_module()
    cfg = verify.build_config(accum=1, work=work)
    try:
        from lifemate.features.config import FeatureConfig  # noqa: PLC0415
    except ImportError:
        cfg.features = None
        return verify._TracedTrainer(
            cfg=cfg, model=verify.build_tiny_model(), tokenizer=verify.CharTokenizer(),
            records=verify.synthetic_records(6), work=work, run_name=run_name,
            planned_total_steps=6,
        )
    features = FeatureConfig()
    features.training.tool_dropout_enabled = bool(enabled)
    cfg.features = features
    return verify._TracedTrainer(
        cfg=cfg, model=verify.build_tiny_model(), tokenizer=verify.CharTokenizer(),
        records=verify.synthetic_records(6), work=work, run_name=run_name,
        planned_total_steps=6,
    )

def _features_available() -> bool:
    try:
        import lifemate.features.config  # noqa: F401,PLC0415 - probe only

        return True
    except ImportError:
        return False

def test_training_feature_change_blocks_an_exact_resume() -> None:
    """Same seed/architecture, one training feature flipped: the resume is refused.

    Also the negative control in the same test: with the SAME features the same
    call resumes exactly, so the refusal is attributable to the flag and not to
    the harness.
    """
    if not _torch_available() or not _transformers_available():
        print("  SKIP training-feature resume test: torch/transformers not installed")
        return
    if not _features_available():
        print("  SKIP training-feature resume test: lifemate.features absent")
        return
    with tempfile.TemporaryDirectory(prefix="resume_feature_") as tmp:
        work = Path(tmp)
        base = _feature_trainers(work, enabled=True, run_name="interrupted")
        base.trainer.stop_after_steps = 3
        base.run()
        checkpoint = base.trainer.checkpoint_manager.latest()
        assert checkpoint is not None
        recorded = read_training_state(checkpoint)
        assert recorded["training_feature_fingerprint"], recorded
        assert recorded["training_features"] == ["training.tool_dropout"], recorded["training_features"]

        # --- refusal: features flipped off ------------------------------------
        other = _feature_trainers(work, enabled=False, run_name="resumed")
        other.prepare_indices()
        try:
            other.trainer.resume(checkpoint)
        except LifeMateError as exc:
            message = str(exc)
            assert "training-feature fingerprint" in message, message
            assert "training.tool_dropout" in message, message
            assert "re-run from scratch" in message, message
        else:
            raise AssertionError("a changed training feature must not resume exactly")

        # --- negative control: SAME features resume --------------------------
        same = _feature_trainers(work, enabled=True, run_name="resumed_same")
        same.prepare_indices()
        assert same.trainer.resume(checkpoint) == 3
        assert same.trainer.exact_resume is True, "unchanged features must resume exactly"
    print(
        "  training features -> flipped flag refused (names tool_dropout); same flag resumes exact"
    )

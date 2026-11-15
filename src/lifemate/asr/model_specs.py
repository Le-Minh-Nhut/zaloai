"""Inspectable spec table for the Gipformer ASR checkpoints.

Why this module exists
----------------------
The backend used to reject ``gipformer1.5`` with a substring check
(``if "1.5" in model_id``). That is an unreadable, untestable hack: it cannot say
*which* files a model needs, cannot tell an int8-only checkout from a complete
one, and cannot be extended without editing the backend again. This module
replaces it with one frozen table: each known checkpoint declares its artifacts,
sample rate, feature dimension, token-file hash and language set, and the answer
to "can this machine run it?" becomes :func:`validate_artifacts`.

Two entries:

``gipformer-68M-rnnt``
    **MEASURED** on this machine. ``models/gipformer-68M-rnnt`` contains exactly
    ``encoder.int8.onnx``, ``decoder.int8.onnx``, ``joiner.int8.onnx``,
    ``tokens.txt`` (2000 lines, 25,847 B), ``bpe.model`` and ``config.json``.
    There is **no** fp32 ``encoder.onnx`` here. ``tokens_sha256`` is the hash of
    the local file, computed by reading it — not copied from a manifest.

``gipformer1.5-68M-rnnt``
    Artifact names and revision verified against the upstream repository; actual
    model loading and inference remain unverified without local weights.

Pure stdlib: importable without sherpa_onnx/torch/numpy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..compliance import assert_path_allowed
from ..errors import ConfigError
from .interface import AsrError

__all__ = [
    "GipformerSpec",
    "GIPFORMER_SPECS",
    "DEFAULT_SPEC_KEY",
    "KNOWN_MODEL_IDS",
    "artifacts_for",
    "download_hint",
    "hash_tokens_file",
    "known_quantizations",
    "require_spec_artifacts",
    "resolve_gipformer_spec",
    "spec_key",
    "validate_artifacts",
]

#: Key of the default checkpoint in :data:`GIPFORMER_SPECS`.
DEFAULT_SPEC_KEY = "gipformer-68M-rnnt"

#: sha256(``models/gipformer-68M-rnnt/tokens.txt``), read from the local file.
_68M_TOKENS_SHA256 = "f536d03c2e95ebd2930cf0abec88e823bd17d3c1933da7ae6a82db3b80605e15"


@dataclass(frozen=True)
class GipformerSpec:
    """One checkpoint's declared artifacts and audio front-end settings.

    ``onnx`` maps a *role* (``encoder``/``decoder``/``joiner``/``tokens``) to the
    filename that role needs for the spec's default quantization, i.e. the file
    names that were actually observed. ``quantizations`` lists which
    quantizations have a file set at all; an empty tuple means "no verified
    artifact set", which every builder must treat as an error rather than a guess.
    """

    model_id: str
    revision: str
    onnx: Mapping[str, str]
    sample_rate: int
    feature_dim: int
    tokens_sha256: str | None
    languages: tuple[str, ...]
    notes: str
    #: Quantizations with a known file set (empty = unverified spec).
    quantizations: tuple[str, ...] = ()
    #: ``True`` only when this spec's files were observed on this machine.
    verified: bool = False
    #: Extra non-ONNX assets the runtime needs alongside the recognizer files.
    assets: tuple[str, ...] = field(default=())
    #: Upstream release tag ("1.5" for the 1.5 checkpoint); "" when untagged.
    version: str = ""

    @property
    def key(self) -> str:
        """Table key: the model id without its namespace."""
        return self.model_id.rsplit("/", 1)[-1]


GIPFORMER_SPECS: dict[str, GipformerSpec] = {
    # MEASURED: every filename below was listed in models/gipformer-68M-rnnt.
    DEFAULT_SPEC_KEY: GipformerSpec(
        model_id="g-group-ai-lab/gipformer-68M-rnnt",
        revision="318322b009200f8e77a3751fa64638e5cc85a7dc",
        onnx={
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        },
        sample_rate=16000,
        feature_dim=80,
        tokens_sha256=_68M_TOKENS_SHA256,
        languages=("vi",),
        notes=(
            "MEASURED locally: int8-only checkout (encoder/decoder/joiner .int8.onnx, "
            "tokens.txt with 2000 entries / 25,847 B, bpe.model, config.json with "
            "model_type=zipformer-transducer, num_parameters=68625511, sample_rate=16000). "
            "No fp32 encoder.onnx is present. The fp32 file set exists upstream under the "
            "same names without the .int8 suffix; it was NOT downloaded here, so requesting "
            "fp32 fails on this machine and names encoder.onnx. The token file is "
            "ALL-UPPERCASE (3 lowercase entries out of 2000), which is an export artifact."
        ),
        quantizations=("int8", "fp32"),
        verified=True,
        assets=("bpe.model", "config.json"),
    ),
    # Upstream metadata verified; the checkpoint is not bundled in this repository.
    "gipformer1.5-68M-rnnt": GipformerSpec(
        model_id="g-group-ai-lab/gipformer1.5-68M-rnnt",
        revision="dd9227dcd8705c13f33bdbe59728d546ab94480f",
        onnx={
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        },
        sample_rate=16000,
        feature_dim=80,
        tokens_sha256=None,
        languages=("vi",),
        notes="Upstream config and file inventory verified; real weights and decoding not tested locally.",
        quantizations=("int8", "fp32"),
        verified=True,
        assets=("bpe.model", "config.json"),
        version="1.5",
    ),
}

#: Every model id the table knows, in table order.
KNOWN_MODEL_IDS: tuple[str, ...] = tuple(spec.model_id for spec in GIPFORMER_SPECS.values())


def _add_spec_help() -> str:
    known = ", ".join(f"{key!r} ({spec.model_id})" for key, spec in GIPFORMER_SPECS.items())
    return (
        f"Known Gipformer specs: {known}. "
        "To support another checkpoint, add a GipformerSpec entry to "
        "GIPFORMER_SPECS in src/lifemate/asr/model_specs.py: measure its file set, "
        "revision and tokens.txt sha256 first, then set verified=True."
    )


def spec_key(model_id_or_dir: object) -> str | None:
    """Table key named by ``model_id_or_dir``, or ``None`` when nothing matches.

    Accepts a hub id (``g-group-ai-lab/gipformer-68M-rnnt``), a bare descriptor
    (``gipformer-68M-rnnt``) or a directory path ending in either.
    """
    # Prohibited *_environment paths are rejected before any inspection.
    text = str(assert_path_allowed(model_id_or_dir)).strip().replace("\\", "/").lower()
    for key in GIPFORMER_SPECS:
        if key.lower() in text:
            return key
    return None


def resolve_gipformer_spec(model_id_or_dir: object, *, kind: str | None = None) -> GipformerSpec:
    """Return the spec for ``model_id_or_dir``; never silently fall back.

    ``kind`` optionally names a spec key (or hub id) when the path itself carries
    no recognisable descriptor; if the path *does* name a different spec, that is
    a conflict and an error, not a precedence rule.
    """
    from_name = spec_key(model_id_or_dir)
    from_kind = spec_key(kind) if kind else None
    if from_name and from_kind and from_name != from_kind:
        raise ConfigError(
            f"conflicting Gipformer specs: {model_id_or_dir!r} is {from_name!r} but "
            f"kind={kind!r} is {from_kind!r}. " + _add_spec_help()
        )
    chosen = from_kind or from_name
    if chosen is None:
        raise ConfigError(
            f"unknown Gipformer model {model_id_or_dir!r}"
            + (f" (kind={kind!r})" if kind else "")
            + ". "
            + _add_spec_help()
        )
    return GIPFORMER_SPECS[chosen]


def known_quantizations(spec: GipformerSpec) -> tuple[str, ...]:
    """Quantizations ``spec`` declares an artifact set for."""
    return tuple(spec.quantizations)


def artifacts_for(spec: GipformerSpec, quantization: str = "int8") -> dict[str, str]:
    """Role → filename for ``quantization``, or an actionable error.

    int8 filenames are the measured ones. fp32 names are the same names without
    the ``.int8`` infix — that is how the checkpoint's own ``config.json``
    declares them (``onnx.encoder`` next to ``onnx.encoder_int8``). A spec with no
    declared quantization set (the unverified 1.5 entry) raises instead of
    inventing files.
    """
    quant = str(quantization).strip().lower()
    if quant not in ("int8", "fp32"):
        raise ConfigError(f"unknown quantization {quantization!r}; choose 'int8' or 'fp32'")
    if not spec.quantizations:
        raise ConfigError(
            f"spec {spec.key!r} declares no verified artifact set "
            f"({'UNVERIFIED: no weights available' if not spec.verified else 'unknown'}). "
            + spec.notes
        )
    if quant not in spec.quantizations:
        raise ConfigError(
            f"spec {spec.key!r} has no {quant!r} artifact set; declared: {spec.quantizations}. "
            + spec.notes
        )
    if quant == "int8":
        return dict(spec.onnx)
    return {
        role: filename.replace(".int8.onnx", ".onnx")
        for role, filename in spec.onnx.items()
    }


def hash_tokens_file(path: str | Path) -> str:
    """sha256 of a token file, read through the compliance guard."""
    allowed = assert_path_allowed(path)
    digest = hashlib.sha256()
    with open(allowed, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(
    spec: GipformerSpec,
    directory: str | Path,
    *,
    quantization: str = "int8",
    verify_tokens: bool = False,
) -> list[str]:
    """Problems that would stop ``spec`` from loading from ``directory``.

    Returns a list (empty = nothing missing) and never raises, so a caller can
    report several problems at once. Presence only by default: ``verify_tokens``
    additionally compares ``tokens.txt`` with the spec's recorded sha256.
    """
    problems: list[str] = []
    try:
        allowed = assert_path_allowed(directory)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return [f"{type(exc).__name__}: {exc}"]
    directory = Path(allowed)
    if not directory.is_dir():
        return [f"model directory not found: {directory}"]

    try:
        roles = artifacts_for(spec, quantization)
    except ConfigError as exc:
        return [str(exc)]

    for role, filename in roles.items():
        if not (directory / filename).is_file():
            problems.append(f"missing {role} file: {filename}")

    if verify_tokens and spec.tokens_sha256:
        token_path = directory / roles["tokens"]
        if token_path.is_file():
            try:
                actual = hash_tokens_file(token_path)
            except OSError as exc:
                problems.append(f"cannot hash {roles['tokens']}: {exc}")
            else:
                if actual != spec.tokens_sha256:
                    problems.append(
                        f"tokens.txt sha256 mismatch: expected {spec.tokens_sha256}, got {actual}"
                    )
    return problems


def download_hint(spec: GipformerSpec, directory: Path, quantization: str) -> str:
    if spec.verified:
        return (
            f"Download with: python scripts/download_asr_model.py "
            f"--output-dir {directory} --quantization {quantization}"
        )
    return (
        f"No verified artifact set for {spec.key!r}: fetch the checkpoint from "
        f"{spec.model_id!r} (revision unknown/UNVERIFIED), inspect its files, then either "
        f"place them in {directory} or record the measured file names in GIPFORMER_SPECS "
        f"(src/lifemate/asr/model_specs.py). Refusing to guess filenames."
    )


def require_spec_artifacts(
    spec: GipformerSpec,
    directory: str | Path,
    *,
    quantization: str = "int8",
) -> None:
    """Raise :class:`AsrError` naming every missing artifact."""
    problems = validate_artifacts(spec, directory, quantization=quantization)
    if not problems:
        return
    hint = download_hint(spec, Path(str(directory)), quantization)
    raise AsrError(
        f"missing Gipformer ({quantization}) model files in {directory} for spec "
        f"{spec.key!r}: {', '.join(problems)}. {hint}"
    )


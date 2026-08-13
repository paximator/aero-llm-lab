"""Build a frozen corpus from immutable NTSB source and chunk manifests."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerollm.common.schemas import Document, Split
from aerollm.data.chunking import ChunkManifest
from aerollm.data.snapshots import atomic_write
from aerollm.data.splits import assign_family_splits
from aerollm.data.validation import CorpusValidationReport, validate_built_corpus
from aerollm.evaluation.corpus import CorpusManifest, SourceManifestEntry


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    version: str
    seed: str
    train_fraction: float
    development_fraction: float
    test_fraction: float
    minimum_chunk_characters: int
    chunk_root: Path
    output: Path
    schema_version: int = 1

    @classmethod
    def load(cls, path: Path) -> CorpusConfig:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version", "version", "seed", "train_fraction",
            "development_fraction", "test_fraction", "minimum_chunk_characters",
            "chunk_root", "output",
        }
        if set(value) != expected:
            raise ValueError("invalid corpus configuration fields")
        return cls(
            version=value["version"], seed=value["seed"],
            train_fraction=value["train_fraction"],
            development_fraction=value["development_fraction"],
            test_fraction=value["test_fraction"],
            minimum_chunk_characters=value["minimum_chunk_characters"],
            chunk_root=Path(value["chunk_root"]), output=Path(value["output"]),
            schema_version=value["schema_version"],
        )

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.version or not self.seed:
            raise ValueError("unsupported or incomplete corpus configuration")
        fractions = self.fractions
        if any(type(value) not in {int, float} or value < 0 for value in fractions.values()):
            raise ValueError("split fractions must be non-negative numbers")
        if self.minimum_chunk_characters < 1:
            raise ValueError("minimum_chunk_characters must be positive")

    @property
    def fractions(self) -> dict[Split, float]:
        return {
            Split.TRAIN: float(self.train_fraction),
            Split.DEVELOPMENT: float(self.development_fraction),
            Split.TEST: float(self.test_fraction),
        }

    @property
    def fingerprint(self) -> str:
        value = {
            "schema_version": self.schema_version, "version": self.version,
            "seed": self.seed, "fractions": {k.value: v for k, v in self.fractions.items()},
            "minimum_chunk_characters": self.minimum_chunk_characters,
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class CorpusBuildResult:
    corpus: CorpusManifest
    validation: CorpusValidationReport
    config_fingerprint: str
    input_manifests: tuple[str, ...]


def build_corpus(
    source_manifest_paths: Sequence[Path], config: CorpusConfig,
    *, explicit_family_splits: Mapping[str, Split] | None = None,
) -> CorpusBuildResult:
    if not source_manifest_paths:
        raise ValueError("at least one source manifest is required")
    records = [_load_report(path, config.chunk_root) for path in source_manifest_paths]
    flattened = [item for group in records for item in group]
    if not flattened:
        raise ValueError("source manifests contain no parsed report documents")
    families = {item.event_family_id for item in flattened}
    if explicit_family_splits is None:
        family_splits = assign_family_splits(
            families, seed=config.seed, fractions=config.fractions,
        )
    else:
        family_splits = dict(explicit_family_splits)
        if set(family_splits) != families:
            missing = sorted(families - set(family_splits))
            extra = sorted(set(family_splits) - families)
            raise ValueError(
                f"explicit family split mismatch; missing={missing}, extra={extra}"
            )
        if not all(isinstance(split, Split) for split in family_splits.values()):
            raise ValueError("explicit family splits must contain Split values")
    sources = tuple(
        SourceManifestEntry(
            item.source_id, item.event_id, item.event_family_id,
            family_splits[item.event_family_id], item.document.source_sha256,
        )
        for item in sorted(flattened, key=lambda value: value.source_id)
    )
    ordered_records = sorted(flattened, key=lambda item: item.document.document_id)
    documents = tuple(item.document for item in ordered_records)
    chunks = tuple(
        chunk
        for item in ordered_records
        for chunk in item.chunks.chunks
    )
    corpus = CorpusManifest(config.version, sources, documents, chunks)
    validation = validate_built_corpus(
        corpus, minimum_chunk_characters=config.minimum_chunk_characters
    )
    if not validation.valid:
        raise ValueError("corpus validation failed: " + "; ".join(validation.errors))
    return CorpusBuildResult(
        corpus, validation, config.fingerprint,
        tuple(path.as_posix() for path in sorted(source_manifest_paths)),
    )


def write_corpus(result: CorpusBuildResult, output: Path) -> tuple[str, Path]:
    content = (json.dumps(result.corpus.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    atomic_write(output, content)
    digest = hashlib.sha256(content).hexdigest()
    build_path = output.with_suffix(".build.json")
    build = {
        "schema_version": 1, "corpus_path": output.as_posix(), "corpus_sha256": digest,
        "config_fingerprint": result.config_fingerprint,
        "input_manifests": list(result.input_manifests),
        "validation": result.validation.to_dict(),
    }
    atomic_write(build_path, (json.dumps(build, indent=2, sort_keys=True) + "\n").encode())
    return digest, build_path


@dataclass(frozen=True, slots=True)
class _ReportRecord:
    source_id: str
    event_id: str
    event_family_id: str
    document: Document
    chunks: ChunkManifest


def _load_report(path: Path, chunk_root: Path) -> list[_ReportRecord]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require_manifest_shape(manifest)
    snapshots = {item["sha256"]: item for item in manifest["snapshots"]}
    records: list[_ReportRecord] = []
    for artifact in manifest["derived_artifacts"]:
        if artifact["kind"] != "parsed-document":
            continue
        parsed_path = Path(artifact["artifact_path"])
        parsed_bytes = parsed_path.read_bytes()
        if hashlib.sha256(parsed_bytes).hexdigest() != artifact["sha256"]:
            raise ValueError(f"parsed artifact digest mismatch: {parsed_path}")
        document = Document.from_dict(json.loads(parsed_bytes))
        snapshot = snapshots.get(document.source_sha256)
        if snapshot is None:
            raise ValueError(f"parsed document has no source snapshot: {document.document_id}")
        source_id = snapshot["source_id"]
        event_id, separator, _ = source_id.partition(":")
        if not separator or not event_id:
            raise ValueError(f"report source_id does not encode an event: {source_id}")
        chunk_path = chunk_root / f"{document.document_id}.json"
        chunks = ChunkManifest.from_dict(json.loads(chunk_path.read_text(encoding="utf-8")))
        if chunks.source_sha256 != document.source_sha256:
            raise ValueError(f"chunk source mismatch: {chunk_path}")
        records.append(_ReportRecord(source_id, event_id, event_id, document, chunks))
    return records


def _require_manifest_shape(value: Any) -> None:
    expected = {
        "schema_version", "source", "created_at", "snapshots",
        "derived_artifacts", "links",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("invalid source manifest fields")
    if value["schema_version"] != 2 or value["source"] != "ntsb":
        raise ValueError("unsupported source manifest")
    if not isinstance(value["snapshots"], list) or not isinstance(value["derived_artifacts"], list):
        raise ValueError("source manifest collections must be lists")

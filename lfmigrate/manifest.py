"""Sharded, resumable manifests.

Design, and why it differs from the Phoenix migration tooling it is modelled on
(``arize-skills/skills/arize-phoenix-migration/scripts/migrate.py``):

That tool keeps every span in ONE JSON manifest, loads the whole file, and
checksums the in-memory list. That is fine for a few thousand spans and will
not survive three million: span payloads carry full prompt/response bodies, so
a single manifest would be tens of gigabytes and the process would die holding
it.

So spans live in a streamable JSONL **data file** (one span per line) and the
manifest holds only metadata: counts, a checksum of the data file, and the
batch ledger addressed by line range. Memory stays bounded by batch size, not
by migration size.

What IS inherited, deliberately and with attribution, is the batch state
machine, because it is the part that makes retries safe:

    pending -> uncertain -> submitted

``uncertain`` is persisted BEFORE the upload is attempted. If the process dies
mid-upload, the batch is found in ``uncertain`` on restart and the tool refuses
to retry it, because a retry after an ambiguous outcome is how you get
duplicates. There is no evidence AX de-duplicates by span ID -- the Phoenix
tooling assumes it does not -- so an ambiguous batch must be resolved by
reading back, never by resending and hoping.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

MANIFEST_VERSION = 2
#: Rows per upload batch. 10_000 matches the SDK's own
#: DEFAULT_PYARROW_MAX_CHUNKSIZE, i.e. the record-batch size it already chunks
#: an Arrow table into, so a batch maps to one internal chunk instead of being
#: 20x smaller than the unit the SDK is built around.
#:
#: Measured on a live space (20k spans, same data, same machine):
#:     batch=500    40 posts   62.7s    319 spans/s
#:     batch=2500    8 posts   19.2s   1044 spans/s
#:     batch=10000   2 posts   13.9s   1442 spans/s
#: Fixed cost per POST is ~1.3s (TLS handshake -- the SDK posts with a bare
#: requests.post, not a Session, so there is no connection reuse -- plus a
#: dataframe copy, validation and an Arrow temp-file write). At 500 rows that
#: overhead is 82% of the call; at 10_000 it is 19%.
DEFAULT_BATCH_SIZE = 10_000

#: Byte budget per batch, as measured on the serialized JSONL. A row count
#: alone is the wrong unit: 10_000 lean spans serialize to ~9 MB, but spans
#: carrying long prompts and responses can be 10x that, and a 100 MB POST
#: risks a timeout or a server-side rejection. Whichever limit is hit first
#: closes the batch.
DEFAULT_BATCH_MAX_BYTES = 32 * 1024 * 1024

PENDING = "pending"
UNCERTAIN = "uncertain"
SUBMITTED = "submitted"


class ManifestError(Exception):
    pass


class AmbiguousBatch(ManifestError):
    """A batch whose upload outcome is unknown. Must be resolved by readback."""


def _atomic_write(path: Path, text: str) -> None:
    """Write via temp file + rename so a crash cannot truncate the ledger.

    A half-written manifest is worse than no manifest: it loses the record of
    which batches already landed, which is the only thing standing between a
    resume and duplicated data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def file_checksum(path: Path, *, chunk: int = 1 << 20) -> str:
    """Streamed SHA-256 of the data file. Never loads it whole."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Shard:
    """One unit of migration: conventionally a single UTC day."""

    root: Path
    name: str

    @property
    def data_path(self) -> Path:
        return self.root / f"{self.name}.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / f"{self.name}.manifest.json"

    def exists(self) -> bool:
        return self.manifest_path.exists()


def write_shard(
    shard: Shard,
    spans: Iterable[dict],
    *,
    source: dict,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_max_bytes: int = DEFAULT_BATCH_MAX_BYTES,
    overwrite: bool = False,
) -> dict:
    """Stream spans to the shard's data file and seal a manifest over them.

    Refuses to clobber an existing shard unless ``overwrite`` is set: an
    accidental re-export after a partial import would reset the batch ledger
    and re-upload everything.

    Duplicate span IDs are rejected. The upload path has no de-duplication to
    fall back on, so a duplicate here becomes a duplicate in AX.
    """
    if shard.exists() and not overwrite:
        raise ManifestError(
            f"Shard {shard.name!r} already exists. Use the existing manifest for "
            f"import/verify, or pass overwrite to discard it."
        )

    shard.root.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    trace_ids: set[str] = set()
    count = 0
    # Batch boundaries are decided while writing, because that is the only
    # point at which the serialized size of each span is known.
    boundaries: list[tuple[int, int]] = []
    batch_start = 0
    batch_bytes = 0

    # Written to a temp file first so a failed export never leaves a data file
    # that a later manifest could be sealed over.
    handle, temp_name = tempfile.mkstemp(dir=str(shard.root), suffix=".jsonl.tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w") as stream:
            for span in spans:
                span_id = (span.get("context") or {}).get("span_id")
                if not span_id:
                    raise ManifestError("Span is missing context.span_id.")
                if span_id in seen:
                    raise ManifestError(
                        f"Duplicate span ID {span_id!r} in shard {shard.name!r}. "
                        "The upload path does not de-duplicate, so this would "
                        "double-count in AX."
                    )
                seen.add(span_id)
                trace_id = (span.get("context") or {}).get("trace_id")
                if trace_id:
                    trace_ids.add(trace_id)
                line = json.dumps(span, default=str, ensure_ascii=False) + "\n"
                stream.write(line)
                count += 1
                batch_bytes += len(line.encode("utf-8"))
                # Cut on rows OR bytes, whichever comes first. A single span
                # larger than the budget still forms a batch of one rather
                # than being dropped or looping.
                if (count - batch_start >= batch_size
                        or batch_bytes >= batch_max_bytes):
                    boundaries.append((batch_start, count))
                    batch_start, batch_bytes = count, 0
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, shard.data_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    # The trailing partial batch, if the stream did not land on a boundary.
    if count > batch_start:
        boundaries.append((batch_start, count))

    manifest = {
        "version": MANIFEST_VERSION,
        "shard": shard.name,
        "source": source,
        "destination": None,
        "span_count": count,
        "trace_count": len(trace_ids),
        "data_file": shard.data_path.name,
        "checksum": file_checksum(shard.data_path),
        "batches": [
            {"start": start, "end": end, "status": PENDING}
            for start, end in boundaries
        ],
    }
    _atomic_write(shard.manifest_path, json.dumps(manifest, indent=2))
    return manifest


def load_manifest(shard: Shard) -> dict:
    """Load and integrity-check a manifest against its data file."""
    if not shard.manifest_path.exists():
        raise ManifestError(f"No manifest for shard {shard.name!r}.")
    manifest = json.loads(shard.manifest_path.read_text())
    if manifest.get("version") != MANIFEST_VERSION:
        raise ManifestError(
            f"Manifest version {manifest.get('version')!r} is not supported "
            f"(expected {MANIFEST_VERSION})."
        )
    if not shard.data_path.exists():
        raise ManifestError(f"Data file {shard.data_path.name!r} is missing.")
    # Verifying the checksum on every load is cheap next to an upload and is the
    # only thing that catches a data file edited or truncated between export and
    # import -- which would otherwise upload silently wrong content.
    if file_checksum(shard.data_path) != manifest["checksum"]:
        raise ManifestError(
            f"Data file for shard {shard.name!r} does not match its manifest "
            "checksum. Re-export this shard; do not import it."
        )
    return manifest


def save_manifest(shard: Shard, manifest: dict) -> None:
    _atomic_write(shard.manifest_path, json.dumps(manifest, indent=2))


def read_span_range(shard: Shard, start: int, end: int) -> list[dict]:
    """Read lines [start, end) without loading the whole data file."""
    spans: list[dict] = []
    with shard.data_path.open() as stream:
        for index, line in enumerate(stream):
            if index >= end:
                break
            if index >= start:
                spans.append(json.loads(line))
    return spans


def iter_spans(shard: Shard) -> Iterator[dict]:
    with shard.data_path.open() as stream:
        for line in stream:
            yield json.loads(line)


def bind_destination(shard: Shard, manifest: dict, destination: dict) -> dict:
    """Pin the shard to one destination, so a resume cannot change target.

    Resuming into a different space or project would scatter one migration
    across two destinations and make the counts irreconcilable.
    """
    existing = manifest.get("destination")
    if existing and existing != destination:
        raise ManifestError(
            f"Shard {shard.name!r} was already bound to "
            f"{existing.get('project_name')!r} in space {existing.get('space_id')!r}. "
            "Resuming into a different destination is refused."
        )
    if not existing:
        manifest["destination"] = destination
        save_manifest(shard, manifest)
    return manifest


def pending_batches(manifest: dict) -> list[dict]:
    """Batches still to upload, refusing to proceed past an ambiguous one."""
    ambiguous = [b for b in manifest["batches"] if b["status"] == UNCERTAIN]
    if ambiguous:
        raise AmbiguousBatch(
            f"Shard {manifest['shard']!r} has {len(ambiguous)} batch(es) with an "
            "unknown upload outcome. Run verify to establish what landed; do not "
            "retry blindly, as the upload path does not de-duplicate."
        )
    return [b for b in manifest["batches"] if b["status"] != SUBMITTED]


def mark(shard: Shard, manifest: dict, batch: dict, status: str) -> None:
    batch["status"] = status
    save_manifest(shard, manifest)


def shard_progress(manifest: dict) -> dict:
    batches = manifest["batches"]
    submitted = [b for b in batches if b["status"] == SUBMITTED]
    return {
        "shard": manifest["shard"],
        "span_count": manifest["span_count"],
        "trace_count": manifest.get("trace_count"),
        "batches_total": len(batches),
        "batches_submitted": len(submitted),
        "spans_submitted": sum(b["end"] - b["start"] for b in submitted),
        "uncertain": sum(1 for b in batches if b["status"] == UNCERTAIN),
        "complete": len(submitted) == len(batches) and bool(batches),
    }


def discover_shards(root: Path) -> list[Shard]:
    """Every shard under ``root``, ordered by name (so oldest day first)."""
    if not root.exists():
        return []
    names = sorted(p.name[: -len(".manifest.json")]
                   for p in root.glob("*.manifest.json"))
    return [Shard(root=root, name=name) for name in names]


def shard_time_range(root: Path) -> tuple[datetime, datetime] | None:
    """The span time range the shards under ``root`` actually cover.

    ``verify`` needs this because it reads back by TIME, not by shard: a
    backfill migrates history, so the rows are as old as the source data, not
    as recent as the upload. Guessing a short recent window makes a correct
    migration look like a partial one -- a 4-day-old shard verified against a
    3-day window reports an empty project.

    Returns ``None`` when no shard records a window, so the caller can fall
    back rather than treat an empty range as "nothing migrated".
    """
    starts, ends = [], []
    for shard in discover_shards(root):
        try:
            source = load_manifest(shard).get("source") or {}
        except ManifestError:
            continue
        for key, bucket in (("window_start", starts), ("window_end", ends)):
            raw = source.get(key)
            if raw:
                try:
                    bucket.append(datetime.fromisoformat(raw))
                except ValueError:
                    continue
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def migration_progress(root: Path) -> dict:
    """Aggregate progress, for a status command and for safe resume."""
    shards = []
    for shard in discover_shards(root):
        try:
            shards.append(shard_progress(load_manifest(shard)))
        except ManifestError as exc:
            shards.append({"shard": shard.name, "error": str(exc)})
    return {
        "root": str(root),
        "shards": shards,
        "spans_total": sum(s.get("span_count") or 0 for s in shards),
        "spans_submitted": sum(s.get("spans_submitted") or 0 for s in shards),
        "shards_complete": sum(1 for s in shards if s.get("complete")),
        "shards_total": len(shards),
        "shards_with_errors": [s["shard"] for s in shards if s.get("error")],
        "shards_uncertain": [s["shard"] for s in shards if s.get("uncertain")],
    }

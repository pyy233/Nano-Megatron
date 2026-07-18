"""Pinned TinyStories download and exact deterministic sampling utilities."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

TINYSTORIES_CANONICAL_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_MIRROR_REVISION = "d71d0182cc67962186d395b75b2c180340904e00"
TINYSTORIES_TRAIN_BYTES = 1_924_281_556
TINYSTORIES_TRAIN_SHA256 = "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f"
TINYSTORIES_TRAIN_RECORDS = 2_119_719
TINYSTORIES_EMPTY_RECORDS = 230
TINYSTORIES_TRAIN_STORIES = TINYSTORIES_TRAIN_RECORDS - TINYSTORIES_EMPTY_RECORDS
TINYSTORIES_SAMPLE_STORIES = 500_000
TINYSTORIES_SAMPLE_SEED = 1234
TINYSTORIES_DELIMITER = b"<|endoftext|>"
TINYSTORIES_TRAIN_URLS = (
    "https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/TinyStories/repo?"
    f"Revision={TINYSTORIES_MIRROR_REVISION}&FilePath=TinyStories-train.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    f"{TINYSTORIES_CANONICAL_REVISION}/TinyStories-train.txt?download=true",
)
TINYSTORIES_VALIDATION_BYTES = 19_447_282
TINYSTORIES_VALIDATION_SHA256 = (
    "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
)
TINYSTORIES_VALIDATION_STORIES = 21_990
TINYSTORIES_VALIDATION_URLS = (
    "https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/TinyStories/repo?"
    f"Revision={TINYSTORIES_MIRROR_REVISION}&FilePath=TinyStories-valid.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    f"{TINYSTORIES_CANONICAL_REVISION}/TinyStories-valid.txt?download=true",
)

_SAMPLE_FORMAT = "nano-megatron-tinystories-sample"
_SAMPLE_VERSION = 1
_SELECTION_ALGORITHM = "blake2b-index-bottom-k-v1"

ProgressCallback = Callable[[int, int], None]
UrlOpener = Callable[[Request], Any]


@dataclass(frozen=True, slots=True)
class FileVerification:
    path: Path
    size_bytes: int
    sha256: str


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_sha256(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("expected_sha256 must be hexadecimal") from error
    return value.lower()


def sha256_file(path: str | Path, *, chunk_size: int = 8 << 20) -> str:
    """Return the SHA256 digest of one file without loading it into memory."""

    _positive_integer("chunk_size", chunk_size)
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: str | Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> FileVerification:
    """Require one file to match its pinned size and digest."""

    source = Path(path)
    size = _positive_integer("expected_size", expected_size)
    expected_digest = _validate_sha256(expected_sha256)
    if not source.is_file():
        raise FileNotFoundError(f"expected file does not exist: {source}")
    actual_size = source.stat().st_size
    if actual_size != size:
        raise ValueError(
            f"file size does not match pinned source: {source}: {actual_size} != {size}"
        )
    actual_digest = sha256_file(source)
    if actual_digest != expected_digest:
        raise ValueError(
            f"file SHA256 does not match pinned source: {source}: "
            f"{actual_digest} != {expected_digest}"
        )
    return FileVerification(source.resolve(), actual_size, actual_digest)


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    if isinstance(status, bool) or not isinstance(status, int):
        raise RuntimeError(f"download response returned invalid HTTP status {status!r}")
    return status


def _download_once(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    chunk_size: int,
    opener: UrlOpener,
    progress: ProgressCallback | None,
) -> FileVerification:
    partial = destination.with_name(f"{destination.name}.part")
    current_size = partial.stat().st_size if partial.exists() else 0
    if current_size > expected_size:
        raise ValueError(
            f"partial download is larger than the pinned source: {partial}: "
            f"{current_size} > {expected_size}"
        )
    if current_size == expected_size:
        try:
            verified = verify_file(
                partial,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
        except ValueError:
            partial.unlink()
            raise
        partial.replace(destination)
        return FileVerification(destination.resolve(), verified.size_bytes, verified.sha256)

    headers = {"User-Agent": "nano-megatron-tinystories/1"}
    if current_size:
        headers["Range"] = f"bytes={current_size}-"
    request = Request(url, headers=headers)

    digest = hashlib.sha256()
    if current_size:
        with partial.open("rb") as existing:
            while chunk := existing.read(chunk_size):
                digest.update(chunk)

    with opener(request) as response:
        status = _response_status(response)
        append = current_size > 0 and status == 206
        if append:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {current_size}-"):
                raise RuntimeError(
                    "download server returned an incompatible Content-Range for resume: "
                    f"{content_range!r}"
                )
        elif status in {200, 206}:
            # Some mirrors ignore Range and return the full body. Restart only
            # the script-owned partial file; never truncate the published target.
            current_size = 0
            digest = hashlib.sha256()
        else:
            raise RuntimeError(f"download server returned HTTP status {status} for {url}")

        mode = "ab" if append else "wb"
        downloaded = current_size
        with partial.open(mode) as stream:
            if progress is not None:
                progress(downloaded, expected_size)
            while chunk := response.read(chunk_size):
                stream.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if downloaded > expected_size:
                    stream.close()
                    partial.unlink(missing_ok=True)
                    raise ValueError(
                        f"download exceeded pinned source size: {downloaded} > {expected_size}"
                    )
                if progress is not None:
                    progress(downloaded, expected_size)
            stream.flush()
            os.fsync(stream.fileno())

    if downloaded != expected_size:
        raise EOFError(
            f"download ended before the pinned source size: {downloaded} != {expected_size}; "
            f"rerun to resume {partial}"
        )
    actual_digest = digest.hexdigest()
    if actual_digest != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(
            "downloaded TinyStories source failed SHA256 verification: "
            f"{actual_digest} != {expected_sha256}"
        )
    partial.replace(destination)
    return FileVerification(destination.resolve(), downloaded, actual_digest)


def download_pinned_file(
    destination: str | Path,
    *,
    urls: Sequence[str] = TINYSTORIES_TRAIN_URLS,
    expected_size: int = TINYSTORIES_TRAIN_BYTES,
    expected_sha256: str = TINYSTORIES_TRAIN_SHA256,
    chunk_size: int = 8 << 20,
    retries: int = 3,
    opener: UrlOpener = urlopen,
    progress: ProgressCallback | None = None,
) -> FileVerification:
    """Download a pinned file with mirror fallback, Range resume, and atomic publish."""

    target = Path(destination)
    size = _positive_integer("expected_size", expected_size)
    digest = _validate_sha256(expected_sha256)
    chunk = _positive_integer("chunk_size", chunk_size)
    attempts = _positive_integer("retries", retries)
    candidates = tuple(urls)
    if not candidates or any(not isinstance(url, str) or not url for url in candidates):
        raise ValueError("urls must contain at least one non-empty URL")
    if target.exists():
        return verify_file(target, expected_size=size, expected_sha256=digest)
    target.parent.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        for url in candidates:
            try:
                return _download_once(
                    url,
                    target,
                    expected_size=size,
                    expected_sha256=digest,
                    chunk_size=chunk,
                    opener=opener,
                    progress=progress,
                )
            except (EOFError, OSError, RuntimeError, ValueError) as error:
                failures.append(f"attempt {attempt} {url}: {type(error).__name__}: {error}")
    details = "; ".join(failures)
    raise RuntimeError(f"could not download pinned TinyStories source: {details}")


def _iter_delimited_story_bytes(
    path: str | Path,
    *,
    delimiter: bytes = TINYSTORIES_DELIMITER,
    chunk_size: int = 8 << 20,
) -> Iterator[bytes]:
    """Yield stripped story payloads without quadratic buffer slicing."""

    if not isinstance(delimiter, bytes) or not delimiter:
        raise ValueError("delimiter must be non-empty bytes")
    _positive_integer("chunk_size", chunk_size)
    source = Path(path)
    buffer = b""
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            buffer += chunk
            parts = buffer.split(delimiter)
            buffer = parts.pop()
            for raw_story in parts:
                stripped = raw_story.strip()
                if stripped:
                    yield stripped
        stripped = buffer.strip()
        if stripped:
            yield stripped


def iter_delimited_stories(
    path: str | Path,
    *,
    delimiter: bytes = TINYSTORIES_DELIMITER,
    chunk_size: int = 8 << 20,
) -> Iterator[str]:
    """Yield complete non-empty UTF-8 stories from the upstream text artifact."""

    for story in _iter_delimited_story_bytes(
        path,
        delimiter=delimiter,
        chunk_size=chunk_size,
    ):
        yield story.decode("utf-8")


def _selection_score(seed: int, index: int) -> int:
    payload = f"{_SELECTION_ALGORITHM}\0{seed}\0{index}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def select_story_indices(
    source: str | Path,
    *,
    sample_size: int,
    seed: int,
    expected_source_records: int | None,
    progress: ProgressCallback | None = None,
) -> tuple[frozenset[int], int]:
    """Select the exact bottom-k deterministic source indices in bounded memory."""

    requested = _positive_integer("sample_size", sample_size)
    selected_seed = _non_negative_integer("seed", seed)
    if expected_source_records is not None:
        _positive_integer("expected_source_records", expected_source_records)
        if requested > expected_source_records:
            raise ValueError("sample_size cannot exceed expected_source_records")

    heap: list[tuple[int, int]] = []
    total = 0
    expected_progress = expected_source_records or requested
    for index, _ in enumerate(_iter_delimited_story_bytes(source)):
        score = _selection_score(selected_seed, index)
        item = (-score, -index)
        if len(heap) < requested:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
        total = index + 1
        if progress is not None and (total % 100_000 == 0):
            progress(total, expected_progress)

    if total < requested:
        raise ValueError(f"source contains only {total} stories; cannot select {requested}")
    if expected_source_records is not None and total != expected_source_records:
        raise ValueError(
            f"TinyStories source record count changed: {total} != {expected_source_records}"
        )
    if progress is not None:
        progress(total, expected_progress)
    return frozenset(-negative_index for _, negative_index in heap), total


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read TinyStories metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"TinyStories metadata must contain a JSON object: {path}")
    return value


def validate_sample_artifact(
    sample_path: str | Path,
    metadata_path: str | Path,
    *,
    sample_size: int,
    seed: int,
    expected_source_records: int,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Validate one previously published exact-sample artifact."""

    sample = Path(sample_path)
    metadata_file = Path(metadata_path)
    if not sample.is_file() or not metadata_file.is_file():
        raise FileNotFoundError("TinyStories sample requires both JSONL and metadata files")
    metadata = _read_json_object(metadata_file)
    selection = metadata.get("selection")
    source = metadata.get("source")
    artifact = metadata.get("artifact")
    if metadata.get("format") != _SAMPLE_FORMAT or metadata.get("version") != _SAMPLE_VERSION:
        raise ValueError("TinyStories sample metadata format/version is unsupported")
    if not isinstance(selection, dict) or not isinstance(source, dict) or not isinstance(
        artifact, dict
    ):
        raise ValueError("TinyStories sample metadata sections are invalid")
    expected_selection = {
        "algorithm": _SELECTION_ALGORITHM,
        "records": sample_size,
        "seed": seed,
        "source_records": expected_source_records,
    }
    if selection != expected_selection:
        raise ValueError(
            f"TinyStories sample selection metadata does not match: "
            f"{selection!r} != {expected_selection!r}"
        )
    if source.get("sha256") != _validate_sha256(expected_source_sha256):
        raise ValueError("TinyStories sample was built from a different source digest")
    actual_size = sample.stat().st_size
    if artifact.get("records") != sample_size or artifact.get("size_bytes") != actual_size:
        raise ValueError("TinyStories sample artifact count/size metadata does not match")
    actual_digest = sha256_file(sample)
    if artifact.get("sha256") != actual_digest:
        raise ValueError("TinyStories sample JSONL failed SHA256 verification")
    return metadata


def create_exact_sample(
    source_path: str | Path,
    sample_path: str | Path,
    metadata_path: str | Path,
    *,
    source_verification: FileVerification,
    sample_size: int = TINYSTORIES_SAMPLE_STORIES,
    seed: int = TINYSTORIES_SAMPLE_SEED,
    expected_source_records: int = TINYSTORIES_TRAIN_STORIES,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Create or validate an exact deterministic TinyStories JSONL sample."""

    source = Path(source_path).resolve()
    output = Path(sample_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    if source != source_verification.path.resolve():
        raise ValueError("source_verification does not describe source_path")
    if not source.is_file() or source.stat().st_size != source_verification.size_bytes:
        raise ValueError("source_path size changed after source verification")
    if output == metadata_file or source in {output, metadata_file}:
        raise ValueError("source, sample, and metadata paths must be distinct")

    if output.exists() or metadata_file.exists():
        if not output.is_file() or not metadata_file.is_file():
            raise FileExistsError(
                "refusing to replace an incomplete TinyStories sample artifact; "
                f"remove both {output} and {metadata_file} to rebuild"
            )
        metadata = validate_sample_artifact(
            output,
            metadata_file,
            sample_size=sample_size,
            seed=seed,
            expected_source_records=expected_source_records,
            expected_source_sha256=source_verification.sha256,
        )
        return {**metadata, "reused": True}

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    selected, total = select_story_indices(
        source,
        sample_size=sample_size,
        seed=seed,
        expected_source_records=expected_source_records,
        progress=progress,
    )

    suffix = uuid.uuid4().hex
    output_temporary = output.with_name(f".{output.name}.{suffix}.tmp")
    metadata_temporary = metadata_file.with_name(f".{metadata_file.name}.{suffix}.tmp")
    sample_digest = hashlib.sha256()
    written = 0
    output_published = False
    try:
        with output_temporary.open("xb") as stream:
            second_pass_records = 0
            for index, story_bytes in enumerate(_iter_delimited_story_bytes(source)):
                second_pass_records = index + 1
                if index not in selected:
                    continue
                text = story_bytes.decode("utf-8")
                encoded = (
                    json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                stream.write(encoded)
                sample_digest.update(encoded)
                written += 1
                if progress is not None and (written % 100_000 == 0):
                    progress(written, sample_size)
            stream.flush()
            os.fsync(stream.fileno())
        if written != sample_size:
            raise RuntimeError(f"sample writer emitted {written} stories, expected {sample_size}")
        if second_pass_records != expected_source_records:
            raise RuntimeError(
                "TinyStories source record count changed between sample passes: "
                f"{second_pass_records} != {expected_source_records}"
            )
        if progress is not None:
            progress(written, sample_size)

        metadata = {
            "artifact": {
                "format": "jsonl",
                "path": str(output),
                "records": written,
                "sha256": sample_digest.hexdigest(),
                "size_bytes": output_temporary.stat().st_size,
                "text_key": "text",
            },
            "format": _SAMPLE_FORMAT,
            "selection": {
                "algorithm": _SELECTION_ALGORITHM,
                "records": sample_size,
                "seed": seed,
                "source_records": total,
            },
            "source": {
                "canonical_revision": TINYSTORIES_CANONICAL_REVISION,
                "empty_records_excluded": (
                    TINYSTORIES_EMPTY_RECORDS
                    if source_verification.sha256 == TINYSTORIES_TRAIN_SHA256
                    else None
                ),
                "mirror_revision": TINYSTORIES_MIRROR_REVISION,
                "non_empty_stories": total,
                "path": str(source),
                "records_including_empty": (
                    TINYSTORIES_TRAIN_RECORDS
                    if source_verification.sha256 == TINYSTORIES_TRAIN_SHA256
                    else None
                ),
                "sha256": source_verification.sha256,
                "size_bytes": source_verification.size_bytes,
                "story_delimiter": TINYSTORIES_DELIMITER.decode("ascii"),
            },
            "version": _SAMPLE_VERSION,
        }
        metadata_temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_temporary.replace(output)
        output_published = True
        metadata_temporary.replace(metadata_file)
    except BaseException:
        output_temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)
        if output_published:
            output.unlink(missing_ok=True)
        raise
    return {**metadata, "reused": False}


def prepare_tinystories_500k(
    data_dir: str | Path = Path("data/raw/tinystories"),
    *,
    urls: Sequence[str] = TINYSTORIES_TRAIN_URLS,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Download the pinned train source and publish the standard 500k sample."""

    directory = Path(data_dir)
    source = directory / "TinyStories-train.txt"
    sample = directory / "train_500k.jsonl"
    metadata = directory / "train_500k.metadata.json"
    verification = download_pinned_file(source, urls=urls, progress=progress)
    sample_metadata = create_exact_sample(
        source,
        sample,
        metadata,
        source_verification=verification,
        progress=progress,
    )
    return {
        "command": "prepare-tinystories-500k",
        "sample": sample_metadata,
        "sample_metadata": str(metadata.resolve()),
        "sample_output": str(sample.resolve()),
        "source": {
            "path": str(verification.path),
            "sha256": verification.sha256,
            "size_bytes": verification.size_bytes,
        },
    }


__all__ = [
    "FileVerification",
    "TINYSTORIES_CANONICAL_REVISION",
    "TINYSTORIES_DELIMITER",
    "TINYSTORIES_EMPTY_RECORDS",
    "TINYSTORIES_MIRROR_REVISION",
    "TINYSTORIES_SAMPLE_SEED",
    "TINYSTORIES_SAMPLE_STORIES",
    "TINYSTORIES_TRAIN_BYTES",
    "TINYSTORIES_TRAIN_RECORDS",
    "TINYSTORIES_TRAIN_SHA256",
    "TINYSTORIES_TRAIN_STORIES",
    "TINYSTORIES_TRAIN_URLS",
    "TINYSTORIES_VALIDATION_BYTES",
    "TINYSTORIES_VALIDATION_SHA256",
    "TINYSTORIES_VALIDATION_STORIES",
    "TINYSTORIES_VALIDATION_URLS",
    "create_exact_sample",
    "download_pinned_file",
    "iter_delimited_stories",
    "prepare_tinystories_500k",
    "select_story_indices",
    "sha256_file",
    "validate_sample_artifact",
    "verify_file",
]

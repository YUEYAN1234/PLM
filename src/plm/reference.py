from __future__ import annotations

import csv
import hashlib
import heapq
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from .fasta import FastaRecord, is_valid_sequence, parse_fasta


@dataclass(frozen=True)
class HeaderInfo:
    record_id: str
    description: str
    tax: str
    tax_id: str
    rep_id: str


def stable_hash_score(value: str) -> int:
    return int.from_bytes(hashlib.sha1(value.encode("utf-8")).digest(), byteorder="big", signed=False)


def parse_uniref_header(header: str) -> HeaderInfo:
    header = header.strip()
    if not header:
        return HeaderInfo("", "", "", "", "")

    parts = header.split(maxsplit=1)
    record_id = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    tax = _first_match(body, r"(?:^|\s)Tax=(.*?)(?=\s+TaxID=|\s+RepID=|$)")
    tax_id = _first_match(body, r"(?:^|\s)TaxID=(\S+)")
    rep_id = _first_match(body, r"(?:^|\s)RepID=(\S+)")
    description = body
    marker_positions = [
        position
        for marker in (" n=", " Tax=", " TaxID=", " RepID=")
        if (position := body.find(marker)) >= 0
    ]
    if marker_positions:
        description = body[: min(marker_positions)]
    return HeaderInfo(record_id, description.strip(), tax.strip(), tax_id.strip(), rep_id.strip())


def _first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def sample_reference_records(
    fasta_path: Path,
    *,
    max_records: int,
    min_length: int = 30,
) -> list[FastaRecord]:
    if max_records <= 0:
        raise ValueError("max_records must be positive")

    heap: list[tuple[int, int, FastaRecord]] = []
    seen = 0
    for record in tqdm(parse_fasta(fasta_path), desc="sampling reference"):
        if len(record.sequence) < min_length or not is_valid_sequence(record.sequence):
            continue
        score = stable_hash_score(record.record_id)
        item = (-score, seen, record)
        seen += 1
        if len(heap) < max_records:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)

    return [item[2] for item in sorted(heap, key=lambda entry: -entry[0])]


def metadata_rows(records: Iterable[FastaRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        header = parse_uniref_header(record.description)
        rows.append(
            {
                "id": record.record_id,
                "description": header.description,
                "tax": header.tax,
                "tax_id": header.tax_id,
                "rep_id": header.rep_id,
                "length": len(record.sequence),
            }
        )
    return rows


def write_metadata_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "description", "tax", "tax_id", "rep_id", "length"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_reference_index(
    *,
    out_dir: Path,
    embeddings: torch.Tensor,
    records: list[FastaRecord],
    source_fasta: Path,
    checkpoint: Path,
    max_records: int,
    min_length: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = metadata_rows(records)
    index = {
        "ids": [record.record_id for record in records],
        "descriptions": [record.description for record in records],
        "metadata": rows,
        "embeddings": embeddings.cpu(),
        "source_fasta": str(source_fasta),
        "checkpoint": str(checkpoint),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "normalized": True,
    }
    torch.save(index, out_dir / "index.pt")
    write_metadata_csv(rows, out_dir / "metadata.csv")
    summary = {
        "source_fasta": str(source_fasta),
        "checkpoint": str(checkpoint),
        "max_records": max_records,
        "min_length": min_length,
        "records": len(records),
        "embedding_dim": index["embedding_dim"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

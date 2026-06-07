from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

from .fasta import is_valid_sequence, parse_fasta
from .tokenizer import ProteinTokenizer


def validation_bucket(record_id: str) -> float:
    digest = hashlib.sha1(record_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def record_split(record_id: str, validation_fraction: float) -> str:
    if validation_fraction <= 0:
        return "train"
    if validation_fraction >= 1:
        return "validation"
    return "validation" if validation_bucket(record_id) < validation_fraction else "train"


def sequence_chunks(sequence: str, payload_length: int, min_length: int) -> Iterator[tuple[int, str]]:
    for chunk_index, start in enumerate(range(0, len(sequence), payload_length)):
        chunk = sequence[start : start + payload_length]
        if len(chunk) >= min_length:
            yield chunk_index, chunk


def encoded_examples(
    fasta_path: str | Path,
    *,
    split: str,
    tokenizer: ProteinTokenizer,
    max_length: int,
    min_length: int,
    validation_fraction: float,
) -> Iterator[dict[str, object]]:
    payload_length = max_length - 2
    if payload_length < min_length:
        raise ValueError("max_length must leave enough payload for min_length after special tokens")

    for record in parse_fasta(fasta_path):
        if len(record.sequence) < min_length or not is_valid_sequence(record.sequence):
            continue
        if record_split(record.record_id, validation_fraction) != split:
            continue
        for chunk_index, chunk in sequence_chunks(record.sequence, payload_length, min_length):
            input_ids = tokenizer.encode(chunk, add_special_tokens=True, max_length=max_length)
            yield {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "sequence_id": record.record_id,
                "chunk_index": chunk_index,
                "length": len(chunk),
            }

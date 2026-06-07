from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

from .tokenizer import RESIDUE_TOKENS


VALID_RESIDUES = set(RESIDUE_TOKENS)


@dataclass(frozen=True)
class FastaRecord:
    record_id: str
    description: str
    sequence: str


def open_text(path: str | Path) -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def parse_fasta(path: str | Path) -> Iterator[FastaRecord]:
    record_id: str | None = None
    description = ""
    chunks: list[str] = []

    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if record_id is not None:
                    yield FastaRecord(record_id, description, clean_sequence("".join(chunks)))
                description = line[1:].strip()
                record_id = description.split()[0]
                chunks = []
            else:
                chunks.append(line)

    if record_id is not None:
        yield FastaRecord(record_id, description, clean_sequence("".join(chunks)))


def clean_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def is_valid_sequence(sequence: str) -> bool:
    return bool(sequence) and all(char in VALID_RESIDUES for char in sequence)

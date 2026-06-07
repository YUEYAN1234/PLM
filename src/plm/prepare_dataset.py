from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value

from .data import encoded_examples
from .tokenizer import ProteinTokenizer


DATASET_FEATURES = Features(
    {
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "sequence_id": Value("string"),
        "chunk_index": Value("int32"),
        "length": Value("int32"),
    }
)


def empty_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [],
            "attention_mask": [],
            "sequence_id": [],
            "chunk_index": [],
            "length": [],
        },
        features=DATASET_FEATURES,
    )


def split_fingerprint(
    *,
    fasta_path: Path,
    split: str,
    max_length: int,
    min_length: int,
    validation_fraction: float,
) -> str:
    stat = fasta_path.stat()
    payload = "|".join(
        [
            str(fasta_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            split,
            str(max_length),
            str(min_length),
            str(validation_fraction),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_split(
    *,
    fasta_path: Path,
    split: str,
    tokenizer: ProteinTokenizer,
    max_length: int,
    min_length: int,
    validation_fraction: float,
    cache_dir: Path | None = None,
) -> Dataset:
    try:
        return Dataset.from_generator(
            encoded_examples,
            gen_kwargs={
                "fasta_path": str(fasta_path),
                "split": split,
                "tokenizer": tokenizer,
                "max_length": max_length,
                "min_length": min_length,
                "validation_fraction": validation_fraction,
            },
            features=DATASET_FEATURES,
            cache_dir=str(cache_dir) if cache_dir else None,
            fingerprint=split_fingerprint(
                fasta_path=fasta_path,
                split=split,
                max_length=max_length,
                min_length=min_length,
                validation_fraction=validation_fraction,
            ),
        )
    except ValueError as exc:
        if "corresponds to no data" in str(exc):
            return empty_dataset()
        raise


def prepare_dataset(
    *,
    input_path: Path,
    out_path: Path,
    max_length: int = 512,
    min_length: int = 30,
    validation_fraction: float = 0.001,
) -> DatasetDict:
    tokenizer = ProteinTokenizer()
    cache_root = Path("data/processed/.generator_cache")
    train = build_split(
        fasta_path=input_path,
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min_length,
        validation_fraction=validation_fraction,
        cache_dir=cache_root / "train",
    )
    validation = build_split(
        fasta_path=input_path,
        split="validation",
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min_length,
        validation_fraction=validation_fraction,
        cache_dir=cache_root / "validation",
    )
    if len(train) == 0:
        raise ValueError("Prepared train split is empty.")
    if len(validation) == 0:
        raise ValueError("Prepared validation split is empty. Increase --validation-fraction for small datasets.")

    dataset = DatasetDict({"train": train, "validation": validation})
    out_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_path))
    tokenizer.save(out_path)
    metadata = {
        "input": str(input_path),
        "max_length": max_length,
        "payload_length": max_length - 2,
        "min_length": min_length,
        "validation_fraction": validation_fraction,
        "train_rows": len(train),
        "validation_rows": len(validation),
    }
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare FASTA data for protein MLM training.")
    parser.add_argument("--input", type=Path, required=True, help="Input FASTA or FASTA.GZ path.")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.001)
    args = parser.parse_args(argv)

    prepare_dataset(
        input_path=args.input,
        out_path=args.out,
        max_length=args.max_length,
        min_length=args.min_length,
        validation_fraction=args.validation_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

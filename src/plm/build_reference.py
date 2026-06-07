from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .embedding import embed_records
from .reference import sample_reference_records, save_reference_index


def build_reference(
    *,
    checkpoint: Path,
    fasta_path: Path,
    out_dir: Path,
    max_records: int = 100000,
    min_length: int = 30,
    batch_size: int = 16,
    max_length: int | None = None,
) -> None:
    records = sample_reference_records(fasta_path, max_records=max_records, min_length=min_length)
    if not records:
        raise ValueError("No valid reference records were selected.")
    result = embed_records(
        records=records,
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_length=max_length,
        normalize=True,
        desc="embedding reference",
    )
    save_reference_index(
        out_dir=out_dir,
        embeddings=result.embeddings,
        records=records,
        source_fasta=fasta_path,
        checkpoint=checkpoint,
        max_records=max_records,
        min_length=min_length,
    )
    print(f"Wrote reference index: {out_dir / 'index.pt'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a protein embedding reference index from FASTA.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/small_esm_mlm/best"))
    parser.add_argument("--fasta", type=Path, default=Path("data/raw/uniref50.fasta.gz"))
    parser.add_argument("--out", type=Path, default=Path("data/apps/uniref50_100k"))
    parser.add_argument("--max-records", type=int, default=100000)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args(argv)

    build_reference(
        checkpoint=args.checkpoint,
        fasta_path=args.fasta,
        out_dir=args.out,
        max_records=args.max_records,
        min_length=args.min_length,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

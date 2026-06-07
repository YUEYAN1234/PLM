from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .embedding import embed_fasta_file


def embed_fasta(
    *,
    checkpoint: Path,
    fasta_path: Path,
    out_path: Path,
    batch_size: int = 8,
    max_length: int | None = None,
) -> None:
    result = embed_fasta_file(
        checkpoint=checkpoint,
        fasta_path=fasta_path,
        batch_size=batch_size,
        max_length=max_length,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ids": result.ids,
            "descriptions": result.descriptions,
            "lengths": result.lengths,
            "embeddings": result.embeddings,
            "checkpoint": str(checkpoint),
        },
        out_path,
    )
    print(f"Wrote embeddings: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export mean-pooled sequence embeddings from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args(argv)

    embed_fasta(
        checkpoint=args.checkpoint,
        fasta_path=args.fasta,
        out_path=args.out,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

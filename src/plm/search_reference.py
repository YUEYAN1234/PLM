from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .embedding import embed_fasta_file
from .search import chunked_topk_cosine, load_torch_file, write_search_results


def search_reference(
    *,
    checkpoint: Path,
    index_path: Path,
    query_fasta: Path,
    out_path: Path,
    top_k: int = 10,
    batch_size: int = 16,
    max_length: int | None = None,
    reference_chunk_size: int = 50000,
) -> None:
    index = load_torch_file(index_path)
    query = embed_fasta_file(
        checkpoint=checkpoint,
        fasta_path=query_fasta,
        batch_size=batch_size,
        max_length=max_length,
        normalize=True,
    )
    if not query.ids:
        raise ValueError("No query sequences were embedded.")
    reference_embeddings = index["embeddings"]
    if not isinstance(reference_embeddings, torch.Tensor):
        raise TypeError("index.pt does not contain tensor embeddings")
    scores, indices = chunked_topk_cosine(
        query.embeddings,
        reference_embeddings,
        top_k=top_k,
        reference_chunk_size=reference_chunk_size,
    )
    write_search_results(query_ids=query.ids, scores=scores, indices=indices, index=index, out_path=out_path)
    print(f"Wrote search results: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search query proteins against a reference embedding index.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/small_esm_mlm/best"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--reference-chunk-size", type=int, default=50000)
    args = parser.parse_args(argv)

    search_reference(
        checkpoint=args.checkpoint,
        index_path=args.index,
        query_fasta=args.query,
        out_path=args.out,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
        reference_chunk_size=args.reference_chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

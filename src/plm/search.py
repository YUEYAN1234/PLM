from __future__ import annotations

import csv
import inspect
from pathlib import Path

import torch

from .embedding import normalize_embeddings


def load_torch_file(path: Path) -> dict[str, object]:
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def chunked_topk_cosine(
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    *,
    top_k: int = 10,
    reference_chunk_size: int = 50000,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if reference_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("query_embeddings and reference_embeddings must be rank-2 tensors")
    if reference_embeddings.shape[0] == 0:
        raise ValueError("reference index is empty")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query = normalize_embeddings(query_embeddings).to(device)
    reference = normalize_embeddings(reference_embeddings)
    k = min(top_k, reference.shape[0])

    best_scores: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None
    for start in range(0, reference.shape[0], reference_chunk_size):
        chunk = reference[start : start + reference_chunk_size].to(device)
        scores = query @ chunk.T
        chunk_k = min(k, scores.shape[1])
        values, indices = torch.topk(scores, k=chunk_k, dim=1)
        indices = indices + start

        if best_scores is None or best_indices is None:
            candidate_scores = values
            candidate_indices = indices
        else:
            candidate_scores = torch.cat([best_scores.to(device), values], dim=1)
            candidate_indices = torch.cat([best_indices.to(device), indices], dim=1)

        best_scores, selected = torch.topk(candidate_scores, k=min(k, candidate_scores.shape[1]), dim=1)
        best_indices = candidate_indices.gather(dim=1, index=selected)

    return best_scores.cpu(), best_indices.cpu()


def write_search_results(
    *,
    query_ids: list[str],
    scores: torch.Tensor,
    indices: torch.Tensor,
    index: dict[str, object],
    out_path: Path,
) -> None:
    ids = list(index["ids"])
    metadata = list(index.get("metadata", []))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "rank",
        "hit_id",
        "cosine_similarity",
        "hit_description",
        "hit_tax",
        "hit_tax_id",
        "hit_rep_id",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for query_row, query_id in enumerate(query_ids):
            for rank, ref_index in enumerate(indices[query_row].tolist(), start=1):
                row = metadata[ref_index] if ref_index < len(metadata) else {}
                writer.writerow(
                    {
                        "query_id": query_id,
                        "rank": rank,
                        "hit_id": ids[ref_index],
                        "cosine_similarity": f"{float(scores[query_row, rank - 1]):.6f}",
                        "hit_description": row.get("description", ""),
                        "hit_tax": row.get("tax", ""),
                        "hit_tax_id": row.get("tax_id", ""),
                        "hit_rep_id": row.get("rep_id", ""),
                    }
                )

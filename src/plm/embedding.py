from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import sequence_chunks
from .fasta import FastaRecord, parse_fasta
from .tokenizer import ProteinTokenizer, load_tokenizer


@dataclass(frozen=True)
class EmbeddingResult:
    ids: list[str]
    descriptions: list[str]
    lengths: list[int]
    embeddings: torch.Tensor
    checkpoint: str


def mean_pool(last_hidden_state: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    special_ids = torch.tensor(sorted(tokenizer.special_token_ids), device=input_ids.device)
    special_mask = torch.isin(input_ids, special_ids)
    token_mask = attention_mask.bool() & ~special_mask
    token_mask = token_mask.unsqueeze(-1)
    summed = (last_hidden_state * token_mask).sum(dim=1)
    counts = token_mask.sum(dim=1).clamp(min=1)
    return summed / counts


def load_embedding_model(checkpoint: Path):
    from transformers import EsmForMaskedLM

    tokenizer = load_tokenizer(checkpoint)
    model = EsmForMaskedLM.from_pretrained(str(checkpoint))
    model.eval()
    return tokenizer, model


def infer_model_max_length(model, tokenizer: ProteinTokenizer, max_length: int | None = None) -> int:
    if max_length is not None:
        return max_length
    return int(getattr(model.config, "max_input_length", model.config.max_position_embeddings - tokenizer.pad_token_id - 1))


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.numel() == 0:
        return embeddings
    return F.normalize(embeddings.float(), dim=1)


def embed_records(
    *,
    records: Sequence[FastaRecord],
    checkpoint: Path,
    batch_size: int = 8,
    max_length: int | None = None,
    normalize: bool = False,
    desc: str = "embedding",
) -> EmbeddingResult:
    tokenizer, model = load_embedding_model(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_max_length = infer_model_max_length(model, tokenizer, max_length)
    payload_length = model_max_length - 2

    ids: list[str] = []
    descriptions: list[str] = []
    lengths: list[int] = []
    embeddings: list[torch.Tensor] = []

    with torch.no_grad():
        for record in tqdm(records, desc=desc):
            chunk_embeddings: list[torch.Tensor] = []
            chunks = list(sequence_chunks(record.sequence, payload_length, min_length=1))
            for start in range(0, len(chunks), batch_size):
                batch_chunks = chunks[start : start + batch_size]
                encoded = [
                    tokenizer.encode(chunk, add_special_tokens=True, max_length=model_max_length)
                    for _, chunk in batch_chunks
                ]
                max_batch_length = max(len(item) for item in encoded)
                input_ids = torch.full(
                    (len(encoded), max_batch_length),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros((len(encoded), max_batch_length), dtype=torch.long, device=device)
                for row, item in enumerate(encoded):
                    values = torch.tensor(item, dtype=torch.long, device=device)
                    input_ids[row, : values.numel()] = values
                    attention_mask[row, : values.numel()] = 1
                output = model.esm(input_ids=input_ids, attention_mask=attention_mask)
                pooled = mean_pool(output.last_hidden_state, input_ids, attention_mask, tokenizer)
                chunk_embeddings.extend(pooled.detach().cpu())
            if chunk_embeddings:
                ids.append(record.record_id)
                descriptions.append(record.description)
                lengths.append(len(record.sequence))
                embeddings.append(torch.stack(chunk_embeddings).mean(dim=0))

    hidden_size = int(model.config.hidden_size)
    tensor = torch.stack(embeddings) if embeddings else torch.empty(0, hidden_size)
    if normalize:
        tensor = normalize_embeddings(tensor)
    return EmbeddingResult(
        ids=ids,
        descriptions=descriptions,
        lengths=lengths,
        embeddings=tensor,
        checkpoint=str(checkpoint),
    )


def embed_fasta_file(
    *,
    checkpoint: Path,
    fasta_path: Path,
    batch_size: int = 8,
    max_length: int | None = None,
    normalize: bool = False,
) -> EmbeddingResult:
    return embed_records(
        records=list(parse_fasta(fasta_path)),
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize,
    )

from __future__ import annotations

import csv
from pathlib import Path

import torch

from plm.reference import parse_uniref_header, sample_reference_records
from plm.search import chunked_topk_cosine, write_search_results


def test_parse_uniref_header() -> None:
    header = "UniRef50_UPI002E2621C6 uncharacterized protein LOC134193701 n=1 Tax=Corticium candelabrum TaxID=121492 RepID=UPI002E2621C6"
    parsed = parse_uniref_header(header)
    assert parsed.record_id == "UniRef50_UPI002E2621C6"
    assert parsed.description == "uncharacterized protein LOC134193701"
    assert parsed.tax == "Corticium candelabrum"
    assert parsed.tax_id == "121492"
    assert parsed.rep_id == "UPI002E2621C6"


def test_sample_reference_records_is_stable(tmp_path: Path) -> None:
    fasta = tmp_path / "reference.fasta"
    fasta.write_text(
        "\n".join(
            [
                ">record_a protein A",
                "A" * 40,
                ">record_b protein B",
                "C" * 40,
                ">record_c protein C",
                "D" * 40,
                ">record_bad invalid",
                "ACD*EFG",
            ]
        ),
        encoding="utf-8",
    )

    first = sample_reference_records(fasta, max_records=2, min_length=30)
    second = sample_reference_records(fasta, max_records=2, min_length=30)

    assert [record.record_id for record in first] == [record.record_id for record in second]
    assert len(first) == 2
    assert all(record.record_id != "record_bad" for record in first)


def test_chunked_topk_cosine_returns_sorted_hits() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reference = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])

    scores, indices = chunked_topk_cosine(query, reference, top_k=2, reference_chunk_size=1, device=torch.device("cpu"))

    assert indices.tolist() == [[0, 1], [2, 1]]
    assert scores[0, 0] >= scores[0, 1]
    assert scores[1, 0] >= scores[1, 1]


def test_write_search_results(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    index = {
        "ids": ["hit_a"],
        "metadata": [
            {
                "description": "protein A",
                "tax": "Example taxon",
                "tax_id": "123",
                "rep_id": "REP_A",
            }
        ],
    }
    write_search_results(
        query_ids=["query_a"],
        scores=torch.tensor([[0.9]]),
        indices=torch.tensor([[0]]),
        index=index,
        out_path=out,
    )

    rows = list(csv.DictReader(out.open("r", encoding="utf-8")))
    assert rows == [
        {
            "query_id": "query_a",
            "rank": "1",
            "hit_id": "hit_a",
            "cosine_similarity": "0.900000",
            "hit_description": "protein A",
            "hit_tax": "Example taxon",
            "hit_tax_id": "123",
            "hit_rep_id": "REP_A",
        }
    ]

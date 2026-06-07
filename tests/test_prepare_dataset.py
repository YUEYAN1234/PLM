import gzip
from pathlib import Path

from datasets import load_from_disk

from plm.prepare_dataset import prepare_dataset


def write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_prepare_dataset_filters_and_chunks(tmp_path: Path) -> None:
    fasta = tmp_path / "toy.fasta.gz"
    write_gzip(
        fasta,
        "\n".join(
            [
                ">train_long example",
                "A" * 70,
                ">invalid",
                "ACD*EFG",
                ">short",
                "ACD",
                ">valid_small",
                "C" * 35,
            ]
        ),
    )

    out = tmp_path / "dataset"
    prepare_dataset(
        input_path=fasta,
        out_path=out,
        max_length=32,
        min_length=10,
        validation_fraction=0.5,
    )
    dataset = load_from_disk(str(out))
    assert set(dataset.keys()) == {"train", "validation"}
    assert len(dataset["train"]) + len(dataset["validation"]) >= 3
    for split in dataset:
        for row in dataset[split]:
            assert len(row["input_ids"]) <= 32
            assert row["length"] >= 10

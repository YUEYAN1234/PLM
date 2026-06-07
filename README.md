# Small ESM-like Protein MLM

This project trains a small ESM-like masked language model on protein FASTA data.
It is designed for `D:\PLM`, a Windows workstation, and an 8 GB NVIDIA GPU.

## 1. Environment

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install CUDA PyTorch first. The current machine previously had CPU-only PyTorch,
so replace it before training on GPU:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .[dev]
```

Check CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 2. Download UniRef50

```powershell
python -m plm.download_uniref --dataset uniref50
```

This downloads:

```text
data/raw/uniref50.fasta.gz
data/raw/uniref50.fasta.gz.md5
```

The downloader verifies the MD5 checksum unless `--skip-md5` is passed.

## 3. Prepare Dataset

```powershell
python -m plm.prepare_dataset --input data/raw/uniref50.fasta.gz --out data/processed/uniref50_len512
```

Defaults:

- `max_length=512`
- payload length is `510` residues because `<cls>` and `<eos>` are added
- sequences shorter than `30` residues are skipped
- invalid amino acid characters are skipped
- long sequences are split into non-overlapping chunks
- validation split uses a stable hash of the UniRef entry id

## 4. Train

```powershell
python -m plm.train_mlm --config configs/small_esm_mlm.yaml
```

Resume from a checkpoint:

```powershell
python -m plm.train_mlm --config configs/small_esm_mlm.yaml --resume-from-checkpoint runs/small_esm_mlm/checkpoint-5000
```

The final or best model is saved to:

```text
runs/small_esm_mlm/best
```

## 5. Export Embeddings

```powershell
python -m plm.embed --checkpoint runs/small_esm_mlm/best --fasta input.fasta --out embeddings.pt
```

The output is a PyTorch file with:

- `ids`: FASTA record ids
- `embeddings`: tensor shaped `[num_sequences, hidden_size]`
- `checkpoint`: checkpoint path

## 6. Downstream Protein Search

Build a UniRef50 subset reference index with the trained model:

```powershell
python -m plm.build_reference --checkpoint runs/small_esm_mlm/best --fasta data/raw/uniref50.fasta.gz --out data/apps/uniref50_100k --max-records 100000
```

This creates:

```text
data/apps/uniref50_100k/index.pt
data/apps/uniref50_100k/metadata.csv
data/apps/uniref50_100k/build_summary.json
```

Search query proteins against the reference index:

```powershell
python -m plm.search_reference --checkpoint runs/small_esm_mlm/best --index data/apps/uniref50_100k/index.pt --query query.fasta --top-k 10 --out results.csv
```

The result CSV contains:

```text
query_id,rank,hit_id,cosine_similarity,hit_description,hit_tax,hit_tax_id,hit_rep_id
```

The UniRef50 index is useful for candidate annotation and nearest-neighbor exploration. It is not a reviewed functional classifier.

## 7. Smoke Test

```powershell
python -m pytest
python -m plm.prepare_dataset --input tests/fixtures/toy.fasta.gz --out data/processed/toy_len128 --max-length 128 --validation-fraction 0.25
python -m plm.train_mlm --config configs/small_esm_mlm.yaml --dataset-path data/processed/toy_len128 --output-dir runs/smoke --max-steps 5
python -m plm.embed --checkpoint runs/smoke/best --fasta tests/fixtures/toy.fasta.gz --out runs/smoke/embeddings.pt
python -m plm.build_reference --checkpoint runs/smoke/best --fasta tests/fixtures/toy.fasta.gz --out data/apps/toy_reference --max-records 3 --min-length 1
python -m plm.search_reference --checkpoint runs/smoke/best --index data/apps/toy_reference/index.pt --query tests/fixtures/toy.fasta.gz --top-k 2 --out runs/smoke/search_results.csv
```

## Notes

Full UniRef50 training is intentionally configured as one epoch. On an 8 GB GPU it
will take a long time; checkpointing every 5000 steps lets you resume safely.

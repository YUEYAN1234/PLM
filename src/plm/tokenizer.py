from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<unk>", "<cls>", "<eos>", "<mask>"]
RESIDUE_TOKENS = list("ACDEFGHIKLMNPQRSTVWYXBZUO")
VOCAB_FILENAME = "protein_vocab.json"


class ProteinTokenizer:
    """Character-level tokenizer for protein sequences."""

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        if vocab is None:
            tokens = SPECIAL_TOKENS + RESIDUE_TOKENS
            vocab = {token: idx for idx, token in enumerate(tokens)}
        self.token_to_id = dict(vocab)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.cls_token = "<cls>"
        self.eos_token = "<eos>"
        self.mask_token = "<mask>"

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.cls_token_id = self.token_to_id[self.cls_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.mask_token_id = self.token_to_id[self.mask_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def residue_token_ids(self) -> list[int]:
        return [self.token_to_id[token] for token in RESIDUE_TOKENS]

    @property
    def special_token_ids(self) -> set[int]:
        return {
            self.pad_token_id,
            self.unk_token_id,
            self.cls_token_id,
            self.eos_token_id,
            self.mask_token_id,
        }

    def encode(
        self,
        sequence: str,
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation: bool = False,
    ) -> list[int]:
        sequence = sequence.strip().upper()
        ids = [self.token_to_id.get(char, self.unk_token_id) for char in sequence]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.eos_token_id]
        if max_length is not None and len(ids) > max_length:
            if not truncation:
                raise ValueError(f"Encoded sequence length {len(ids)} exceeds max_length={max_length}")
            ids = ids[:max_length]
            if add_special_tokens:
                ids[-1] = self.eos_token_id
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for token_id in ids:
            token = self.id_to_token.get(int(token_id), self.unk_token)
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix:
            vocab_path = path
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            vocab_path = path / VOCAB_FILENAME
        vocab_path.write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")
        return vocab_path

    def save_pretrained(self, save_directory: str | Path, **_: object) -> tuple[str]:
        vocab_path = self.save(save_directory)
        return (str(vocab_path),)

    @classmethod
    def load(cls, path: str | Path) -> "ProteinTokenizer":
        path = Path(path)
        vocab_path = path / VOCAB_FILENAME if path.is_dir() else path
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        return cls({str(token): int(idx) for token, idx in vocab.items()})


def load_tokenizer(path: str | Path | None = None) -> ProteinTokenizer:
    if path is None:
        return ProteinTokenizer()
    path = Path(path)
    if path.is_dir() and not (path / VOCAB_FILENAME).exists():
        return ProteinTokenizer()
    if not path.exists():
        return ProteinTokenizer()
    return ProteinTokenizer.load(path)

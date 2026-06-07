from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .tokenizer import ProteinTokenizer


@dataclass
class ProteinMLMCollator:
    tokenizer: ProteinTokenizer
    mlm_probability: float = 0.15
    pad_to_multiple_of: int | None = None

    def __call__(self, examples: Sequence[dict[str, object]]) -> dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        input_ids = torch.full(
            (len(examples), max_length),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)

        for row, example in enumerate(examples):
            ids = torch.tensor(example["input_ids"], dtype=torch.long)
            length = ids.numel()
            input_ids[row, :length] = ids
            attention_mask[row, :length] = 1

        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_ids = torch.tensor(sorted(self.tokenizer.special_token_ids), dtype=torch.long)
        special_mask = torch.isin(input_ids, special_ids) | attention_mask.eq(0)
        probability_matrix.masked_fill_(special_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        replace_probability = torch.full(labels.shape, 0.8)
        indices_replaced = torch.bernoulli(replace_probability).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        random_probability = torch.full(labels.shape, 0.5)
        indices_random = torch.bernoulli(random_probability).bool() & masked_indices & ~indices_replaced
        residue_ids = torch.tensor(self.tokenizer.residue_token_ids, dtype=torch.long)
        random_indices = torch.randint(low=0, high=len(residue_ids), size=labels.shape)
        random_tokens = residue_ids[random_indices]
        input_ids[indices_random] = random_tokens[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

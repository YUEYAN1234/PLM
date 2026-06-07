import torch

from plm.mlm_collator import ProteinMLMCollator
from plm.tokenizer import ProteinTokenizer


def test_collator_masks_only_residue_tokens() -> None:
    torch.manual_seed(1)
    tokenizer = ProteinTokenizer()
    collator = ProteinMLMCollator(tokenizer=tokenizer, mlm_probability=1.0)
    batch = collator(
        [
            {"input_ids": tokenizer.encode("ACDE"), "attention_mask": [1] * 6},
            {"input_ids": tokenizer.encode("FG"), "attention_mask": [1] * 4},
        ]
    )

    labels = batch["labels"]
    input_ids = batch["input_ids"]
    assert labels[0, 0].item() == -100
    assert labels[0, 5].item() == -100
    assert labels[1, 0].item() == -100
    assert labels[1, 3].item() == -100
    assert labels[1, 4].item() == -100
    assert labels[0, 1:5].ne(-100).all()
    assert input_ids.shape == labels.shape

import pytest

from plm.mlm_collator import ProteinMLMCollator
from plm.modeling import create_mlm_model
from plm.tokenizer import ProteinTokenizer


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_small_esm_forward_loss() -> None:
    pytest.importorskip("transformers")
    tokenizer = ProteinTokenizer()
    model = create_mlm_model(
        tokenizer,
        {
            "hidden_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 96,
            "max_position_embeddings": 64,
        },
    )
    collator = ProteinMLMCollator(tokenizer=tokenizer, mlm_probability=1.0)
    batch = collator([{"input_ids": tokenizer.encode("ACDEFGHIK"), "attention_mask": [1] * 11}])
    output = model(**batch)
    assert output.loss is not None
    assert output.loss.item() > 0


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_esm_forward_accepts_configured_max_length() -> None:
    pytest.importorskip("transformers")
    tokenizer = ProteinTokenizer()
    model = create_mlm_model(
        tokenizer,
        {
            "hidden_size": 48,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 96,
            "max_position_embeddings": 512,
        },
    )
    ids = tokenizer.encode("A" * 510)
    assert len(ids) == 512
    batch = ProteinMLMCollator(tokenizer=tokenizer, mlm_probability=1.0)(
        [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    )
    output = model(**batch)
    assert output.loss is not None

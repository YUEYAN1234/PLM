from __future__ import annotations

from .tokenizer import ProteinTokenizer


def build_esm_config(tokenizer: ProteinTokenizer, model_config: dict[str, object]):
    from transformers import EsmConfig

    max_input_length = int(model_config.get("max_position_embeddings", 512))
    # HF ESM position ids start at padding_idx + 1 for non-padding tokens.
    # With pad_token_id=0, a 512-token input needs position rows 0..512.
    position_embedding_rows = max_input_length + tokenizer.pad_token_id + 1

    config = EsmConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(model_config.get("hidden_size", 384)),
        num_hidden_layers=int(model_config.get("num_hidden_layers", 6)),
        num_attention_heads=int(model_config.get("num_attention_heads", 6)),
        intermediate_size=int(model_config.get("intermediate_size", 1536)),
        hidden_dropout_prob=float(model_config.get("hidden_dropout_prob", 0.1)),
        attention_probs_dropout_prob=float(model_config.get("attention_probs_dropout_prob", 0.1)),
        max_position_embeddings=position_embedding_rows,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    config.max_input_length = max_input_length
    return config


def create_mlm_model(tokenizer: ProteinTokenizer, model_config: dict[str, object]):
    from transformers import EsmForMaskedLM

    config = build_esm_config(tokenizer, model_config)
    return EsmForMaskedLM(config)

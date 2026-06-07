from plm.tokenizer import ProteinTokenizer


def test_tokenizer_roundtrip_and_unknown() -> None:
    tokenizer = ProteinTokenizer()
    ids = tokenizer.encode("ACD?")
    assert ids[0] == tokenizer.cls_token_id
    assert ids[-1] == tokenizer.eos_token_id
    assert tokenizer.unk_token_id in ids
    assert tokenizer.decode(ids) == "ACD"


def test_tokenizer_truncates_with_eos() -> None:
    tokenizer = ProteinTokenizer()
    ids = tokenizer.encode("ACDEFG", max_length=5, truncation=True)
    assert len(ids) == 5
    assert ids[-1] == tokenizer.eos_token_id

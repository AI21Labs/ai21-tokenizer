import json
from pathlib import Path

import pytest

from jurassic_tokenization import JurassicTokenizer


def test_tokenizer_encode_decode(tokenizer: JurassicTokenizer):
    text = "Hello world!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_tokenizer_encode_set(tokenizer: JurassicTokenizer, resources_path: Path):
    tokenized_docs_path = resources_path / "200_tokenized_C4_val_docs.jsonl"
    with tokenized_docs_path.open("r") as tokenized_docs_file:
        for i, tokenized_doc_line in enumerate(tokenized_docs_file.readlines()):
            tokenized_doc = json.loads(tokenized_doc_line)

            assert tokenized_doc["token_ids_start_true"] == tokenizer.encode(
                tokenized_doc["doc_text"]
            ), f"Not equal at doc {i}"


def test_tokenizer_create__when_receives_unknown_name__should_raise():
    with pytest.raises(ValueError):
        JurassicTokenizer.from_pretrained("unknown_name")


def test_tokenizer__convert_tokens_to_ids(tokenizer: JurassicTokenizer):
    expected_tokens = ["▁hello"]
    ids = tokenizer.encode("hello")
    actual_tokens = tokenizer.convert_ids_to_tokens(ids)

    assert actual_tokens == expected_tokens


def test_tokenizer__convert_ids_to_tokens(tokenizer: JurassicTokenizer):
    expected_ids = [30671]
    ids = tokenizer.encode("hello")

    tokens = tokenizer.convert_ids_to_tokens(ids)
    actual_ids = tokenizer.convert_tokens_to_ids(tokens)

    assert actual_ids == expected_ids

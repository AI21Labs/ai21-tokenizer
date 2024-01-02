import json
from pathlib import Path
from typing import Union, List

import pytest

from ai21_tokenizer.jurassic_tokenizer import JurassicTokenizer


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


def test_tokenizer_encode_set_when_is_start_false(tokenizer: JurassicTokenizer, resources_path: Path):
    tokenized_docs_path = resources_path / "200_tokenized_C4_val_docs.jsonl"
    with tokenized_docs_path.open("r") as tokenized_docs_file:
        for i, tokenized_doc_line in enumerate(tokenized_docs_file.readlines()):
            tokenized_doc = json.loads(tokenized_doc_line)

            assert tokenized_doc["token_ids_start_false"] == tokenizer.encode(
                tokenized_doc["doc_text"], is_start=False
            ), f"Not equal at doc {i}"


def test_tokenizer_decode_set_with_offsets(tokenizer: JurassicTokenizer, resources_path: Path):
    tokenized_docs_path = resources_path / "200_tokenized_C4_val_docs.jsonl"
    with tokenized_docs_path.open("r") as tokenized_docs_file:
        for i, tokenized_doc_line in enumerate(tokenized_docs_file.readlines()):
            tokenized_doc = json.loads(tokenized_doc_line)

            tokens = tokenized_doc["token_ids_start_true"]
            decoded_text, offsets = tokenizer.decode_with_offsets(tokens)
            assert tokenized_doc["decoded_text_from_start_true"] == decoded_text, f"Not equal at doc {i}"
            assert [
                tuple(x) for x in tokenized_doc["decoded_offsets_from_start_true"]
            ] == offsets, f"Not equal at doc {i}"


@pytest.mark.parametrize(
    ids=[
        "when_single_int__should_return_single_str",
        "when_list_of_int__should_return_list_of_str",
    ],
    argnames=["ids", "expected_tokens"],
    argvalues=[
        (30671, "▁hello"),
        ([7463, 1754], ["▁Hello", "▁world"]),
    ],
)
def test_tokenizer__convert_ids_to_tokens(
    ids: Union[int, List[int]], expected_tokens: Union[str, List[str]], tokenizer: JurassicTokenizer
):
    actual_tokens = tokenizer.convert_ids_to_tokens(ids)

    assert actual_tokens == expected_tokens


@pytest.mark.parametrize(
    ids=[
        "when_single_str__should_return_single_int",
        "when_list_of_str__should_return_list_of_ints",
    ],
    argnames=["tokens", "expected_ids"],
    argvalues=[
        ("▁hello", 30671),
        (["▁Hello", "▁world"], [7463, 1754]),
    ],
)
def test_tokenizer__convert_tokens_to_ids(
    tokens: Union[str, List[str]], expected_ids: Union[int, List[int]], tokenizer: JurassicTokenizer
):
    actual_ids = tokenizer.convert_tokens_to_ids(tokens)

    assert actual_ids == expected_ids


@pytest.mark.parametrize(
    ids=[
        "when_start_of_line__should_return_no_leading_whitespace",
        "when_not_start_of_line__should_return_leading_whitespace",
    ],
    argnames=["tokens", "start_of_line", "expected_text"],
    argvalues=[
        ([30671], True, "hello"),
        ([30671], False, " hello"),
    ],
)
def test_tokenizer__decode_with_start_of_line(
    tokens: List[int], start_of_line: bool, expected_text: str, tokenizer: JurassicTokenizer
):
    actual_text = tokenizer.decode(tokens, start_of_line=start_of_line)

    assert actual_text == expected_text

from pathlib import Path
from typing import List, Union
from unittest.mock import patch

import pytest
from ai21_tokenizer import JambaInstructTokenizer
from ai21_tokenizer.tokenizer_factory import JAMABA_TOKENIZER_HF_PATH


def test_tokenizer_encode_decode(jamba_instruct_tokenizer: JambaInstructTokenizer):
    text = "Hello world!"
    encoded = jamba_instruct_tokenizer.encode(text)
    decoded = jamba_instruct_tokenizer.decode(encoded)

    assert decoded == text


@pytest.mark.parametrize(
    ids=[
        "when_single_int__should_return_single_str",
        "when_list_of_int__should_return_list_of_str",
    ],
    argnames=["ids", "expected_tokens"],
    argvalues=[
        (27164, "▁hello"),
        ([22560, 2620], ["▁Hello", "▁world"]),
    ],
)
def test_tokenizer__convert_ids_to_tokens(
    ids: Union[int, List[int]], expected_tokens: Union[str, List[str]], jamba_instruct_tokenizer: JambaInstructTokenizer
):
    actual_tokens = jamba_instruct_tokenizer.convert_ids_to_tokens(ids)

    assert actual_tokens == expected_tokens


@pytest.mark.parametrize(
    ids=[
        "when_single_str__should_return_single_int",
        "when_list_of_str__should_return_list_of_ints",
    ],
    argnames=["tokens", "expected_ids"],
    argvalues=[
        ("▁hello", 27164),
        (["▁Hello", "▁world"], [22560, 2620]),
    ],
)
def test_tokenizer__convert_tokens_to_ids(
    tokens: Union[str, List[str]], expected_ids: Union[int, List[int]], jamba_instruct_tokenizer: JambaInstructTokenizer
):
    actual_ids = jamba_instruct_tokenizer.convert_tokens_to_ids(tokens)

    assert actual_ids == expected_ids


@pytest.mark.parametrize(
    ids=[
        "when_skip_special_tokens__should_return_no_leading_whitespace",
        "when_not_skip_special_tokens__should_return_leading_whitespace",
    ],
    argnames=["tokens", "skip_special_tokens", "expected_text"],
    argvalues=[
        ([1, 26928], False, "<|startoftext|>hello"),
        ([1, 26928], True, "hello"),
    ],
)
def test_tokenizer__decode_with_start_of_line(
    tokens: List[int], skip_special_tokens: bool, expected_text: str, jamba_instruct_tokenizer: JambaInstructTokenizer
):
    actual_text = jamba_instruct_tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    print("Hello")
    assert actual_text == expected_text


def test_tokenizer__when_cache_dir_not_exists__should_save_tokenizer_in_cache_dir(tmp_path: Path):
    assert not (tmp_path / "tokenizer.json").exists()
    JambaInstructTokenizer(JAMABA_TOKENIZER_HF_PATH, tmp_path)

    assert (tmp_path / "tokenizer.json").exists()


def test_tokenizer__when_cache_dir_exists__should_load_from_cache(tmp_path: Path):
    # Creating tokenizer once from repo
    assert not (tmp_path / "tokenizer.json").exists()
    JambaInstructTokenizer(JAMABA_TOKENIZER_HF_PATH, tmp_path)

    # Creating tokenizer again to load from cache
    with patch.object(JambaInstructTokenizer, JambaInstructTokenizer._load_from_cache.__name__) as mock_load_from_cache:
        JambaInstructTokenizer(JAMABA_TOKENIZER_HF_PATH, tmp_path)

    # assert load_from_cache was called
    mock_load_from_cache.assert_called_once()

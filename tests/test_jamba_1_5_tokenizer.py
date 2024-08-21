from pathlib import Path
from typing import Union, List
from unittest.mock import patch

import pytest
from pytest_lazyfixture import lazy_fixture

from ai21_tokenizer.jamba_1_5_tokenizer import Jamba1_5Tokenizer, AsyncJamba1_5Tokenizer
from ai21_tokenizer.tokenizer_factory import JAMBA_1_5_MINI_TOKENIZER_HF_PATH, JAMBA_1_5_LARGE_TOKENIZER_HF_PATH


@pytest.mark.parametrize(
    ids=[
        "when_mini",
        "when_large",
    ],
    argnames=["tokenizer"],
    argvalues=[
        (lazy_fixture("jamba_1_5_mini_tokenizer"),),
        (lazy_fixture("jamba_1_5_large_tokenizer"),),
    ],
)
def test_tokenizer_mini_encode_decode(tokenizer: Jamba1_5Tokenizer):
    text = "Hello world!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


@pytest.mark.parametrize(
    ids=[
        "when_mini_single_int__should_return_single_str",
        "when_large_single_int__should_return_single_str",
        "when_mini_list_of_int__should_return_list_of_str",
        "when_large_list_of_int__should_return_list_of_str",
    ],
    argnames=["ids", "expected_tokens", "tokenizer"],
    argvalues=[
        (27164, "▁hello", lazy_fixture("jamba_1_5_mini_tokenizer")),
        (27164, "▁hello", lazy_fixture("jamba_1_5_large_tokenizer")),
        ([22560, 2620], ["▁Hello", "▁world"], lazy_fixture("jamba_1_5_mini_tokenizer")),
        ([22560, 2620], ["▁Hello", "▁world"], lazy_fixture("jamba_1_5_large_tokenizer")),
    ],
)
def test_tokenizer_mini__convert_ids_to_tokens(
    ids: Union[int, List[int]],
    expected_tokens: Union[str, List[str]],
    tokenizer: Jamba1_5Tokenizer,
):
    actual_tokens = tokenizer.convert_ids_to_tokens(ids)

    assert actual_tokens == expected_tokens


@pytest.mark.parametrize(
    ids=[
        "when_mini_single_str__should_return_single_int",
        "when_large_single_str__should_return_single_int",
        "when_mini_list_of_str__should_return_list_of_ints",
        "when_large_list_of_str__should_return_list_of_ints",
    ],
    argnames=["tokens", "expected_ids", "tokenizer"],
    argvalues=[
        ("▁hello", 27164, lazy_fixture("jamba_1_5_mini_tokenizer")),
        ("▁hello", 27164, lazy_fixture("jamba_1_5_large_tokenizer")),
        (["▁Hello", "▁world"], [22560, 2620], lazy_fixture("jamba_1_5_mini_tokenizer")),
        (["▁Hello", "▁world"], [22560, 2620], lazy_fixture("jamba_1_5_large_tokenizer")),
    ],
)
def test_tokenizer__convert_tokens_to_ids(
    tokens: Union[str, List[str]],
    expected_ids: Union[int, List[int]],
    tokenizer: Jamba1_5Tokenizer,
):
    actual_ids = tokenizer.convert_tokens_to_ids(tokens)

    assert actual_ids == expected_ids


@pytest.mark.parametrize(
    ids=[
        "when_mini_skip_special_tokens__should_return_no_leading_whitespace",
        "when_large_skip_special_tokens__should_return_no_leading_whitespace",
        "when_mini_not_skip_special_tokens__should_return_leading_whitespace",
        "when_large_not_skip_special_tokens__should_return_leading_whitespace",
    ],
    argnames=["tokens", "skip_special_tokens", "expected_text", "tokenizer"],
    argvalues=[
        ([1, 26928], False, "<|startoftext|>hello", lazy_fixture("jamba_1_5_mini_tokenizer")),
        ([1, 26928], False, "<|startoftext|>hello", lazy_fixture("jamba_1_5_large_tokenizer")),
        ([1, 26928], True, "hello", lazy_fixture("jamba_1_5_mini_tokenizer")),
        ([1, 26928], True, "hello", lazy_fixture("jamba_1_5_large_tokenizer")),
    ],
)
def test_tokenizer__decode_with_start_of_line(
    tokens: List[int],
    skip_special_tokens: bool,
    expected_text: str,
    tokenizer: Jamba1_5Tokenizer,
):
    actual_text = tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    assert actual_text == expected_text


@pytest.mark.parametrize(
    ids=[
        "when_mini",
        "when_large",
    ],
    argnames=["hf_path"],
    argvalues=[
        (JAMBA_1_5_MINI_TOKENIZER_HF_PATH,),
        (JAMBA_1_5_LARGE_TOKENIZER_HF_PATH,),
    ],
)
def test_tokenizer__when_cache_dir_not_exists__should_save_tokenizer_in_cache_dir(tmp_path: Path, hf_path: str):
    assert not (tmp_path / "tokenizer.json").exists()
    Jamba1_5Tokenizer(hf_path, tmp_path)

    assert (tmp_path / "tokenizer.json").exists()


@pytest.mark.parametrize(
    ids=[
        "when_mini",
        "when_large",
    ],
    argnames=["hf_path"],
    argvalues=[
        (JAMBA_1_5_MINI_TOKENIZER_HF_PATH,),
        (JAMBA_1_5_LARGE_TOKENIZER_HF_PATH,),
    ],
)
def test_tokenizer__when_cache_dir_exists__should_load_from_cache(tmp_path: Path, hf_path: str):
    # Creating tokenizer once from repo
    assert not (tmp_path / "tokenizer.json").exists()
    Jamba1_5Tokenizer(hf_path, tmp_path)

    # Creating tokenizer again to load from cache
    with patch.object(Jamba1_5Tokenizer, Jamba1_5Tokenizer._load_from_cache.__name__) as mock_load_from_cache:
        Jamba1_5Tokenizer(hf_path, tmp_path)

    # assert load_from_cache was called
    mock_load_from_cache.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ids=[
        "when_mini",
        "when_large",
    ],
    argnames=["async_tokenizer"],
    argvalues=[
        (lazy_fixture("async_jamba_1_5_mini_tokenizer"),),
        (lazy_fixture("async_jamba_1_5_large_tokenizer"),),
    ],
)
async def test_async_tokenizer_encode_decode(async_tokenizer: AsyncJamba1_5Tokenizer):
    text = "Hello world!"
    encoded = await async_tokenizer.encode(text)
    decoded = await async_tokenizer.decode(encoded)

    assert decoded == text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ids=[
        "when_mini_single_int__should_return_single_str",
        "when_large_single_int__should_return_single_str",
        "when_mini_list_of_int__should_return_list_of_str",
        "when_large_list_of_int__should_return_list_of_str",
    ],
    argnames=["ids", "expected_tokens", "async_tokenizer"],
    argvalues=[
        (27164, "▁hello", lazy_fixture("async_jamba_1_5_mini_tokenizer")),
        (27164, "▁hello", lazy_fixture("async_jamba_1_5_large_tokenizer")),
        ([22560, 2620], ["▁Hello", "▁world"], lazy_fixture("async_jamba_1_5_mini_tokenizer")),
        ([22560, 2620], ["▁Hello", "▁world"], lazy_fixture("async_jamba_1_5_large_tokenizer")),
    ],
)
async def test_async_tokenizer__convert_ids_to_tokens(
    ids: Union[int, List[int]],
    expected_tokens: Union[str, List[str]],
    async_tokenizer: AsyncJamba1_5Tokenizer,
):
    actual_tokens = await async_tokenizer.convert_ids_to_tokens(ids)

    assert actual_tokens == expected_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ids=[
        "when_mini_single_str__should_return_single_int",
        "when_large_single_str__should_return_single_int",
        "when_mini_list_of_str__should_return_list_of_ints",
        "when_large_list_of_str__should_return_list_of_ints",
    ],
    argnames=["tokens", "expected_ids", "async_tokenizer"],
    argvalues=[
        ("▁hello", 27164, lazy_fixture("async_jamba_1_5_mini_tokenizer")),
        ("▁hello", 27164, lazy_fixture("async_jamba_1_5_large_tokenizer")),
        (["▁Hello", "▁world"], [22560, 2620], lazy_fixture("async_jamba_1_5_mini_tokenizer")),
        (["▁Hello", "▁world"], [22560, 2620], lazy_fixture("async_jamba_1_5_large_tokenizer")),
    ],
)
async def test_async_tokenizer__convert_tokens_to_ids(
    tokens: Union[str, List[str]],
    expected_ids: Union[int, List[int]],
    async_tokenizer: AsyncJamba1_5Tokenizer,
):
    actual_ids = await async_tokenizer.convert_tokens_to_ids(tokens)

    assert actual_ids == expected_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ids=[
        "when_mini_skip_special_tokens__should_return_no_leading_whitespace",
        "when_large_skip_special_tokens__should_return_no_leading_whitespace",
        "when_mini_not_skip_special_tokens__should_return_leading_whitespace",
        "when_large_not_skip_special_tokens__should_return_leading_whitespace",
    ],
    argnames=["tokens", "skip_special_tokens", "expected_text", "async_tokenizer"],
    argvalues=[
        ([1, 26928], False, "<|startoftext|>hello", lazy_fixture("async_jamba_1_5_mini_tokenizer")),
        ([1, 26928], False, "<|startoftext|>hello", lazy_fixture("async_jamba_1_5_large_tokenizer")),
        ([1, 26928], True, "hello", lazy_fixture("async_jamba_1_5_mini_tokenizer")),
        ([1, 26928], True, "hello", lazy_fixture("async_jamba_1_5_large_tokenizer")),
    ],
)
async def test_async_tokenizer__decode_with_start_of_line(
    tokens: List[int],
    skip_special_tokens: bool,
    expected_text: str,
    async_tokenizer: AsyncJamba1_5Tokenizer,
):
    actual_text = await async_tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    assert actual_text == expected_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ids=[
        "when_mini",
        "when_large",
    ],
    argnames=["hf_path"],
    argvalues=[
        (JAMBA_1_5_MINI_TOKENIZER_HF_PATH,),
        (JAMBA_1_5_LARGE_TOKENIZER_HF_PATH,),
    ],
)
async def test_async_tokenizer_encode_caches_tokenizer__should_have_tokenizer_in_cache_dir(
    tmp_path: Path, hf_path: str
):
    assert not (tmp_path / "tokenizer.json").exists()
    jamba_tokenizer = await AsyncJamba1_5Tokenizer.create(hf_path, tmp_path)
    _ = await jamba_tokenizer.encode("Hello world!")
    assert (tmp_path / "tokenizer.json").exists()


def test_async_tokenizer_initialized_directly__should_raise_error():
    with pytest.raises(ValueError):
        AsyncJamba1_5Tokenizer()

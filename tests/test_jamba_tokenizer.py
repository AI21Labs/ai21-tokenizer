from pathlib import Path
from typing import List, Union
from unittest.mock import patch, AsyncMock

import pytest
from ai21_tokenizer import JambaInstructTokenizer, AsyncJambaInstructTokenizer
from ai21_tokenizer.tokenizer_factory import JAMBA_TOKENIZER_HF_PATH


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

    assert actual_text == expected_text


def test_tokenizer__when_cache_dir_not_exists__should_save_tokenizer_in_cache_dir(tmp_path: Path):
    assert not (tmp_path / "tokenizer.json").exists()
    JambaInstructTokenizer(JAMBA_TOKENIZER_HF_PATH, tmp_path)

    assert (tmp_path / "tokenizer.json").exists()


def test_tokenizer__when_cache_dir_exists__should_load_from_cache(tmp_path: Path):
    # Creating tokenizer once from repo
    assert not (tmp_path / "tokenizer.json").exists()
    JambaInstructTokenizer(JAMBA_TOKENIZER_HF_PATH, tmp_path)

    # Creating tokenizer again to load from cache
    with patch.object(JambaInstructTokenizer, JambaInstructTokenizer._load_from_cache.__name__) as mock_load_from_cache:
        JambaInstructTokenizer(JAMBA_TOKENIZER_HF_PATH, tmp_path)

    # assert load_from_cache was called
    mock_load_from_cache.assert_called_once()


@pytest.mark.asyncio
async def test_async_tokenizer_encode_decode(async_jamba_instruct_tokenizer: AsyncJambaInstructTokenizer):
    text = "Hello world!"
    encoded = await async_jamba_instruct_tokenizer.encode(text)
    decoded = await async_jamba_instruct_tokenizer.decode(encoded)

    assert decoded == text


@pytest.mark.asyncio
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
async def test_async_tokenizer__convert_ids_to_tokens(
    ids: Union[int, List[int]],
    expected_tokens: Union[str, List[str]],
    async_jamba_instruct_tokenizer: AsyncJambaInstructTokenizer,
):
    actual_tokens = await async_jamba_instruct_tokenizer.convert_ids_to_tokens(ids)

    assert actual_tokens == expected_tokens


@pytest.mark.asyncio
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
async def test_async_tokenizer__convert_tokens_to_ids(
    tokens: Union[str, List[str]],
    expected_ids: Union[int, List[int]],
    async_jamba_instruct_tokenizer: AsyncJambaInstructTokenizer,
):
    actual_ids = await async_jamba_instruct_tokenizer.convert_tokens_to_ids(tokens)

    assert actual_ids == expected_ids


@pytest.mark.asyncio
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
async def test_async_tokenizer__decode_with_start_of_line(
    tokens: List[int],
    skip_special_tokens: bool,
    expected_text: str,
    async_jamba_instruct_tokenizer: AsyncJambaInstructTokenizer,
):
    actual_text = await async_jamba_instruct_tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    assert actual_text == expected_text


@pytest.mark.asyncio
async def test_async_tokenizer_encode_caches_tokenizer__should_have_tokenizer_in_cache_dir(
    tmp_path: Path,
):
    assert not (tmp_path / "tokenizer.json").exists()
    tokenizer = AsyncJambaInstructTokenizer(JAMBA_TOKENIZER_HF_PATH, tmp_path)
    _ = await tokenizer.encode("Hello world!")
    assert (tmp_path / "tokenizer.json").exists()


@pytest.mark.asyncio
@patch("ai21_tokenizer.jamba_instruct_tokenizer._load_from_cache", new_callable=AsyncMock)
async def test_async_tokenizer_when_cache_dir_exists__should_load_from_cache(
    tmp_path: Path,
    mock_async_jamba_instruct_tokenizer: AsyncJambaInstructTokenizer,
):
    # Creating tokenizer once from repo
    assert not (tmp_path / "tokenizer.json").exists()
    tokenizer = AsyncJambaInstructTokenizer(JAMBA_TOKENIZER_HF_PATH, tmp_path)
    _ = await tokenizer.encode("Hello world!")

    assert (tmp_path / "tokenizer.json").exists()

    tokenizer2 = AsyncJambaInstructTokenizer(JAMBA_TOKENIZER_HF_PATH, tmp_path)
    assert (tmp_path / "tokenizer.json").exists()

    _ = await tokenizer2.encode("Hello world!")

    # Assert that _load_from_cache was called once
    mock_async_jamba_instruct_tokenizer._load_from_cache.assert_called_once()


# @pytest.mark.asyncio
# async def test_async_tokenizer__when_cache_dir_not_exists__should_save_tokenizer_in_cache_dir(tmp_path: Path):
#     assert not (tmp_path / "tokenizer.json").exists()
#     AsyncJambaInstructTokenizer(JAMABA_TOKENIZER_HF_PATH, tmp_path)
#
#     assert (tmp_path / "tokenizer.json").exists()


# @pytest.mark.asyncio
# async def test_async_tokenizer__when_cache_dir_exists__should_load_from_cache(tmp_path: Path):
#     # Creating tokenizer once from repo
#     assert not (tmp_path / "tokenizer.json").exists()
#     AsyncJambaInstructTokenizer(JAMABA_TOKENIZER_HF_PATH, tmp_path)
#
#     # Creating tokenizer again to load from cache
#     with patch.object(
#         AsyncJambaInstructTokenizer, AsyncJambaInstructTokenizer._load_from_cache.__name__
#     ) as mock_load_from_cache:
#         AsyncJambaInstructTokenizer(JAMABA_TOKENIZER_HF_PATH, tmp_path)
#
#     # assert load_from_cache was called
#     mock_load_from_cache.assert_called_once()

import pytest

from tokenizer import TokenizerFactory, PreTrainedTokenizers
from tokenizer.jurassic_tokenizer import JurassicTokenizer


def test_tokenizer_factory__get_tokenizer__when_receives_unknown_name__should_raise():
    with pytest.raises(ValueError):
        TokenizerFactory.get_tokenizer("unknown_name")


def test_tokenizer_factory__get_tokenizer__when_receives_j2_tokenizer__should_return_jurassic_tokenizer():
    tokenizer = TokenizerFactory.get_tokenizer(PreTrainedTokenizers.J2Tokenizer)

    assert tokenizer is not None
    assert isinstance(tokenizer, JurassicTokenizer)

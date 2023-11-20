from tokenizer import Tokenizer
from tokenizer.jurassic_tokenizer import JurassicTokenizer


def test_tokenizer_factory__get_tokenizer___should_return_jurassic_tokenizer():
    tokenizer = Tokenizer.get_tokenizer()

    assert tokenizer is not None
    assert isinstance(tokenizer, JurassicTokenizer)

from pathlib import Path

from jurassic_tokenization import JurassicTokenizer


def test_tokenizer_encode_decode(tokenizer: JurassicTokenizer, resources_path: Path):
    text = "Hello world!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text

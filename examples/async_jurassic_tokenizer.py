import asyncio
from pathlib import Path

from ai21_tokenizer import AsyncJurassicTokenizer
from ai21_tokenizer.utils import load_json

resource_path = Path(__file__).parent.parent / "ai21_tokenizer" / "resources"

model_path = resource_path / "j2-tokenizer/j2-tokenizer.model"
config_path = resource_path / "j2-tokenizer/config.json"
config = load_json(config_path)


async def main():
    tokenizer = AsyncJurassicTokenizer(model_path=model_path, config=config)

    example_sentence = "This sentence should be encoded and then decoded. Hurray!!!!"
    encoded = await tokenizer.encode(example_sentence)
    decoded = await tokenizer.decode(encoded)

    assert decoded == example_sentence
    print("Example sentence: " + example_sentence)
    print("Encoded and decoded: " + decoded)


asyncio.run(main())

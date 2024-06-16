import asyncio

from ai21_tokenizer import Tokenizer, PreTrainedTokenizers


async def main():
    tokenizer = await Tokenizer.get_async_tokenizer(PreTrainedTokenizers.JAMBA_INSTRUCT_TOKENIZER)

    example_sentence = "This sentence should be encoded and then decoded. Hurray!!!!"
    encoded = await tokenizer.encode(example_sentence)
    decoded = await tokenizer.decode(encoded)

    assert decoded == example_sentence
    print("Example sentence: " + example_sentence)
    print("Encoded and decoded: " + decoded)


asyncio.run(main())

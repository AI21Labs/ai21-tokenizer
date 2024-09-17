import asyncio

from ai21_tokenizer import Tokenizer

"""
If you wish to use the tokenizers for `Jamba 1.5 Mini` or `Jamba 1.5 Large`,
you will need to request access to the relevant model's HuggingFace repo:
* https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini
* https://huggingface.co/ai21labs/AI21-Jamba-1.5-Large
"""


async def main():
    tokenizer = await Tokenizer.get_async_tokenizer()
    example_sentence = "This sentence should be encoded and then decoded. Hurray!!"
    encoded = await tokenizer.encode(example_sentence)
    decoded = await tokenizer.decode(encoded)

    assert decoded == example_sentence
    print("Example sentence: " + example_sentence)
    print("Encoded and decoded: " + decoded)


asyncio.run(main())

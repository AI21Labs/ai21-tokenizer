from ai21_tokenizer import Jamba1_5Tokenizer

"""
If you wish to use the tokenizers for `Jamba 1.5 Mini` or `Jamba 1.5 Large`,
you will need to request access to the relevant model's HuggingFace repo:
* https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini
* https://huggingface.co/ai21labs/AI21-Jamba-1.5-Large
"""

model_path = "ai21labs/AI21-Jamba-1.5-Mini"

tokenizer = Jamba1_5Tokenizer(model_path=model_path)

example_sentence = "This sentence should be encoded and then decoded. Hurray!!!!"
encoded = tokenizer.encode(example_sentence)
decoded = tokenizer.decode(encoded)

assert decoded == example_sentence

from ai21_tokenizer import SyncJambaTokenizer


"""
If you wish to use the tokenizers for `Jamba Mini` or `Jamba Large`,
you will need to request access to the relevant model's HuggingFace repo:
* https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.6
* https://huggingface.co/ai21labs/AI21-Jamba-Large-1.6
"""

model_path = "ai21labs/AI21-Jamba-Mini-1.6"

tokenizer = SyncJambaTokenizer(model_path=model_path)

example_sentence = "This sentence should be encoded and then decoded. Hurray!!!!"
encoded = tokenizer.encode(example_sentence)
decoded = tokenizer.decode(encoded)

assert decoded == example_sentence

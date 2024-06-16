from ai21_tokenizer import JambaInstructTokenizer

model_path = "ai21labs/Jamba-v0.1"

tokenizer = JambaInstructTokenizer(model_path=model_path)

example_sentence = "This sentence should be encoded and then decoded. Hurray!!!!"
encoded = tokenizer.encode(example_sentence)
decoded = tokenizer.decode(encoded)

assert decoded == example_sentence

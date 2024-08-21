from ai21_tokenizer import Jamba1_5Tokenizer

model_path = "ai21labs/AI21-Jamba-1.5-Mini"

tokenizer = Jamba1_5Tokenizer(model_path=model_path)

example_sentence = "This sentence should be encoded and then decoded. Hurray!!!!"
encoded = tokenizer.encode(example_sentence)
decoded = tokenizer.decode(encoded)

assert decoded == example_sentence

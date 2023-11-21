from ai21_tokenizer import Tokenizer

tokenizer = Tokenizer.get_tokenizer()
example_sentence = "This sentence should be encoded and then decoded. Hurray!!"
encoded = tokenizer.encode(example_sentence)
decoded = tokenizer.decode(encoded)

assert decoded == example_sentence

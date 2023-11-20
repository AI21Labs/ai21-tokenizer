from tokenizer import TokenizerFactory

tokenizer = TokenizerFactory.get_tokenizer()
example_sentence = "This sentence should be encoded and then decoded. Hurray!"
encoded = tokenizer.encode(example_sentence)
decoded = tokenizer.decode(encoded)

print(decoded == example_sentence)

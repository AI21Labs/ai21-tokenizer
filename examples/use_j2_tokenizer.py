from jurassic_tokenization import JurassicTokenizer

tokenizer = JurassicTokenizer.from_pretrained("j2-tokenizer")
example_sentence = "This sentence should be encoded and then decoded. Hurray!"
encoded = tokenizer.encode(example_sentence)
decoded = tokenizer.decode(encoded)

assert decoded == example_sentence

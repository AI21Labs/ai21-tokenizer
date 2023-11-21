from pathlib import Path

from ai21_tokenizer import JurassicTokenizer, load_json

resource_path = Path(__file__).parent.parent / "ai21_tokenizer" / "resources"

model_path = resource_path / "j2-tokenizer/j2-tokenizer.model"
config_path = resource_path / "j2-tokenizer/config.json"
config = load_json(config_path)
tokenizer = JurassicTokenizer(model_path=model_path, config=config)

example_sentence = "This sentence should be encoded and then decoded. Hurray!!!!"
encoded = tokenizer.encode(example_sentence)
decoded = tokenizer.decode(encoded)

assert decoded == example_sentence

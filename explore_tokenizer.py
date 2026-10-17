"""Inspect GPT-2 tokens, token IDs, decoding, and vocabulary lookups."""

import sys

from transformers import AutoTokenizer


MODEL_NAME = "openai-community/gpt2"
SAMPLE_TEXT = "Hello, world! Transformers learn from text."


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    vocabulary = tokenizer.get_vocab()

    tokens = tokenizer.tokenize(SAMPLE_TEXT)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    decoded_text = tokenizer.decode(token_ids)

    print(f"Input text: {SAMPLE_TEXT}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Token count: {len(tokens)}")
    print("\nToken strings, IDs, and vocabulary lookups:")
    for token, token_id in zip(tokens, token_ids):
        print(f"{token!r} -> {token_id} (vocabulary lookup: {vocabulary[token]})")

    print(f"\nDecoded text: {decoded_text}")
    print(f"Round-trip matches input: {decoded_text == SAMPLE_TEXT}")


if __name__ == "__main__":
    main()

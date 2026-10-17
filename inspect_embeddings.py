"""Inspect GPT-2 token and positional embeddings for a short prompt."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "openai-community/gpt2"
PROMPT = "Hello, world!"
NUM_VALUES_TO_DISPLAY = 5


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    model.to("cpu")
    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    batch_size, sequence_length = input_ids.shape
    position_ids = torch.arange(sequence_length).unsqueeze(0)

    with torch.inference_mode():
        token_embeddings = model.transformer.wte(input_ids)
        position_embeddings = model.transformer.wpe(position_ids)
        combined_embeddings = token_embeddings + position_embeddings

    print(f"Prompt: {PROMPT}")
    print(f"Input IDs shape: {tuple(input_ids.shape)}")
    print(f"Position IDs shape: {tuple(position_ids.shape)}")
    print(f"Token embeddings shape: {tuple(token_embeddings.shape)}")
    print(f"Position embeddings shape: {tuple(position_embeddings.shape)}")
    print(f"Combined embeddings shape: {tuple(combined_embeddings.shape)}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Hidden size: {model.config.n_embd}")

    print("\nFirst token embedding (first 5 values):")
    print(token_embeddings[0, 0, :NUM_VALUES_TO_DISPLAY].tolist())
    print("First position embedding (first 5 values):")
    print(position_embeddings[0, 0, :NUM_VALUES_TO_DISPLAY].tolist())
    print("Their sum before embedding dropout (first 5 values):")
    print(combined_embeddings[0, 0, :NUM_VALUES_TO_DISPLAY].tolist())


if __name__ == "__main__":
    main()

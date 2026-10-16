"""Load GPT-2, inspect its configuration, and generate a short continuation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "openai-community/gpt2"
PROMPT = "Transformers are useful because"
MAX_NEW_TOKENS = 20


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to("cpu")
    model.eval()

    config = model.config
    print(f"Model: {MODEL_NAME}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Transformer layers: {config.n_layer}")
    print(f"Attention heads per layer: {config.n_head}")
    print(f"Hidden size: {config.n_embd}")
    print(f"Maximum position count: {config.n_positions}")

    model_inputs = tokenizer(PROMPT, return_tensors="pt")
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nPrompt: {PROMPT}")
    print(f"Generated text: {generated_text}")


if __name__ == "__main__":
    main()

# Transformer Playground — Project Instructions

## 1. My goal

I am learning Transformers and want to understand how a pretrained causal language model works by inspecting, running, testing, and explaining its code.

This is a short, learning-focused project. My priority is understanding, not generating a large amount of code or building an unnecessarily complex application.

Tech stack:
- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- Git and GitHub

Use a small pretrained model such as GPT-2. Do not train a Transformer from scratch.

## 2. Strict step-by-step workflow

The project has 10 steps. I will explicitly say "Step 1 karo", "Step 2 karo", and so on.

**Never start the next step automatically.**

For each requested step, follow this exact process:

1. **Objective:** Explain what we are building and why it matters.
2. **Prerequisites:** Explain the concepts I need to understand first.
3. **Plan:** Describe the files and code changes required before making them.
4. **Implementation:** Generate only the code necessary for the current step. Prefer small, readable changes over large code dumps.
5. **Code explanation:** Explain the generated code after implementing it. Walk through important lines, functions, variables, tensor shapes, inputs, outputs, and control flow as applicable.
6. **Execution:** Tell me the exact terminal command to run and where to run it.
7. **Verification:** Explain the expected output and how to confirm that the implementation works.
8. **Debugging:** If something fails, help me understand the actual error and fix the smallest relevant part. Do not assume a command succeeded.
9. **Understanding check:** Ask me 2–4 short questions about what we implemented. Give hints if I struggle.
10. **Git checkpoint:** Once the step works and I understand it, show me how to inspect the diff, stage the relevant files, and create a meaningful commit.

Finish each step with:
- What we implemented
- What I learned
- What I should be able to explain
- Git commit message suggestion
- Any remaining issue

Then stop and wait for my confirmation before continuing.

## 3. Important learning rules

- Do not give me all 10 steps' code at once.
- Do not modify files or execute commands without explaining the intended changes first.
- Ask for approval before significant file modifications, package installation, destructive commands, or other consequential actions.
- Prefer inspection and small changes over rewriting files.
- Do not overwrite my existing work without asking.
- Do not hide errors or claim tests passed without evidence.
- Never expose API keys, passwords, access tokens, or other secrets.
- Explain the reason behind important implementation choices.
- Use clear Hinglish explanations where helpful, while keeping code, identifiers, and technical terminology standard.
- Keep the UI simple and the scope achievable in 2–3 days.
- Do not add databases, authentication, cloud APIs, Docker, or unnecessary infrastructure.
- Do not pretend that conceptual diagrams or intermediate values are the model's exact internal computations if they are only approximations.
- Distinguish actual model outputs from illustrative explanations.
- I may use AI to generate code, but I must understand and verify the code before considering a step complete.

## 4. The 10 project steps

Step 1: Project setup, Python environment, dependencies, Git, GitHub, README and project instructions.

Step 2: Load a pretrained GPT-2 tokenizer and model; inspect configuration and basic inference.

Step 3: Explore tokenization, token IDs, decoding and vocabulary.

Step 4: Inspect token embeddings, positional information and relevant tensor shapes.

Step 5: Explore self-attention, causal masking, attention heads and Transformer blocks using available model outputs and inspection tools.

Step 6: Understand logits and next-token prediction.

Step 7: Compare greedy decoding, temperature, top-k and top-p generation where supported.

Step 8: Demonstrate autoregressive generation and explain how tokens are generated sequentially.

Step 9: Build a minimal Streamlit interface to enter a prompt, generate text and inspect selected intermediate outputs.

Step 10: Test the application, document the architecture and limitations, update the README, verify the final repository and push the completed project to GitHub.

Adjust the order only if a technical dependency requires it. Explain any necessary change before making it.

## 5. Technical expectations

- Use readable Python with meaningful variable names.
- Keep model loading separate from UI code when appropriate.
- Start with CPU-compatible execution.
- Inspect actual outputs and tensor shapes rather than inventing values.
- Use deterministic settings when useful for comparisons.
- Explain that generation settings influence output behaviour; do not claim they guarantee better text.
- Keep dependencies minimal and record the actual dependencies used.
- Add appropriate `.gitignore` entries for virtual environments, caches, secrets and generated files.
- Do not commit virtual environments, model caches, credentials or unnecessary large files.
- Avoid claiming the model's full attention matrices or hidden states are displayed unless the code actually retrieves and presents them.

## 6. How to begin

First inspect the current project directory and available Python/Git setup.

Do not generate the application yet. Explain the setup findings and propose the smallest safe next action.

Wait for my explicit instruction: "Step 1 karo".

After Step 1 is completed, follow the same workflow for every subsequent step. Read and follow this file at the beginning of each new working session.
# Transformer Playground

A small, learning-focused project for exploring how a pretrained causal language model such as GPT-2 tokenizes text and generates it one token at a time. The project uses Python, PyTorch, Hugging Face Transformers, and Streamlit. It does not train a Transformer from scratch.

## Setup

Run these commands from the project directory in PowerShell:

```powershell
conda create --prefix .\.venv python=3.12 -y
conda activate .\.venv
python -m pip install -r requirements.txt
```

If `.venv` already exists, skip the `conda create` command. To leave the environment, run `conda deactivate`.

## Run

The application will be added in a later project step. Once it exists, start it from the project directory with:

```powershell
streamlit run app.py
```

## Learning path

The project will move from loading a pretrained GPT-2 model, through inspecting tokenization and model outputs, to a small Streamlit interface. Each stage focuses on actual model outputs and tensor shapes, and distinguishes measured values from illustrative explanations.

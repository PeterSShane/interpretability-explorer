# Interpretability Explorer

An interactive tool for visualizing what a sentiment classification model (DistilBERT) pays attention to when making predictions, and using that to audit for bias and debug errors.

## Problem

AI language models are black boxes. They make predictions but nobody can easily see *why*, whether they're picking up on legitimate signals or biased/spurious ones.

## Goal

Give anyone a way to see inside a real sentiment classifier: what it focuses on, whether it treats sensitive attributes (gender, race, religion, age, disability, class) consistently, and what causes it to make mistakes.

## How It Works

DistilBERT computes self-attention across 6 layers and 12 heads per layer, 72 attention patterns per sentence. This tool extracts those weights directly using Hugging Face's `transformers` library and renders them as interactive heatmaps with Plotly, inside a Streamlit dashboard.

## Features

- **Free Text Mode**: type any sentence, see a live attention heatmap and a plain-English summary of what drove the prediction
- **Bias Audit Mode**: compare contrastive sentence pairs (e.g. swapping "he"/"she", a name, a religion, an age) and get an automatic verdict on whether the model's confidence shifted meaningfully
- **Error Analysis Mode**: browse real misclassified examples from the SST-2 validation set, with a diagnosis of what the model focused on and how that mismatched the correct sentiment

## Example Finding

For the sentence *"holden caulfield did it better."* (a sarcastic negative reference, true label NEGATIVE), the model predicted POSITIVE with 99% confidence, focusing 30% of its final-layer attention on the word "better." This is a clear, explainable failure mode: the model leans on surface-level word sentiment and misses sarcasm/comparison framing.

## Tech Stack

Python, PyTorch, Hugging Face `transformers` + `datasets`, Streamlit, Plotly

## Caveat

Attention weights are a useful but imperfect lens into model behavior, some interpretability researchers argue attention correlates with importance but isn't a full causal explanation of the decision. This tool is best used as an exploratory/diagnostic aid, not a definitive account of model reasoning.

## Live Demo

https://interpretability-explorer-zpeguigzfvlgmfpq6fc7zt.streamlit.app

## Planned Extensions

- 3D UMAP visualization of sentence embeddings to see how the model clusters similar-sentiment sentences
- Applying the same interpretability approach to a network traffic classifier (website fingerprinting attack/defense research)
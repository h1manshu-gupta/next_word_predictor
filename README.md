# Gutenberg Next-Word Predictor

A word-level next-word prediction model built with PyTorch and trained on cleaned Project Gutenberg text.

Given a sequence of previous words, the model predicts the most likely next word. It uses an Embedding layer followed by stacked LSTM layers and supports top-k sampling for text generation.

## Features

- Text cleaning and word-level tokenization
- Fixed-length context windows for next-word prediction
- PyTorch LSTM language model
- CUDA GPU acceleration when available
- Early stopping based on validation loss
- Top-1 and top-5 accuracy evaluation
- Top-k sampling for less repetitive generated text
- Saved model checkpoint and tokenizer/vocabulary artifacts

## Project Structure

```text
next-word-predictor/
├── next_word_lstm_pytorch.ipynb   # Training, evaluation, and generation notebook
├── data.txt                       # Gutenberg training text
├── best_next_word_lstm.pt         # Best model checkpoint
├── tokenizer.pkl                  # Saved tokenizer/vocabulary
├── .gitignore
└── README.md
```

## Model Pipeline

```text
Gutenberg text
    -> cleaning
    -> tokenization
    -> fixed-length input windows
    -> Embedding layer
    -> stacked LSTM layers
    -> softmax vocabulary probabilities
    -> predicted next word
```

For each training example, the model receives a context of previous words and learns to predict one integer word ID as the target. Cross-entropy loss is used during training.

## Dataset

The model is trained on cleaned text from Project Gutenberg.

The raw book text is not included in this repository. Download public-domain texts responsibly and follow Project Gutenberg’s Terms of Use. Avoid automated bulk downloading from its website. [Project Gutenberg Terms of Use](https://www.gutenberg.org/policy/terms_of_use.html)

---

## Results

Training used early stopping based on validation loss.

| Metric | Best epoch | Result |
|---|---:|---:|
| Validation loss | 6 | 5.5313 |
| Validation top-1 accuracy | 9 | 15.86% |
| Validation top-5 accuracy | 10 | 33.99% |
| Training stopped | 11 | Early stopping |

The lowest validation loss occurred at epoch 6, so `best_next_word_lstm.pt` is the checkpoint used for final text generation. Although top-1 and top-5 accuracy continued to increase after epoch 6, validation loss increased after that point. This suggests mild overfitting: the model became more confident on some exact predictions while its overall probability estimates on unseen text became worse.

The model continued to improve on the training set after epoch 5, while validation loss increased. This indicates overfitting, so the epoch-5 checkpoint is used for generation.

Example generated continuation:

```text
Prompt: once upon a time
Output: once upon a time and we would be the
```

Generated output is expected to be short and sometimes generic because this is a relatively small LSTM trained from scratch on a limited corpus.

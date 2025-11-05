# Overview

Predicts emojis from text inputs using a deep learning model trained on tweet data in Python using TensorFlow with GloVe embeddings and two bidirectional LSTM layers.

The trained model, GloVe embeddings, and cleaned tweets dataset are too large to include in the repository, so they have been left out.

## Project Files

### app.py
Runs the trained emoji prediction model to take text input and output the predicted emoji.

### training_code.py
Prepares data, trains the emoji prediction model using GloVe embeddings and bidirectional LSTMs, and saves the model.

### mapping.csv
Contains the mapping between emojis and a number that the trained model outputs.

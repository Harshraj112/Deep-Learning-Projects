# Sentiment Analysis with an LSTM Neural Network

This notebook builds a **recurrent neural network (RNN)** — specifically an **LSTM (Long Short-Term Memory)** model — to classify consumer product reviews as positive or negative (a binary **sentiment polarity** classification task). This README explains the theory behind each stage of the pipeline.

---

## 1. The Problem: Sentiment Classification

The dataset consists of consumer reviews, each labeled with a polarity: `__label__2` (positive) or another label (negative), parsed into a binary target (1 = positive, 0 = negative). The goal is to train a model that reads a review's text and predicts whether the sentiment is positive or negative.

Unlike the earlier numeric/image classification notebooks, the input here is **unstructured natural language text**, which cannot be fed directly into a neural network — it must first be converted into numbers in a way that preserves meaningful structure. This is the central theoretical challenge this notebook addresses.

---

## 2. Text Cleaning

```python
def data_prep(in_tex):
    out_tex = re.sub('[^a-zA-Z]', ' ', in_tex)   # remove punctuation/numbers
    out_tex = out_tex.lower()                     # normalize case
    out_tex = re.sub(r"\s+[a-zA-Z]\s+", ' ', out_tex)  # remove single characters
    return out_tex
```

Before feeding text into a model, it's standard practice to reduce noise and vocabulary size:
- **Removing punctuation/numbers** ensures the model focuses on meaningful words rather than incidental symbols.
- **Lowercasing** ensures "Great" and "great" are treated as the same word, rather than two different tokens.
- **Removing single characters** strips out stray letters left behind by prior cleaning steps (e.g., leftover "s" from an apostrophe removal), which usually carry no sentiment signal.

This preprocessing reduces the **vocabulary size** and **sparsity** of the data, which generally improves a model's ability to learn generalizable patterns instead of memorizing noise.

---

## 3. Tokenization

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
```

Neural networks operate on numbers, not words. **Tokenization** is the process of building a vocabulary — a mapping from each unique word in the training data to a unique integer ID (its `word_index`). For example, "good" might map to 42 and "terrible" might map to 187.

`tokenizer.texts_to_sequences()` then converts each review from a string of words into a sequence of integers, according to this mapping. Words never seen during training (out-of-vocabulary) are simply dropped or ignored.

---

## 4. Padding Sequences

```python
max_length = 100
x_train = pad_sequences(x_train, padding='post', maxlen=max_length)
```

Reviews naturally vary in length — one might be 10 words, another 200. However, neural networks require **fixed-size inputs** for batch processing. **Padding** solves this by:
- Truncating sequences longer than `max_length` (100 tokens here).
- Padding shorter sequences with zeros (`padding='post'` adds zeros *after* the actual content) until they reach exactly 100 tokens.

This creates uniformly-shaped input matrices the model can process efficiently in batches.

---

## 5. The Embedding Layer

```python
model.add(Embedding(total_size, 20, input_length=max_length))
```

Simply representing words as arbitrary integers (e.g., "good" = 42) is problematic: it implies a numeric relationship between words that doesn't reflect their actual meaning, and it doesn't capture that "good" and "great" are semantically similar.

An **embedding layer** solves this by learning a **dense vector representation** for each word — in this case, a 20-dimensional vector per word (`total_size` is the vocabulary size, 20 is the embedding dimension). Rather than being handcrafted, these vectors are learned during training such that words with similar meanings or usage patterns end up with similar vector representations (closer together in the 20-dimensional embedding space). This is conceptually similar to (though independently learned from) well-known embedding techniques like Word2Vec or GloVe.

---

## 6. The LSTM Layer

```python
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
```

### Why not a simple Dense layer for text?
A standard feedforward (Dense) layer treats inputs independently and has no concept of **order** or **sequence**. But in language, word order matters immensely ("not good" means the opposite of "good"). This calls for a **recurrent neural network (RNN)**, which processes sequences one element (word) at a time while maintaining an internal "memory" (hidden state) that carries information forward from earlier words to later ones.

### Why LSTM specifically?
Plain RNNs suffer from the **vanishing gradient problem** — over long sequences, gradients used to update earlier time steps shrink toward zero during backpropagation, making it hard for the network to learn long-range dependencies (e.g., connecting a sentiment-bearing word at the start of a review to context at the end).

**LSTM (Long Short-Term Memory)** networks solve this with a more sophisticated internal structure involving three gates:
- **Forget gate** — decides what information from the previous state to discard.
- **Input gate** — decides what new information to add to the memory (cell state).
- **Output gate** — decides what part of the memory to output as the hidden state.

These gates are themselves small learned neural network components (sigmoid-activated), allowing the LSTM to selectively remember or forget information across long sequences, making it far more effective than a plain RNN at tasks like sentiment analysis, where relevant sentiment cues can appear anywhere in a review.

### Dropout and Recurrent Dropout
```
dropout=0.2, recurrent_dropout=0.2
```
**Dropout** is a regularization technique that randomly "turns off" a fraction of neurons (here, 20%) during each training step, forcing the network to not overly rely on any single neuron or pathway. This helps prevent **overfitting** — memorizing the training data rather than learning generalizable patterns.
- `dropout` applies to the inputs of the LSTM layer.
- `recurrent_dropout` applies to the recurrent connections (the hidden-state pathway that carries information across time steps).

---

## 7. Output Layer: Sigmoid

```python
model.add(Dense(1, activation='sigmoid'))
```

Since this is a **binary classification** problem (positive vs. negative), a single output neuron with a sigmoid activation is appropriate — it outputs a value between 0 and 1, representing the predicted probability that the review is positive.

---

## 8. Loss Function and Optimizer

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
```

- **Binary cross-entropy** is the standard loss function for two-class problems, penalizing the model based on how far its predicted probability is from the true 0/1 label (mathematically identical to the sigmoid cross-entropy loss used in the earlier perceptron notebook).
- **Adam optimizer** adaptively adjusts the learning rate per parameter, generally leading to faster, more stable convergence than plain gradient descent — particularly valuable here given the larger number of parameters in the embedding and LSTM layers compared to a simple perceptron.

---

## 9. Training with Validation

```python
model.fit(x_train, y_train, batch_size=128, epochs=5,
          verbose=1, validation_data=(x_test, y_test))
```

- **Batch size (128)**: instead of updating weights after every single example (slow, noisy) or the entire dataset at once (memory-intensive), the model processes the data in **mini-batches** of 128 reviews at a time, striking a balance between training speed, memory efficiency, and gradient stability.
- **Epochs (5)**: the model passes over the entire training set 5 times.
- **Validation data**: at the end of each epoch, the model's performance is checked against the held-out test set (data it never trains on). This is critical for monitoring **overfitting** — if training accuracy keeps rising while validation accuracy stalls or drops, the model is starting to memorize rather than generalize.

---

## 10. Saving and Loading the Model

```python
model.save("model.h5")
model = keras.models.load_model("model.h5")
```

Once trained, the model's architecture, learned weights, and optimizer state can be serialized to disk (`.h5` format) so it can be reloaded later for inference or further training without repeating the entire training process — a standard practice for deploying trained models into production or sharing them across sessions.

---

## Summary of the Workflow

1. **Parse** raw text reviews and polarity labels from `train.csv`.
2. **Clean** text (remove punctuation/numbers, lowercase, strip stray characters).
3. **Split** data into training and test sets.
4. **Tokenize** words into integer sequences and **pad** them to a fixed length (100 tokens).
5. **Embed** each word into a learned 20-dimensional dense vector space.
6. **Process** the sequence through an LSTM layer (32 units) to capture contextual and order-dependent information, with dropout for regularization.
7. **Classify** using a sigmoid output neuron, trained with binary cross-entropy loss and the Adam optimizer.
8. **Validate** performance on held-out data across 5 epochs.
9. **Save/reload** the trained model for reuse.

This notebook illustrates the standard architecture for text classification before the era of Transformer-based models (like BERT or GPT): **Embedding → Recurrent layer (LSTM/GRU) → Dense output**. It remains a solid, interpretable baseline for sentiment analysis and other sequence classification tasks.
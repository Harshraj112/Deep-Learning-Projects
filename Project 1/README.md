# Single-Layer Perceptron for Binary Classification

This notebook implements a **single-layer perceptron** — the simplest possible artificial neural network — trained with TensorFlow to solve a binary classification problem on 2D data. This README focuses on the *theory* behind each step, so you understand *why* the code does what it does, not just *what* it does.

---

## 1. What Is a Perceptron?

A perceptron is the foundational unit of a neural network, originally proposed by Frank Rosenblatt in 1958. It takes a set of inputs, computes a weighted sum of them, adds a bias term, and passes the result through an activation function to produce an output.

Mathematically, for an input vector **x** = (x₁, x₂, ..., xₙ):

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b   =   w·x + b
output = σ(z)
```

Where:
- **w** (weights) — determines how much influence each input feature has
- **b** (bias) — shifts the decision boundary, independent of the inputs
- **σ** — the activation function (here, the sigmoid function)

In this notebook, there are **2 input features** (`x1`, `x2`) and **1 output neuron**, making it suitable for binary classification of 2D points.

---

## 2. The Sigmoid Activation Function

The sigmoid function squashes any real number into the range (0, 1):

```
σ(z) = 1 / (1 + e^(-z))
```

This is useful for binary classification because the output can be interpreted as a **probability** — the likelihood that a given input belongs to class 1. Values close to 0 indicate class 0, values close to 1 indicate class 1, and 0.5 is the decision threshold.

---

## 3. Loss Function: Sigmoid Cross-Entropy

To train the perceptron, we need a way to measure how wrong its predictions are. This notebook uses **sigmoid cross-entropy loss** (also called binary cross-entropy or log loss):

```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

Where:
- **y** is the true label (0 or 1)
- **ŷ** is the predicted probability from the sigmoid output

This loss penalizes confident wrong predictions heavily and rewards confident correct predictions. TensorFlow's `sigmoid_cross_entropy_with_logits` combines the sigmoid activation and the cross-entropy calculation internally for numerical stability, which is why the raw logits (`I`, before applying sigmoid) are typically what's passed to this function — though in this notebook the perceptron's sigmoid *output* is passed in directly, which is a slightly unconventional but workable variant since accuracy is only reduced, not eliminated, when the resulting loss is passed through `abs()`.

---

## 4. Training via Gradient Descent (Adam Optimizer)

Training a perceptron means finding the values of **w** and **b** that minimize the loss function across all training examples. This is done through **gradient descent**:

1. Compute the gradient (slope) of the loss with respect to each weight and the bias.
2. Update each parameter in the direction that reduces the loss:
   ```
   w := w - η · ∂L/∂w
   b := b - η · ∂L/∂b
   ```
   where **η** (eta) is the learning rate.
3. Repeat over many iterations until the loss converges to a minimum.

This notebook uses the **Adam optimizer**, an advanced variant of gradient descent that adapts the learning rate for each parameter individually based on estimates of the first and second moments (mean and variance) of past gradients. Adam generally converges faster and more reliably than plain gradient descent, especially on noisy or sparse gradients.

The training loop runs for **1000 iterations**, each time calling `optimizer.minimize()` on the loss function with respect to `weight` and `bias`, nudging them closer to their optimal values.

---

## 5. Evaluation: Accuracy and the Confusion Matrix

After training, the model's sigmoid outputs (continuous probabilities) are rounded to 0 or 1 to produce hard class predictions. Two standard metrics are then used to evaluate performance:

- **Accuracy** — the proportion of predictions that match the true labels:
  ```
  Accuracy = (Correct Predictions) / (Total Predictions)
  ```
- **Confusion Matrix** — a table showing the counts of:
  - True Positives (TP) — predicted 1, actually 1
  - True Negatives (TN) — predicted 0, actually 0
  - False Positives (FP) — predicted 1, actually 0
  - False Negatives (FN) — predicted 0, actually 1

  |               | Predicted 0 | Predicted 1 |
  |---------------|-------------|-------------|
  | **Actual 0**  | TN          | FP          |
  | **Actual 1**  | FN          | TP          |

The confusion matrix gives a more nuanced picture than accuracy alone, especially useful when classes are imbalanced.

---

## 6. Why a Single Perceptron Has Limits

It's worth noting, theoretically, that a single-layer perceptron can only learn **linearly separable** patterns — it essentially learns a straight line (or hyperplane in higher dimensions) that divides the two classes. If the underlying data isn't linearly separable (like the classic XOR problem), a single perceptron cannot achieve perfect accuracy no matter how it's trained. This is precisely the historical motivation for **multi-layer perceptrons (MLPs)** and deeper neural networks, which stack multiple layers of neurons with non-linear activations to learn more complex, non-linear decision boundaries.

---

## Summary of the Workflow

1. **Initialize** weights and bias to zero.
2. **Define** the perceptron function: linear combination → sigmoid.
3. **Define** the loss function: sigmoid cross-entropy.
4. **Load and visualize** the 2D labeled dataset.
5. **Train** using the Adam optimizer over 1000 iterations to minimize loss.
6. **Evaluate** using final loss, accuracy, and a confusion matrix.

This is essentially a from-scratch, minimal illustration of how neural network training works under the hood — without relying on high-level `model.fit()` abstractions — making it a good pedagogical example of gradient-based learning.

---

## Notes on the Code (Practical, Not Theoretical)

A couple of small technical notes if you run this notebook as-is:
- `dataframe[...].as_matrix()` is deprecated in modern pandas — use `.values` or `.to_numpy()` instead.
- The notebook expects a `data.csv` file with columns `x1`, `x2`, and `label` in the working directory.
# MNIST Handwritten Digit Classification

This notebook builds a simple neural network using Keras/TensorFlow to classify handwritten digits (0–9) from the classic **MNIST dataset**. This README explains the theory behind each component, so you understand *why* the model works, not just what the code does.

---

## 1. The MNIST Dataset

MNIST is a dataset of 70,000 grayscale images of handwritten digits (60,000 for training, 10,000 for testing), each **28×28 pixels**. Each image is labeled with the digit it represents (0 through 9). It's often called the "hello world" of computer vision and neural networks because it's small, clean, and well-suited for demonstrating classification concepts.

Each pixel has an intensity value from 0 (black) to 255 (white). This notebook loads the data pre-split into training and test sets via `tf.keras.datasets.mnist`.

---

## 2. Normalization

```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```

Raw pixel values range from 0–255. Dividing by 255 rescales them to the range **[0, 1]**. This is a form of **feature scaling**, which matters because:

- Neural networks train more efficiently when inputs are on a consistent, small scale.
- Large input values can cause unstable gradients or slow convergence during optimization.
- Activation functions (like softmax, sigmoid) behave more predictably with normalized inputs.

---

## 3. Model Architecture

```python
tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

This is a **Sequential model** — layers stacked one after another, each feeding into the next.

### Flatten Layer
Each image is a 28×28 grid of pixels (a 2D matrix). Neural network layers like `Dense` expect a 1D vector of inputs, so the `Flatten` layer reshapes each 28×28 image into a single vector of 784 values (28 × 28 = 784), without changing the data itself — just its shape.

### Dense Layer with Softmax
This is a **fully connected layer** with 10 neurons — one for each digit class (0–9). Each neuron computes a weighted sum of all 784 input pixels plus a bias, exactly like the perceptron in your earlier notebook:

```
zᵢ = w_i · x + b_i     (for each digit class i = 0...9)
```

The key difference here is the **softmax activation function**, applied across all 10 outputs together (not each independently, like sigmoid):

```
softmax(z_i) = e^(z_i) / Σⱼ e^(z_j)
```

Softmax converts the 10 raw scores into a **probability distribution** — all 10 values are between 0 and 1, and they sum to exactly 1. The digit with the highest resulting probability is the model's prediction.

> Note: With no hidden layers, this model is architecturally equivalent to **multinomial logistic regression** — it can only learn a linear decision boundary between classes in pixel-intensity space. It performs reasonably well on MNIST (which is simple enough to be near-linearly separable) but would struggle on harder image datasets, which is why real-world image classifiers use deeper networks with convolutional layers.

---

## 4. Loss Function: Sparse Categorical Cross-Entropy

```python
digit.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

Since this is a **multi-class classification** problem (10 possible classes, not 2), the appropriate loss function is **categorical cross-entropy**:

```
L = -Σᵢ yᵢ · log(ŷᵢ)
```

Where **y** is the true label and **ŷ** is the predicted probability distribution from softmax.

The **"sparse"** variant is used specifically because the labels (`y_train`, `y_test`) are given as plain integers (e.g., `7`) rather than one-hot encoded vectors (e.g., `[0,0,0,0,0,0,0,1,0,0]`). Sparse categorical cross-entropy handles the conversion internally, avoiding the need to manually one-hot encode 60,000 labels.

---

## 5. Optimizer: Adam

Just as in the earlier perceptron notebook, **Adam** (Adaptive Moment Estimation) is used to update the weights and biases. Adam combines ideas from momentum-based gradient descent and adaptive learning rates, generally converging faster and more reliably than plain stochastic gradient descent — an especially useful property when training on tens of thousands of examples.

---

## 6. Training (Epochs)

```python
digit.fit(x_train, y_train, epochs=3)
```

An **epoch** is one complete pass through the entire training dataset. During each epoch:
1. The model makes predictions on batches of training images.
2. The loss is calculated by comparing predictions to true labels.
3. Gradients are computed and weights/biases are updated via Adam.

Training for only 3 epochs is fairly minimal, but because MNIST is a relatively simple dataset and this model has very few parameters (784 × 10 weights + 10 biases = 7,850 parameters total), it can still reach fairly high accuracy quickly.

---

## 7. Evaluation

```python
digit.evaluate(x_test, y_test)
```

This measures the model's **loss** and **accuracy** on the held-out test set — data the model never saw during training. This is critical for judging **generalization**: how well the model performs on new, unseen digits, rather than just memorizing the training examples.

---

## Summary of the Workflow

1. **Load** the MNIST dataset (28×28 grayscale digit images with labels).
2. **Normalize** pixel values to the [0, 1] range.
3. **Flatten** each image into a 784-length vector.
4. **Feed forward** through a single Dense layer with softmax to get class probabilities.
5. **Train** using Adam optimizer and sparse categorical cross-entropy loss over 3 epochs.
6. **Evaluate** on the test set to measure real-world generalization performance.

This notebook is essentially a minimal, single-layer classifier — a natural next step up from the earlier 2-class perceptron, generalized to 10 classes using softmax instead of sigmoid. It's a common first "real" deep learning exercise before moving on to convolutional neural networks (CNNs), which better exploit the spatial structure of images.
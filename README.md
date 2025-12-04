# Neural Network from Scratch - MNIST Digit Classification

A fully-connected neural network implementation built from scratch using only NumPy. This project demonstrates fundamental deep learning concepts by manually implementing forward propagation, backpropagation, and gradient descent without relying on deep learning frameworks.

## Project Overview

This project classifies handwritten digits (0-9) from the MNIST dataset using a 5-layer neural network with all core operations implemented manually in NumPy.

## Network Architecture

```
Input Layer:    784 neurons (28×28 flattened images)
                    ↓
Hidden Layer 1: 500 neurons + Leaky ReLU
                    ↓
Hidden Layer 2: 300 neurons + Leaky ReLU
                    ↓
Hidden Layer 3: 100 neurons + Leaky ReLU
                    ↓
Output Layer:   10 neurons + Softmax
```

## Results

- **Test Accuracy:** 97.23%
- **Training Time:** 30 epochs
- **Dataset Split:** MNIST (50,000 training + 10,000 validation + 10,000 test images)

## Key Features

- **Pure NumPy Implementation** - No PyTorch, TensorFlow, or Keras for the neural network
- **Xavier Weight Initialization** - Ensures stable gradient flow during training
- **Mini-Batch Gradient Descent** - Efficient training with batch size of 60
- **Leaky ReLU Activation** (α=0.01) - Prevents dying neuron problem
- **Learning Rate Decay** - Improves convergence over time
- **Cross-Entropy Loss** - Standard classification loss function
- **Train/Validation Split** - Monitors overfitting during training
- **Visualization** - Training curves and sample predictions

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch Size | 60 |
| Initial Learning Rate | 0.01 |
| Learning Rate Decay | 0.005 |
| Validation Set Size | 10,000 images |

## Dependencies (or libraries used)

```bash
numpy
matplotlib
torchvision
```

Install dependencies:
```bash
pip install numpy matplotlib torchvision
```

Note: PyTorch is only used for loading the MNIST dataset via `torchvision.datasets`, not for building the neural network.

## Usage

1. Clone the repository
2. Install dependencies
3. Run the notebook sequentially

The training process will:
- Download MNIST dataset automatically
- Train the network for 30 epochs
- Display training/validation metrics
- Save the trained model to `trained_model.npz`
- Evaluate on the test set

## Training Curves

The implementation tracks and visualizes:
- Training accuracy over epochs
- Validation accuracy over epochs
- Training loss over epochs
- Validation loss over epochs

## Model Persistence

Trained weights and biases are saved using:
```python
np.savez_compressed('trained_model.npz', w1=w1, w2=w2, ...)
```

## Implementation Details

### Core Components

**Activation Functions:**
- Leaky ReLU for hidden layers
- Softmax for output layer

**Loss Function:**
- Cross-entropy loss

**Weight Initialization:**
- Xavier/Glorot uniform initialization

**Optimization:**
- Mini-batch gradient descent with learning rate decay (helped in achieving a smoother learning curve)

### Forward Propagation
For each layer: `z = xW + b` → `a = activation(z)`

### Backward Propagation
Manual computation of gradients using chain rule for all weights and biases.

### Gradient Descent
Weights are updated using: `W = W - learning_rate × gradient`

## What I Learned

This project demonstrates understanding of:
- How neural networks actually work under the hood
- The math behind backpropagation and gradient computation
- Weight initialization strategies
- Training loop management
- Overfitting prevention techniques
- Vectorized operations in NumPy


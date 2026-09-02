"""Hand-written NumPy forward/backward passes and vanilla SGD for the
Actor and Critic MLPs used in One-Step Actor-Critic.

No torch/sklearn/auto-diff. All gradients are derived by hand via the chain
rule and applied with vanilla SGD (no momentum).
"""

import numpy as np


def forward_with_cache(net, x):
    """Forward pass, caching (input, pre-activation, activation) per layer.

    Returns (output, acts) where acts[i] = (a_in_i, z_i, a_i) for layer i.
      a_in_i : input (activations of layer i-1, or x for i=0)
      z_i    : pre-activation  z = W a_in + b
      a_i    : activation       a = ReLU(z) for hidden, a = z for output
    """
    x = np.asarray(x, dtype=np.float32)
    acts = []
    a_in = x
    h = x
    for i, (W, b) in enumerate(net.layers):
        z = W @ h + b
        a = z if i == len(net.layers) - 1 else np.maximum(z, 0)  # ReLU
        acts.append((a_in, z, a))
        a_in = a
        h = a
    return h, acts


def backward(net, acts, grad_out):
    """Backprop through the network, returning grads[i] = (dW, db) per layer.

    grad_out is dL/d(output) (same shape as the network output).

    Chain rule for a linear layer (W, b) followed by activation:
      z   = W a_{in} + b
      a   = act(z)
      g   = dL/da          (gradient flowing into this layer's output)
      for hidden layers, apply ReLU': g *= (z > 0)
      dW = outer(g, a_in)
      db = g
      propagate to previous layer: g = W.T @ g
    """
    n_layers = len(net.layers)
    g = np.asarray(grad_out, dtype=np.float32)
    grads = []
    for i in range(n_layers - 1, -1, -1):
        a_in, z, a = acts[i]
        if i < n_layers - 1:
            g = g * (z > 0)  # ReLU derivative: 1 if z>0 else 0
        W, b = net.layers[i]
        dW = np.outer(g, a_in)  # dW = outer(gradient, input)
        db = g.copy()            # db = gradient
        grads.append((dW, db))
        if i > 0:
            g = W.T @ g  # gradient_to_previous = W.T @ g
    grads.reverse()
    return grads


def sgd_step(net, grads, lr):
    """Vanilla SGD update (no momentum, no velocity).

        W <- W - lr * dW
        b <- b - lr * db
    """
    for i, (dW, db) in enumerate(grads):
        W, b = net.layers[i]
        W[:] = W - lr * dW
        b[:] = b - lr * db
        net.layers[i] = (W, b)


def softmax(logits):
    """Numerically stable softmax returning proper probabilities.

    Probabilities are clamped to avoid exact 0/1 and normalized so they sum
    exactly to 1 (avoids np.random.multinomial float-precision rejection).
    """
    logits = np.asarray(logits, dtype=np.float32)
    e = np.exp(logits - np.max(logits))
    p = e / e.sum()
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    p = p / p.sum()  # exact normalization
    return p


def sample(p):
    """Sample an action index from normalized probability vector p.

    multinomial rejects pvals whose float64 sum exceeds 1 by a tiny amount,
    so we force the trailing element to make the sum exactly 1.
    """
    p = np.asarray(p, dtype=float).copy()
    p = p / p.sum()
    p = np.clip(p, 0.0, 1.0)
    p[-1] = 1.0 - p[:-1].sum()  # force exact sum == 1
    if p[-1] < 0:  # guard against rounding
        p = np.clip(p, 0.0, 1.0)
        p = p / p.sum()
        p[-1] = 1.0 - p[:-1].sum()
    return int(np.random.multinomial(1, p).argmax())

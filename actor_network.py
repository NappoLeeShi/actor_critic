"""NumPy-only MLP Actor Network for One-Step Actor-Critic.

Architecture:
    Input(64) -> Linear(64->128) -> ReLU -> Linear(128->64) -> ReLU -> Linear(64->4) -> logits

Actions are sampled from softmax(logits) during training.
Deterministic policy uses argmax(logits) after training.
"""

import numpy as np


class ActorNetwork:
    """NumPy MLP policy network: state -> action logits."""

    def __init__(self, input_size=64, hidden_sizes=(128, 64), n_actions=4, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.layers = []
        prev = input_size
        for h in hidden_sizes:
            W = np.random.randn(h, prev) * 0.1
            b = np.zeros(h)
            self.layers.append((W, b))
            prev = h
        W = np.random.randn(n_actions, prev) * 0.1
        b = np.zeros(n_actions)
        self.layers.append((W, b))
        self.params = list(self.layers)

    def forward(self, x):
        """Forward pass returning raw logits (1D array of length n_actions)."""
        x = np.asarray(x, dtype=np.float32)
        h = x
        for i, (W, b) in enumerate(self.layers):
            out = W @ h + b
            if i < len(self.layers) - 1:
                out = np.maximum(out, 0)  # ReLU
            h = out
        return h

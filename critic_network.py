"""NumPy-only MLP Critic Network for One-Step Actor-Critic.

Architecture:
    Input(64) -> Linear(64->128) -> ReLU -> Linear(128->64) -> ReLU -> Linear(64->1) -> V(s)

Output is a scalar state-value estimate V(s).
"""

import numpy as np


class CriticNetwork:
    """NumPy MLP value network: state -> scalar V(s)."""

    def __init__(self, input_size=64, hidden_sizes=(128, 64), seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.layers = []
        prev = input_size
        for h in hidden_sizes:
            W = np.random.randn(h, prev) * 0.1
            b = np.zeros(h)
            self.layers.append((W, b))
            prev = h
        W = np.random.randn(1, prev) * 0.1
        b = np.zeros(1)
        self.layers.append((W, b))
        self.params = list(self.layers)

    def forward(self, x):
        """Forward pass returning scalar V(s)."""
        x = np.asarray(x, dtype=np.float32)
        h = x
        for i, (W, b) in enumerate(self.layers):
            out = W @ h + b
            if i < len(self.layers) - 1:
                out = np.maximum(out, 0)  # ReLU
            h = out
        return float(h[0])

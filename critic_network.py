# ---------------------------------------------------------------------------
# Prompt for reproducing this file (critic_network.py):
#
# Write a NumPy-only MLP "CriticNetwork" used in an Actor-Critic RL agent for
# a 4x4 Wumpus World. Do NOT use torch or sklearn; only numpy.
#
# Contract (exact):
#   - Constructor: CriticNetwork(input_size=5, hidden_sizes=(128, 64), seed=None)
#   - Hidden layers use ReLU; output layer is a single scalar V(s).
#   - Store layers as a list of (W, b). Initialize weights with:
#       W = np.random.randn(n_out, n_in) * 0.1,  b = np.zeros(n_out)
#   - If seed is not None, call np.random.seed(seed) before init.
#   - Attributes: .layers (list of (W,b)), keep .params = list(self.layers).
#   - .forward(x): x is 1D np.float32 array-ish (len == input_size). Return a
#     numpy float64 scalar (use .item()/float()). Hidden layers: W@in+b then
#     ReLU(max(0,.)). Final layer: linear scalar output, no activation.
#   - Follow numpy conventions, plain-Python class, no auto-diff library.
# ---------------------------------------------------------------------------

import numpy as np


class CriticNetwork:
    """NumPy MLP value network: state -> scalar V(s) (separate from actor)."""

    def __init__(self, input_size=5, hidden_sizes=(128, 64), seed=None):
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
        x = np.asarray(x, dtype=np.float32)
        h = x
        for i, (W, b) in enumerate(self.layers):
            out = W @ h + b
            if i < len(self.layers) - 1:
                out = np.maximum(out, 0)  # ReLU on hidden layers
            h = out
        return float(h[0])

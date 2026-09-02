# ---------------------------------------------------------------------------
# Prompt for reproducing this file (actor_network.py):
#
# Write a NumPy-only MLP "ActorNetwork" used in an Actor-Critic RL agent for
# a 4x4 Wumpus World. Do NOT use torch or sklearn; only numpy.
#
# Contract (exact):
#   - Constructor: ActorNetwork(input_size=5, hidden_sizes=(128, 64), n_actions=4, seed=None)
#   - Hidden layers use ReLU; output layer produces n_actions logits (no softmax).
#   - Store layers as a list of (W, b). Initialize weights with:
#       W = np.random.randn(n_out, n_in) * 0.1,  b = np.zeros(n_out)
#   - If seed is not None, call np.random.seed(seed) before init.
#   - Attributes: .layers (list of (W,b)), keep a copy .params = [(W,b),...]
#     so optimizers can mutate weights in place.
#   - .forward(x): x is 1D np.float32 array-ish (len == input_size). Return a
#     1D np.ndarray of n_actions logits. Hidden layers: out = W@in + b, then
#     ReLU(max(0,.)). Final layer: linear output, no activation.
#   - Follow numpy conventions and keep it a self-contained plain-Python class.
# ---------------------------------------------------------------------------

import numpy as np


class ActorNetwork:
    """NumPy MLP policy network: state -> action logits (separate from critic)."""

    def __init__(self, input_size=5, hidden_sizes=(128, 64), n_actions=4, seed=None):
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
        x = np.asarray(x, dtype=np.float32)
        h = x
        for i, (W, b) in enumerate(self.layers):
            out = W @ h + b
            if i < len(self.layers) - 1:
                out = np.maximum(out, 0)  # ReLU on hidden layers
            h = out
        return h

"""Wumpus World 4x4 grid environment for One-Step Actor-Critic RL.

Fully observable variant: state encodes the full 4x4 map using 4 channels
(Agent, Gold, Wumpus, Pit) flattened to a 64-dimensional vector.

Actions:
    0 = Up, 1 = Down, 2 = Left, 3 = Right

Rewards:
    normal move = -1, gold = +100, pit = -100, wumpus = -100

Episode ends on: gold reached, pit/wumpus reached, or max_steps exceeded.
"""

import random

import numpy as np

SIZE = 4
MAX_STEPS = 50
N_CHANNELS = 4  # agent, gold, wumpus, pit
STATE_DIM = SIZE * SIZE * N_CHANNELS  # 64

ACTION_DELTAS = {
    0: (0, -1),  # Up
    1: (0, 1),   # Down
    2: (-1, 0),  # Left
    3: (1, 0),   # Right
}

ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

REWARD_MOVE = -1
REWARD_GOLD = 100
REWARD_PIT = -100
REWARD_WUMPUS = -100


class WumpusWorldEnv:
    """Fully observable 4x4 Wumpus World environment (no Gym dependency)."""

    def __init__(self, size=SIZE, max_steps=MAX_STEPS, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.start = (0, 0)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._gen_map()
        self.agent_pos = self.start
        self.steps = 0

    def _gen_map(self):
        """Generate a random map with 1 gold, 1 wumpus, 2-4 pits."""
        self.gold_pos = None
        self.wumpus_pos = None
        self.pit_positions = []

        cells = [(x, y) for x in range(self.size) for y in range(self.size)]
        cells.remove(self.start)
        random.shuffle(cells)

        self.gold_pos = cells.pop()
        self.wumpus_pos = cells.pop()
        n_pits = random.randint(2, 4)
        self.pit_positions = cells[:n_pits]

        self.map = np.zeros((self.size, self.size), dtype=int)
        self.map[self.gold_pos] = 1
        self.map[self.wumpus_pos] = 2
        for p in self.pit_positions:
            self.map[p] = 3

    def _get_state(self):
        """Return 64-dim fully observable state vector.

        4 channels x 16 cells:
          channel 0: agent position (one-hot)
          channel 1: gold position (one-hot)
          channel 2: wumpus position (one-hot)
          channel 3: pit positions (multi-hot)
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        ax, ay = self.agent_pos
        state[0 * 16 + ay * self.size + ax] = 1.0

        gx, gy = self.gold_pos
        state[1 * 16 + gy * self.size + gx] = 1.0

        wx, wy = self.wumpus_pos
        state[2 * 16 + wy * self.size + wx] = 1.0

        for px, py in self.pit_positions:
            state[3 * 16 + py * self.size + px] = 1.0

        return state

    def reset(self, seed=None):
        """Reset environment with optional new seed for map generation."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._gen_map()
        self.agent_pos = self.start
        self.steps = 0
        return self._get_state()

    def step(self, action):
        """Execute action, return (next_state, reward, done, truncated, info)."""
        reward = REWARD_MOVE
        done = False
        info = {"outcome": "move"}

        dx, dy = ACTION_DELTAS[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        if 0 <= nx < self.size and 0 <= ny < self.size:
            self.agent_pos = (nx, ny)

        pos = self.agent_pos
        if pos == self.gold_pos:
            reward = REWARD_GOLD
            done = True
            info["outcome"] = "gold"
        elif pos in self.pit_positions:
            reward = REWARD_PIT
            done = True
            info["outcome"] = "pit"
        elif pos == self.wumpus_pos:
            reward = REWARD_WUMPUS
            done = True
            info["outcome"] = "wumpus"

        self.steps += 1
        truncated = self.steps >= self.max_steps
        if truncated and not done:
            info["outcome"] = "max_steps"

        state = self._get_state()
        return state, reward, done or truncated, truncated, info

    def get_state_for_cell(self, x, y):
        """Return the 64-dim state for a hypothetical agent at (x, y)
        on the current map. Used for policy/value grid visualization."""
        saved = self.agent_pos
        self.agent_pos = (x, y)
        state = self._get_state()
        self.agent_pos = saved
        return state

    def render(self):
        symbols = {0: ".", 1: "G", 2: "W", 3: "P"}
        print("+" + "---+" * self.size)
        for y in range(self.size):
            row = "|"
            for x in range(self.size):
                if (x, y) == self.agent_pos:
                    cell = "A"
                else:
                    cell = symbols[self.map[y, x]]
                row += f" {cell} |"
            print(row)
            print("+" + "---+" * self.size)

    def __repr__(self):
        return (
            f"WumpusWorldEnv(agent={self.agent_pos}, "
            f"gold={self.gold_pos}, wumpus={self.wumpus_pos}, "
            f"pits={self.pit_positions})"
        )

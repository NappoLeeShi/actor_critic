"""Wumpus World 4x4 grid environment for Actor-Critic RL.

State representation (input to Actor and Critic networks):
    s = [breeze, stench, glitter, x, y]
where x and y are the agent's column/row normalized by SATURATE.

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
SATURATE = 0.4

ACTION_DELTAS = {
    0: (0, -1),  # Up
    1: (0, 1),   # Down
    2: (-1, 0),  # Left
    3: (1, 0),   # Right
}

REWARD_MOVE = -1
REWARD_GOLD = 100
REWARD_PIT = -100
REWARD_WUMPUS = -100


class WumpusWorldEnv:
    """A Gym-compatible Wumpus World environment (no Gym dependency)."""

    def __init__(self, size=SIZE, max_steps=MAX_STEPS, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.seed = seed
        self.start = (1, 1)  # (x, y)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()

    def _gen_map(self):
        self.map = np.zeros((self.size, self.size), dtype=int)
        cells = [(x, y) for x in range(self.size) for y in range(self.size)]
        cells = [c for c in cells if c != self.start]
        random.shuffle(cells)

        self.gold_pos = cells.pop()
        self.wumpus_pos = cells.pop()
        n_pits = random.randint(2, 4)
        self.pit_positions = cells[:n_pits]

        self.map[self.gold_pos] = 1          # 1 = gold
        self.map[self.wumpus_pos] = 2        # 2 = wumpus
        for p in self.pit_positions:
            self.map[p] = 3                  # 3 = pit

    def _cell_has(self, pos, cell_type):
        x, y = pos
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        if cell_type == "gold":
            return pos == self.gold_pos
        if cell_type == "wumpus":
            return pos == self.wumpus_pos
        if cell_type == "pit":
            return pos in self.pit_positions
        return False

    def _is_breeze(self, pos):
        x, y = pos
        for dx, dy in ACTION_DELTAS.values():
            nxt = (x + dx, y + dy)
            if self._cell_has(nxt, "pit"):
                return True
        return False

    def _is_stench(self, pos):
        x, y = pos
        for dx, dy in ACTION_DELTAS.values():
            nxt = (x + dx, y + dy)
            if self._cell_has(nxt, "wumpus"):
                return True
        return False

    def _is_glitter(self, pos):
        return self._cell_has(pos, "gold")

    def _get_state(self, pos):
        x, y = pos
        state = [
            float(self._is_breeze(pos)),
            float(self._is_stench(pos)),
            float(self._is_glitter(pos)),
            x * SATURATE,
            y * SATURATE,
        ]
        return np.array(state, dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
        self._gen_map()
        self.agent_pos = self.start
        self.steps = 0
        return self._get_state(self.agent_pos)

    def step(self, action):
        reward = REWARD_MOVE
        done = False
        info = {"outcome": "move"}

        dx, dy = ACTION_DELTAS[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        if 0 <= nx < self.size and 0 <= ny < self.size:
            self.agent_pos = (nx, ny)

        pos = self.agent_pos
        if self._cell_has(pos, "gold"):
            reward = REWARD_GOLD
            done = True
            info["outcome"] = "gold"
        elif self._cell_has(pos, "pit"):
            reward = REWARD_PIT
            done = True
            info["outcome"] = "pit"
        elif self._cell_has(pos, "wumpus"):
            reward = REWARD_WUMPUS
            done = True
            info["outcome"] = "wumpus"

        self.steps += 1
        truncated = self.steps >= self.max_steps
        if truncated and not done:
            info["outcome"] = "max_steps"

        state = self._get_state(self.agent_pos)
        return state, reward, done or truncated, truncated, info

    def explain_state(self, pos=None):
        pos = pos if pos is not None else self.agent_pos
        x, y = pos
        return {
            "pos": pos,
            "breeze": float(self._is_breeze(pos)),
            "stench": float(self._is_stench(pos)),
            "glitter": float(self._is_glitter(pos)),
            "state": self._get_state(pos).tolist(),
        }

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
        info = self.explain_state()
        return (
            f"WumpusWorldEnv(agent={info['pos']}, "
            f"breeze={info['breeze']}, stench={info['stench']}, "
            f"glitter={info['glitter']}, state={info['state']})"
        )

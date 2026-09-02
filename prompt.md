# Prompt: Actor-Critic with Wumpus World (4x4)

Use this prompt to reproduce the exact same code. Any AI given only this prompt
must produce functionally identical code.

## Goal
Implement an Actor-Critic reinforcement learning agent that navigates a
randomly generated 4x4 Wumpus World grid to collect the gold while avoiding
pits and the Wumpus.

## Files to create

### 1) `wumpus_env.py`
A custom Gymnasium-style Wumpus World environment (no Gym dependency needed;
use plain Python + numpy with a Gym-compatible `reset` / `step` interface).

Environment definition:

- **Grid size**: 4x4, cells indexed `(x, y)` with `x` = column, `y` = row.
  The agent starts at `(1, 1)`.
- **Objects**: exactly **1 gold**, exactly **1 Wumpus**, and between **2 and 4
  pits**, all placed at random cells, never on the start cell.
- **Actions**: `0=Up`, `1=Down`, `2=Left`, `3=Right`.
  Movement that would leave the grid is blocked (agent stays in place and
  still receives the normal move reward).
- **Rewards**:
  - Normal movement: `-1`
  - Reaching gold: `+100`
  - Reaching a pit: `-100`
  - Reaching the Wumpus: `-100`
- **Episode ends** when the agent:
  - reaches the gold, OR
  - reaches a pit / the Wumpus, OR
  - exceeds `max_steps` (default 50).
- **State representation** (the input to Actor and Critic networks), a vector:
  ```
  s = [breeze, stench, glitter, x, y]
  saturate = 0.4   # normalization factor used below
  x_n = x * saturate
  y_n = y * saturate
  ```
  Featurization rules at the agent's current cell `(x, y)`:
  - `breeze  = 1.0` if any orthogonally-adjacent neighbor holds a pit, else `0.0`
  - `stench  = 1.0` if any orthogonally-adjacent neighbor holds the Wumpus, else `0.0`
  - `glitter = 1.0` if this cell holds the gold, else `0.0`
  - `x` and `y` are normalized by multiplying by `saturate`.
  - Concretely, for the example "agent at (2,2), breeze=1, stench=0, glitter=0"
    this gives: `s = [1.0, 0.0, 0.0, 0.8, 0.8]`.
- Environment methods:
  - `reset(seed=None) -> state`: rebuild a brand-new random map, return state.
  - `step(action) -> (state, reward, done, truncated, info)`.
  - `render()`: ASCII grid printing the map.
  - Attribute `map` holding the grid.
  - Add a `__repr__`/inline state explanation helper.

### 2) `actor_network.py`
A separate **NumPy-only** MLP class `ActorNetwork` (no torch/sklearn/auto-diff).
- Constructor `ActorNetwork(input_size=5, hidden_sizes=(128, 64), n_actions=4, seed=None)`.
- Layers stored as a list of `(W, b)`; init `W = randn*0.1`, `b = zeros`.
  If `seed` given, call `np.random.seed(seed)` first.
- Expose `.params = list(self.layers)` and a `forward(x)` returning logits
  (no softmax). Hidden layers ReLU, final layer linear.

### 3) `critic_network.py`
A separate **NumPy-only** MLP class `CriticNetwork` (no torch/sklearn/auto-diff).
- Constructor `CriticNetwork(input_size=5, hidden_sizes=(128, 64), seed=None)`.
- Same weight-init convention and `.params`; `forward(x)` returns a scalar float
  `V(s)`. Hidden layers ReLU, final layer linear scalar.

### 4) `train_actor_critic.ipynb`
A Jupyter notebook that:
1. Imports `ActorNetwork` from `actor_network` and `CriticNetwork` from
   `critic_network` (kept as **separate modules**, not combined).
2. Imports the `WumpusWorldEnv` from `wumpus_env`.
3. Implements a standard **Actor-Critic (A2C)** training loop entirely with
   **hand-written NumPy** backprop (NO torch/sklearn/auto-diff):
   - Sample an action from the actor's softmax categorical distribution.
   - Compute **advantage** = `clip(r/REWARD_SCALE + gamma*V(s') - V(s), -10, 10)`
     (reward scaled by `REWARD_SCALE=100` to keep the bootstrapped target stable).
   - **Critic update**: MSE of `advantage` via manual backprop through the MLP.
   - **Actor update**: policy gradient `-advantage*(probs - onehot)` plus a correct
     entropy gradient `-ENTROPY_COEF*probs*(log probs + H)`; clip `d_logits` to ±5.
   - Use plain SGD (no momentum) with per-parameter gradient clipping (`±1`) and
     hard weight clipping (`±20`) to prevent value-network divergence.
4. Hyperparameters: `lr_actor=1e-3`, `lr_critic=1e-3`, `gamma=0.99`,
   `reward_scale=100.0`, `adv_clip=10.0`, `grad_clip=1.0`,
   `max_steps=50`, `episodes=2000`, `entropy_coef=0.02`.
5. Logs cumulative reward per episode (prints avg of last 200 every 200).
6. Plots the learning curve (episode reward + moving avg, window=100).
7. Prints the final trained policy as a deterministic best-action policy grid.
8. Runs one greedy sample episode on a fresh seed map and prints step outcomes.

## Requirements
- Python 3, NumPy, Matplotlib, Jupyter. **No torch, no sklearn, no auto-diff
  library** — backprop is written by hand using numpy arrays.
- Do NOT merge the actor and critic into one combined network/class.
- Any AI reproducing this prompt must generate the same files and same logic.

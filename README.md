# One-Step Actor–Critic in a 4×4 Wumpus World

Manual / from-scratch implementation of a **One-Step Actor–Critic** reinforcement
learning agent that navigates a randomly generated 4×4 Wumpus World grid to
collect the gold while avoiding pits and the Wumpus. Everything — environment,
policy (actor) network, value (critic) network, forward/backward pass, and
optimizer — is written by hand in NumPy. **No PyTorch, TensorFlow, Keras,
sklearn, Gym dependency, or automatic differentiation is used.**

> **Vibecoded:** this project was built with [opencode](https://opencode.ai)
> (model `opencode/big-pickle`). See the
> [`prompts_and_sessions/`](#prompts--sessions) directory for the actual
> conversation and the prompts behind it.

## Requirements

- Python 3
- NumPy
- Matplotlib
- Jupyter (for the notebook)

## Usage

```sh
python train.py      # train the agent (15,000 episodes, ~100 s) -> saves results/
python evaluate.py   # evaluate on 200 unseen maps, produce plots + sample episodes
python tests.py      # gradient-direction, finite-difference, and environment tests
```

Training saves the learned weights and metrics into `results/`:
`actor_weights.npz`, `critic_weights.npz`, and `training_results.npz` plus three
plots (`learning_curve.png`, `success_curve.png`, `policy_value_map.png`).
Evaluation loads the trained weights and reports success / hazard / timeout
rates, average return, and steps on held-out test maps (seeds `10000–10199`, never
seen during training).

## Directory Structure

```
.
├── wumpus_env.py                 # Fully-observable 4×4 Wumpus World env (Gym-compatible interface, no Gym)
├── actor_network.py              # NumPy MLP policy network (64 → 128 → 64 → 4 logits)
├── critic_network.py             # NumPy MLP value network (64 → 128 → 64 → 1, scalar V(s))
├── nn.py                         # Hand-written forward/backward, vanilla SGD, softmax, sampling
├── train.py                      # One-Step Actor–Critic training (TD learning, reward shaping, SGD)
├── evaluate.py                   # Metrics on unseen maps, plots, sample episodes
├── tests.py                      # Gradient-direction + finite-difference + env correctness tests
├── train_actor_critic.ipynb      # Annotated training notebook (same algorithm as train.py)
├── report.md                     # Full report (mechanism, formulas, results, hyperparameters)
├── results/                      # Learned weights + plots + training metrics
│   ├── actor_weights.npz         #   saved actor parameters
│   ├── critic_weights.npz        #   saved critic parameters
│   ├── training_results.npz      #   episode returns + eval success curves
│   ├── learning_curve.png        #   Plot 1: return + moving average
│   ├── success_curve.png         #   Plot 2: success rate during training
│   └── policy_value_map.png      #   Plot 3: deterministic policy + value map
└── prompts_and_sessions/         # Exported opencode sessions + the prompts that drove them
    ├── session-actor-critic.json #   Record of the session that built this project
    ├── prompt.md                 #   Reproducible prompt that generated this codebase
    └── guidelines.md             #   Rewrite guidelines the code was built against
```

## How It Works

The state is a **64-dimensional fully-observable Markov state**: the entire
4×4 map encoded as 4 channels (agent, gold, wumpus, pits) × 16 cells,
flattened. This lets the agent generalize to unseen maps rather than memorize a
single gold position.

Each environment step produces one update:

```
State → Actor → Action → Environment → Reward + Next State
      → Critic → TD Target → TD Error → Actor/Critic Update
```

Using a one-step TD target `y = r' + γ(1−done)V(s')` and TD error
`δ = y − V(s)` (a one-step estimator of the advantage), the **actor** is updated
with the policy gradient `−log π(a|s)·δ` plus an entropy bonus, and the
**critic** with `½(y − V)²`, backpropagated by hand. Both use vanilla SGD
(no momentum) with gradient clipping. A potential-based reward-shaping bonus
guides learning without changing the optimal policy.

## Prompts & Sessions

`prompts_and_sessions/` stores the **opencode session export** and the **prompt
files** that drove the work, so the codebase is fully reproducible:

| File | What it is | What it is for |
|------|-----------|----------------|
| `session-actor-critic.json` | Exported opencode session (agent, model, messages, diffs, metadata) | Record of the actual conversation that built and reworked this project; can be re-imported into opencode and resumed |
| `prompt.md` | The original task specification prompt for an AI assistant | Gives an AI that prompt alone the requirement to produce functionally identical code — regenerates the whole project from scratch |
| `guidelines.md` | Detailed rewrite instructions for the project (in Vietnamese) | Specifies the corrected One-Step Actor–Critic design: NumPy-only, fully-observable 64-dim state, correct TD/advantage math, hand-written backprop, metrics and report requirements |

The session JSON is a snapshot of a conversation with opencode (agent, model,
messages, file diffs, metadata). The prompt files are the inputs that spawned it:
`prompt.md` is a self-contained spec for reproducing the codebase, and
`guidelines.md` documents the mathematical-correctness rewrite applied afterward
(e.g. the critic gradient sign `∂L/∂V = −δ`, the distinction between TD error and
advantage, and the strict train/test seed split).

### Importing the session back into opencode

Install opencode, then in the project directory:

```sh
opencode import prompts_and_sessions/session-actor-critic.json
```

The session is restored into the local opencode database and can be resumed with:

```sh
opencode session list               # find the imported session ID
opencode -s <session-id>            # continue a specific session
opencode -s <session-id> --fork     # continue as a new fork
```

Or use the TUI: run `opencode`, press `<leader> s` (session switcher) and pick the imported session.

### Exporting a session (for reference)

To save/share a session from the CLI:

```sh
opencode session list               # list sessions and IDs
opencode export <session-id>        # export JSON (prints to stdout)
opencode export <session-id> > prompts_and_sessions/session-<name>.json
```

The TUI also supports exporting the current session (`/export` or keybind `gx`). Use
`opencode export <session-id> --sanitize` to redact transcript/file data when sharing
publicly.
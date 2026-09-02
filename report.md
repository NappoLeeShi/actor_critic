# One-Step Actor–Critic in a 4×4 Wumpus World

A NumPy-only, hand-implemented **One-Step Actor–Critic** agent for a **fully-observable** variant of the Wumpus World (4×4 grid). Every piece — environment, policy (actor) network, value (critic) network, forward/backward pass, and optimizer — is written by hand in NumPy. No PyTorch, TensorFlow, Keras, sklearn, Gym dependency, or automatic differentiation is used. Backpropagation is derived and implemented manually.

> This is **One-Step Actor–Critic**, *not* A2C. A2C in the strict sense collects a full rollout and updates once per batch; here we perform one parameter update per environment step. We therefore use the accurate name "One-Step Actor–Critic."

The mechanism demonstrated:

$$
State \rightarrow Actor \rightarrow Action \rightarrow Environment \rightarrow Reward + Next~State \rightarrow Critic \rightarrow TD~Target \rightarrow TD~Error \rightarrow Actor/Critic~Update
$$

---

## 1. Files

| File | Role |
|------|------|
| `wumpus_env.py` | Fully-observable 4×4 Wumpus World environment (NumPy, no Gym) |
| `actor_network.py` | NumPy MLP policy network `64 → 128 → 64 → 4` (logits) |
| `critic_network.py` | NumPy MLP value network `64 → 128 → 64 → 1` (`V(s)`) |
| `nn.py` | Hand-written `forward_with_cache`, `backward`, `sgd_step`, `softmax` |
| `train.py` | One-Step Actor–Critic training script (vanilla SGD, train/test split) |
| `evaluate.py` | Test on unseen maps, all metrics, 3 plots, sample episodes |
| `tests.py` | Gradient direction + finite-difference + environment correctness tests |
| `train_actor_critic.ipynb` | Annotated training notebook (same algorithm as `train.py`) |
| `report.md` | This report |

---

## 2. Environment: Wumpus World 4×4

Each episode generates a random map with:

- 1 **Agent** starting at a fixed position `(0,0)`.
- 1 **Gold** (`+100`).
- 1 **Wumpus** (`−100`).
- 2–4 **Pits** (`−100`).

Gold/Wumpus/Pits are placed randomly but never on the agent start cell.

**Actions:**
```
0 = Up, 1 = Down, 2 = Left, 3 = Right
```

**Deterministic transition.** If the agent is at `(x,y)` and chooses `Right`, it moves to `(x+1,y)` provided that is inside the grid; otherwise it stays in place. The transition probability is essentially deterministic:

$$
P(s' \mid s, a) \in \{0, 1\}.
$$

An action that would push the agent outside the grid leaves it in place and still yields the normal move reward (`−1`).

**Episode termination:** gold reached, pit reached, wumpus reached, or `MAX_STEPS=50` exceeded.

**Environment rewards (raw):**

```
normal move = -1
gold        = +100
pit         = -100
wumpus      = -100
```

**Training reward vs raw reward.** The learner optimizes a *training reward* that equals the raw environment reward plus a **potential-based reward-shaping** bonus. Reward shaping is explained in §7. The metrics reported in §9 use the **raw** environment reward only.

---

## 3. State Representation (fully observable)

We use a **fully-observable Markov state**: the entire 4×4 map is encoded as 4 one-hot/multi-hot channels, flattened into a 64-dimensional vector:

| Channel | Cells | Contents |
|---------|-------|----------|
| 0 (agent) | 16 | `1` at the agent's current cell |
| 1 (gold)  | 16 | `1` at the gold's cell |
| 2 (wumpus)| 16 | `1` at the wumpus's cell |
| 3 (pits)  | 16 | `1` on every pit cell |

$$
4 \times 4 \times 4 = 64 \text{ features.}
$$

This state is sufficient to **uniquely identify the current situation**: for the same `(x,y)` position, two random maps that differ in where Gold/Pits/Wumpus are placed will produce different state vectors. This makes it meaningful to train on many random maps and test on maps the agent has **never seen**. Because the map is fully visible, this is a *fully-observable Wumpus variant*, not the classic partially-observable Wumpus World.

### Why this satisfies the Markov property

The state contains all the information needed so that the next state and the expected future reward depend only on the current state plus the chosen action, not on the whole history:

$$
P(s_{t+1} \mid s_t, a_t, \text{history}) = P(s_{t+1} \mid s_t, a_t).
$$

In this deterministic environment, e.g. the agent at `(2,2)` choosing `Right` always transitions to `(3,2)` with probability `1`.

---

## 4. Actor Network

The actor is a NumPy MLP mapping state → 4 logits:

```
Input (64)
  ↓
Linear(64 → 128)   # W0, b0
  ↓
ReLU
  ↓
Linear(128 → 64)   # W1, b1
  ↓
ReLU
  ↓
Linear(64 → 4)     # W2, b2   (logits, no softmax in the layer)
  ↓
logits
```

The policy is obtained with a **numerically stable softmax**:

$$
\pi_\theta(a \mid s) = \operatorname{softmax}(z), \qquad
p_i = \frac{e^{z_i - \max z}}{\sum_j e^{z_j - \max z}}.
$$

During **training** we **sample** an action from this categorical distribution (`np.random.multinomial`), we do **not** use `argmax`. After training we use `argmax` to form the **deterministic policy**.

**Actor parameters** are the weights and biases of all three linear layers. They are **kept across episodes** (never reset).

---

## 5. Critic Network

The critic is a NumPy MLP mapping state → a scalar `V(s)`:

```
Input (64)
  ↓
Linear(64 → 128)
  ↓
ReLU
  ↓
Linear(128 → 64)
  ↓
ReLU
  ↓
Linear(64 → 1)
  ↓
V(s)
```

The final output is a **scalar linear output** (no softmax for the critic).

**Critic parameters** are the weights and biases, also kept across episodes.

---

## 6. ReLU Activation

Hidden layers use

$$
\operatorname{ReLU}(z) = \max(0, z),
$$

implemented as `np.maximum(z, 0)`, with derivative

$$
\operatorname{ReLU}'(z) =
\begin{cases}
1, & z > 0 \\
0, & z \le 0.
\end{cases}
$$

In backprop this is applied as `gradient *= (z > 0)`.

---

## 7. One-Step TD Learning

For each transition `(s_t, a_t, r_t, s_{t+1}, done)` the critic computes

$$
V_t = V_\phi(s_t).
$$

With a terminal state we set `V_{t+1} = 0`; otherwise `V_{t+1} = V_\phi(s_{t+1})`.

**TD Target** (one-step bootstrap):

$$
\boxed{y_t = r_t + \gamma\,(1 - done)\,V_\phi(s_{t+1})}
$$

> The TD target is **not** the prediction of the next state. It is the *current reward* plus the *discounted value prediction* of the next state.

**TD Error**:

$$
\boxed{\delta_t = y_t - V_\phi(s_t)}.
$$

**Advantage.** The true advantage is defined as

$$
A(s,a) = Q(s,a) - V(s).
$$

The TD error is **not** identical to the advantage by definition; it is used as a **one-step estimator of the advantage**:

$$
\boxed{\delta_t \approx A(s_t, a_t)}.
$$

We do **not** train a Q-network in this implementation.

### Reward shaping (training reward)

To help the sparse-reward navigation task, the training reward is

$$
r'_t = r_t + \gamma\,\Phi(s_{t+1}) - \Phi(s_t),
$$

with a potential function

$$
\Phi(s) = -k_a\,d(s,\text{gold}) + k_h\,d(s,\text{nearest~hazard}),
$$

where `d` is the Manhattan distance. Because this is **potential-based shaping**, it does **not** change the optimal policy; it only densifies the reward. Metrics in §9 report the **raw unshaped** return.

---

## 8. Losses, Gradients, Backpropagation, Optimizer

### Critic loss

$$
\boxed{L_{critic} = \tfrac{1}{2}\big(y_t - V_\phi(s_t)\big)^2}
$$

so that

$$
\frac{\partial L_{critic}}{\partial V} = V - y = -\delta.
$$

**Critical gradient sign.** Because the optimizer performs `θ ← θ − η·∇_θ L`, the backward pass for the critic must use

$$
\boxed{\frac{\partial L}{\partial V} = -\delta}.
$$

This is verified by a unit test (§10, `tests.py`): with prediction `2` and target `8`, the critic must update so that its prediction moves **toward 8**, never downward.

### Actor loss

Policy-gradient estimator:

$$
\boxed{L_{actor} = -\log \pi_\theta(a_t \mid s_t)\,\delta_t}
$$

with gradient w.r.t. the logits:

$$
\boxed{\frac{\partial L_{actor}}{\partial \text{logits}} = -\delta_t\,(p - \text{onehot}(a_t))}.
$$

The `sign` is then flipped by the SGD step. Checking the sign convention: if `δ > 0` the just-chosen action should be **reinforced**; if `δ < 0` it should be **discouraged**. In code we pass `d_logits = δ·(p − onehot)` to `backward`, so that the update `W ← W − η·dW` raises the chosen action's probability when `δ > 0`. Verified by unit test (§10).

### Entropy bonus

To avoid premature policy collapse we add an entropy bonus `ENTROPY_COEF·H(π)` with

$$
H(\pi) = -\sum_a p(a)\log p(a).
$$

Its analytic gradient `∂H/∂logits = p·(\log p + H)` is included with the correct sign, and `ENTROPY_COEF = 0.05` (tuned by experiment, not assumed optimal).

### Backpropagation

Each training step follows:

```
forward
  → cache (z, activation, input) per layer
  → loss
  → backward from output to input
  → gradients dW, db
  → parameter update
```

The core rules (in `nn.py`):

```python
dW = np.outer(gradient, input)   # dW
db = gradient                    # db
gradient_to_previous = W.T @ gradient
gradient *= (z > 0)              # ReLU derivative
```

Gradient correctness is verified against **finite differences** in `tests.py`.

### Optimizer: Vanilla SGD (no momentum)

The chosen optimizer is **vanilla SGD**:

$$
W \leftarrow W - \eta\,\nabla_W L, \qquad b \leftarrow b - \eta\,\nabla_b L.
$$

There is **no velocity accumulation and no momentum** — this satisfies the guidelines' "parallel SGD, no momentum" requirement consistently. We use separate learning rates for actor and critic (`LR_ACTOR = LR_CRITIC = 1e-3`) and clip gradients per-parameter to `±20` for stability. The optimizer is **not** called Adam.

---

## 9. Training Setup and Train/Test Split

- Train for `EPISODES = 15000`, each episode with a fresh random map (training map seeds `0–9999`).
- Actor and Critic weights are **initialized once** and **never reset** between episodes.
- `MAX_STEPS = 50`, `GAMMA = 0.99`, `ENTROPY_COEF = 0.05`, `LR_ACTOR = LR_CRITIC = 1e-3`.

Pseudo-code:

```
initialize_actor(); initialize_critic()
for episode in range(EPISODES):
    env.reset_random_map()
    state = env.get_state()
    while not done:
        logits = actor.forward(state)
        probs = softmax(logits); action = sample(probs)
        next_state, reward, done, info = env.step(action)
        value = critic.forward(state)
        next_value = 0 if done else critic.forward(next_state)
        target = shaped_reward + gamma * next_value
        td_error = target - value
        update_critic(td_error)   # backward with -td_error
        update_actor(td_error)    # backward with td_error*(p-onehot)
        state = next_state
```

**Strict train/test split.** Training uses random map seeds `0–9999`. Evaluation uses **held-out test seeds `10000–10199`** that are never seen during training. We therefore report **training performance** and **test performance on unseen maps** separately, never merged into one number.

---

## 10. Correctness & Validation (`tests.py`)

`tests.py` checks:

1. **Critic gradient direction** — with `prediction=2, target=8` the critic moves toward 8 (up, not down).
2. **Actor gradient direction** — `δ>0` increases the chosen action's probability; `δ<0` decreases it.
3. **Critic finite-difference gradient check** (with ReLU-boundary rows excluded).
4. **Actor finite-difference gradient check** on the scalar loss.
5. **Environment correctness** — 64-dim state, deterministic movement, boundary handling, random maps, train/test seed separation.
6. **Softmax numerical stability** — no NaN/Inf, probabilities sum to 1, non-negative, valid for extreme logits.
7. **Terminal handling** — next value fixed to `0` when `done`.

All tests pass. Additional runtime checks confirm no NaN/Inf in weights and probabilities sum to 1.

---

## 11. Experimental Results (real run)

The training run took ~100 s and produced the following real numbers.

### Learning curve

Episode return improves from about `−64` early in training to positive values, with the moving average rising steadily (Plot 1). Success rate on held-out maps rises from `0%` to about `44%` over 15000 episodes (Plot 2):

| Training episode | Moving-avg return | Held-out success rate |
|------------------|-------------------|------------------------|
| 1,000  | −50 | 14% |
| 3,000  | −29 | 33% |
| 5,000  | −13 | 39% |
| 7,000  | −3  | 35% |
| 9,000  | +12 | 43% |
| 12,000 | +11 | 29% |
| 15,000 | +14 | 44% |

### Final test performance on 200 unseen maps (seeds 10000–10199)

These numbers come from `evaluate.py` running the trained deterministic policy on maps the agent has never seen:

| Metric | Value |
|--------|-------|
| **Success rate** (gold wins) | **46.5%** (93/200) |
| **Hazard rate** (pit + wumpus deaths) | **7.5%** (pit 4.0%, wumpus 3.5%) |
| **Timeout rate** | **46.0%** |
| **Average episode return** | **+15.27** |
| **Average steps (all episodes)** | 24.27 |
| **Average return on successes** | **+98.85** |
| **Average steps on successes** | **2.15** |

For comparison: a uniform-random policy achieves ~22% success on the same held-out maps, and the theoretical maximum success (when the gold is reachable while avoiding known hazards) is ~84.5%.

**Interpretation.** The agent clearly **learns** versus its random baseline and versus the early-training policy:
- Success rate rises from 0–14% to ~44–46%, roughly **double the random-policy rate**.
- Average return rises from about `−64` to positive `+15`, i.e. episodes end much sooner (survival is much more efficient).
- When the agent does find gold, it does so in **~2.15 steps on average** with near-maximal return `+98.85`, showing it has learned to navigate directly to the goal.

The success rate (~46%) is well below the theoretical ceiling (~84%). This gap is analyzed in §12.

### Sample episodes (real, from `evaluate.py`)

Two example **successes** on unseen maps (deterministic policy):

```
Map seed 10004  (gold reached):
  Action sequence: Down -> Right -> Down -> Right
  Total reward = 97   Steps = 4   Outcome = GOLD

Map seed 10005:
  Action sequence: Down -> Right -> Right -> Right
  Total reward = 97   Steps = 4   Outcome = GOLD
```

and two **failures** (real, not cherry-picked), where the policy times out:

```
Map seed 10000:
  Action sequence: Down -> Down -> Down -> ... (50 steps)
  Total reward = -50   Steps = 50   Outcome = MAX_STEPS

Map seed 10001:
  Action sequence: Down -> Left -> Left -> ... (50 steps)
  Total reward = -50   Steps = 50   Outcome = MAX_STEPS
```

In the seed-10001 failure the gold is at `(1,2)` (right next to the agent) but the agent repeatedly walks `Left` against the left wall; the full-observability policy did not always select the shortest/obvious path, leaving room for improvement.

### Plots

- **Plot 1** `learning_curve.png` — episode return + moving average.
- **Plot 2** `success_curve.png` — success rate vs training progress.
- **Plot 3** `policy_value_map.png` — deterministic (argmax) policy arrows + critic value map `V(s)` on one held-out test map. The value map shows higher `V(s)` in states near the gold and lower `V(s)` in dangerous states.

> The policy arrows are produced by **greedy `argmax`**, i.e. the **deterministic policy after training**.

---

## 12. Analysis: Why success rate is (~46%) not (~84%)

The remaining gap is expected for this setup and is not hidden:

1. **Sparse, delayed credit.** Gold gives `+100` but only at a single terminal state. Discovering it requires stochastically reaching it during training; many early episodes simply time out at `−50` before ever observing a gold reward, so the value/advantage signal for "go toward gold" is weak relative to the "avoid deaths" signal. This is the well-known exploration burden of sparse-reward navigation.
2. **One-step TD bootstrap.** We deliberately use a pure **one-step** TD update (per the project's educational goal). One-step TD with a high discount propagates the gold bonus back through the network slowly compared to Monte-Carlo or n-step returns, so distant gold states are learned last.
3. **Random, adversarial maps.** Only ~84.5% of maps have a gold reachable without crossing a known hazard. On the remaining maps success is impossible. Even on reachable maps, the MLP must generalize the *pathfinding* rule across arbitrary configurations of pits/wumpus — a hard function for a small feed-forward net trained online with high-variance policy gradients.
4. **High policy-gradient variance.** Online, single-sample updates give a noisy advantage `δ`, slowing and periodically destabilizing learning (visible in the non-monotonic success curve, e.g. episodes 10500–13000).

The agent nonetheless demonstrates genuine learning (positive return, ~2× random success, near-optimal path length on successes), which is the point of the educational exercise. We do **not** claim full convergence or that the model "solves" the task; the honest result is a modest but clear improvement over chance with a correct, explainable implementation.

---

## 13. Parameters vs Hyperparameters

**Parameters** (learned by gradient descent using the training data):

- Actor weights and biases `W0,b0,W1,b1,W2,b2`.
- Critic weights and biases `W0,b0,W1,b1,W2,b2`.

**Hyperparameters** (set by the experimenter, not learned):

| Hyperparameter | Value |
|----------------|-------|
| Learning rate (actor) | `1e-3` |
| Learning rate (critic) | `1e-3` |
| Discount factor `γ` | `0.99` |
| Number of layers / hidden size | 3 linear layers, hidden 128 & 64 |
| Activation | ReLU |
| Entropy coefficient | `0.05` |
| Number of episodes | `15000` |
| Max steps | `50` |
| Shaping coefficients | `k_a=1.0`, `k_h=0.5` |
| Gradient clip | `20` |

---

## 14. Bellman Equation

The state-value Bellman equation for a policy `π`:

$$
V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \big[ R(s, a, s') + \gamma\, V^\pi(s') \big].
$$

In words: the value of the current state equals the **expected current reward plus the discounted expected future value**, taken over the actions the policy chooses and the states the environment transitions to. The one-step TD target we use,

$$
y_t = r_t + \gamma\,V(s_{t+1}),
$$

is exactly the **one-step bootstrap approximation** of this Bellman idea: it replaces the full expectation with a single sampled transition and a single bootstrapped value.

---

## 15. Final Validation Checklist

**Correctness**
- [x] Critic gradient has the correct sign (unit tested).
- [x] Actor gradient has the correct sign (unit tested).
- [x] Terminal next value = 0.
- [x] Softmax numerically stable.
- [x] No NaN (checked).
- [x] No invalid probabilities (checked).
- [x] Probabilities sum ≈ 1 (checked).
- [x] Weights update across episodes.
- [x] Actor/Critic not reset after each episode.

**Environment**
- [x] Movement deterministic.
- [x] Rewards correct.
- [x] Episode termination correct.
- [x] Random training maps (seeds 0–9999).
- [x] Unseen test maps (seeds 10000–10199).

**Evaluation**
- [x] Success rate, hazard rate, timeout rate, average return, average steps.
- [x] Training curve.
- [x] Test performance on unseen maps (reported separately from training).

**Report**
- [x] Called One-Step Actor–Critic (not A2C).
- [x] Does not call Q-learning the "Critic".
- [x] Does not claim `Q(s,a) = V(s)`.
- [x] Does not claim TD error ≡ advantage by definition; explains TD error ≈ advantage.
- [x] Explains TD target.
- [x] Explains Bellman.
- [x] Explains state representation.
- [x] Explains backpropagation.
- [x] Distinguishes parameter vs hyperparameter.
- [x] Reports numbers from the real run (not fabricated).

---

## 16. Deliverables

1. Corrected environment code (`wumpus_env.py`).
2. Corrected actor network (`actor_network.py`).
3. Corrected critic network (`critic_network.py`).
4. Hand-written `nn.py` (forward/backward/vanilla SGD/softmax).
5. Training script (`train.py`) and annotated notebook (`train_actor_critic.ipynb`).
6. Evaluation script (`evaluate.py`) and unit tests (`tests.py`).
7. Result plots (`results/learning_curve.png`, `success_curve.png`, `policy_value_map.png`).
8. Sample episodes on unseen maps (printed by `evaluate.py`).
9. This report, updated to match the real code and results.

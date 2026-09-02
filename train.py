"""One-Step Actor-Critic training for a fully-observable 4x4 Wumpus World.

NumPy-only. Trains Actor (policy) and Critic (value) networks via one-step TD
learning with vanilla SGD (no momentum), on many random maps with strict
train/test seed separation.

Training reward (used by the learner) = raw environment reward + a
potential-based reward-shaping bonus that guides the agent toward the gold and
away from hazards:

    Phi(s) = -ka * manhattan_dist(agent, gold) + kh * manhattan_dist(agent, nearest_hazard)
    shaped_r = r + gamma * Phi(s') - Phi(s)

Because this is potential-based shaping, it does not change the optimal policy
but accelerates learning in this sparse-reward navigation task.

Run:  python train.py
"""

import os
import random

import numpy as np

from actor_network import ActorNetwork
from critic_network import CriticNetwork
from nn import forward_with_cache, backward, sgd_step, softmax, sample
from wumpus_env import WumpusWorldEnv, STATE_DIM, MAX_STEPS, ACTION_NAMES

# --- Hyperparameters (tuned by experiment; see report) -------------------
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.99
ENTROPY_COEF = 0.05
SHAPE_KA = 1.0   # potential weight: pull toward gold
SHAPE_KH = 0.5   # potential weight: push away from hazards
MAX_STEPS = 50
EPISODES = 15000
EVAL_EVERY = 500
EVAL_EPISODES = 100
GRAD_CLIP = 20.0  # clip per-parameter gradient magnitude for stability

# Training seeds 0-9999 (map seeds = ep % 10000); test seeds 10000-10199
TRAIN_START_SEED = 0
TEST_SEED_BASE = 10000


def potential(env, ka=SHAPE_KA, kh=SHAPE_KH):
    """Potential function Phi(s) for reward shaping (bounded by grid size)."""
    ax, ay = env.agent_pos
    gx, gy = env.gold_pos
    d_gold = abs(ax - gx) + abs(ay - gy)
    hazards = env.pit_positions + [env.wumpus_pos]
    d_hazard = min((abs(ax - hx) + abs(ay - hy)) for hx, hy in hazards)
    return -ka * d_gold + kh * d_hazard


def sgd_step_clip(net, grads, lr, grad_clip):
    """Vanilla SGD with per-parameter gradient clipping (no momentum)."""
    for i, (dW, db) in enumerate(grads):
        W, b = net.layers[i]
        dW = np.clip(dW, -grad_clip, grad_clip)
        db = np.clip(db, -grad_clip, grad_clip)
        W[:] = W - lr * dW
        b[:] = b - lr * db
        net.layers[i] = (W, b)


def train(episodes=EPISODES, verbose=True):
    random.seed(0)
    np.random.seed(0)

    actor = ActorNetwork(input_size=STATE_DIM, n_actions=4, seed=0)
    critic = CriticNetwork(input_size=STATE_DIM, seed=0)
    env = WumpusWorldEnv()

    episode_rewards = []          # raw env reward per episode (for reporting)
    success_flags = []
    eval_returns = []
    eval_success = []
    eval_ep_numbers = []

    for episode in range(episodes):
        seed = (TRAIN_START_SEED * 10000 + episode) % 10000
        state = np.asarray(env.reset(seed=seed), dtype=np.float32)
        done = False
        total_reward = 0.0
        gold = False
        term = None
        pot_prev = potential(env)

        while not done:
            logits, actor_acts = forward_with_cache(actor, state)
            probs = softmax(logits)  # numerically stable
            action = sample(probs)   # sample from categorical (no argmax in training)

            next_state_np, reward, done, _truncated, info = env.step(action)
            total_reward += reward
            next_state = np.asarray(next_state_np, dtype=np.float32)
            if info["outcome"] == "gold":
                gold = True
                term = "gold"
            elif info["outcome"] == "pit":
                term = "pit"
            elif info["outcome"] == "wumpus":
                term = "wumpus"
            elif info["outcome"] == "max_steps":
                term = "timeout"

            # Reward shaping (potential-based): does not change optimal policy
            pot_next = 0.0 if done else potential(env)
            shaped_reward = reward + (GAMMA * pot_next - pot_prev)

            # --- Critic values & TD target ---
            value = critic.forward(state)
            next_value = 0.0 if done else critic.forward(next_state)
            # One-Step TD target: y = r' + gamma*(1-done)*V(s')
            target = shaped_reward + GAMMA * next_value
            td_error = target - value  # delta_t = y - V(s)

            # --- Critic update -------------------------------------------
            # L = 0.5*(y - V)^2 ; dL/dV = V - y = -td_error
            _, critic_acts = forward_with_cache(critic, state)
            grad_v = -np.array([td_error], dtype=np.float32)
            critic_grads = backward(critic, critic_acts, grad_v)
            sgd_step_clip(critic, critic_grads, LR_CRITIC, GRAD_CLIP)

            # --- Actor update --------------------------------------------
            # L = -log(pi(a|s))*delta ; dL/dlogits = delta*(p - onehot)
            onehot = np.zeros_like(probs)
            onehot[action] = 1.0
            d_logits = td_error * (probs - onehot)
            # entropy bonus: maximize H -> subtract ENTROPY_COEF*dH/dlogits
            entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
            d_logits = d_logits - ENTROPY_COEF * probs * (np.log(probs + 1e-8) + entropy)
            d_logits = np.clip(d_logits, -5.0, 5.0).astype(np.float32)
            actor_grads = backward(actor, actor_acts, d_logits)
            sgd_step_clip(actor, actor_grads, LR_ACTOR, GRAD_CLIP)

            state = next_state
            pot_prev = pot_next if not done else pot_prev

        episode_rewards.append(total_reward)
        success_flags.append(1.0 if gold else 0.0)

        if (episode + 1) % EVAL_EVERY == 0:
            sr, ar = evaluate(actor, env, EVAL_EPISODES)
            eval_returns.append(ar)
            eval_success.append(sr)
            eval_ep_numbers.append(episode + 1)
            if verbose:
                print(
                    f"Ep {episode+1:5d}/{episodes}  "
                    f"avg_r(last{EVAL_EVERY})={np.mean(episode_rewards[-EVAL_EVERY:]):7.2f}  "
                    f"eval_sr={sr*100:5.1f}%  eval_avg_r={ar:7.2f}"
                )

    return {
        "actor": actor,
        "critic": critic,
        "episode_rewards": np.array(episode_rewards),
        "success_flags": np.array(success_flags),
        "eval_returns": np.array(eval_returns),
        "eval_success": np.array(eval_success),
        "eval_ep_numbers": np.array(eval_ep_numbers),
    }


def evaluate(actor, env, n_episodes=200):
    """Evaluate deterministic (argmax) policy on held-out unseen maps.

    Uses test seeds 10000..10000+n-1 which are NEVER used during training.
    """
    returns = []
    n_gold = 0
    for i in range(n_episodes):
        seed = TEST_SEED_BASE + i
        state = np.asarray(env.reset(seed=seed), dtype=np.float32)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < MAX_STEPS:
            logits = actor.forward(state)
            action = int(np.argmax(logits))  # deterministic policy after training
            state_np, reward, done, _trunc, info = env.step(action)
            state = np.asarray(state_np, dtype=np.float32)
            total += reward
            steps += 1
            if info["outcome"] == "gold":
                n_gold += 1
        returns.append(total)
    success_rate = n_gold / n_episodes
    avg_return = float(np.mean(returns))
    return success_rate, avg_return


def save_results(result, outdir="results"):
    os.makedirs(outdir, exist_ok=True)

    def pack_layers(net):
        return {f"W{i}": W for i, (W, b) in enumerate(net.layers)} | {
            f"b{i}": b for i, (W, b) in enumerate(net.layers)
        }

    np.savez(
        os.path.join(outdir, "actor_weights.npz"), **pack_layers(result["actor"])
    )
    np.savez(
        os.path.join(outdir, "critic_weights.npz"), **pack_layers(result["critic"])
    )
    np.savez(
        os.path.join(outdir, "training_results.npz"),
        episode_rewards=result["episode_rewards"],
        success_flags=result["success_flags"],
        eval_returns=result["eval_returns"],
        eval_success=result["eval_success"],
        eval_ep_numbers=result["eval_ep_numbers"],
    )
    print(f"Saved weights + results to {outdir}/")


if __name__ == "__main__":
    result = train(episodes=EPISODES)
    save_results(result)
    env = WumpusWorldEnv()
    sr, ar = evaluate(result["actor"], env, 200)
    print(f"\nFinal test (200 unseen maps): success rate = {sr*100:.1f}%, "
          f"avg return = {ar:.2f}")

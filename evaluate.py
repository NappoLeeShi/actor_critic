"""Evaluation of the trained One-Step Actor-Critic on unseen test maps.

Loads the trained Actor/Critic weights saved by train.py, evaluates on held-out
maps (test seeds 10000+), reports all required metrics, produces the four
required plots, and prints sample episodes.

Run:  python evaluate.py
Outputs plots to results/ (learning_curve.png, success_curve.png,
policy_map.png, value_map.png).
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from actor_network import ActorNetwork
from critic_network import CriticNetwork
from nn import softmax
from wumpus_env import WumpusWorldEnv, STATE_DIM, MAX_STEPS, ACTION_NAMES

RESULT_DIR = "results"
TEST_SEED_BASE = 10000
N_TEST = 200


def build_network(cls, weights_file, **kwargs):
    net = cls(**kwargs)
    d = np.load(weights_file)
    new_layers = []
    for i in range(len(net.layers)):
        W = d[f"W{i}"]
        b = d[f"b{i}"]
        new_layers.append((W, b))
    net.layers = new_layers
    net.params = list(net.layers)
    return net


def run_episode(actor, env, seed):
    """Run one deterministic (argmax) episode. Returns (outcome, steps, rewards, actions, states)."""
    state_np = env.reset(seed=seed)
    state = np.asarray(state_np, dtype=np.float32)
    done = False
    steps = 0
    actions = []
    rewards = []
    outcome = "move"
    states = [env.agent_pos]
    while not done and steps < MAX_STEPS:
        logits = actor.forward(state)
        action = int(np.argmax(logits))  # deterministic policy after training
        state_np, reward, done, _trunc, info = env.step(action)
        state = np.asarray(state_np, dtype=np.float32)
        actions.append(action)
        rewards.append(reward)
        steps += 1
        states.append(env.agent_pos)
        outcome = info["outcome"]
    return outcome, steps, rewards, actions, states


def metrics(actor, env, n=N_TEST):
    """Compute all required metrics over n unseen test episodes."""
    n_gold = 0
    n_pit = 0
    n_wumpus = 0
    n_timeout = 0
    gold_returns = []
    all_returns = []
    gold_steps = []
    all_steps = []
    for i in range(n):
        seed = TEST_SEED_BASE + i
        outcome, steps, rewards, actions, states = run_episode(actor, env, seed)
        ret = sum(rewards)
        all_returns.append(ret)
        all_steps.append(steps)
        if outcome == "gold":
            n_gold += 1
            gold_returns.append(ret)
            gold_steps.append(steps)
        elif outcome == "pit":
            n_pit += 1
        elif outcome == "wumpus":
            n_wumpus += 1
        elif outcome == "max_steps":
            n_timeout += 1
    m = {
        "n": n,
        "n_gold": n_gold,
        "success_rate": n_gold / n,
        "hazard_rate": (n_pit + n_wumpus) / n,
        "pit_rate": n_pit / n,
        "wumpus_rate": n_wumpus / n,
        "timeout_rate": n_timeout / n,
        "avg_return": float(np.mean(all_returns)),
        "avg_steps": float(np.mean(all_steps)),
        "avg_gold_return": float(np.mean(gold_returns)) if gold_returns else float("nan"),
        "avg_gold_steps": float(np.mean(gold_steps)) if gold_steps else float("nan"),
        "gold_returns": np.array(gold_returns),
    }
    return m


def plot_learning_curve(episode_rewards, outpath):
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards, alpha=0.3, label="episode return")
    window = 100
    ma = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    plt.plot(ma, color="red", label=f"moving average (window={window})")
    plt.axhline(0, color="gray", ls="--", lw=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title("Plot 1: Learning Curve (raw episode return + moving average)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def plot_success_curve(ep_numbers, eval_success, outpath):
    plt.figure(figsize=(8, 4))
    plt.plot(ep_numbers, np.array(eval_success) * 100, marker="o", ms=4)
    plt.xlabel("Training episode")
    plt.ylabel("Success rate on held-out maps (%)")
    plt.title("Plot 2: Success Rate During Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def plot_policy_map(actor, critic, env, seed, outpath):
    """Plot 3: deterministic policy + value map on one test map."""
    env.reset(seed=seed)
    res = np.arange(4)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # --- Policy map ---
    ax = axes[0]
    symb = {0: ".", 1: "G", 2: "W", 3: "P"}
    for y in range(4):
        for x in range(4):
            cell = env.map[y, x]
            st = env.get_state_for_cell(x, y)
            action = int(np.argmax(actor.forward(st)))
            arrow = {"Up": "^", "Down": "v", "Left": "<", "Right": ">"}[ACTION_NAMES[action]]
            ax.text(x, -y, arrow, ha="center", va="center", fontsize=14)
            if cell == 0:
                color = "lightgray"
            elif cell == 1:
                color = "gold"
            elif cell == 2:
                color = "lightcoral"
            else:
                color = "olive"
            if (x, y) == env.start:
                color = "lightblue"
            rect = plt.Rectangle((x - 0.5, -y - 0.5), 1, 1, fill=True, facecolor=color, alpha=0.6)
            ax.add_patch(rect)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-3.6, 0.6)
    ax.set_aspect("equal")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_yticklabels(range(4)[::-1])
    ax.set_title("Deterministic policy (argmax) on a test map")
    ax.grid(True)

    # --- Value map ---
    ax = axes[1]
    vals = np.zeros((4, 4))
    for y in range(4):
        for x in range(4):
            st = env.get_state_for_cell(x, y)
            vals[y, x] = critic.forward(st)
    im = ax.imshow(vals, origin="upper", cmap="viridis")
    for y in range(4):
        for x in range(4):
            ax.text(x, y, f"{vals[y, x]:.1f}", ha="center", va="center", fontsize=10)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
    ax.set_title("Critic value map V(s) on the same test map")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()


def print_sample_episodes(actor, env):
    print("\n" + "=" * 60)
    print("SAMPLE EPISODES ON UNSEEN TEST MAPS (deterministic policy)")
    print("=" * 60)

    # Collect outcomes for seeds 10000..10199 to pick success & failure samples
    seeds_success = []
    seeds_fail = []
    for i in range(N_TEST):
        seed = TEST_SEED_BASE + i
        outcome, _, _, _, _ = run_episode(actor, env, seed)
        if outcome == "gold":
            seeds_success.append(seed)
        else:
            seeds_fail.append(seed)

    chosen = seeds_success[:2] + seeds_fail[:2]

    for seed in chosen:
        env.reset(seed=seed)
        print(f"\n--- Test map seed {seed} ---")
        outcome, steps, rewards, actions, states = run_episode(actor, env, seed)
        env.render()
        path_str = " -> ".join(ACTION_NAMES[a] for a in actions[:8])
        if len(actions) > 8:
            path_str += " ... "
        print(f"Action sequence: {path_str}")
        print(f"Total reward = {sum(rewards):.0f}  Steps = {steps}  Outcome = {outcome.upper()}")


def main():
    actor = build_network(ActorNetwork, os.path.join(RESULT_DIR, "actor_weights.npz"),
                          input_size=STATE_DIM, hidden_sizes=(128, 64), n_actions=4)
    critic = build_network(CriticNetwork, os.path.join(RESULT_DIR, "critic_weights.npz"),
                           input_size=STATE_DIM, hidden_sizes=(128, 64))
    env = WumpusWorldEnv()

    # --- Load training history for curves ---
    tr = np.load(os.path.join(RESULT_DIR, "training_results.npz"))
    episode_rewards = tr["episode_rewards"]
    eval_success = tr["eval_success"]
    eval_ep_numbers = tr["eval_ep_numbers"]

    # --- Metrics on unseen test maps ---
    m = metrics(actor, env, N_TEST)
    print("=" * 60)
    print(f"TEST PERFORMANCE on {N_TEST} unseen maps (seeds {TEST_SEED_BASE}-{TEST_SEED_BASE+N_TEST-1})")
    print("=" * 60)
    print(f"  Success rate      : {m['success_rate']*100:5.1f}%  ({m['n_gold']}/{m['n']} gold wins)")
    print(f"  Hazard rate       : {m['hazard_rate']*100:5.1f}%  (pit {m['pit_rate']*100:.1f}%, "
          f"wumpus {m['wumpus_rate']*100:.1f}%)")
    print(f"  Timeout rate      : {m['timeout_rate']*100:5.1f}%")
    print(f"  Avg episode return: {m['avg_return']:7.2f}")
    print(f"  Avg steps (all)   : {m['avg_steps']:5.2f}")
    if not np.isnan(m["avg_gold_return"]):
        print(f"  Avg return on successes : {m['avg_gold_return']:7.2f}")
        print(f"  Avg steps on successes  : {m['avg_gold_steps']:5.2f}")
    print(f"  (Theoretical max success if gold reachable: ~84.5%)")

    # --- Plots ---
    os.makedirs(RESULT_DIR, exist_ok=True)
    plot_learning_curve(episode_rewards, os.path.join(RESULT_DIR, "learning_curve.png"))
    plot_success_curve(eval_ep_numbers, eval_success, os.path.join(RESULT_DIR, "success_curve.png"))
    test_seed = TEST_SEED_BASE  # a test map
    plot_policy_map(actor, critic, env, test_seed, os.path.join(RESULT_DIR, "policy_value_map.png"))
    print(f"\nSaved plots to {RESULT_DIR}/")

    # --- Sample episodes ---
    print_sample_episodes(actor, env)


if __name__ == "__main__":
    main()

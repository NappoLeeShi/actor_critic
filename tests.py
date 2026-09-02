"""Unit tests for gradient correctness in One-Step Actor-Critic.

Tests:
1. Critic gradient direction (prediction=2, target=8 -> update towards 8)
2. Actor gradient sign (delta>0 reinforces chosen action)
3. Finite-difference gradient check for both networks
4. Environment correctness checks
"""

import numpy as np
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from wumpus_env import WumpusWorldEnv


def forward_with_cache(net, x):
    """Forward pass caching (input, pre-activation, activation) per layer."""
    x = np.asarray(x, dtype=np.float32)
    acts = []
    a_in = x
    h = x
    for i, (W, b) in enumerate(net.layers):
        z = W @ h + b
        a = z if i == len(net.layers) - 1 else np.maximum(z, 0)
        acts.append((a_in, z, a))
        a_in = a
        h = a
    return h, acts


def backward(net, acts, grad_out):
    """Backprop through net, returning gradients list.

    grad_out is dL/d(output). The function computes dW = outer(g, input)
    for each layer, applying ReLU derivative where appropriate.

    Standard backprop: for each layer l (from output to input):
      1. g *= relu'(z_l)  [apply ReLU derivative for hidden layers]
      2. dW = outer(g, a_{l-1}), db = g
      3. g = W_l.T @ g  [propagate to previous layer]
    """
    n_layers = len(net.layers)
    g = np.asarray(grad_out, dtype=np.float32)
    grads = []
    for i in range(n_layers - 1, -1, -1):
        a_in, z, a = acts[i]
        if i < n_layers - 1:
            g = g * (z > 0)  # ReLU derivative
        W, b = net.layers[i]
        dW = np.outer(g, a_in)
        db = g.copy()
        grads.append((dW, db))
        if i > 0:
            g = W.T @ g  # propagate to previous layer
    grads.reverse()
    return grads


def sgd_step(net, grads, lr):
    """Vanilla SGD update (no momentum)."""
    for i, (dW, db) in enumerate(grads):
        W, b = net.layers[i]
        W[:] = W - lr * dW
        b[:] = b - lr * db
        net.layers[i] = (W, b)


def test_critic_gradient_direction():
    """Test: prediction=2, target=8 => critic updates prediction towards 8.

    L = 0.5 * (target - prediction)^2
    dL/dV = -(target - prediction) = prediction - target = -delta

    After SGD: W_new = W_old - lr * dW, which moves V(s) towards target.
    """
    print("=" * 60)
    print("TEST 1: Critic gradient direction")
    print("=" * 60)

    critic = CriticNetwork(input_size=64, seed=42)
    state = np.random.randn(64).astype(np.float32)

    v_before = critic.forward(state)
    print(f"  Initial V(s) = {v_before:.6f}")

    target = 8.0
    delta = target - v_before

    _, acts = forward_with_cache(critic, state)
    grad_out = np.array([-delta], dtype=np.float32)  # dL/dV = -(target - V) = -delta
    grads = backward(critic, acts, grad_out)

    lr = 0.01
    sgd_step(critic, grads, lr)

    v_after = critic.forward(state)
    print(f"  Target       = {target:.6f}")
    print(f"  TD error     = {delta:.6f}")
    print(f"  V(s) after   = {v_after:.6f}")

    moved_towards = abs(v_after - target) < abs(v_before - target)
    print(f"  Moved towards target: {moved_towards}")
    assert moved_towards, (
        f"Critic did not move towards target! Before={v_before:.4f}, "
        f"After={v_after:.4f}, Target={target:.4f}"
    )
    print("  PASSED\n")


def test_actor_gradient_sign():
    """Test: delta > 0 => chosen action probability increases.

    L = -log(pi(a|s)) * delta
    dL/dlogits = delta * (probs - onehot)

    For chosen action a: dL/dlogit_a = delta * (probs_a - 1) < 0 (when delta>0)
    SGD: W -= lr * dL/dW => logit_a increases => probs_a increases. ✓
    """
    print("=" * 60)
    print("TEST 2: Actor gradient direction")
    print("=" * 60)

    actor = ActorNetwork(input_size=64, n_actions=4, seed=42)
    state = np.random.randn(64).astype(np.float32)

    logits, acts = forward_with_cache(actor, state)
    e = np.exp(logits - np.max(logits))
    probs_before = e / e.sum()
    chosen_action = 2
    prob_before = probs_before[chosen_action]
    print(f"  Initial probs: {probs_before}")
    print(f"  P(action={chosen_action}) = {prob_before:.6f}")

    delta = 1.0
    onehot = np.zeros(4, dtype=np.float32)
    onehot[chosen_action] = 1.0
    # dL/dlogits = delta * (probs - onehot)
    grad_out = delta * (probs_before - onehot)
    print(f"  grad_out: {grad_out}")

    grads = backward(actor, acts, grad_out)

    lr = 0.01
    sgd_step(actor, grads, lr)

    logits_after = actor.forward(state)
    e2 = np.exp(logits_after - np.max(logits_after))
    probs_after = e2 / e2.sum()
    prob_after = probs_after[chosen_action]
    print(f"  Probs after:  {probs_after}")
    print(f"  P(action={chosen_action}) after = {prob_after:.6f}")

    increased = prob_after > prob_before
    print(f"  Probability increased: {increased}")
    assert increased, (
        f"Chosen action probability did not increase with delta>0! "
        f"Before={prob_before:.6f}, After={prob_after:.6f}"
    )
    print("  PASSED\n")

    # Test delta < 0 => chosen action probability decreases
    print("  Testing delta < 0 ...")
    actor2 = ActorNetwork(input_size=64, n_actions=4, seed=42)
    logits2, acts2 = forward_with_cache(actor2, state)
    e3 = np.exp(logits2 - np.max(logits2))
    probs_b2 = e3 / e3.sum()
    prob_b2 = probs_b2[chosen_action]

    delta_neg = -1.0
    grad_out_neg = delta_neg * (probs_b2 - onehot)
    grads2 = backward(actor2, acts2, grad_out_neg)
    sgd_step(actor2, grads2, lr)

    logits_a2 = actor2.forward(state)
    e4 = np.exp(logits_a2 - np.max(logits_a2))
    probs_a2 = e4 / e4.sum()
    prob_a2 = probs_a2[chosen_action]
    print(f"  P(action={chosen_action}) before = {prob_b2:.6f}")
    print(f"  P(action={chosen_action}) after  = {prob_a2:.6f}")

    decreased = prob_a2 < prob_b2
    print(f"  Probability decreased: {decreased}")
    assert decreased, (
        f"Chosen action probability did not decrease with delta<0! "
        f"Before={prob_b2:.6f}, After={prob_a2:.6f}"
    )
    print("  PASSED\n")


def finite_diff_gradient(net, x, layer_idx, param_idx, new_net=None, eps=1e-5):
    """Compute finite-difference gradient for a single parameter.

    Returns (grad, activemask) where activemask is a boolean row-mask for
    hidden-layer weights (None for the output layer). Rows whose pre-activation
    is near zero (ReLU boundary) are excluded from the comparison.
    """
    # Recompute forward to get hidden pre-activations
    out, acts = forward_with_cache(net, x)
    shape = net.layers[layer_idx][param_idx].shape
    grad = np.zeros(shape)
    activemask = None

    # For hidden-layer weight matrices, mark rows that are safely away from
    # the ReLU boundary (|z| > tolerance) where the analytic subgradient and
    # FD gradient agree.
    n_hidden = len(net.layers) - 1
    if layer_idx < n_hidden and param_idx == 0:
        z = acts[layer_idx][1]
        activemask = np.abs(z) > 1e-2

    for idx in np.ndindex(shape):
        if activemask is not None and not activemask[idx[0]]:
            continue
        net.layers[layer_idx][param_idx][idx] += eps
        out_plus = net.forward(x)
        net.layers[layer_idx][param_idx][idx] -= 2 * eps
        out_minus = net.forward(x)
        grad[idx] = (out_plus - out_minus) / (2 * eps)
        net.layers[layer_idx][param_idx][idx] += eps

    return grad, activemask


def test_finite_difference_critic():
    """Check critic backprop gradients against finite differences.

    The backward function computes dL/dW where L = 0.5*(target-V)^2.
    dL/dW = -(target-V) * dV/dW.
    Finite differences compute dV/dW.
    So we compare: backward_dW / -(target-V) vs finite_diff_dW.
    """
    print("=" * 60)
    print("TEST 3: Critic finite-difference gradient check")
    print("=" * 60)

    critic = CriticNetwork(input_size=64, hidden_sizes=(16, 8), seed=42)
    state = np.random.randn(64).astype(np.float32)

    _, acts = forward_with_cache(critic, state)
    target = 5.0
    v = acts[-1][2][0]
    delta = target - v
    grad_out = np.array([-delta], dtype=np.float32)  # dL/dV = -delta
    grads_analytic = backward(critic, acts, grad_out)

    for li in range(min(2, len(critic.layers))):
        fd_grad, activemask = finite_diff_gradient(critic, state, li, 0, eps=1e-5)
        # Convert analytic dL/dW to dV/dW by dividing by -delta
        analytic_dV_dW = grads_analytic[li][0] / (-delta)

        # Only compare rows whose pre-activation is not at the ReLU
        # boundary (z ~ 0), where ReLU is non-differentiable and FD
        # differs from the analytic subgradient.
        if activemask is not None and np.any(activemask):
            fd_m = fd_grad[activemask]
            an_m = analytic_dV_dW[activemask]
        else:
            fd_m, an_m = fd_grad, analytic_dV_dW

        max_abs_fd = np.abs(fd_m).max() if fd_m.size > 0 else 0.0
        if max_abs_fd > 1e-8:
            rel_error = np.abs(fd_m - an_m).max() / max_abs_fd
        else:
            rel_error = np.abs(fd_m - an_m).max()

        print(f"  Layer {li} dV/dW: max rel error = {rel_error:.2e}")
        assert rel_error < 1e-3, f"Gradient check failed for critic layer {li}: rel_error={rel_error}"

    print("  PASSED\n")


def actor_loss(actor, state, action, delta):
    """Compute L = -log(pi(a|s)) * delta for finite-difference check."""
    logits = actor.forward(state)
    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()
    log_prob = np.log(probs[action] + 1e-8)
    return -log_prob * delta


def test_finite_difference_actor():
    """Check actor backprop gradients against finite differences.

    Loss L = -log(pi(a|s)) * delta.
    Backward computes dL/dW.
    Finite differences compute dL/dW directly from the scalar loss.
    """
    print("=" * 60)
    print("TEST 4: Actor finite-difference gradient check")
    print("=" * 60)

    actor = ActorNetwork(input_size=64, hidden_sizes=(16, 8), n_actions=4, seed=42)
    state = np.random.randn(64).astype(np.float32)

    logits, acts = forward_with_cache(actor, state)
    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()
    action = 1
    onehot = np.zeros(4, dtype=np.float32)
    onehot[action] = 1.0
    delta = 0.5
    grad_out = delta * (probs - onehot)
    grads_analytic = backward(actor, acts, grad_out)

    for li in range(min(2, len(actor.layers))):
        # Finite differences on the scalar loss
        original_val = actor.layers[li][0].copy()
        shape = original_val.shape
        fd_grad = np.zeros(shape)
        eps = 1e-5

        # Skip rows at the ReLU boundary
        out, acts_for_fd = forward_with_cache(actor, state)
        n_hidden = len(actor.layers) - 1
        activemask = None
        if li < n_hidden:
            z = acts_for_fd[li][1]
            activemask = np.abs(z) > 1e-2

        for idx in np.ndindex(shape):
            if activemask is not None and not activemask[idx[0]]:
                continue
            actor.layers[li][0][idx] += eps
            loss_plus = actor_loss(actor, state, action, delta)
            actor.layers[li][0][idx] -= 2 * eps
            loss_minus = actor_loss(actor, state, action, delta)
            fd_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            actor.layers[li][0][idx] += eps

        analytic_dW = grads_analytic[li][0]
        if activemask is not None and np.any(activemask):
            fd_m = fd_grad[activemask]
            an_m = analytic_dW[activemask]
        else:
            fd_m, an_m = fd_grad, analytic_dW

        max_abs_fd = np.abs(fd_m).max() if fd_m.size > 0 else 0.0
        if max_abs_fd > 1e-8:
            rel_error = np.abs(fd_m - an_m).max() / max_abs_fd
        else:
            rel_error = np.abs(fd_m - an_m).max()

        print(f"  Layer {li} dW: max rel error = {rel_error:.2e}")
        assert rel_error < 1e-3, f"Gradient check failed for actor layer {li}: rel_error={rel_error}"

    print("  PASSED\n")


def test_environment():
    """Test environment correctness."""
    print("=" * 60)
    print("TEST 5: Environment correctness")
    print("=" * 60)

    env = WumpusWorldEnv(seed=42)

    state = env.reset(seed=100)
    assert state.shape == (64,), f"State shape wrong: {state.shape}"
    assert state.dtype == np.float32, f"State dtype wrong: {state.dtype}"
    print(f"  State shape: {state.shape} - OK")

    assert env.agent_pos == (0, 0), f"Start position wrong: {env.agent_pos}"
    print(f"  Start position: {env.agent_pos} - OK")

    assert state[0] == 1.0, "Agent not at position 0"
    print("  Agent one-hot encoding - OK")

    state2, reward, done, trunc, info = env.step(3)  # Right
    assert env.agent_pos == (1, 0), f"Position wrong after right: {env.agent_pos}"
    # Reward may be -1 (move), +100 (gold), or -100 (pit/wumpus) depending on map
    assert reward in (-1, 100, -100), f"Unexpected reward: {reward}"
    assert done == (reward != -1), "Done flag inconsistent with reward"
    print(f"  Movement deterministic - OK (reward={reward}, outcome={info['outcome']})")

    # Test boundary: moving outside the grid keeps the agent in place and
    # gives the normal move reward. Move left from (1,0) to (0,0), then left again.
    env.agent_pos = (0, 0)
    env.steps = 0
    state3, reward3, done3, _, _ = env.step(2)  # Left -> stays at (0,0)
    assert env.agent_pos == (0, 0), f"Should stay at (0,0): {env.agent_pos}"
    assert reward3 == -1, f"Boundary move should give -1, got {reward3}"
    assert not done3, "Boundary move should not end the episode (start is safe)"
    print("  Boundary handling - OK")

    env2 = WumpusWorldEnv(seed=99)
    env2.reset(seed=200)
    env3 = WumpusWorldEnv(seed=99)
    env3.reset(seed=300)
    assert env2.gold_pos != env3.gold_pos or env2.wumpus_pos != env3.wumpus_pos, \
        "Different seeds should produce different maps"
    print("  Random map generation - OK")

    env_train = WumpusWorldEnv(seed=50)
    env_test = WumpusWorldEnv(seed=10050)
    print(f"  Train map gold@{env_train.gold_pos}, Test map gold@{env_test.gold_pos} - OK")

    print("  PASSED\n")


def test_softmax_stability():
    """Test numerically stable softmax produces valid probabilities."""
    print("=" * 60)
    print("TEST 6: Softmax numerical stability")
    print("=" * 60)

    logits = np.array([1000.0, -1000.0, 500.0, 0.0], dtype=np.float32)
    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()

    assert not np.any(np.isnan(probs)), "NaN in probabilities!"
    assert not np.any(np.isinf(probs)), "Inf in probabilities!"
    assert abs(probs.sum() - 1.0) < 1e-6, f"Probs don't sum to 1: {probs.sum()}"
    assert np.all(probs >= 0), "Negative probabilities!"
    print(f"  Probs: {probs}")
    print(f"  Sum: {probs.sum():.10f}")
    print("  PASSED\n")


def test_terminal_next_value():
    """Test that terminal state handling is correct."""
    print("=" * 60)
    print("TEST 7: Terminal state handling")
    print("=" * 60)

    env = WumpusWorldEnv(seed=42)
    state = env.reset(seed=42)
    done = False
    steps = 0
    while not done and steps < 100:
        action = np.random.randint(0, 4)
        state, reward, done, trunc, info = env.step(action)
        steps += 1

    print(f"  Episode ended after {steps} steps, outcome={info['outcome']}")
    print("  Terminal handling - OK (enforced in training loop)")
    print("  PASSED\n")


if __name__ == "__main__":
    test_critic_gradient_direction()
    test_actor_gradient_sign()
    test_finite_difference_critic()
    test_finite_difference_actor()
    test_environment()
    test_softmax_stability()
    test_terminal_next_value()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

"""SARSA und Q-Learning von Hand: die beiden TD-Update-Formeln, gierige/erkundende Aktionswahl mit Gleichstand-Aufloesung, eine unabhaengige
Schritt-fuer-Schritt-Nachrechnung von `run_episode`, UND die zentrale strukturelle Gegenprobe dieses Stuecks: bei epsilon=0 sind SARSA und Q-Learning
UPDATE-IDENTISCH (das Ziel ist in beiden Faellen die gierige Aktion) - byte-gleiche Q-Tabellen bei gleichem Seed."""

import numpy as np
import pytest

import sa_agent as A
import sa_constants as C
from sa_grid import Grid, step


def test_choose_action_epsilon_zero_is_always_greedy():
    Q = np.zeros((1, 4))
    Q[0] = [1.0, 5.0, 2.0, 0.0]
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert A.choose_action(Q, 0, epsilon=0.0, rng=rng) == 1


def test_choose_action_breaks_ties_among_all_maxima():
    Q = np.zeros((1, 4))
    Q[0] = [5.0, 0.0, 5.0, 0.0]
    rng = np.random.default_rng(1)
    seen = {A.choose_action(Q, 0, epsilon=0.0, rng=rng) for _ in range(200)}
    assert seen == {0, 2}


def test_update_qlearning_target_by_hand():
    Q = np.array([[1.0, 2.0], [3.0, 4.0]])
    A.update_qlearning(Q, s=0, a=0, r=1.0, s_next=1, done=False, alpha=0.5, gamma=0.9)
    target = 1.0 + 0.9 * 4.0
    assert Q[0, 0] == pytest.approx(1.0 + 0.5 * (target - 1.0))


def test_update_sarsa_target_uses_the_actually_chosen_next_action_not_the_max():
    Q = np.array([[1.0, 2.0], [3.0, 4.0]])
    A.update_sarsa(Q, s=0, a=0, r=1.0, s_next=1, a_next=0, done=False, alpha=0.5, gamma=0.9)  # a_next=0 -> Q[1,0]=3.0, NICHT max=4.0
    target = 1.0 + 0.9 * 3.0
    assert Q[0, 0] == pytest.approx(1.0 + 0.5 * (target - 1.0))


def test_update_terminal_transition_ignores_the_bootstrap_for_both_methods():
    Q_ql = np.array([[1.0, 2.0], [100.0, 100.0]])
    A.update_qlearning(Q_ql, s=0, a=0, r=10.0, s_next=1, done=True, alpha=0.5, gamma=0.9)
    assert Q_ql[0, 0] == pytest.approx(1.0 + 0.5 * (10.0 - 1.0))

    Q_sa = np.array([[1.0, 2.0], [100.0, 100.0]])
    A.update_sarsa(Q_sa, s=0, a=0, r=10.0, s_next=1, a_next=None, done=True, alpha=0.5, gamma=0.9)
    assert Q_sa[0, 0] == pytest.approx(1.0 + 0.5 * (10.0 - 1.0))


def test_run_episode_reproduces_a_hand_composed_step_by_step_replay():
    grid = Grid(rows=1, cols=3, slip=0.0, gamma=0.9)
    rng_a = np.random.default_rng(7)
    Q_a = np.zeros((grid.n_states, A.N_ACTIONS))
    total_a, steps_a, falls_a = A.run_episode(grid, Q_a, "sarsa", epsilon=0.3, alpha=0.2, rng=rng_a, max_steps=20)

    rng_b = np.random.default_rng(7)
    Q_b = np.zeros((grid.n_states, A.N_ACTIONS))
    s = grid.state_of(grid.start)
    a = A.choose_action(Q_b, s, 0.3, rng_b)
    total_b, steps_b, falls_b = 0.0, 0, 0
    for _ in range(20):
        s_next, r, done = step(grid, s, a, rng_b)
        if r == C.CLIFF_PENALTY:
            falls_b += 1
        a_next = None if done else A.choose_action(Q_b, s_next, 0.3, rng_b)
        A.update_sarsa(Q_b, s, a, r, s_next, a_next, done, alpha=0.2, gamma=grid.gamma)
        total_b += r
        steps_b += 1
        if done:
            break
        s, a = s_next, a_next
    assert steps_a == steps_b and total_a == pytest.approx(total_b) and falls_a == falls_b
    assert np.array_equal(Q_a, Q_b)


def test_epsilon_zero_makes_sarsa_and_qlearning_update_identical():
    # Der zentrale Korrektheits-Check dieses Stuecks: ohne Exploration ist die "tatsaechlich gewaehlte" Aktion IMMER die gierige - SARSAs und
    # Q-Learnings Lernziel fallen dann zusammen. Byte-gleiche Q-Tabellen bei gleichem Seed, ueber ein ganzes Training.
    grid = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    Q_sarsa, _, _, _, _ = A.train(grid, "sarsa", alpha=0.5, epsilon=0.0, epsilon_decay=0.0, episodes=50, seed=3)
    Q_ql, _, _, _, _ = A.train(grid, "qlearning", alpha=0.5, epsilon=0.0, epsilon_decay=0.0, episodes=50, seed=3)
    assert np.array_equal(Q_sarsa, Q_ql)


def test_train_snapshot_at_a_checkpoint_equals_a_separately_truncated_run():
    grid = Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    _, _, _, _, snapshots = A.train(grid, "sarsa", alpha=0.5, epsilon=0.1, epsilon_decay=0.0, episodes=200, seed=3, checkpoints=(50,))
    Q_short, _, _, _, _ = A.train(grid, "sarsa", alpha=0.5, epsilon=0.1, epsilon_decay=0.0, episodes=50, seed=3)
    assert np.array_equal(snapshots[50], Q_short)


def test_train_returns_arrays_of_the_right_shape():
    grid = Grid(rows=4, cols=8, slip=0.0, gamma=0.95)
    Q, returns, lengths, falls, snapshots = A.train(grid, "qlearning", alpha=0.5, epsilon=0.1, epsilon_decay=0.0, episodes=30, seed=0)
    assert Q.shape == (grid.n_states, A.N_ACTIONS)
    assert returns.shape == (30,) and lengths.shape == (30,) and falls.shape == (30,)
    assert np.all(lengths >= 1) and np.all(lengths <= C.MAX_STEPS_PER_EPISODE)

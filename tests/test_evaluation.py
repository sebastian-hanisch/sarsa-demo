"""Analyse und die drei Experimente: Aufbau, und die Korrektheits-Kette - bei epsilon=0 muessen beide Verfahren dieselbe (fast optimale) Policy
lernen. Exakte Zahlen stehen in test_claims.py."""

import numpy as np
import pytest

import sa_evaluation as E


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(E.Settings(episodes=100, seed=0))


def test_analyse_wiring(analysis):
    a = analysis
    for m in ("qlearning", "sarsa"):
        assert a.Q[m].shape == (a.grid.n_states, 4)
        assert a.returns[m].shape == (100,)
        assert a.total_falls[m] == int(a.falls[m].sum())


def test_analyse_is_deterministic_given_the_same_seed():
    s = E.Settings(rows=3, cols=4, episodes=50, seed=5)
    a1, a2 = E.analyse(s), E.analyse(s)
    for m in ("qlearning", "sarsa"):
        assert np.array_equal(a1.Q[m], a2.Q[m])


def test_epsilon_zero_gives_identical_analysis_for_both_methods():
    a = E.analyse(E.Settings(episodes=50, epsilon=0.0, seed=1))
    assert np.array_equal(a.Q["qlearning"], a.Q["sarsa"])
    assert a.gap["qlearning"] == pytest.approx(a.gap["sarsa"])


def test_epsilon_experiment_shape():
    exp = E.epsilon_experiment(levels=(0.0, 0.1), seeds=range(5))
    assert len(exp["rows"]) == 2
    for row in exp["rows"]:
        for m in ("qlearning", "sarsa"):
            assert "steady" in row[m] and "falls_mean" in row[m]


def test_alpha_experiment_shape():
    exp = E.alpha_experiment(levels=(0.1, 0.5), seeds=range(5))
    assert len(exp["rows"]) == 2


def test_slip_experiment_shape():
    exp = E.slip_experiment(levels=(0.0, 0.1), seeds=range(5))
    assert len(exp["rows"]) == 2


def test_without_slip_sarsa_beats_qlearning_on_steady_reward_and_falls():
    # Der klassische Sutton/Barto-Befund (kleine Stichprobe fuer Testgeschwindigkeit, grosszuegige Schwelle).
    exp = E.epsilon_experiment(levels=(0.10,), seeds=range(10), base=E.Settings(slip=0.0, alpha=0.5, episodes=300))
    row = exp["rows"][0]
    assert row["sarsa"]["steady"]["mean"] > row["qlearning"]["steady"]["mean"]
    assert row["sarsa"]["falls_mean"] < row["qlearning"]["falls_mean"]

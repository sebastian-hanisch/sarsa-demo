"""Jede Zahl aus dem README, nachgerechnet ueber die echten Auswertungsfunktionen. Beide Verfahren sind stochastisch: Mehr-Seed-Zahlen tragen
grosszuegige Baender (CI-robust), der Standardfall (fester Seed) ist exakt."""

import numpy as np
import pytest

import sa_constants as C
import sa_evaluation as E
from sa_grid import EAST


def _east_cells(grid, policy):
    row = grid.rows - 2
    return sum(1 for c in range(grid.cols - 1) if policy[grid.state_of((row, c))] == EAST)


def test_standard_case():
    a = E.analyse(E.Settings())
    assert a.steady["qlearning"] == pytest.approx(-27.3, abs=1.0)
    assert a.steady["sarsa"] == pytest.approx(-9.1, abs=1.0)
    assert a.total_falls["qlearning"] == pytest.approx(120, abs=10)
    assert a.total_falls["sarsa"] == pytest.approx(35, abs=10)
    assert _east_cells(a.grid, a.policy["qlearning"]) == 7
    assert _east_cells(a.grid, a.policy["sarsa"]) == 3


def test_epsilon_zero_makes_sarsa_and_qlearning_update_identical():
    a = E.analyse(E.Settings(epsilon=0.0, episodes=100, seed=2))
    assert np.array_equal(a.Q["qlearning"], a.Q["sarsa"])


def test_epsilon_experiment():
    ee = E.epsilon_experiment()
    rows = {r["epsilon"]: r for r in ee["rows"]}
    lo, hi = rows[0.01], rows[0.30]
    lo_gap = abs(lo["sarsa"]["steady"]["mean"] - lo["qlearning"]["steady"]["mean"])
    hi_gap = abs(hi["sarsa"]["steady"]["mean"] - hi["qlearning"]["steady"]["mean"])
    assert lo_gap < hi_gap  # der Unterschied waechst mit Epsilon
    assert hi["sarsa"]["steady"]["mean"] > hi["qlearning"]["steady"]["mean"]  # SARSA gewinnt bei starker Exploration


def test_alpha_experiment():
    ae = E.alpha_experiment()
    for row in ae["rows"]:
        assert row["sarsa"]["steady"]["mean"] > row["qlearning"]["steady"]["mean"]  # Rangfolge bleibt ueber alle Lernraten erhalten


def test_slip_experiment():
    se = E.slip_experiment()
    rows = {r["slip"]: r for r in se["rows"]}
    r0, r_last = rows[0.0], rows[0.30]
    assert r0["sarsa"]["steady"]["mean"] > r0["qlearning"]["steady"]["mean"]  # ohne Rutschen gewinnt SARSA im Ertrag
    assert r0["sarsa"]["falls_mean"] < r0["qlearning"]["falls_mean"]          # ... und stuerzt seltener ab
    assert r_last["sarsa"]["steady"]["mean"] < r_last["qlearning"]["steady"]["mean"]  # bei starkem Rutschen dreht sich der Ertrag um
    assert r_last["sarsa"]["falls_mean"] > r_last["qlearning"]["falls_mean"]          # ... und SARSA verliert auch den Sicherheitsvorsprung

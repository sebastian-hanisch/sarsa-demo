"""Unabhängige Orakel für SARSA/Q-Learning: Übergangsmodell per Koordinaten-Neuimplementierung, V* per LP (scipy HiGHS), Policy-Wert per Lineargleichungssystem,
beide Lernverfahren Schritt für Schritt gegen eine Schleifen-Neuimplementierung auf identischem Zufallsstrom (inkl. Abstürze und Schrittzahl), SARSA-Ziel in
Erwartung gleich der ε-gierigen Erwartung über Q(s',·), und das Urteil der App (mehr Ertrag UND weniger Abstürze) gegen die Messwerte."""

from pathlib import Path

import numpy as np
import pytest

import sa_agent as A
import sa_evaluation as E
import sa_grid as G

linprog = pytest.importorskip("scipy.optimize").linprog

_MOVES = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
_NAME = {0: "N", 1: "S", 2: "E", 3: "W"}
_SIDES = {"N": ("E", "W"), "S": ("E", "W"), "E": ("N", "S"), "W": ("N", "S")}


def _outcomes(rows, cols, slip, s, a):
    r, c = divmod(s, cols)
    goal = (rows - 1, cols - 1)
    if (r, c) == goal:
        return [(1.0, s, 0.0, True)]
    cliff = {(rows - 1, k) for k in range(1, cols - 1)}
    m = _NAME[a]
    out = []
    for p, d in ((1 - slip, m), (slip / 2, _SIDES[m][0]), (slip / 2, _SIDES[m][1])):
        nr, nc = r + _MOVES[d][0], c + _MOVES[d][1]
        if not (0 <= nr < rows and 0 <= nc < cols):
            nr, nc = r, c
        if (nr, nc) == goal:
            out.append((p, nr * cols + nc, 10.0, True))
        elif (nr, nc) in cliff:
            out.append((p, (rows - 1) * cols, -100.0, False))
        else:
            out.append((p, nr * cols + nc, -1.0, False))
    return out


def _vstar_lp(P, Rw, gamma):
    S, nA = Rw.shape
    rows = []
    for s in range(S):
        for a in range(nA):
            row = gamma * P[s, a].copy()
            row[s] -= 1.0
            rows.append(row)
    return linprog(np.ones(S), A_ub=np.array(rows), b_ub=-Rw.reshape(-1), bounds=[(None, None)] * S, method="highs").x


def _replay(method, rows, cols, slip, gamma, alpha, eps0, decay, episodes, seed):
    rng = np.random.default_rng(seed)
    Q = [[0.0] * 4 for _ in range(rows * cols)]

    def pick(s, eps):
        if rng.random() < eps:
            return int(rng.integers(4))
        best = [k for k in range(4) if Q[s][k] == max(Q[s])]
        return best[0] if len(best) == 1 else int(rng.choice(np.array(best)))

    rets, lens, falls = [], [], []
    for e in range(episodes):
        eps = eps0 if decay <= 0 else max(0.01, eps0 / (1 + decay * e))
        s = (rows - 1) * cols
        a = pick(s, eps)
        tot, n, nf = 0.0, 0, 0
        for _ in range(400):
            u = rng.random()
            k = 0 if (slip == 0 or u < 1 - slip) else (1 if u < 1 - slip / 2 else 2)
            _, s2, r, d = _outcomes(rows, cols, slip, s, a)[k]
            nf += r == -100.0
            a2 = None if d else pick(s2, eps)
            boot = 0.0 if d else (Q[s2][a2] if method == "sarsa" else max(Q[s2]))
            Q[s][a] += alpha * (r + gamma * boot - Q[s][a])
            tot, n = tot + r, n + 1
            if d:
                break
            s, a = s2, a2
        rets.append(tot)
        lens.append(n)
        falls.append(nf)
    return np.array(Q), np.array(rets), np.array(lens), np.array(falls)


def test_both_methods_replay_step_by_step_on_the_same_random_stream():
    rng = np.random.default_rng(11)
    for _ in range(20):
        rows, cols = int(rng.integers(3, 6)), int(rng.integers(4, 8))
        slip, gamma, alpha = float(rng.choice([0, 0.1, 0.3])), float(rng.choice([0.8, 0.95])), float(rng.choice([0.05, 0.5]))
        eps0, decay, ep, seed = float(rng.choice([0.0, 0.1, 0.3])), float(rng.choice([0, 0.02])), int(rng.integers(2, 30)), int(rng.integers(1000))
        g = G.Grid(rows, cols, slip, gamma)
        for m in ("sarsa", "qlearning"):
            Q, rets, lens, falls, _ = A.train(g, m, alpha, eps0, decay, ep, seed)
            Qo, ro, lo, fo = _replay(m, rows, cols, slip, gamma, alpha, eps0, decay, ep, seed)
            assert np.allclose(Q, Qo, atol=1e-12, rtol=0)
            assert np.array_equal(rets, ro) and np.array_equal(lens, lo) and np.array_equal(falls, fo)


def test_model_reference_and_displayed_metrics_match_independent_computations():
    rng = np.random.default_rng(5)
    for _ in range(8):
        rows, cols = int(rng.integers(3, 5)), int(rng.integers(4, 7))
        s = E.Settings(rows, cols, float(rng.choice([0, 0.1, 0.3])), float(rng.choice([0.9, 0.95])), 0.3, float(rng.choice([0.05, 0.1, 0.3])), 0.0, int(rng.integers(5, 150)), seed=int(rng.integers(100)))
        a = E.analyse(s)
        g = a.grid
        P, Rw = G.build_model(g)
        goal_s = g.state_of(g.goal)
        for st in range(g.n_states):
            for act in range(4):
                Pm, Rm = np.zeros(g.n_states), 0.0
                for p, s2, r, d in _outcomes(rows, cols, s.slip, st, act):
                    Pm[goal_s if d else s2] += p
                    Rm += p * r
                assert np.allclose(P[st, act], Pm) and Rw[st, act] == pytest.approx(Rm)
        Vlp = _vstar_lp(P, Rw, g.gamma)
        assert np.max(np.abs(a.V_star - Vlp)) < 1e-5
        s0, idx = g.state_of(g.start), np.arange(g.n_states)
        for m in ("sarsa", "qlearning"):
            Vpi = np.linalg.solve(np.eye(g.n_states) - g.gamma * P[idx, a.policy[m]], Rw[idx, a.policy[m]])
            assert a.gap[m] == pytest.approx(Vlp[s0] - Vpi[s0], abs=1e-4)
            _, ro, _, fo = _replay(m, rows, cols, s.slip, g.gamma, 0.3, s.epsilon, 0.0, s.episodes, s.seed)
            assert a.steady[m] == pytest.approx(ro[-100:].mean()) and a.total_falls[m] == int(fo.sum())


def test_sarsa_bootstrap_action_is_distributed_like_the_epsilon_greedy_policy():
    for i, (eps, row) in enumerate([(0.0, [1.0, 1.0, 0.0, 0.0]), (0.1, [3.0, 1.0, 2.0, 0.0]), (0.3, [0.0, 0.0, 0.0, 5.0])]):
        Q = np.zeros((2, 4))
        Q[1] = row
        best = np.flatnonzero(Q[1] == Q[1].max())
        pi = np.full(4, eps / 4)
        pi[best] += (1 - eps) / len(best)
        rng = np.random.default_rng(i)
        picks = [A.choose_action(Q, 1, eps, rng) for _ in range(8000)]
        assert np.mean(Q[1][picks]) == pytest.approx(pi @ Q[1], abs=0.1)


def test_verdict_only_claims_what_the_measurements_support():
    assert E.verdict_kind(-5.0, -20.0, 30, 100) == "both"
    assert E.verdict_kind(-30.0, -20.0, 30, 100) == "safer_only"
    assert E.verdict_kind(-5.0, -20.0, 43, 35) == "return_only"
    assert E.verdict_kind(-5.0, -20.0, 35, 35) == "return_only"                                       # Gleichstand heißt nicht "seltener"
    assert E.verdict_kind(-30.0, -20.0, 60, 32) == "neither"


def test_app_does_not_claim_fewer_falls_when_sarsa_falls_more_often():
    from streamlit.testing.v1 import AppTest

    a = E.analyse(E.Settings(slip=0.1, epsilon=0.01, alpha=0.1, episodes=300))
    assert a.steady["sarsa"] > a.steady["qlearning"] and a.total_falls["sarsa"] >= a.total_falls["qlearning"]
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=600)
    for k, v in {"slip_slider": 0.1, "epsilon_slider": 0.01, "alpha_slider": 0.1, "episodes_slider": 300}.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception
    assert not any("stürzt seltener ab" in el.value and "UND" in el.value for el in at.success)
    assert any("nicht seltener" in el.value for el in at.warning)

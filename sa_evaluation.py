"""Auswertung: SARSA gegen Q-Learning auf demselben Modell mit derselben Verhaltenspolitik, dazu drei Experimente (Wirkung von Epsilon, Wirkung von
Alpha, Wirkung der Rutsch-Wahrscheinlichkeit auf den Ertragsvergleich). Kennzahlen je Verfahren: der Ertrag waehrend des Trainings im eingeschwungenen
Zustand (Sutton & Bartos Vergleichsgroesse), die Zahl der Klippen-Abstuerze insgesamt, und der Wert-Abstand der GIERIGEN Endpolicy zu V*(Start)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import sa_agent as A
import sa_constants as C
import sa_grid as G
import sa_reference as R


@dataclass(frozen=True)
class Settings:
    rows: int = C.DEFAULT_ROWS
    cols: int = C.DEFAULT_COLS
    slip: float = C.DEFAULT_SLIP
    gamma: float = C.DEFAULT_GAMMA
    alpha: float = C.DEFAULT_ALPHA
    epsilon: float = C.DEFAULT_EPSILON
    epsilon_decay: float = C.DEFAULT_EPSILON_DECAY
    episodes: int = C.DEFAULT_EPISODES
    seed: int = 0

    @property
    def grid(self):
        return G.Grid(self.rows, self.cols, self.slip, self.gamma)


@lru_cache(maxsize=64)
def _reference(rows, cols, slip, gamma):
    grid = G.Grid(rows, cols, slip, gamma)
    P, Rw = G.build_model(grid)
    V_star, Q_star, pi_star = R.value_iteration(P, Rw, gamma)
    return grid, P, Rw, V_star, Q_star, pi_star


@dataclass
class Analysis:
    settings: Settings
    grid: G.Grid
    V_star: np.ndarray
    pi_star: np.ndarray
    Q: dict
    returns: dict
    lengths: dict
    falls: dict
    policy: dict
    V_pi: dict
    gap: dict
    steady: dict
    total_falls: dict
    snapshots: dict


def analyse(s, checkpoints=()):
    grid, P, Rw, V_star, Q_star, pi_star = _reference(s.rows, s.cols, s.slip, s.gamma)
    start_s = grid.state_of(grid.start)
    Q, returns, lengths, falls, policy, V_pi, gap, steady, total_falls, snapshots = ({} for _ in range(10))
    for m in C.METHODS:
        Qm, rm, lm, fm, sm = A.train(grid, m, s.alpha, s.epsilon, s.epsilon_decay, s.episodes, s.seed, checkpoints=checkpoints)
        Q[m], returns[m], lengths[m], falls[m], snapshots[m] = Qm, rm, lm, fm, sm
        policy[m] = Qm.argmax(axis=1)
        V_pi[m] = R.policy_evaluation(P, Rw, policy[m], grid.gamma)
        gap[m] = float(V_star[start_s] - V_pi[m][start_s])
        steady[m] = float(rm[-min(C.STEADY_WINDOW, len(rm)):].mean())
        total_falls[m] = int(fm.sum())
    return Analysis(s, grid, V_star, pi_star, Q, returns, lengths, falls, policy, V_pi, gap, steady, total_falls, snapshots)


def _run_metrics(rows, cols, slip, gamma, alpha, epsilon, episodes, seed):
    grid, P, Rw, V_star, _, _ = _reference(rows, cols, slip, gamma)
    start_s = grid.state_of(grid.start)
    out = {}
    for m in C.METHODS:
        Q, returns, lengths, falls, _ = A.train(grid, m, alpha, epsilon, 0.0, episodes, seed)
        policy = Q.argmax(axis=1)
        V_pi = R.policy_evaluation(P, Rw, policy, gamma)
        steady = float(returns[-min(C.STEADY_WINDOW, len(returns)):].mean())
        out[m] = {"steady": steady, "falls": int(falls.sum()), "gap": float(V_star[start_s] - V_pi[start_s])}
    return out


def _summary(values):
    values = np.asarray(values, dtype=float)
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {"mean": float(values.mean()), "se": se}


# --- Experiment 1: Wirkung von Epsilon --------------------------------------------------------------------------------------------------------------

def epsilon_experiment(levels=None, seeds=None, base=None):
    levels = C.EXP_EPSILON_LEVELS if levels is None else levels
    seeds = range(C.EXP_SEEDS) if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for eps in levels:
        per_method = {m: {"steady": [], "falls": []} for m in C.METHODS}
        for sd in seeds:
            metrics = _run_metrics(base.rows, base.cols, base.slip, base.gamma, base.alpha, eps, base.episodes, sd)
            for m in C.METHODS:
                per_method[m]["steady"].append(metrics[m]["steady"])
                per_method[m]["falls"].append(metrics[m]["falls"])
        rows.append({"epsilon": eps, **{m: {"steady": _summary(per_method[m]["steady"]), "falls_mean": float(np.mean(per_method[m]["falls"]))} for m in C.METHODS}})
    return {"n_seeds": len(list(seeds)) if not isinstance(seeds, range) else len(seeds), "levels": tuple(levels), "rows": rows}


# --- Experiment 2: Wirkung von Alpha ----------------------------------------------------------------------------------------------------------------

def alpha_experiment(levels=None, seeds=None, base=None):
    levels = C.EXP_ALPHA_LEVELS if levels is None else levels
    seeds = range(C.EXP_SEEDS) if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for alpha in levels:
        per_method = {m: {"steady": [], "falls": []} for m in C.METHODS}
        for sd in seeds:
            metrics = _run_metrics(base.rows, base.cols, base.slip, base.gamma, alpha, base.epsilon, base.episodes, sd)
            for m in C.METHODS:
                per_method[m]["steady"].append(metrics[m]["steady"])
                per_method[m]["falls"].append(metrics[m]["falls"])
        rows.append({"alpha": alpha, **{m: {"steady": _summary(per_method[m]["steady"]), "falls_mean": float(np.mean(per_method[m]["falls"]))} for m in C.METHODS}})
    return {"n_seeds": len(list(seeds)) if not isinstance(seeds, range) else len(seeds), "levels": tuple(levels), "rows": rows}


# --- Experiment 3: Rutsch-Wahrscheinlichkeit - dreht sich der Ertragsvergleich um? ------------------------------------------------------------------

def slip_experiment(levels=None, seeds=None, base=None):
    levels = (0.0, 0.05, 0.10, 0.20, 0.30) if levels is None else levels
    seeds = range(C.EXP_SEEDS) if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for slip in levels:
        per_method = {m: {"steady": [], "falls": []} for m in C.METHODS}
        for sd in seeds:
            metrics = _run_metrics(base.rows, base.cols, slip, base.gamma, base.alpha, base.epsilon, base.episodes, sd)
            for m in C.METHODS:
                per_method[m]["steady"].append(metrics[m]["steady"])
                per_method[m]["falls"].append(metrics[m]["falls"])
        rows.append({"slip": slip, **{m: {"steady": _summary(per_method[m]["steady"]), "falls_mean": float(np.mean(per_method[m]["falls"]))} for m in C.METHODS}})
    return {"n_seeds": len(list(seeds)) if not isinstance(seeds, range) else len(seeds), "levels": tuple(levels), "rows": rows}

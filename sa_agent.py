"""SARSA (Rummery & Niranjan 1994) gegen Q-Learning (Watkins & Dayan 1992): beide TD-Kontrollverfahren, beide mit derselben epsilon-gierigen
Verhaltenspolitik. Der einzige Unterschied ist das Lernziel: Q-Learning bootstrapt IMMER von der gierigen Aktion des Folgezustands (off-policy),
SARSA bootstrapt von der Aktion, die die Verhaltenspolitik im Folgezustand TATSAECHLICH waehlt (on-policy) - bei epsilon=0 sind beide Ziele identisch."""

import numpy as np

import sa_constants as C
from sa_grid import ACTIONS, step

N_ACTIONS = len(ACTIONS)


def epsilon_by_episode(episode, start, decay, eps_min=C.EPSILON_MIN):
    """decay=0 (Standard fuer diese Demo) haelt epsilon konstant bei `start` - der Kontrast SARSA/Q-Learning braucht dauerhafte Exploration,
    anders als das GLIE-Schema in `q-learning-demo`."""
    if decay <= 0.0:
        return start
    return max(eps_min, start / (1.0 + decay * episode))


def choose_action(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    row = Q[state]
    best = np.flatnonzero(row == row.max())
    return int(best[0]) if best.size == 1 else int(rng.choice(best))


def update_qlearning(Q, s, a, r, s_next, done, alpha, gamma):
    target = r if done else r + gamma * Q[s_next].max()
    Q[s, a] += alpha * (target - Q[s, a])


def update_sarsa(Q, s, a, r, s_next, a_next, done, alpha, gamma):
    target = r if done else r + gamma * Q[s_next, a_next]
    Q[s, a] += alpha * (target - Q[s, a])


def run_episode(grid, Q, method, epsilon, alpha, rng, max_steps=C.MAX_STEPS_PER_EPISODE):
    """Beide Verfahren waehlen JEDEN Schritt dieselbe epsilon-gierige Folgeaktion (identische RNG-Ziehungsreihenfolge bei gleichem Seed) - nur das
    TD-Ziel unterscheidet sich. Rueckgabe: Ertrag, Schrittzahl, Zahl der Klippen-Abstuerze in dieser Episode."""
    s = grid.state_of(grid.start)
    a = choose_action(Q, s, epsilon, rng)
    total_reward, steps, falls = 0.0, 0, 0
    for _ in range(max_steps):
        s_next, r, done = step(grid, s, a, rng)
        if r == C.CLIFF_PENALTY:
            falls += 1
        a_next = None if done else choose_action(Q, s_next, epsilon, rng)
        if method == "qlearning":
            update_qlearning(Q, s, a, r, s_next, done, alpha, grid.gamma)
        else:
            update_sarsa(Q, s, a, r, s_next, a_next, done, alpha, grid.gamma)
        total_reward += r
        steps += 1
        if done:
            break
        s, a = s_next, a_next
    return total_reward, steps, falls


def train(grid, method, alpha, epsilon, epsilon_decay, episodes, seed, max_steps=C.MAX_STEPS_PER_EPISODE, checkpoints=()):
    """Trainiert `episodes` Episoden mit Verfahren `method` ("qlearning" oder "sarsa"). `checkpoints` liefern zusaetzlich eine Kopie der Q-Tabelle
    nach genau dieser vielen Episoden - fuer die Episode-fuer-Episode-Ansicht."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((grid.n_states, N_ACTIONS))
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    falls = np.zeros(episodes, dtype=int)
    snapshots = {}
    checkpoint_set = set(checkpoints)
    for e in range(episodes):
        eps = epsilon_by_episode(e, epsilon, epsilon_decay)
        total_reward, steps, n_falls = run_episode(grid, Q, method, eps, alpha, rng, max_steps)
        returns[e] = total_reward
        lengths[e] = steps
        falls[e] = n_falls
        if (e + 1) in checkpoint_set:
            snapshots[e + 1] = Q.copy()
    return Q, returns, lengths, falls, snapshots

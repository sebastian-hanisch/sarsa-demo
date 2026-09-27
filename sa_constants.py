"""Konstanten der Demo "SARSA" (Stück 4 der Reinforcement-Learning-Linie): dasselbe Raster wie `q-learning-demo`/`value-iteration-demo`, dazu die
Lernparameter. Anders als bei Q-Learning ist der Standard-Epsilon-Zerfall hier 0 (konstant explorativ) - der Kontrast SARSA gegen Q-Learning
(Sutton & Barto 2018, Beispiel 6.6/Abbildung 6.4) braucht dauerhafte, nicht gegen null fallende Exploration."""

EPS = 1e-9
SEED_MAX = 999999

# --- Das Raster (wie value-iteration-demo/q-learning-demo, Cliff-Walking-Vorlage) --------------------------------------------------------------------
STEP_COST = -1.0
CLIFF_PENALTY = -100.0
GOAL_REWARD = 10.0

ROWS_MIN, ROWS_MAX, DEFAULT_ROWS = 3, 6, 4
COLS_MIN, COLS_MAX, DEFAULT_COLS = 4, 12, 8
# DEFAULT_SLIP=0 weicht bewusst von value-iteration-demo/q-learning-demo (0,10) ab: nur ohne Rutschen reproduziert der Kontrast SARSA/Q-Learning
# exakt Sutton & Bartos klassisches Cliff-Walking-Beispiel (Abbildung 6.4). Mit Rutschen dreht sich der Ertragsvergleich teilweise um (gemessen,
# siehe README) - das ist ein eigener, zusaetzlicher Befund dieser Demo, kein Widerspruch.
SLIP_MIN, SLIP_MAX, SLIP_STEP, DEFAULT_SLIP = 0.0, 0.30, 0.02, 0.0
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.80, 0.99, 0.01, 0.95

# --- Referenzlösung (Value Iteration, nur zur Gegenprobe) -------------------------------------------------------------------------------------------
VI_TOL = 1e-8
VI_MAX_ITER = 5000

# --- Lernparameter (SARSA gegen Q-Learning, dieselbe epsilon-gierige Verhaltenspolitik) ---------------------------------------------------------------
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP, DEFAULT_ALPHA = 0.05, 0.50, 0.05, 0.50
EPSILON_MIN_SLIDER, EPSILON_MAX_SLIDER, EPSILON_STEP, DEFAULT_EPSILON = 0.01, 0.30, 0.01, 0.10
EPSILON_DECAY_MIN, EPSILON_DECAY_MAX, EPSILON_DECAY_STEP, DEFAULT_EPSILON_DECAY = 0.0, 0.02, 0.001, 0.0
EPSILON_MIN = 0.01

EPISODES_MIN, EPISODES_MAX, EPISODES_STEP, DEFAULT_EPISODES = 100, 2000, 100, 500
MAX_STEPS_PER_EPISODE = 400
STEADY_WINDOW = 100

METHODS = ("qlearning", "sarsa")
METHOD_LABELS = {"qlearning": "Q-Learning", "sarsa": "SARSA"}

# --- Experimente (feste Konfigurationen) --------------------------------------------------------------------------------------------------------------
EXP_SEEDS = 20
EXP_EPISODES = 500
EXP_EPSILON_LEVELS = (0.01, 0.05, 0.10, 0.20, 0.30)
EXP_ALPHA_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.50)
EXP_EPISODE_CHECKPOINTS = (50, 100, 200, 300, 500, 1000, 2000)

NEAR_OPTIMAL_GAP = 0.5
CLIFF_GAP_THRESHOLD = 50.0

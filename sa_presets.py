"""SETTING_SPECS-Permalink-Muster, Presets und Regler-Grenzen (Standardmuster des Portfolios)."""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import sa_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "rows_slider": SettingSpec("rows", int, C.DEFAULT_ROWS, C.ROWS_MIN, C.ROWS_MAX),
    "cols_slider": SettingSpec("cols", int, C.DEFAULT_COLS, C.COLS_MIN, C.COLS_MAX),
    "slip_slider": SettingSpec("slip", float, C.DEFAULT_SLIP, C.SLIP_MIN, C.SLIP_MAX),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
    "alpha_slider": SettingSpec("alpha", float, C.DEFAULT_ALPHA, C.ALPHA_MIN, C.ALPHA_MAX),
    "epsilon_slider": SettingSpec("eps", float, C.DEFAULT_EPSILON, C.EPSILON_MIN_SLIDER, C.EPSILON_MAX_SLIDER),
    "episodes_slider": SettingSpec("episodes", int, C.DEFAULT_EPISODES, C.EPISODES_MIN, C.EPISODES_MAX),
}
PRESET_KEYS = {
    "rows": "rows_slider", "cols": "cols_slider", "slip": "slip_slider", "gamma": "gamma_slider",
    "alpha": "alpha_slider", "epsilon": "epsilon_slider", "episodes": "episodes_slider",
}
STEPS = {
    "slip_slider": C.SLIP_STEP, "gamma_slider": C.GAMMA_STEP, "alpha_slider": C.ALPHA_STEP,
    "epsilon_slider": C.EPSILON_STEP, "episodes_slider": C.EPISODES_STEP,
}


def _p(**kw):
    base = {
        "rows": C.DEFAULT_ROWS, "cols": C.DEFAULT_COLS, "slip": C.DEFAULT_SLIP, "gamma": C.DEFAULT_GAMMA,
        "alpha": C.DEFAULT_ALPHA, "epsilon": C.DEFAULT_EPSILON, "episodes": C.DEFAULT_EPISODES,
    }
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _p(),
    "Mit Rutsch-Risiko": _p(slip=0.10),
    "Kaum Exploration": _p(epsilon=0.01),
    "Stark explorativ": _p(epsilon=0.30),
    "Niedrige Lernrate": _p(alpha=0.05),
    "Großes Raster": _p(rows=6, cols=12, episodes=1000),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = round(float(snapped), 4)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


PRESET_HELP = {
    "Standardfall": "Ohne Rutschen (α=0,50, ε=0,10, 500 Episoden) reproduziert dieses Setting Sutton & Bartos klassisches Cliff-Walking-Beispiel: SARSA lernt eine sichere Route weit von der Klippe, Q-Learning die riskante direkte Route entlang der Klippe - und bekommt dafür während des Trainings spürbar weniger Ertrag.",
    "Mit Rutsch-Risiko": "Rutschen 0,10: die Umgebung selbst ist jetzt schon riskant. SARSAs Vorteil dreht sich teilweise um - es stürzt zwar weiterhin seltener ab, der Umweg kostet aber mehr Schritte, als die vermiedenen Stürze einsparen.",
    "Kaum Exploration": "ε=0,01: kaum Unterschied zwischen SARSA und Q-Learning - mit wenig Exploration ist das Risiko durch die eigene Verhaltenspolitik gering, der Kontrast verschwindet fast.",
    "Stark explorativ": "ε=0,30: der Unterschied wird extrem - Q-Learnings riskante Route in Kombination mit sehr viel zufälligem Verhalten führt zu deutlich mehr Abstürzen als bei SARSA.",
    "Niedrige Lernrate": "α=0,05: beide Verfahren lernen langsamer, der qualitative Unterschied (SARSA sicherer, Q-Learning direkter) bleibt aber erhalten.",
    "Großes Raster": "6×12-Raster (72 States, 1000 Episoden): derselbe Kontrast auf einem größeren Problem.",
}

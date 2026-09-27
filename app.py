"""SARSA gegen Q-Learning - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück (Kontrast zu Q-Learning) der Reinforcement-Learning-Linie der "Konzepte"-Reihe: derselbe Lagerroboter, dieselbe epsilon-gierige
Verhaltenspolitik - aber SARSA lernt on-policy (das Ziel ist die Aktion, die tatsächlich als nächstes getan wird), Q-Learning off-policy (das Ziel
ist immer die gierige Aktion). Der klassische Kontrast: SARSA lernt eine sicherere, Q-Learning eine riskantere Route (Sutton & Barto 2018).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import sa_constants as C
from sa_evaluation import Settings, alpha_experiment, analyse, epsilon_experiment, slip_experiment
from sa_grid import EAST
from sa_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, sync_query_params
from sa_visualization import build_alpha, build_epsilon, build_falls, build_grid, build_learning_curves, build_slip

st.set_page_config(page_title="SARSA gegen Q-Learning – Sebastian Hanisch", layout="wide")


def de(x, digits=2):
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _analyse(settings, checkpoints):
    return analyse(settings, checkpoints=checkpoints)


@st.cache_data(show_spinner=False)
def _epsilon_exp(base):
    return epsilon_experiment(base=base)


@st.cache_data(show_spinner=False)
def _alpha_exp(base):
    return alpha_experiment(base=base)


@st.cache_data(show_spinner=False)
def _slip_exp(base):
    return slip_experiment(base=base)


def _east_cells(grid, policy):
    row = grid.rows - 2
    return sum(1 for c in range(grid.cols - 1) if policy[grid.state_of((row, c))] == EAST)


st.title("🧗 SARSA gegen Q-Learning")
st.markdown(
    """
Derselbe Lagerroboter wie bei **Q-Learning** (Stück 3) - Raster, Klippe, Packstation. Diesmal lernen **zwei** Verfahren gleichzeitig, mit derselben
**epsilon-gierigen Verhaltenspolitik**: **SARSA** (Rummery & Niranjan 1994) korrigiert seine Q-Schätzung anhand der Aktion, die tatsächlich als
nächstes getan wird (**on-policy**) - Q-Learning korrigiert immer anhand der gierigen Aktion, egal was als nächstes wirklich passiert (**off-policy**).
Der klassische Unterschied (Sutton & Barto 2018): SARSA lernt eine **sicherere** Route, weil es das Risiko seiner eigenen Exploration in seine
Schätzung einpreist - Q-Learning lernt die **riskantere**, direkte Route und stürzt während des Trainings häufiger ab. Alle Daten sind erzeugt.
"""
)
st.caption(
    "Viertes Stück (Kontrast) der **Reinforcement-Learning-Linie** der \"Konzepte\"-Reihe: derselbe Nachfolger von Q-Learning (Stück 3), aber "
    "on-policy statt off-policy. **Bezug:** bei Epsilon = 0 sind SARSA und Q-Learning identisch (strukturell geprüft) - der ganze Unterschied "
    "entsteht ausschließlich durch fortgesetzte Exploration."
)

with st.expander("So unterscheiden sich SARSA und Q-Learning", expanded=True):
    st.markdown(
        r"""
1. **Dieselbe Verhaltenspolitik.** Beide Verfahren handeln epsilon-gierig: mit Wahrscheinlichkeit $\varepsilon$ zufällig, sonst die aktuell beste Aktion. Anders als bei Q-Learning (Stück 3) bleibt $\varepsilon$ hier **konstant** - kein Zerfall gegen null, sonst würde der Unterschied am Ende verschwinden.
2. **SARSA-Update (on-policy):** $Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma\,Q(s',a') - Q(s,a)\big)$, wobei $a'$ die Aktion ist, die die Verhaltenspolitik im Folgezustand **tatsächlich wählt**.
3. **Q-Learning-Update (off-policy):** $Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma\,\max_{a'} Q(s',a') - Q(s,a)\big)$ - das Ziel ignoriert, was als nächstes wirklich getan wird.
4. **Die Folge:** SARSAs Ziel "weiß", dass in der Nähe der Klippe auch mal eine zufällige (gefährliche) Aktion passieren kann, und lernt deshalb, dieser Nähe auszuweichen. Q-Learnings Ziel unterstellt immer optimales Verhalten und ignoriert dieses Risiko - die gelernte Policy ist optimal, solange man sich daran hält, aber riskant, solange man noch (epsilon-gierig) explorieren muss.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Raster**")
    rows = st.slider("Zeilen", *bounds("rows_slider"), key="rows_slider")
    cols = st.slider("Spalten", *bounds("cols_slider"), key="cols_slider")
    slip = st.slider("Rutsch-Wahrscheinlichkeit", *bounds("slip_slider"), key="slip_slider", step=C.SLIP_STEP, format="%.2f", help="0 = klassisches, deterministisches Cliff Walking (Sutton & Barto).")
    gamma = st.slider("Diskontfaktor γ", *bounds("gamma_slider"), key="gamma_slider", step=C.GAMMA_STEP, format="%.2f")
    st.markdown("**Das Lernen**")
    alpha = st.slider("Lernrate α", *bounds("alpha_slider"), key="alpha_slider", step=C.ALPHA_STEP, format="%.2f")
    epsilon = st.slider("Epsilon (konstant)", *bounds("epsilon_slider"), key="epsilon_slider", step=C.EPSILON_STEP, format="%.2f", help="Bleibt über das gesamte Training konstant - kein Zerfall.")
    episodes = st.slider("Trainingsepisoden", *bounds("episodes_slider"), key="episodes_slider", step=C.EPISODES_STEP)

sync_query_params({
    "rows_slider": int(rows), "cols_slider": int(cols), "slip_slider": round(float(slip), 3), "gamma_slider": round(float(gamma), 3),
    "alpha_slider": round(float(alpha), 3), "epsilon_slider": round(float(epsilon), 3), "episodes_slider": int(episodes),
})

settings = Settings(int(rows), int(cols), round(float(slip), 3), round(float(gamma), 3), round(float(alpha), 3), round(float(epsilon), 3), 0.0, int(episodes), seed=0)
with st.spinner("SARSA und Q-Learning trainieren ..."):
    a = _analyse(settings, ())
grid = a.grid
s0 = grid.state_of(grid.start)

# --- Lernkurven --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ertrag während des Trainings")
st.plotly_chart(build_learning_curves(a.returns), width="stretch", key="curves")
st.caption("Gleitender Durchschnitt des Ertrags je Episode (Fenster 20) - die grüne Referenzlinie fehlt bewusst: anders als bei Q-Learning (Stück 3) ist hier nicht V*(Start) das Ziel, sondern der Ertrag UNTER der epsilon-gierigen Verhaltenspolitik selbst (Sutton & Bartos Vergleichsgröße).")

st.markdown("---")

# --- Kernfrage -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wer bekommt während des Trainings mehr Ertrag - und wer stürzt seltener ab?")
mcols = st.columns(4)
mcols[0].metric("Q-Learning: Ertrag (eingeschwungen)", de(a.steady["qlearning"], 1))
mcols[1].metric("SARSA: Ertrag (eingeschwungen)", de(a.steady["sarsa"], 1), delta=de(a.steady["sarsa"] - a.steady["qlearning"], 1))
mcols[2].metric("Q-Learning: Abstürze insgesamt", a.total_falls["qlearning"])
mcols[3].metric("SARSA: Abstürze insgesamt", a.total_falls["sarsa"], delta=a.total_falls["sarsa"] - a.total_falls["qlearning"], delta_color="inverse")

east_ql = _east_cells(grid, a.policy["qlearning"])
east_sa = _east_cells(grid, a.policy["sarsa"])
sarsa_better = a.steady["sarsa"] > a.steady["qlearning"]
if sarsa_better:
    st.success(f"✅ SARSA bekommt hier mehr Ertrag ({de(a.steady['sarsa'],1)} gegen {de(a.steady['qlearning'],1)}) UND stürzt seltener ab ({a.total_falls['sarsa']} gegen {a.total_falls['qlearning']}) - der klassische Sutton/Barto-Befund. SARSAs Policy hält sich in {east_sa} von {grid.cols-1} Zellen direkt an der Klippe an Osten, Q-Learnings Policy in {east_ql} von {grid.cols-1}.")
else:
    st.warning(f"⚠️ SARSA stürzt zwar seltener ab ({a.total_falls['sarsa']} gegen {a.total_falls['qlearning']}), bekommt aber **weniger** Ertrag ({de(a.steady['sarsa'],1)} gegen {de(a.steady['qlearning'],1)}) - der Umweg von der Klippe weg kostet hier mehr Schritte, als die vermiedenen Abstürze einsparen. Das ist KEIN Widerspruch zu Sutton & Barto, sondern ein Effekt des Rutschens (siehe Experiment 3): ohne Rutschen (Preset \"Standardfall\") kippt das Ergebnis zugunsten von SARSA.")
g1, g2 = st.columns(2)
with g1:
    st.markdown("##### Q-Learning: gelernte Policy")
    st.plotly_chart(build_grid(grid, a.Q["qlearning"].max(axis=1), a.policy["qlearning"]), width="stretch", key="grid_ql")
with g2:
    st.markdown("##### SARSA: gelernte Policy")
    st.plotly_chart(build_grid(grid, a.Q["sarsa"].max(axis=1), a.policy["sarsa"]), width="stretch", key="grid_sa")
st.caption("Farbe/Zahl: gelernter Schätzwert max Q(s,·). Pfeile: die gierige Aktion nach dem Training.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie stark wirkt Epsilon auf den Unterschied?")
st.caption(f"Standardraster (ohne Rutschen), α={de(C.DEFAULT_ALPHA,2)}, {C.DEFAULT_EPISODES} Episoden, {C.EXP_SEEDS} Seeds je Epsilon-Stufe. Dauer bis zu einer Minute.")
if st.button("Epsilon-Stufen durchrechnen", key="epsilon_start"):
    st.session_state["epsilon_on"] = True
if st.session_state.get("epsilon_on"):
    ee = _epsilon_exp(Settings())
    c1, c2 = st.columns(2)
    c1.plotly_chart(build_epsilon(ee), width="stretch", key="epsilon_chart")
    c2.plotly_chart(build_falls(ee, "epsilon", "Epsilon"), width="stretch", key="epsilon_falls_chart")
    lo, hi = ee["rows"][0], ee["rows"][-1]
    st.warning(
        f"**Befund:** Bei kaum Exploration (ε={de(lo['epsilon'],2)}) ist der Unterschied klein (SARSA {de(lo['sarsa']['steady']['mean'],1)} gegen Q-Learning {de(lo['qlearning']['steady']['mean'],1)}); bei starker Exploration (ε={de(hi['epsilon'],2)}) "
        f"wird er groß (SARSA {de(hi['sarsa']['steady']['mean'],1)} gegen Q-Learning {de(hi['qlearning']['steady']['mean'],1)}) - **der ganze Unterschied entsteht durch Exploration selbst**, nicht durch die Lernrate oder das Raster."
    )

st.markdown("---")

st.subheader("🔬 Wie stark wirkt die Lernrate α?")
st.caption(f"Standardraster (ohne Rutschen), ε={de(C.DEFAULT_EPSILON,2)}, {C.DEFAULT_EPISODES} Episoden, {C.EXP_SEEDS} Seeds je Lernrate. Dauer bis zu einer Minute.")
if st.button("Lernraten durchrechnen", key="alpha_start"):
    st.session_state["alpha_on"] = True
if st.session_state.get("alpha_on"):
    al = _alpha_exp(Settings())
    st.plotly_chart(build_alpha(al), width="stretch", key="alpha_chart")
    st.warning("**Befund:** Der qualitative Unterschied (SARSA sicherer und ertragreicher) bleibt über alle gemessenen Lernraten erhalten - die Lernrate verändert nur, wie schnell beide Verfahren dorthin konvergieren, nicht die Rangfolge.")

st.markdown("---")

st.subheader("🔬 Dreht sich der Vorteil bei Rutschen um?")
st.caption(f"Standardraster, α={de(C.DEFAULT_ALPHA,2)}, ε={de(C.DEFAULT_EPSILON,2)}, {C.DEFAULT_EPISODES} Episoden, {C.EXP_SEEDS} Seeds je Rutsch-Stufe. Dauer bis zu einer Minute.")
if st.button("Rutschraten durchrechnen", key="slip_start"):
    st.session_state["slip_on"] = True
if st.session_state.get("slip_on"):
    se = _slip_exp(Settings())
    c1, c2 = st.columns(2)
    c1.plotly_chart(build_slip(se), width="stretch", key="slip_chart")
    c2.plotly_chart(build_falls(se, "slip", "Rutsch-Wahrscheinlichkeit"), width="stretch", key="slip_falls_chart")
    r0 = se["rows"][0]
    r_last = se["rows"][-1]
    sarsa_falls_less_at_end = r_last["sarsa"]["falls_mean"] < r_last["qlearning"]["falls_mean"]
    st.warning(
        f"**Befund:** Ohne Rutschen ({de(r0['slip'],2)}) gewinnt SARSA klar in Ertrag UND Abstürzen (Ertrag {de(r0['sarsa']['steady']['mean'],1)} gegen {de(r0['qlearning']['steady']['mean'],1)}; Abstürze {de(r0['sarsa']['falls_mean'],0)} gegen {de(r0['qlearning']['falls_mean'],0)}). "
        f"Mit wachsendem Rutschen dreht sich zuerst der ERTRAGS-Vergleich um (schon ab ~0,05 liegt Q-Learning vorn) - und bei starkem Rutschen ({de(r_last['slip'],2)}) verliert SARSA sogar seinen Sicherheitsvorsprung: es stürzt dann "
        + ("**immer noch seltener** ab" if sarsa_falls_less_at_end else f"**genauso oft oder häufiger** ab ({de(r_last['sarsa']['falls_mean'],0)} gegen {de(r_last['qlearning']['falls_mean'],0)})")
        + " als Q-Learning. Der ganze Vorteil von SARSA gilt also nur, solange die Umgebung selbst nicht schon riskant ist - keiner der beiden Effekte (Ertrag, Sicherheit) ist eine feste Eigenschaft des Verfahrens."
    )

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Konstante, dauerhafte Exploration** | Zerfällt Epsilon gegen null (wie bei Q-Learning, Stück 3), konvergieren SARSA und Q-Learning auf DIESELBE Policy - der ganze Unterschied verschwindet. | - |
| **Die Umgebung selbst ist (noch) nicht riskant** | Kommt schon Rutschen hinzu, kostet SARSAs zusätzlicher Sicherheitsabstand mehr Schritte, als er an vermiedenen Abstürzen einspart - bei starkem Rutschen stürzt SARSA sogar nicht mehr seltener ab als Q-Learning (gemessen, Experiment 3). Weder der Ertragsvorteil noch der Sicherheitsvorteil sind feste Eigenschaften des Verfahrens. | - |
| **Endlich viele States und Actions (Tabelle)** | Ein sehr großes oder stetiges Raster macht eine dichte Q-Tabelle unhandlich - für beide Verfahren gleichermaßen. | Funktionsapproximation / DQN (Stück 6) |
| **Jede Erfahrung wird nur einmal genutzt, dann verworfen** | Teuer gesammelte Erfahrung wird nicht wiederverwendet - für beide Verfahren gleichermaßen. | Dyna-Q (Stück 5) |
"""
)
st.caption("Die Linie: Bandit → Value Iteration und Policy Iteration → Q-Learning → **SARSA** (dieses Stück) → Dyna-Q / Funktionsapproximation (DQN) / Policy Gradient → Actor-Critic.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Das Modell** ist identisch zu `q-learning-demo`/`value-iteration-demo`. Beide Verfahren beobachten nur Übergänge $(s, a, r, s')$, nie das Modell selbst.

**SARSA-Update** (Rummery & Niranjan 1994, benannt nach $S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}$): $Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma\,Q(s',a') - Q(s,a)\big)$, wobei $a'$ epsilon-gierig aus $s'$ gewählt und danach TATSÄCHLICH ausgeführt wird.

**Q-Learning-Update** (Watkins & Dayan 1992): $Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma\,\max_{a'} Q(s',a') - Q(s,a)\big)$ - $a'$ wird zwar ebenfalls epsilon-gierig gewählt und ausgeführt (dieselbe Verhaltenspolitik), aber NICHT für das Lernziel verwendet.

**Reduktion:** bei $\varepsilon=0$ ist die epsilon-gierige Wahl immer die gierige Aktion, also $a' = \arg\max_{a'} Q(s',a')$ - damit sind SARSA- und Q-Learning-Ziel identisch, beide Verfahren erzeugen bei gleichem Seed byte-gleiche Q-Tabellen (struktureller Regressionstest).

Implementiert in `sa_grid.py` (Vehikel), `sa_agent.py` (beide Verfahren, gemeinsame Verhaltenspolitik), `sa_reference.py` (Value Iteration, nur zur Gegenprobe), `sa_evaluation.py` (Analyse, drei Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)

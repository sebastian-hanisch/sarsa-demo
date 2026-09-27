"""Plotly-Abbildungen der Demo "SARSA gegen Q-Learning". Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import sa_constants as C
import sa_grid as G

CLIFF_COLOR = "#3a3a3a"
GOAL_COLOR = "#2e7d32"
START_COLOR = "#8c6bb1"
TEXT_LIGHT = "#ffffff"
TEXT_DARK = "#14233B"
METHOD_COLOR = {"qlearning": "#d62728", "sarsa": "#1f77b4"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_grid(grid, V, policy=None):
    R, Cc = grid.rows, grid.cols
    z = np.full((R, Cc), np.nan)
    text = [["" for _ in range(Cc)] for _ in range(R)]
    for r in range(R):
        for c in range(Cc):
            s = grid.state_of((r, c))
            kind = grid.cell_kind((r, c))
            z[r, c] = np.nan if kind == "cliff" else V[s]
            if kind == "goal":
                text[r][c] = "Ziel"
            elif kind == "cliff":
                text[r][c] = ""
            elif policy is not None:
                text[r][c] = G.ACTION_ARROWS[policy[s]]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z, colorscale="RdYlGn", zmid=0, showscale=True, text=[[de(v, 1) if not np.isnan(v) else "" for v in row] for row in z],
                              hovertemplate="Zeile %{y}, Spalte %{x}: Q=%{z:.2f}<extra></extra>", colorbar=dict(title="max Q(s,·)", thickness=14)))
    cliff_x = [c for r in range(R) for c in range(Cc) if grid.cell_kind((r, c)) == "cliff"]
    cliff_y = [r for r in range(R) for c in range(Cc) if grid.cell_kind((r, c)) == "cliff"]
    if cliff_x:
        fig.add_trace(go.Scatter(x=cliff_x, y=cliff_y, mode="markers", marker=dict(symbol="square", size=34, color=CLIFF_COLOR), showlegend=False, hovertemplate="Klippe<extra></extra>"))
    for r in range(R):
        for c in range(Cc):
            kind = grid.cell_kind((r, c))
            if text[r][c]:
                color = TEXT_LIGHT if kind == "goal" else TEXT_DARK
                fig.add_annotation(x=c, y=r, text=text[r][c], showarrow=False, font=dict(size=18, color=color))
    sr, sc = grid.start
    fig.add_shape(type="rect", x0=sc - 0.45, x1=sc + 0.45, y0=sr - 0.45, y1=sr + 0.45, line=dict(color=START_COLOR, width=3))
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return _base(fig, 90 * grid.rows + 60)


def build_learning_curves(returns, window=20):
    """Ertrag je Episode (gleitender Durchschnitt) fuer beide Verfahren uebereinander - wie Sutton & Barto, Abbildung 6.4."""
    fig = go.Figure()
    for m in C.METHODS:
        y = np.asarray(returns[m], dtype=float)
        x = np.arange(1, len(y) + 1)
        if len(y) >= window:
            smooth = np.convolve(y, np.ones(window) / window, mode="valid")
            x_smooth = x[window - 1:]
        else:
            smooth, x_smooth = y, x
        fig.add_trace(go.Scatter(x=x_smooth, y=smooth, mode="lines", line=dict(color=METHOD_COLOR[m], width=2), name=C.METHOD_LABELS[m]))
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text=f"Ertrag (gleitender Durchschnitt, {window})")
    return _base(fig, 300).update_layout(legend=dict(orientation="h", y=-0.3))


def _grouped(exp, level_key, metric_extractor, y_title):
    levels = [r[level_key] for r in exp["rows"]]
    fig = go.Figure()
    for m in C.METHODS:
        y = [metric_extractor(r, m) for r in exp["rows"]]
        fig.add_trace(go.Bar(x=[de(x, 2) for x in levels], y=y, name=C.METHOD_LABELS[m], marker=dict(color=METHOD_COLOR[m])))
    fig.update_layout(barmode="group", yaxis=dict(title=y_title))
    fig.update_xaxes(type="category")
    return fig


def build_epsilon(exp):
    fig = _grouped(exp, "epsilon", lambda r, m: r[m]["steady"]["mean"], "Ertrag im eingeschwungenen Zustand")
    fig.update_xaxes(title_text="Epsilon")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_alpha(exp):
    fig = _grouped(exp, "alpha", lambda r, m: r[m]["steady"]["mean"], "Ertrag im eingeschwungenen Zustand")
    fig.update_xaxes(title_text="Lernrate α")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_slip(exp):
    fig = _grouped(exp, "slip", lambda r, m: r[m]["steady"]["mean"], "Ertrag im eingeschwungenen Zustand")
    fig.update_xaxes(title_text="Rutsch-Wahrscheinlichkeit")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_falls(exp, level_key, title):
    levels = [r[level_key] for r in exp["rows"]]
    fig = go.Figure()
    for m in C.METHODS:
        y = [r[m]["falls_mean"] for r in exp["rows"]]
        fig.add_trace(go.Bar(x=[de(x, 2) for x in levels], y=y, name=C.METHOD_LABELS[m], marker=dict(color=METHOD_COLOR[m])))
    fig.update_layout(barmode="group", yaxis=dict(title="Klippen-Abstürze insgesamt"))
    fig.update_xaxes(title_text=title, type="category")
    return _base(fig, 300).update_layout(legend=dict(orientation="h", y=-0.3))

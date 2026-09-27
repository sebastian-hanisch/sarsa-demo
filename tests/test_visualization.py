"""Regressionstest fuer den in `value-iteration-demo`/`q-learning-demo` gefundenen Plotly-Bug: deutsch formatierte Dezimal-Strings ("0,10") werden
von Plotly.js als US-tausendergruppierte Ganzzahlen interpretiert, wenn die x-Achse nicht explizit kategorial ist. In diesem Repo von Anfang an mit
type="category" vermieden - dieser Test haelt das fest."""

import sa_evaluation as E
from sa_visualization import build_alpha, build_epsilon, build_slip


def test_epsilon_chart_x_axis_is_categorical():
    exp = E.epsilon_experiment(levels=(0.01, 0.10), seeds=range(3))
    fig = build_epsilon(exp)
    assert fig.layout.xaxis.type == "category"
    assert list(fig.data[0].x) == ["0,01", "0,10"]


def test_alpha_chart_x_axis_is_categorical():
    exp = E.alpha_experiment(levels=(0.10, 0.30), seeds=range(3))
    fig = build_alpha(exp)
    assert fig.layout.xaxis.type == "category"
    assert list(fig.data[0].x) == ["0,10", "0,30"]


def test_slip_chart_x_axis_is_categorical():
    exp = E.slip_experiment(levels=(0.0, 0.10), seeds=range(3))
    fig = build_slip(exp)
    assert fig.layout.xaxis.type == "category"
    assert list(fig.data[0].x) == ["0,00", "0,10"]

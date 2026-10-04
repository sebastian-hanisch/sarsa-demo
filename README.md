# 🧗 SARSA gegen Q-Learning

**[→ Demo live ausprobieren](https://sebastianhanisch-sarsa-demo.streamlit.app/)**

Viertes Stück (Kontrast zu Q-Learning) der **Reinforcement-Learning-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Derselbe Lagerroboter wie bei [Q-Learning](https://github.com/sebastian-hanisch/q-learning-demo) (Stück 3), aber diesmal lernen **zwei** Verfahren gleichzeitig mit derselben epsilon-gierigen Verhaltenspolitik: **SARSA** (Rummery & Niranjan 1994) korrigiert sein Lernziel anhand der Aktion, die tatsächlich als nächstes getan wird (**on-policy**) – **Q-Learning** (Watkins & Dayan 1992) korrigiert immer anhand der gierigen Aktion, egal was als nächstes wirklich passiert (**off-policy**).

## Kernfrage

**Wer bekommt während des Trainings mehr Ertrag – und wer stürzt seltener ab?** Der klassische Befund (Sutton & Barto 2018, Abbildung 6.4): SARSA lernt eine sicherere Route, weil sein Lernziel das Risiko der eigenen Exploration einpreist; Q-Learning lernt die riskante, direkte Route entlang der Klippe. Gilt das auch, wenn die Umgebung selbst schon riskant ist (Rutschen)?

## Modell

- **Vehikel** (`sa_grid.py`): dasselbe Raster wie `q-learning-demo`/`value-iteration-demo`. Beide Agenten sehen nur einzelne Übergänge, nie das Modell selbst.
- **Dieselbe Verhaltenspolitik.** Beide Verfahren handeln epsilon-gierig mit **konstantem** $\varepsilon$ – kein Zerfall gegen null wie bei Q-Learning (Stück 3), sonst würde der Unterschied am Ende verschwinden.
- **SARSA-Update:** $Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma\,Q(s',a') - Q(s,a)\big)$, $a'$ = die Aktion, die die Verhaltenspolitik im Folgezustand **tatsächlich wählt**.
- **Q-Learning-Update:** $Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma\,\max_{a'} Q(s',a') - Q(s,a)\big)$ – das Ziel ignoriert, was als nächstes wirklich getan wird.
- **Reduktion:** bei $\varepsilon=0$ ist die tatsächlich gewählte Aktion immer die gierige – beide Ziele fallen zusammen, beide Verfahren erzeugen bei gleichem Seed byte-gleiche Q-Tabellen.

**Abweichung vom Rest der Linie:** `DEFAULT_SLIP=0` (statt 0,10 bei den Geschwistern) – nur ohne Rutschen reproduziert der Kontrast exakt Sutton & Bartos klassisches Beispiel. Mit Rutschen zeigt sich ein eigener, zusätzlicher Befund (siehe unten).

## Methodik

Bewertet wird der Ertrag **im eingeschwungenen Zustand** (Mittel der letzten 100 Trainingsepisoden – Sutton & Bartos Vergleichsgröße, nicht der exakte Wert der Endpolicy), dazu die Zahl der Klippen-Abstürze insgesamt. Drei Experimente über mehrere Seeds: Wirkung von Epsilon, Wirkung der Lernrate α, und ob sich der Vergleich mit wachsendem Rutschen ändert.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (ohne Rutschen, α=0,50, ε=0,10, 500 Episoden, Seed 0) | Q-Learning: Ertrag −27,3, 120 Abstürze, Policy hält sich in 7 von 7 Zellen direkt an der Klippe. SARSA: Ertrag −9,1, 35 Abstürze, nur 3 von 7 Zellen an der Klippe – der klassische Sutton/Barto-Befund. | `test_standard_case` |
| **Reduktion bei ε=0** | SARSA und Q-Learning erzeugen bei ε=0 (keine Exploration) byte-gleiche Q-Tabellen bei gleichem Seed – der ganze Unterschied entsteht ausschließlich durch fortgesetzte Exploration. | `test_epsilon_zero_makes_sarsa_and_qlearning_update_identical` |
| **Wie stark wirkt Epsilon?** (0,01 bis 0,30, 20 Seeds je Stufe) | Bei kaum Exploration (ε=0,01) ist der Unterschied klein (SARSA −1,4 gegen Q-Learning 0,0); bei starker Exploration (ε=0,30) wird er groß (SARSA −39,5 gegen Q-Learning −99,3) – der Unterschied entsteht durch Exploration selbst. | `test_epsilon_experiment` |
| **Wie stark wirkt die Lernrate α?** (0,05 bis 0,50, 20 Seeds je Stufe) | Der qualitative Unterschied (SARSA sicherer und ertragreicher) bleibt über alle gemessenen Lernraten erhalten – α verändert nur die Konvergenzgeschwindigkeit, nicht die Rangfolge. | `test_alpha_experiment` |
| **Dreht sich der Vorteil bei Rutschen um?** (0,00 bis 0,30, 20 Seeds je Stufe) | Ohne Rutschen gewinnt SARSA klar in Ertrag UND Abstürzen (−10,3/37 gegen −20,9/104). Mit wachsendem Rutschen dreht sich zuerst der Ertragsvergleich um (ab ~0,05 liegt Q-Learning vorn); bei starkem Rutschen (0,30) verliert SARSA sogar seinen Sicherheitsvorsprung – es stürzt dann **häufiger** ab als Q-Learning (242 gegen 160 Abstürze, Ertrag −88,7 gegen −47,4). **Weder der Ertrags- noch der Sicherheitsvorteil sind feste Eigenschaften des Verfahrens.** | `test_slip_experiment` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Konstante, dauerhafte Exploration** | Zerfällt Epsilon gegen null (wie bei Q-Learning, Stück 3), konvergieren SARSA und Q-Learning auf DIESELBE Policy – der Unterschied verschwindet. | – |
| **Die Umgebung selbst ist (noch) nicht riskant** | Kommt schon Rutschen hinzu, kostet SARSAs Sicherheitsabstand mehr Schritte, als er einspart – bei starkem Rutschen stürzt SARSA sogar nicht mehr seltener ab als Q-Learning (gemessen). Weder Ertrags- noch Sicherheitsvorteil sind feste Eigenschaften des Verfahrens. | – |
| **Endlich viele States und Actions (Tabelle)** | Ein sehr großes oder stetiges Raster macht eine dichte Q-Tabelle unhandlich – für beide Verfahren gleichermaßen. | Funktionsapproximation / DQN (Stück 6) |
| **Jede Erfahrung wird nur einmal genutzt, dann verworfen** | Teuer gesammelte Erfahrung wird nicht wiederverwendet – für beide Verfahren gleichermaßen. | Dyna-Q (Stück 5) |

## Tests

`tests/` prüft das Vehikel (`sa_grid.py`, identisch zu `q-learning-demo`), die Referenzlösung (`sa_reference.py`: Bellman-Formel von Hand), beide Lernverfahren (`sa_agent.py`: TD-Update von Hand für SARSA UND Q-Learning, terminale und nicht-terminale Übergänge, eine unabhängige Schritt-für-Schritt-Nachrechnung von `run_episode`, ein struktureller Kürzungstest [Snapshot = kürzerer Lauf] UND die zentrale Korrektheits-Kette: bei ε=0 sind SARSA und Q-Learning **byte-gleich**), die Auswertung und drei Experimente, die Presets und Permalinks, die Plotly-Achsen (`type="category"`, derselbe Bug wie in `value-iteration-demo`/`q-learning-demo` von Anfang an vermieden), die App (AppTest: Standard, jedes Preset, Permalink-Klemmen/-Einrasten, Extremwerte, drei Experimente auf Abruf) und jede Zahl dieses READMEs (`test_claims.py`). Beide Verfahren sind stochastisch – Einzelläufe sind exakt (fester Seed), Mehr-Seed-Aussagen tragen großzügige Bänder. 59 Tests, Laufzeit gut eine Minute; die CI läuft bei jedem Push und wöchentlich.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Lernkurven beider Verfahren, Kernfrage, drei Experimente auf Abruf, Grenzen, Formeln |
| `sa_constants.py` | Regler-Grenzen, feste Rewards, Lernparameter, Experimentkonstanten |
| `sa_grid.py` | Das Vehikel: Raster, `step` (Einzelübergang), `build_model` (nur für die Referenz) |
| `sa_agent.py` | SARSA und Q-Learning: gemeinsame Verhaltenspolitik, beide TD-Updates, Trainingsschleife mit Snapshots |
| `sa_reference.py` | Value Iteration und Policy Evaluation – nur zur Gegenprobe |
| `sa_evaluation.py` | Analyse, drei Experimente |
| `sa_visualization.py` | Plotly-Abbildungen (Lernkurven, Raster-Heatmaps, Balkenvergleiche) |
| `sa_presets.py` | Presets, Permalink |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Erwartungswert-SARSA / n-Schritt-SARSA** – Varianten, die die Verhaltenspolitik expliziter mitteln bzw. über mehrere Schritte bootstrappen.
- **Wiederverwendung der gesammelten Erfahrung** (ein Modell lernen und damit planen) – das ist Dyna-Q (Stück 5).
- **Funktionsapproximation** – die Q-Tabelle bleibt hier dicht und klein genug, um sie vollständig zu speichern (Stück 6).

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly und numpy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Reinforcement Learning: Bandit bis Actor-Critic](https://sebastianhanisch.net/konzepte-reinforcement-learning.html).

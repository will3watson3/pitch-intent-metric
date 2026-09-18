# Pitch Intent Metric

**Estimating a plausible finish location for breaking and offspeed pitches from the catcher's glove target and a pitcher's history.**

Pitch location alone cannot distinguish a missed target from an intentional chase pitch. This project studies whether earlier pitches with similar catcher setups, counts, and movement profiles can help estimate where a pitch was meant to finish.

This repository contains the statistical core of a larger local video-analysis project: validated Statcast joins, historical-comparable inference, a runnable example, and regression tests. Built with Python, pandas, and NumPy.

## Try it

Use Python 3.9–3.11 and run from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/demo.py
python -m unittest discover -s scripts/tests -p test_intent_synthetic.py -v
```

The demo uses invented pitches and runs all three inference variants. It prints estimated locations, historical support, comparable weights, and empirical bounds as JSON. It needs no credentials or downloaded data. Synthetic output demonstrates software behavior, not predictive accuracy.

## How it works

1. Validate pitch IDs and join reviewed glove targets to pitch-level Statcast records.
2. Find earlier-game pitches from the same pitcher and pitch type against the same batter side, near the reviewed target.
3. Weight comparables by target proximity, count, historical movement profile, and outcome.
4. Transfer a supported historical target-to-finish offset to the new glove target.
5. Abstain when history or effective support is insufficient.

The current pitch's realized movement, endpoint, and outcome are excluded from inference. The estimator uses weighted comparables; all estimates remain provisional. [Model features, data requirements, and evaluation details](MODEL.md).

## What's in the code

| File | Purpose |
| --- | --- |
| [`scripts/run_intent_pilot.py`](scripts/run_intent_pilot.py) | Inference, model variants, baselines, and temporal evaluation |
| [`scripts/build_webb_sweeper_intent_pilot.py`](scripts/build_webb_sweeper_intent_pilot.py) | Validated joins for the Webb, Cease, and Skubal pilot groups |
| [`scripts/demo.py`](scripts/demo.py) | Small, reproducible synthetic example and input schema |
| [`scripts/tests/test_intent_synthetic.py`](scripts/tests/test_intent_synthetic.py) | Leakage, abstention, coordinate conversion, support, and row-order checks |

GitHub Actions runs the tests and demo on Python 3.11. Video processing, browser review tools, datasets, and experiments remain in the local research workspace and are outside this focused release.

## Current evidence

The reviewed pilot contains **59 accepted pitches**: 22 Cease sliders, 21 Webb sweepers, and 16 Skubal changeups. The full model produced 30 estimates. On the **29 pitches shared by every variant**, mean endpoint errors were:

| Method | Mean endpoint error, feet |
| --- | ---: |
| Comparables without outcome weighting | 1.023 |
| Past average offset baseline | 1.079 |
| Comparables without shape weighting | 1.093 |
| Full weighted comparables | 1.111 |
| Glove target baseline | 1.127 |

The full model did not outperform the average-offset baseline. Empirical bounds covered only 34.5% of observed endpoints in this sample. Endpoint prediction is not proof of recovered intent, and independent true-intent labels are unavailable. This remains a research pilot, with limitations and next steps documented in [MODEL.md](MODEL.md).

The empirical snapshot comes from local reviewed data; the public synthetic example cannot reproduce those results.

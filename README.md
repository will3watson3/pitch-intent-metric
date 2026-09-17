# Pitch Intent Metric

**Estimating where a pitcher meant to throw, starting with the catcher's glove target.**

Pitch location alone cannot distinguish a missed target from an intentional chase pitch. This research project connects pitch-level Statcast data to broadcast video, identifies the catcher's setup, supports manual review, and estimates a plausible finish location for breaking and offspeed pitches using earlier comparable pitches.

Built with Python, pandas, NumPy, OpenCV, and plain JavaScript. The project includes data ingestion, computer vision, human review tools, temporal inference, and regression tests.

**Status:** working local research pipeline and audit tools; intent estimates remain provisional. The current model uses weighted historical comparables. Independent true-intent labels are not available, and the full model has not outperformed the average-offset baseline in the small pilot.

## How it works

```mermaid
flowchart LR
    A[Pitch-level Statcast] --> B[Match video by pitch ID]
    B --> C[Extract delivery frames]
    C --> D[Glove detection and zone geometry]
    D --> E[Review setup frame and target]
    E --> F[Earlier-game comparable pitches]
    F --> G[Provisional finish area]
    G --> H[Compare models and review video]
```

- **Reliable joins:** stable pitch identifiers and one-to-one checks connect labels, video frames, and Statcast rows.
- **Frame-specific geometry:** overlays use the selected frame's zone, preserve manual corrections, and test all four edges.
- **Reviewable predictions:** browser tools display glove targets, model variants, actual locations, and video together.
- **Leakage controls:** inference excludes the current game's and future pitches' movement, locations, and outcomes.
- **Explicit abstention:** inadequate history or weak comparable support produces no estimate.

## Try the offline demo

From the repository root, use Python 3.9–3.11. The local development environment uses Python 3.9; GitHub Actions is configured for 3.11.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python examples/synthetic_demo.py
```

The demo runs the actual inference code for all three model variants using invented pitches. It prints the glove target, estimated finish, comparable IDs and weights, support counts, and empirical bounds as JSON. No credentials, videos, or private data are required. Synthetic output demonstrates behavior, not predictive accuracy.

## Review tools

With your own local data, start the shared server:

```sh
python scripts/labeler_server.py --port 8765
```

| Page | Purpose |
| --- | --- |
| [Labeler](http://127.0.0.1:8765/) | Label catcher setup targets |
| [Dashboard](http://127.0.0.1:8765/dashboard/) | Explore pitch and target summaries |
| [Frame review](http://127.0.0.1:8765/frame-review/) | Inspect frames and correct zone geometry |
| [Target audit](http://127.0.0.1:8765/target-audit/) | Confirm or correct the selected glove target |
| [Intent audit](http://127.0.0.1:8765/intent-audit/) | Compare inferred targets, actual location, and video |

These interfaces need local datasets; a new checkout has no real pitches to display. See [setup and workflows](docs/WORKFLOWS.md) for inputs and commands. The server is a local review tool and binds to loopback by default.

## Current evidence

The audited pilot contains **59 accepted pitches**: 22 Dylan Cease sliders, 21 Logan Webb sweepers, and 16 Tarik Skubal changeups. The full model produced 30 estimates. On the **29 pitches shared by every variant**, mean endpoint errors were:

| Method | Mean endpoint error, feet |
| --- | ---: |
| Comparables without outcome weighting | 1.023 |
| Past average offset baseline | 1.079 |
| Comparables without shape weighting | 1.093 |
| Full weighted comparables | 1.111 |
| Glove target baseline | 1.127 |

Endpoint error measures where the pitch finished, **not whether intended location was recovered**. Empirical bounds covered only 34.5% of observed endpoints in this shared sample. These bounds are not calibrated confidence regions, and the exploratory comparison does not establish a validated best model. See the [model and evaluation notes](docs/MODEL.md).

## Tests

```sh
python -m unittest discover -s scripts/tests -p 'test_*.py' -v
node scripts/tests/test_frame_overlay.js
```

The published Python tests use synthetic inputs or mocks: no network calls, API keys, or private fixtures. They exercise inference leakage, abstention, coordinate conversion, zone fitting, and frame-specific overlays. Node.js is only needed for the JavaScript check. [Testing details](docs/TESTING.md).

## Repository guide

| Folder | Contents |
| --- | --- |
| [`scripts/`](scripts/README.md) | Pipeline, model, review server, and tests |
| [`examples/`](examples/README.md) | Runnable synthetic inference demo |
| `labeler/`, `dashboard/`, `frame-review/`, `target-audit/`, `intent-audit/` | Browser interfaces, each with its own README |
| [`docs/`](docs/README.md) | Workflows, model features, data contracts, and testing |
| `data/`, `reports/`, `models/` | Local inputs and generated artifacts; only directory documentation is published |

Raw broadcast footage, downloaded datasets, manual audit records, credentials, model binaries, historical experiment scripts, and the separate ScoutDeck website are excluded from this repository. Existing local research files retain their original paths. See [publication scope](docs/PUBLICATION.md).

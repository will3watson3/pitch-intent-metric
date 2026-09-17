# Setup and workflows

Run all commands from the repository root with the virtual environment activated. Start with the offline instructions in the main README.

## Optional full video pipeline

```sh
python -m pip install -r requirements.txt
export ROBOFLOW_API_KEY='your-private-key'
```

Replace the placeholder locally. A real key is needed only for hosted inference, with access to the configured Roboflow model/workflow. `.env.example` documents the variable; scripts do not load dotenv files automatically. External model access and upstream data/video availability are not guaranteed by this repository. Additional packages support Statcast downloads, FFmpeg extraction, hosted detection, pose estimation, and plotting.

Inspect configuration without downloading data:

```sh
python scripts/run_autonomous_pitch_pipeline.py --help
python scripts/run_autonomous_pitch_pipeline.py \
  --pitcher-id 657277 --pitcher-name 'Logan Webb' \
  --game-date 2025-06-01 --run-name example-run --plan
```

To execute a run, supply a known pitcher/game or date range and omit `--plan`. Start with `--limit-pitches 5`; it downloads video and can use billed hosted inference. Outputs are placed under `data/runs/<run-name>/`. Stages are dataset → videos → frames → inference → pose → selection → quality → dashboard. Use `--start-at` / `--stop-after` to scope work. The pipeline normally excludes pitches with a runner on second.

Per-run output includes `pitch_intent.csv`, `dashboard.html`, `summary.json`, a review queue, frame detections, and final vision predictions. Open the generated `dashboard.html` for that run; the shared review server's research views use their own dataset filenames and do not automatically import arbitrary run folders.

Some recovery code can use local broadcast-logo templates. Those assets are intentionally excluded; see [`models/README.md`](../models/README.md). The full external pipeline is separate from the offline demo and CI checks.

## Collect and label your own pitches

```sh
python scripts/build_pitcher_dataset.py --help
python scripts/match_woo_2025_videos.py --help
python scripts/create_labeling_queue.py \
  --input data/my_video_matches.csv --output data/my_labeling_queue.csv --limit 25
python scripts/labeler_server.py --port 8765
```

The video matcher's historical filename is retained for compatibility; inspect its arguments before supplying input/output paths. The labeler discovers `data/*labeling_queue*.csv`. Without local input files the review pages have no real records. If port 8765 is occupied, use `--port 8766` and update the browser URL.

Review setup frames before the catcher drops the glove to receive/frame the pitch. Save the target on the actual reviewed frame. Target audit statuses are `confirmed`, `corrected`, `needs_review`, or `unusable`; model audit statuses are `plausible`, `uncertain`, or `wrong`.

## Reproduce the local three-pitcher pilot

This section requires the unpublished files listed in [Data contracts](DATA.md). The named research scripts retain these dataset-specific defaults.

```sh
python scripts/build_webb_sweeper_intent_pilot.py --group webb
python scripts/build_webb_sweeper_intent_pilot.py --group cease
python scripts/build_webb_sweeper_intent_pilot.py --group skubal
python scripts/run_intent_pilot.py
```

The merge writes audited/model-ready tables under `data/`. Inference writes `data/audited_intent_pilot_predictions.csv` and variant predictions, benchmarks, and summary under `reports/intent_pilot/`. Open `/intent-audit/` on the running review server to compare model targets with actual location and video.

`analyze_labeled_targets.py` retains earlier descriptive/heuristic analysis needed by the labeling workflow. Its historical intent scores are separate from the provisional comparable-pitch estimator documented here.

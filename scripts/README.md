# Source guide

Keep these scripts in place: imports and subprocess calls use this layout. Commands run from the repository root. Most command-line entry points expose `--help`; the named pilot merge, inference, and descriptive analysis retain research-dataset defaults.

| Script | Responsibility |
| --- | --- |
| `run_autonomous_pitch_pipeline.py` | Orchestrate the video pipeline with stage/resume controls |
| `build_pitcher_dataset.py` | Collect pitch-level Statcast records and stable IDs |
| `match_woo_2025_videos.py` | Match pitch records to MLB/Savant video; historical filename |
| `extract_setup_frames.py` | Extract timestamped delivery frames with FFmpeg |
| `run_roboflow_workflow.py` | Run hosted detections and save frame-level outputs |
| `detect_pitcher_leg_kick.py` | Estimate delivery timing from pose landmarks |
| `detect_setup_frame.py` | Identify stable camera views/setup timing |
| `select_best_glove_frame.py` | Select the catcher setup frame using motion/pose evidence |
| `build_statcast_zone_predictions.py` | Build frame-associated zone geometry |
| `refine_broadcast_zone.py` | Fit visible zone edges and register neighboring frames |
| `apply_calibrated_zones_to_predictions.py` | Validate frame alignment and recompute target coordinates |
| `detect_fox_zone.py` | Detect transparent broadcast-zone candidates and classify broadcast cues |
| `fox_zone_tracker.py` | Track/validate broadcast zones across frames |
| `fox_zone_crop.py` | Crop/restore native-resolution detector coordinates |
| `cache_recovery_videos.py` | Cache videos used by zone recovery |
| `recover_unverified_zones.py` | Recover usable zones for unverified predictions |
| `build_vision_review_queue.py` | Assign quality flags and prepare human review |
| `build_autonomous_pitch_dataset.py` | Assemble pitch-level run outputs |
| `build_autonomous_dashboard.py` | Generate a standalone dashboard for a run |
| `create_labeling_queue.py` | Create a review queue from video-matched pitches |
| `create_additional_labeling_queues.py` | Extend the existing local research labeling sample |
| `analyze_labeled_targets.py` | Summarize labels and older heuristic target analyses |
| `labeler_server.py` | Serve five local interfaces and persist review records |
| `build_webb_sweeper_intent_pilot.py` | Validate/merge reviewed targets with Statcast for Webb, Cease, or Skubal |
| `run_intent_pilot.py` | Infer finish estimates and compare model variants with baselines |
| `tests/` | Portable synthetic and mocked regression checks |

See [workflows](../docs/WORKFLOWS.md), [model features](../docs/MODEL.md), and [testing](../docs/TESTING.md). Extra scripts in the original research workspace remain local and are not required by the published pipeline.

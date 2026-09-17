# Data contracts

## Pitch identity

Use one row per pitch. `pitch_uid` is the string `game_pk_at_bat_number_pitch_number`; MLB `play_id` connects a pitch to its video. Aggregated pitcher or pitch-type tables cannot support these joins.

The intent merge rejects duplicate pitch IDs and frame IDs, incomplete group coverage, missing Statcast rows, missing frames, and frames belonging to another pitch. Never join video by row order.

## Frame and coordinate identity

`frame_uid` identifies a specific extracted image and belongs to exactly one `pitch_uid`. A manifest records frame time and image path. Zone center/width/height and glove center are image-pixel quantities belonging to that frame. An overlay from another frame can be wrong even within the same pitch.

Normalized targets use the pitcher/broadcast view: x increases left to right; y increases bottom to top. Zone edges are 0 and 1. Chase targets may lie outside that range and must not be clamped.

```text
x_01 = (glove_x - zone_left) / zone_width
y_01 = (zone_bottom - glove_y) / zone_height
plate_x_ft = (0.5 - x_01) * (17 / 12)
plate_z_ft = sz_bot + y_01 * (sz_top - sz_bot)
```

The horizontal flip aligns these target coordinates with Statcast's catcher-view `plate_x`. The conversion approximates the broadcast box as a 17-inch plate width and the batter-specific strike-zone height; it is not full camera calibration.

## Inputs for the audited intent pilot

| Local file | Role |
| --- | --- |
| `data/target_audit_labels.csv` | Reviewed frame, original and corrected target, status, notes |
| `data/labeled_target_analysis_sample_400.csv` | Pitch labels, pitcher metadata, video links |
| `data/setup_frame_manifest_31_1200_2700.csv` | Frame-to-pitch mapping, extraction time, image path, status |
| `data/{webb,cease,skubal}_2025_pitch_level.csv` | Pitch-level Statcast history with stable IDs |

The merge script creates `{webb_sweeper,cease_slider,skubal_changeup}_intent_pilot_model_ready.csv`. `confirmed` and `corrected` audits are accepted; `needs_review` and `unusable` are excluded. Accepted records must have finite target, endpoint, strike-zone, movement, speed, and spin fields. Original audit coordinates are retained separately.

The inference function's minimum in-memory schema is demonstrated directly in [`examples/synthetic_demo.py`](../examples/synthetic_demo.py). Dates must be consistently formatted ISO `YYYY-MM-DD`; count and measurement columns must be numeric. Callers should validate incoming data before calling `infer`.

## Local artifacts

Labels and predictions are CSV; geometry overrides and run metadata are JSON; images and video remain local. The review server writes changes to local CSV/JSON files. Back up reviewed data before intentional relabeling or regeneration. Credentials belong in environment variables, not data files, source code, or screenshots.

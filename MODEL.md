# Breaking and offspeed intent pilot

Implementation: [`scripts/run_intent_pilot.py`](scripts/run_intent_pilot.py), especially `infer(query, audited, history)`.

The working hypothesis is that a pitcher's historical finishes near a similar catcher setup provide evidence about the intended finish area. This implementation is a transparent weighted-comparables estimator, not a trained random forest or a supervised true-intent model.

## Inputs and features

| Input | Use |
| --- | --- |
| Pitcher ID and pitch type | Restrict both the historical movement profile and reviewed comparables |
| Game date | Require strictly earlier dates, excluding same-day games as well as the current game |
| Batter side (`stand`) | Require comparables against the same batter side |
| Reviewed glove target (`model_target_x_01`, `model_target_y_01`) | Filter and weight comparables by setup location |
| Balls and strikes | Downweight comparables with dissimilar counts |
| Historical `pfx_x`, `pfx_z` | Horizontal and vertical movement in feet |
| Historical `release_speed` | Release velocity in mph |
| Historical `release_spin_rate` | Spin rate in rpm; spin axis is not an input |
| Historical description and contact xwOBA | Modest outcome weights for whiffs, called strikes, and weak contact |
| Historical actual finish minus reviewed target | The offset from setup to finish that the model transfers to the new target |
| Current `sz_bot`, `sz_top` | Convert normalized output to approximate Statcast feet |

The query's realized movement, spin, speed, endpoint, and outcome are never inference features. Historical candidates use their observed movement; the expected profile for the query is a median over up to 100 earlier same-pitcher/type pitches. Zone height is a Statcast-supplied coordinate reference: this is a retrospective audit workflow, not yet a fully pre-pitch deployment.

## Selection and weighting

1. Require at least 20 earlier profile pitches and three earlier reviewed comparables.
2. Keep same-side comparables within Euclidean distance 0.6 of the target in normalized zone coordinates.
3. Weight target proximity with `exp(-0.5 * (distance / 0.3)^2)`.
4. Weight count similarity with `exp(-0.35 * (abs(delta_balls) + abs(delta_strikes)))`.
5. For the full model, compare each candidate's shape with the historical median profile. Scales are 0.3 feet horizontal movement, 0.3 feet vertical movement, 3 mph, and 300 rpm; weight is `exp(-0.5 * mean(scaled_difference^2))`.
6. Apply outcome multipliers: whiff 1.5, called strike 1.2, ball in play with xwOBA ≤ 0.25 is 1.25, otherwise 1.0. Good outcomes are not assumed to prove good execution.
7. Normalize weights and require effective sample size `1 / sum(weight^2)` ≥ 2.5.
8. Select the weighted medoid of historical two-dimensional finish offsets and add it to the new target. This chooses an observed joint offset rather than averaging separate clusters.

Marginal weighted 10th and 90th percentiles describe historical offset spread. They are not a calibrated joint 80% region. Sparse history, distant targets, missing profile values, or concentrated weights produce an explicit abstention status. All emitted estimates are marked `provisional` with `low` evidence.

## Variants and baselines

- `comparables`: all weights above.
- `no_outcome_weight`: identical procedure with outcome weights disabled.
- `no_shape_weight`: identical procedure with shape weights disabled; a valid historical profile is still required.
- `glove_only`: use the glove target as the expected endpoint.
- `past_average_offset`: add the earlier same-pitcher/type mean offset in feet to the glove target, without same-side or nearby-target filtering.

## Evaluation and limitations

The pilot uses earlier dates only for every query. Comparisons use the intersection of pitches on which all three variants emitted estimates, so an abstaining model does not receive an easier evaluation set.

The locally generated `reports/intent_pilot/summary.json` and `benchmark.csv` supplied the aggregate snapshot in the project README. The underlying reviewed pitches are excluded from the public repository. The synthetic demo cannot reproduce those empirical metrics.

The sample is small and selected from reviewed video. Glove placement, framing timing, broadcast geometry, and coordinate conversion introduce measurement error. The full model did not beat the simple average-offset baseline. Removing outcome weights improved endpoint error in this exploratory sample, but that result needs an independent holdout before model selection.

No independent true-intent labels exist here. Lower endpoint error alone cannot validate recovered intent or support command grading. Next useful work is independent review of inferred areas, additional earlier-game labels, and a separately held-out evaluation with coverage calibration.

## Data and reproduction

### Pitch identity

Use one row per pitch. `pitch_uid` is the string `game_pk_at_bat_number_pitch_number`; MLB `play_id` connects a pitch to its video. Aggregated pitcher or pitch-type tables cannot support these joins.

The intent merge rejects duplicate pitch IDs and frame IDs, incomplete group coverage, missing Statcast rows, missing frames, and frames belonging to another pitch. Never join video by row order.

### Frame and coordinate identity

`frame_uid` identifies a specific extracted image and belongs to exactly one `pitch_uid`. A manifest records frame time and image path. Zone center/width/height and glove center are image-pixel quantities belonging to that frame. An overlay from another frame can be wrong even within the same pitch.

Normalized targets use the pitcher/broadcast view: x increases left to right; y increases bottom to top. Zone edges are 0 and 1. Chase targets may lie outside that range and must not be clamped.

```text
x_01 = (glove_x - zone_left) / zone_width
y_01 = (zone_bottom - glove_y) / zone_height
plate_x_ft = (0.5 - x_01) * (17 / 12)
plate_z_ft = sz_bot + y_01 * (sz_top - sz_bot)
```

The horizontal flip aligns these target coordinates with Statcast's catcher-view `plate_x`. The conversion approximates the broadcast box as a 17-inch plate width and the batter-specific strike-zone height; it is not full camera calibration.

### Inputs for the audited intent pilot

| Local file | Role |
| --- | --- |
| `data/target_audit_labels.csv` | Reviewed frame, original and corrected target, status, notes |
| `data/labeled_target_analysis_sample_400.csv` | Pitch labels, pitcher metadata, video links |
| `data/setup_frame_manifest_31_1200_2700.csv` | Frame-to-pitch mapping, extraction time, image path, status |
| `data/{webb,cease,skubal}_2025_pitch_level.csv` | Pitch-level Statcast history with stable IDs |

The merge script creates `{webb_sweeper,cease_slider,skubal_changeup}_intent_pilot_model_ready.csv`. `confirmed` and `corrected` audits are accepted; `needs_review` and `unusable` are excluded. Accepted records must have finite target, endpoint, strike-zone, movement, speed, and spin fields. Original audit coordinates are retained separately.

The inference function's minimum in-memory schema is demonstrated directly in [`scripts/demo.py`](scripts/demo.py). Dates must be consistently formatted ISO `YYYY-MM-DD`; count and measurement columns must be numeric. Callers should validate incoming data before calling `infer`.

With the local audited inputs listed above, run from the repository root:

```sh
python scripts/build_webb_sweeper_intent_pilot.py --group webb
python scripts/build_webb_sweeper_intent_pilot.py --group cease
python scripts/build_webb_sweeper_intent_pilot.py --group skubal
python scripts/run_intent_pilot.py
```

These dataset-specific commands write merged tables under `data/`, then `data/audited_intent_pilot_predictions.csv` and evaluation artifacts under `reports/intent_pilot/`. Real inputs are not bundled. Use `python scripts/demo.py` for an entirely offline example.

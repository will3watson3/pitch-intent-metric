"""Past-game-only comparable-pitch inference; endpoints are proxies, not intent labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
REPORT = ROOT / 'reports' / 'intent_pilot'
GROUPS = [('webb', 'webb_sweeper'), ('cease', 'cease_slider'), ('skubal', 'skubal_changeup')]
SHAPE = ['pfx_x', 'pfx_z', 'release_speed', 'release_spin_rate']
# Fixed pilot settings, not tuned on the evaluation results.
SHAPE_SCALE = np.array([0.3, 0.3, 3.0, 300.0])
VARIANTS = ['comparables', 'no_outcome_weight', 'no_shape_weight']


def infer(query, audited, history, outcome_weight=True, shape_weight=True):
    """Never access the query's realized movement, endpoint, or outcome."""
    earlier = audited[(audited.game_date < query.game_date)
                      & (audited.pitcher == query.pitcher)
                      & (audited.pitch_type == query.pitch_type)]
    candidates = earlier[earlier.stand == query.stand].copy()
    profile_rows = history[(history.game_date < query.game_date)
                           & (history.pitcher == query.pitcher)
                           & (history.pitch_type == query.pitch_type)]
    profile_rows = profile_rows.sort_values(['game_date', 'at_bat_number', 'pitch_number']).tail(100)
    result = {'status': 'insufficient_history', 'prior_group_n': len(earlier),
              'profile_n': len(profile_rows), 'candidate_n': 0, 'effective_n': 0.0}
    if len(profile_rows) < 20 or len(candidates) < 3:
        return result
    target = np.array([query.model_target_x_01, query.model_target_y_01], float)
    delta = candidates[['model_target_x_01', 'model_target_y_01']].to_numpy() - target
    distance = np.linalg.norm(delta, axis=1)
    candidates = candidates.loc[distance <= 0.6].copy()
    distance = distance[distance <= 0.6]
    result['candidate_n'] = len(candidates)
    if len(candidates) < 3:
        result['status'] = 'insufficient_nearby_targets'
        return result
    weights = np.exp(-0.5 * (distance / 0.3) ** 2)
    count_distance = (candidates.balls - query.balls).abs() + (candidates.strikes - query.strikes).abs()
    weights *= np.exp(-0.35 * count_distance.to_numpy())
    profile = profile_rows[SHAPE].median().to_numpy(float)
    if not np.isfinite(profile).all():
        result['status'] = 'missing_shape_history'
        return result
    if shape_weight:
        shape_distance = np.mean(((candidates[SHAPE].to_numpy() - profile) / SHAPE_SCALE) ** 2, axis=1)
        weights *= np.exp(-0.5 * shape_distance)
    if outcome_weight:
        # Modest evidence weights, not declarations of correctly executed intent.
        multiplier = np.ones(len(candidates))
        multiplier[candidates.description.isin(['swinging_strike', 'swinging_strike_blocked'])] = 1.5
        multiplier[candidates.description.eq('called_strike')] = 1.2
        weak_contact = candidates.description.eq('hit_into_play') & candidates.estimated_woba_using_speedangle.le(0.25)
        multiplier[weak_contact] = 1.25
        weights *= multiplier
    weights /= weights.sum()
    effective_n = 1 / np.sum(weights ** 2)
    result['effective_n'] = float(effective_n)
    if effective_n < 2.5:
        result['status'] = 'insufficient_effective_support'
        return result
    offsets = candidates[['model_miss_x_01', 'model_miss_y_01']].to_numpy(float)
    # Weighted medoid selects a supported joint offset instead of averaging clusters.
    pairwise = np.linalg.norm(offsets[:, None, :] - offsets[None, :, :], axis=2)
    center = offsets[np.argmin(pairwise @ weights)]
    finish = target + center
    def quantile(values, probability):
        order = np.argsort(values)
        return float(np.interp(probability, np.cumsum(weights[order]), values[order]))
    x_low, x_high = (target[0] + quantile(offsets[:, 0], q) for q in (0.1, 0.9))
    y_low, y_high = (target[1] + quantile(offsets[:, 1], q) for q in (0.1, 0.9))
    width_ft = 17 / 12
    height_ft = query.sz_top - query.sz_bot
    result.update(status='provisional', intent_x_01=float(finish[0]), intent_y_01=float(finish[1]),
                  intent_plate_x_ft=float((0.5 - finish[0]) * width_ft),
                  intent_plate_z_ft=float(query.sz_bot + finish[1] * height_ft),
                  region_x_low_01=float(x_low), region_x_high_01=float(x_high),
                  region_y_low_01=float(y_low), region_y_high_01=float(y_high),
                  region_area_sqft=float((x_high-x_low)*(y_high-y_low)*width_ft*height_ft),
                  evidence_tier='low',
                  comparable_pitch_uids=';'.join(candidates.pitch_uid),
                  comparable_weights=';'.join(f'{w:.6f}' for w in weights),
                  newest_comparable_date=str(candidates.game_date.max()))
    for name, value in zip(SHAPE, profile):
        result[f'expected_{name}'] = float(value)
    return result


def load_data():
    audited = pd.concat([pd.read_csv(DATA / f'{slug}_intent_pilot_model_ready.csv') for _, slug in GROUPS], ignore_index=True)
    history = pd.concat([pd.read_csv(DATA / f'{name}_2025_pitch_level.csv') for name, _ in GROUPS], ignore_index=True)
    if not audited.pitch_uid.is_unique or not history.pitch_uid.is_unique:
        raise ValueError('Duplicate pitch keys')
    return audited.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number']), history


def run():
    audited, history = load_data()
    REPORT.mkdir(parents=True, exist_ok=True)
    predictions = []
    for query in audited.itertuples(index=False):
        for variant in VARIANTS:
            estimate = infer(query, audited, history,
                             outcome_weight=variant != 'no_outcome_weight',
                             shape_weight=variant != 'no_shape_weight')
            row = dict(pitch_uid=query.pitch_uid, sample_pitcher_name=query.sample_pitcher_name,
                       pitch_type=query.pitch_type, game_date=query.game_date, variant=variant, **estimate)
            if estimate['status'] == 'provisional':
                earlier = audited[(audited.game_date < query.game_date) & (audited.pitcher == query.pitcher)
                                  & (audited.pitch_type == query.pitch_type)]
                baseline_x = query.model_target_plate_x_ft + earlier.model_miss_x_ft.mean()
                baseline_z = query.model_target_plate_z_ft + earlier.model_miss_z_ft.mean()
                row.update(endpoint_error_ft=float(np.hypot(query.plate_x - estimate['intent_plate_x_ft'],
                                                           query.plate_z - estimate['intent_plate_z_ft'])),
                           glove_error_ft=query.model_miss_distance_ft,
                           average_offset_error_ft=float(np.hypot(query.plate_x-baseline_x, query.plate_z-baseline_z)),
                           endpoint_in_region=bool(estimate['region_x_low_01'] <= query.actual_x_01 <= estimate['region_x_high_01']
                                                   and estimate['region_y_low_01'] <= query.actual_y_01 <= estimate['region_y_high_01']))
            predictions.append(row)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(REPORT / 'variant_predictions.csv', index=False)
    primary = predictions[predictions.variant == 'comparables'].copy()
    audited.merge(primary.drop(columns=['sample_pitcher_name', 'pitch_type', 'game_date']),
                  on='pitch_uid', validate='one_to_one').to_csv(DATA / 'audited_intent_pilot_predictions.csv', index=False)
    available = predictions[predictions.status == 'provisional']
    common = set.intersection(*(set(available.loc[available.variant == v, 'pitch_uid']) for v in VARIANTS))
    benchmark = []
    for group in ['All'] + sorted(audited.sample_pitcher_name.unique()):
        same = available[available.pitch_uid.isin(common)]
        if group != 'All':
            same = same[same.sample_pitcher_name == group]
        for variant in VARIANTS:
            sample = same[same.variant == variant]
            if sample.empty:
                continue
            benchmark.append(dict(group=group, model=variant, n=len(sample),
                                  mean_endpoint_error_ft=sample.endpoint_error_ft.mean(),
                                  median_endpoint_error_ft=sample.endpoint_error_ft.median(),
                                  endpoint_region_coverage=sample.endpoint_in_region.mean(),
                                  mean_region_area_sqft=sample.region_area_sqft.mean()))
        sample = same[same.variant == 'comparables']
        for name, col in [('glove_only', 'glove_error_ft'), ('past_average_offset', 'average_offset_error_ft')]:
            if len(sample):
                benchmark.append(dict(group=group, model=name, n=len(sample),
                                      mean_endpoint_error_ft=sample[col].mean(), median_endpoint_error_ft=sample[col].median()))
    metrics = pd.DataFrame(benchmark)
    metrics.to_csv(REPORT / 'benchmark.csv', index=False)
    summary = dict(accepted_pitches=len(audited),
                   accepted_by_pitcher=audited.sample_pitcher_name.value_counts().to_dict(),
                   prediction_statuses=primary.status.value_counts().to_dict(),
                   common_evaluation_pitches=len(common), model_version='weighted_comparables_v1')
    (REPORT / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Intent pilot results', '', json.dumps(summary, indent=2), '',
             'Evaluation holds out every current game and uses earlier dates only. Settings were fixed before running.',
             'Profiles use up to 100 earlier same-pitcher/type pitches; glove comparables additionally match batter side and nearby targets.',
             'Current pitch movement, location and outcome never enter inference. Historical outcomes receive at most 1.5x weight.',
             'Three comparable pitches and effective support >= 2.5 are required. All estimates remain provisional/low evidence.', '',
             '## Same-pitch endpoint benchmarks', '', '```', metrics.to_string(index=False), '```', '',
             'Endpoint error measures finish prediction, not recovered intent or command accuracy.',
             'Regions are empirical marginal 10–90% bounds, not calibrated joint confidence regions.',
             'Observed region coverage is too low to present these bounds as reliable intended-finish areas.',
             'The full model did not outperform the past-average-offset baseline on this run. Removing outcome weights improved endpoint error; this is exploratory, not an independently validated model selection.',
             'Decision: keep this as a research pilot, with no production command grading or high-confidence intent claims.',
             'A good outcome is not proof of intended execution. No model has independent true-intent labels here.',
             'Do not promote this model based solely on reducing endpoint error; review inferred finish areas independently.',
             'Statcast plate conversion uses the existing project approximation, with x flipped from pitcher-view targets to catcher-view plate_x.',
             'Realized movement columns in the merged output are diagnostic; inference uses only historical profiles.',
             'Original audit coordinates are preserved. No detector, zone archive, or dashboard scoring was changed.']
    (REPORT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))
    print(metrics[metrics.group == 'All'].to_string(index=False))


if __name__ == '__main__':
    run()

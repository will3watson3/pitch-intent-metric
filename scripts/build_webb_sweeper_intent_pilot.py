"""Merge a selected audited pitcher/pitch group with authoritative Statcast data."""

from pathlib import Path
import argparse
import numpy as np

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLATE_EDGE_FT = 17 / 12 / 2

AUDIT_PATH = DATA_DIR / "target_audit_labels.csv"
ANALYSIS_PATH = DATA_DIR / "labeled_target_analysis_sample_400.csv"
MANIFEST_PATH = DATA_DIR / "setup_frame_manifest_31_1200_2700.csv"

MODEL_READY_STATUSES = {"confirmed", "corrected"}

def require_unique(df: pd.DataFrame, key: str, source: Path) -> None:
    duplicates = df.loc[df[key].duplicated(keep=False), key].dropna().unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate {key} values in {source}: {duplicates[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['webb', 'cease', 'skubal'], default='webb')
    args = parser.parse_args()
    name, pitch_type, slug = {
        'webb': ('Logan Webb', 'ST', 'webb_sweeper'),
        'cease': ('Dylan Cease', 'SL', 'cease_slider'),
        'skubal': ('Tarik Skubal', 'CH', 'skubal_changeup'),
    }[args.group]
    statcast_path = DATA_DIR / f'{args.group}_2025_pitch_level.csv'
    full_output = DATA_DIR / f'{slug}_target_audit_merged.csv'
    model_output = DATA_DIR / f'{slug}_intent_pilot_model_ready.csv'
    audit = pd.read_csv(AUDIT_PATH, dtype={"pitch_uid": str, "frame_uid": str})
    audit = audit[(audit.sample_pitcher_name == name) & (audit.pitch_type == pitch_type)].copy()
    analysis = pd.read_csv(ANALYSIS_PATH, dtype={"pitch_uid": str})
    statcast = pd.read_csv(statcast_path, dtype={"pitch_uid": str})
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"pitch_uid": str, "frame_uid": str})

    require_unique(audit, "pitch_uid", AUDIT_PATH)
    require_unique(analysis, "pitch_uid", ANALYSIS_PATH)
    require_unique(statcast, "pitch_uid", statcast_path)
    require_unique(manifest, "frame_uid", MANIFEST_PATH)

    analysis = analysis[
        (analysis["sample_pitcher_name"] == name)
        & (analysis["pitch_type"] == pitch_type)
    ].copy()
    statcast = statcast[statcast["pitch_type"] == pitch_type].copy()
    if set(audit.pitch_uid) != set(analysis.pitch_uid):
        raise ValueError('Audit must cover the complete selected group')
    frame_pitch = manifest.set_index('frame_uid').pitch_uid
    if not audit.frame_uid.map(frame_pitch).eq(audit.pitch_uid).all():
        raise ValueError('Audit frame belongs to a different pitch or is missing')
    # Retain original label metadata, but remove obsolete inferred-intent outputs.
    label_cols = ['pitch_uid', 'sample_pitcher_name', 'source_file',
                  'target_x_01', 'target_y_01', 'target_confidence_1_to_5',
                  'setup_visible', 'label_notes', 'direct_mp4_url', 'savant_video_url']
    analysis = analysis[label_cols].merge(statcast, on='pitch_uid', validate='one_to_one')

    audit = audit.rename(
        columns={
            "frame_uid": "audit_frame_uid",
            "audited_target_x_01": "audit_target_x_01",
            "audited_target_y_01": "audit_target_y_01",
            "sample_pitcher_name": "audit_sample_pitcher_name",
            "pitch_type": "audit_pitch_type",
        }
    )
    frame_lookup = manifest[
        ["frame_uid", "frame_time_sec", "image_path", "extraction_status"]
    ].rename(
        columns={
            "frame_uid": "audit_frame_uid",
            "frame_time_sec": "audit_frame_time_sec",
            "image_path": "audit_image_path",
            "extraction_status": "audit_frame_extraction_status",
        }
    )
    audit = audit.merge(frame_lookup, on="audit_frame_uid", how="left", validate="one_to_one")

    missing_analysis = sorted(set(audit["pitch_uid"]) - set(analysis["pitch_uid"]))
    missing_statcast = sorted(set(audit["pitch_uid"]) - set(statcast["pitch_uid"]))
    missing_frames = audit.loc[audit["audit_frame_time_sec"].isna(), "audit_frame_uid"].tolist()
    if missing_analysis or missing_statcast or missing_frames:
        raise ValueError(
            "Incomplete merge: "
            f"analysis={missing_analysis}, statcast={missing_statcast}, frames={missing_frames}"
        )

    merged = audit.merge(analysis, on="pitch_uid", how="left", validate="one_to_one")
    merged['actual_x_01'] = 0.5 - merged.plate_x / (2 * PLATE_EDGE_FT)
    merged['actual_y_01'] = (merged.plate_z - merged.sz_bot) / (merged.sz_top - merged.sz_bot)

    merged["model_ready"] = merged["audit_status"].isin(MODEL_READY_STATUSES)
    merged["target_was_corrected"] = merged["audit_status"].eq("corrected")
    merged["target_correction_dx_01"] = (
        pd.to_numeric(merged["audit_target_x_01"], errors="coerce")
        - pd.to_numeric(merged["original_target_x_01"], errors="coerce")
    )
    merged["target_correction_dy_01"] = (
        pd.to_numeric(merged["audit_target_y_01"], errors="coerce")
        - pd.to_numeric(merged["original_target_y_01"], errors="coerce")
    )

    ready = merged["model_ready"]
    merged["model_target_x_01"] = pd.to_numeric(
        merged["audit_target_x_01"], errors="coerce"
    ).where(ready)
    merged["model_target_y_01"] = pd.to_numeric(
        merged["audit_target_y_01"], errors="coerce"
    ).where(ready)
    required = ['model_target_x_01', 'model_target_y_01', 'plate_x', 'plate_z',
                'sz_top', 'sz_bot', 'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z']
    if not np.isfinite(merged.loc[ready, required].to_numpy(dtype=float)).all():
        raise ValueError('Accepted audit has missing or nonfinite core data')
    if not (merged.loc[ready, 'sz_top'] > merged.loc[ready, 'sz_bot']).all():
        raise ValueError('Invalid strike-zone height')
    merged["model_target_source"] = "excluded_" + merged["audit_status"].astype(str)
    merged.loc[merged["audit_status"].eq("confirmed"), "model_target_source"] = (
        "audited_confirmed_original"
    )
    merged.loc[merged["audit_status"].eq("corrected"), "model_target_source"] = (
        "audited_manual_correction"
    )

    merged["model_target_plate_x_ft"] = PLATE_EDGE_FT * (
        1 - (2 * merged["model_target_x_01"])
    )
    merged["model_target_plate_z_ft"] = merged["sz_bot"] + (
        merged["model_target_y_01"] * (merged["sz_top"] - merged["sz_bot"])
    )
    merged["model_miss_x_01"] = merged["actual_x_01"] - merged["model_target_x_01"]
    merged["model_miss_y_01"] = merged["actual_y_01"] - merged["model_target_y_01"]
    merged["model_miss_distance_01"] = (
        merged["model_miss_x_01"] ** 2 + merged["model_miss_y_01"] ** 2
    ) ** 0.5
    merged["model_miss_x_ft"] = merged["plate_x"] - merged["model_target_plate_x_ft"]
    merged["model_miss_z_ft"] = merged["plate_z"] - merged["model_target_plate_z_ft"]
    merged["model_miss_distance_ft"] = (
        merged["model_miss_x_ft"] ** 2 + merged["model_miss_z_ft"] ** 2
    ) ** 0.5

    priority = [
        "pitch_uid",
        "audit_status",
        "model_ready",
        "audit_frame_uid",
        "audit_frame_time_sec",
        "audit_image_path",
        "audited_at",
        "audit_notes",
        "original_target_x_01",
        "original_target_y_01",
        "audit_target_x_01",
        "audit_target_y_01",
        "target_was_corrected",
        "target_correction_dx_01",
        "target_correction_dy_01",
        "model_target_x_01",
        "model_target_y_01",
        "model_target_plate_x_ft",
        "model_target_plate_z_ft",
        "model_target_source",
        "model_miss_x_01",
        "model_miss_y_01",
        "model_miss_distance_01",
        "model_miss_x_ft",
        "model_miss_z_ft",
        "model_miss_distance_ft",
    ]
    remaining = [column for column in merged.columns if column not in priority]
    merged = merged[priority + remaining].sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )

    merged.to_csv(full_output, index=False)
    model_ready = merged.loc[merged["model_ready"]].copy()
    model_ready.to_csv(model_output, index=False)

    print(f"Wrote {len(merged)} audited pitches to {full_output.relative_to(ROOT)}")
    print(f"Wrote {len(model_ready)} accepted pitches to {model_output.relative_to(ROOT)}")
    print(f"Corrected targets: {int(model_ready['target_was_corrected'].sum())}")


if __name__ == "__main__":
    main()

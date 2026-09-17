import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLATE_EDGE_FT = 17 / 12 / 2
FASTBALL_PITCHES = {"FF", "SI", "FC", "FA"}
BREAKING_PITCHES = {"SL", "ST", "CU", "KC", "SV", "CS"}
OFFSPEED_PITCHES = {"CH", "FS", "FO", "SC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join autonomous vision results to Statcast and calculate target metrics."
    )
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--pitch-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def number(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed


def bounded(value, default: float = 0.0) -> float:
    parsed = number(value)
    if parsed is None:
        return default
    return max(0.0, min(1.0, parsed))


def pitch_family(value) -> str:
    pitch_type = str(value or "").upper()
    if pitch_type in FASTBALL_PITCHES:
        return "fastball"
    if pitch_type in BREAKING_PITCHES:
        return "breaking"
    if pitch_type in OFFSPEED_PITCHES:
        return "offspeed"
    return "other"


def is_away_target(row: pd.Series) -> bool:
    stand = str(row.get("stand") or "").upper()
    target_x = number(row.get("target_x"))
    if target_x is None:
        return False
    if stand == "R":
        return target_x <= 0.15
    if stand == "L":
        return target_x >= 0.85
    return target_x <= 0.05 or target_x >= 0.95


def intent_label(row: pd.Series) -> str:
    target_x = number(row.get("target_x"))
    target_y = number(row.get("target_y"))
    balls = number(row.get("balls"))
    strikes = number(row.get("strikes"))
    family = str(row.get("pitch_family") or "")
    if target_x is None or target_y is None:
        return "unknown"

    two_strikes = strikes is not None and strikes >= 2
    ahead = balls is not None and strikes is not None and strikes > balls
    far_outside = target_x <= -0.15 or target_x >= 1.15
    if two_strikes and (target_y > 1.18 or target_y < -0.12 or target_x < -0.35 or target_x > 1.35):
        return "waste"
    if target_y >= 1.02 and (two_strikes or family == "fastball"):
        return "chase_up"
    if family in {"breaking", "offspeed"} and is_away_target(row) and (two_strikes or ahead or far_outside):
        return "chase_away"
    if target_y < 0:
        return "chase_down"
    if family == "breaking" and far_outside:
        return "frontdoor_backdoor"
    if 0 <= target_x <= 1 and 0 <= target_y <= 1:
        return "zone_attack"
    return "unknown"


def confidence_for_row(row: pd.Series) -> tuple[float, str, str, str]:
    zone_confidence = bounded(row.get("zone_confidence"))
    glove_confidence = bounded(row.get("glove_confidence"))
    zone_quality = bounded((number(row.get("zone_quality_score")) or 0.0) / 100)
    track_stability = bounded(row.get("glove_track_stability"))
    track_frame_count = number(row.get("glove_track_frame_count")) or 0.0
    stable_run_count = number(row.get("glove_stable_run_count")) or 0.0
    track_support = bounded(track_frame_count / 8)
    stable_support = bounded(stable_run_count / 4)
    pose_coverage = bounded(row.get("pose_coverage"), default=0.5)
    drop_confirmed = str(row.get("glove_drop_confirmed") or "").lower() == "yes"
    strong_stable_hold = (
        str(row.get("strong_stable_glove_hold") or "").lower() == "yes"
        or (
            glove_confidence >= 0.50
            and track_frame_count >= 3
            and track_stability >= 0.80
            and stable_run_count >= 3
        )
    )
    setup_evidence_score = 1.0 if drop_confirmed or strong_stable_hold else 0.35

    score = 100 * (
        (0.24 * zone_confidence)
        + (0.28 * glove_confidence)
        + (0.14 * zone_quality)
        + (0.14 * track_stability)
        + (0.08 * track_support)
        + (0.05 * stable_support)
        + (0.04 * pose_coverage)
        + (0.03 * setup_evidence_score)
    )

    reasons = []
    zone_source = str(row.get("zone_anchor_source") or "")
    source_penalties = {
        "dense_pitch_consensus": 3,
        "dense_plate_appearance_consensus": 7,
        "broadcast_plate_appearance_template": 2,
        "broadcast_pitch_edge_template": 1,
        "broadcast_batter_horizontal_template": 4,
        "broadcast_game_template_statcast": 6,
        "dense_selected_center_manual_size": 8,
        "manual_game_fallback": 22,
        "detected_global_median": 25,
    }
    score -= source_penalties.get(zone_source, 0)
    reasons.append(zone_source or "unknown_zone_source")

    zone_status = str(row.get("zone_quality_status") or "")
    if zone_status == "warn":
        score *= 0.92
        reasons.append("zone_warning")
    elif zone_status in {"poor", "bad_reviewed"}:
        score *= 0.72
        reasons.append("poor_zone")
    elif zone_status == "missing":
        score = 0
        reasons.append("missing_zone")

    if glove_confidence < 0.50:
        reasons.append("low_glove_confidence")
    if track_stability >= 0.70 and track_support >= 0.50:
        reasons.append("stable_glove_track")
    else:
        reasons.append("limited_glove_track")
    if drop_confirmed:
        reasons.append("drop_confirmed")
    elif strong_stable_hold:
        reasons.append("strong_stable_hold")
    else:
        reasons.append("setup_evidence_limited")

    if str(row.get("selected_after_delivery_start") or "").lower() == "no":
        score *= 0.90
        reasons.append("before_delivery_signal")

    target_x = number(row.get("vision_target_x_01"))
    target_y = number(row.get("vision_target_y_01"))
    status = str(row.get("status") or "")
    filter_status = str(row.get("vision_filter_status") or "")
    excluded = filter_status == "excluded"
    if target_x is None or target_y is None:
        score = 0
        reasons.append("missing_target")
    elif target_x < -0.75 or target_x > 1.75 or target_y < -0.60 or target_y > 1.50:
        score *= 0.70
        reasons.append("extreme_target")
    if status != "ok":
        score = min(score, 20)
        reasons.append("prediction_not_ok")
    if filter_status == "needs_review":
        score = min(score, 79)
        reasons.append("quality_review_required")
    if excluded:
        score = 0
        reasons.append("excluded")

    score = round(max(0.0, min(100.0, score)), 1)
    tier = "high" if score >= 80 else "medium" if score >= 65 else "low"
    ready = "yes" if score >= 80 and status == "ok" and filter_status == "trusted" else "no"
    return score, tier, ready, ";".join(dict.fromkeys(reasons))


def component_confidence_for_row(row: pd.Series) -> tuple[float, str, float, str]:
    zone_confidence = bounded(row.get("zone_confidence"))
    zone_quality = bounded((number(row.get("zone_quality_score")) or 0.0) / 100)
    zone_score = 100 * ((0.65 * zone_confidence) + (0.35 * zone_quality))
    zone_source = str(row.get("zone_anchor_source") or "")
    zone_score -= {
        "dense_pitch_consensus": 3,
        "dense_plate_appearance_consensus": 7,
        "broadcast_plate_appearance_template": 2,
        "broadcast_pitch_edge_template": 1,
        "broadcast_batter_horizontal_template": 4,
        "broadcast_game_template_statcast": 6,
        "dense_game_consensus": 10,
        "manual_game_fallback": 22,
        "detected_global_median": 25,
    }.get(zone_source, 0)
    zone_status = str(row.get("zone_quality_status") or "")
    if zone_status == "warn":
        zone_score *= 0.90
    elif zone_status in {"poor", "bad_reviewed"}:
        zone_score *= 0.65
    elif zone_status == "missing":
        zone_score = 0

    glove_confidence = bounded(row.get("glove_confidence"))
    track_stability = bounded(row.get("glove_track_stability"))
    track_frames = number(row.get("glove_track_frame_count")) or 0.0
    stable_frames = number(row.get("glove_stable_run_count")) or 0.0
    track_support = bounded(track_frames / 8)
    stable_support = bounded(stable_frames / 4)
    pose_coverage = bounded(row.get("pose_coverage"), default=0.5)
    setup_evidence = str(row.get("glove_setup_evidence") or "")
    evidence_score = 1.0 if setup_evidence in {"confirmed_drop", "stable_hold"} else 0.25
    target_score = 100 * (
        (0.40 * glove_confidence)
        + (0.24 * track_stability)
        + (0.13 * track_support)
        + (0.10 * stable_support)
        + (0.05 * pose_coverage)
        + (0.08 * evidence_score)
    )
    target_x = number(row.get("vision_target_x_01"))
    target_y = number(row.get("vision_target_y_01"))
    if target_x is None or target_y is None or str(row.get("status") or "") != "ok":
        target_score = 0
    elif target_x < -0.75 or target_x > 1.75 or target_y < -0.60 or target_y > 1.50:
        target_score *= 0.70

    zone_score = round(max(0.0, min(100.0, zone_score)), 1)
    target_score = round(max(0.0, min(100.0, target_score)), 1)
    zone_tier = "high" if zone_score >= 80 else "medium" if zone_score >= 65 else "low"
    target_tier = "high" if target_score >= 80 else "medium" if target_score >= 65 else "low"
    return target_score, target_tier, zone_score, zone_tier


def add_location_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["plate_x", "plate_z", "sz_top", "sz_bot", "vision_target_x_01", "vision_target_y_01"]:
        df[column] = pd.to_numeric(df.get(column), errors="coerce")

    zone_height = df["sz_top"] - df["sz_bot"]
    df["actual_x_01"] = (-df["plate_x"] + PLATE_EDGE_FT) / (2 * PLATE_EDGE_FT)
    df["actual_y_01"] = (df["plate_z"] - df["sz_bot"]) / zone_height
    df["target_x"] = df["vision_target_x_01"]
    df["target_y"] = df["vision_target_y_01"]
    df["target_source"] = "autonomous_vision"
    df["target_plate_x_ft"] = PLATE_EDGE_FT * (1 - (2 * df["target_x"]))
    df["target_plate_z_ft"] = df["sz_bot"] + (df["target_y"] * zone_height)
    df["setup_miss_x"] = df["actual_x_01"] - df["target_x"]
    df["setup_miss_y"] = df["actual_y_01"] - df["target_y"]
    df["setup_miss"] = np.hypot(df["setup_miss_x"], df["setup_miss_y"])
    df["setup_miss_x_ft"] = df["plate_x"] - df["target_plate_x_ft"]
    df["setup_miss_z_ft"] = df["plate_z"] - df["target_plate_z_ft"]
    df["setup_miss_ft"] = np.hypot(df["setup_miss_x_ft"], df["setup_miss_z_ft"])
    df["pitch_family"] = df.get("pitch_type", pd.Series(index=df.index, dtype=str)).map(pitch_family)
    df["intent_label"] = df.apply(intent_label, axis=1)
    return df


def main() -> None:
    args = parse_args()
    quality_path = resolve(args.quality)
    pitch_data_path = resolve(args.pitch_data)
    output_path = resolve(args.output)
    summary_path = resolve(args.summary)
    if not quality_path.exists():
        raise SystemExit(f"Missing quality data: {quality_path}")
    if not pitch_data_path.exists():
        raise SystemExit(f"Missing pitch data: {pitch_data_path}")

    quality = pd.read_csv(quality_path)
    pitch_data = pd.read_csv(pitch_data_path)
    quality["pitch_uid"] = quality["pitch_uid"].astype(str)
    pitch_data["pitch_uid"] = pitch_data["pitch_uid"].astype(str)
    extra_columns = [
        column for column in pitch_data.columns if column == "pitch_uid" or column not in quality.columns
    ]
    merged = quality.merge(
        pitch_data[extra_columns].drop_duplicates("pitch_uid", keep="first"),
        on="pitch_uid",
        how="left",
        validate="one_to_one",
    )
    merged = add_location_metrics(merged)
    confidence = merged.apply(confidence_for_row, axis=1, result_type="expand")
    merged["vision_confidence_score"] = confidence[0]
    merged["vision_confidence_tier"] = confidence[1]
    merged["autonomous_ready"] = confidence[2]
    merged["vision_confidence_reasons"] = confidence[3]
    component_confidence = merged.apply(component_confidence_for_row, axis=1, result_type="expand")
    merged["target_confidence_score"] = component_confidence[0]
    merged["target_confidence_tier"] = component_confidence[1]
    merged["zone_confidence_score"] = component_confidence[2]
    merged["zone_confidence_tier"] = component_confidence[3]
    merged["review_recommended"] = merged["autonomous_ready"].map({"yes": "no", "no": "yes"})

    sort_columns = [
        column
        for column in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if column in merged.columns
    ]
    if sort_columns:
        merged = merged.sort_values(sort_columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.fillna("").to_csv(output_path, index=False)

    target_count = int(
        pd.to_numeric(merged["vision_target_x_01"], errors="coerce").notna().sum()
    )
    summary = {
        "pitch_rows_requested": int(len(pitch_data)),
        "pitch_rows_with_vision": int(len(merged)),
        "targets_created": target_count,
        "autonomous_ready": int((merged["autonomous_ready"] == "yes").sum()),
        "mean_confidence": round(float(merged["vision_confidence_score"].mean()), 1) if len(merged) else 0,
        "confidence_tiers": {
            str(key): int(value)
            for key, value in merged["vision_confidence_tier"].value_counts().items()
        },
        "vision_statuses": {
            str(key): int(value)
            for key, value in merged["vision_filter_status"].value_counts().items()
        },
        "zone_sources": {
            str(key): int(value)
            for key, value in merged["zone_anchor_source"].value_counts().items()
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Rows: {len(merged)}")
    print(f"Targets: {target_count}")
    print(f"Autonomous-ready: {summary['autonomous_ready']}")
    print(f"Mean confidence: {summary['mean_confidence']}")


if __name__ == "__main__":
    main()

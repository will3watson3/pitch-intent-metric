import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_PREDICTIONS = DATA_DIR / "roboflow_workflow_predictions_pose_drop_hybrid_100ms_50ms.csv"
DEFAULT_ZONES = DATA_DIR / "game_calibrated_statcast_zones_50ms_400.csv"
DEFAULT_OUTPUT = DATA_DIR / "roboflow_workflow_predictions_pose_drop_hybrid_zone_calibrated.csv"

ZONE_COLUMNS = [
    "zone_confidence",
    "zone_refinement_status",
    "image_width",
    "image_height",
    "zone_edge_score",
    "zone_edge_min_support",
    "zone_refinement_frame_support",
    "zone_refinement_reference_frame_uid",
    "zone_baseline_source",
    "zone_source",
    "zone_anchor_source",
    "zone_registration_source",
    "zone_anchor_pitch_support",
    "zone_anchor_frame_support",
    "zone_anchor_center_mad_px",
    "zone_plate_appearance_key",
    "zone_calibration_reference_pitch_uid",
    "zone_calibration_reference_status",
    "statcast_sz_top",
    "statcast_sz_bot",
    "statcast_zone_height_ft",
    "zone_x",
    "zone_y",
    "zone_width",
    "zone_height",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply updated zone geometry without changing selected glove frames."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def number(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def recompute_target(row: pd.Series) -> tuple[Optional[float], Optional[float]]:
    zone_x = number(row.get("zone_x"))
    zone_y = number(row.get("zone_y"))
    zone_width = number(row.get("zone_width"))
    zone_height = number(row.get("zone_height"))
    glove_x = number(row.get("glove_x"))
    glove_y = number(row.get("glove_y"))
    if None in {zone_x, zone_y, zone_width, zone_height, glove_x, glove_y}:
        return None, None
    if zone_width <= 0 or zone_height <= 0:
        return None, None

    zone_left = zone_x - (zone_width / 2)
    zone_top = zone_y - (zone_height / 2)
    return (
        (glove_x - zone_left) / zone_width,
        1 - ((glove_y - zone_top) / zone_height),
    )


def validate_zone_frames(predictions: pd.DataFrame, zones: pd.DataFrame) -> None:
    for name, table in [("predictions", predictions), ("zones", zones)]:
        if table.pitch_uid.isna().any() or table.pitch_uid.duplicated().any():
            raise ValueError(f"{name} must contain one row per nonempty pitch_uid")
    shared = predictions.merge(zones, on="pitch_uid", suffixes=("_prediction", "_zone"))
    for column in ["frame_uid", "image_path"]:
        left, right = f"{column}_prediction", f"{column}_zone"
        if left in shared and right in shared:
            mismatch = shared[left].fillna("").astype(str) != shared[right].fillna("").astype(str)
            if mismatch.any():
                ids = shared.loc[mismatch, "pitch_uid"].head(5).tolist()
                raise ValueError(f"Zone {column} does not match selected frame for {ids}; rebuild zones using these predictions")


def main() -> None:
    args = parse_args()
    predictions_path = resolve(args.predictions)
    zones_path = resolve(args.zones)
    output_path = resolve(args.output)
    if not predictions_path.exists():
        raise SystemExit(f"Missing predictions: {predictions_path}")
    if not zones_path.exists():
        raise SystemExit(f"Missing zones: {zones_path}")

    predictions = pd.read_csv(predictions_path)
    zones = pd.read_csv(zones_path)
    validate_zone_frames(predictions, zones)
    zone_columns = ["pitch_uid", *[column for column in ZONE_COLUMNS if column in zones.columns]]
    merged = predictions.drop(
        columns=[column for column in ZONE_COLUMNS if column in predictions.columns],
        errors="ignore",
    ).merge(zones[zone_columns], on="pitch_uid", how="left", validate="one_to_one")

    targets = merged.apply(recompute_target, axis=1, result_type="expand")
    merged["vision_target_x_01"] = targets[0]
    merged["vision_target_y_01"] = targets[1]
    geometry = merged[["zone_x", "zone_y", "zone_width", "zone_height"]].apply(pd.to_numeric, errors="coerce")
    has_zone = geometry.map(lambda value: math.isfinite(value)).all(axis=1) & geometry.zone_width.gt(0) & geometry.zone_height.gt(0)
    has_glove = merged[["glove_x", "glove_y"]].notna().all(axis=1)
    merged["status"] = "ok"
    merged.loc[~has_zone, "status"] = "missing_required_detection"
    merged.loc[has_zone & ~has_glove, "status"] = "missing_required_detection"
    merged["error"] = ""
    merged.loc[~has_zone, "error"] = "missing_calibrated_zone"
    merged.loc[has_zone & ~has_glove, "error"] = "missing_vision_target"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.fillna("").to_csv(output_path, index=False)
    print(f"Output: {output_path}")
    print(f"Rows: {len(merged)}")
    print(f"Selected gloves preserved: {int(has_glove.sum())}")
    print(f"Targets recalculated: {int((has_zone & has_glove).sum())}")
    if "zone_calibration_reference_status" in merged:
        print("Calibration references:")
        print(merged["zone_calibration_reference_status"].fillna("automatic").value_counts().to_string())


if __name__ == "__main__":
    main()

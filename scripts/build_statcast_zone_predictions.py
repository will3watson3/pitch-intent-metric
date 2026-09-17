import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from refine_broadcast_zone import ZoneRefiner


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_MANIFEST = DATA_DIR / "setup_frame_manifest_31_1200_2700.csv"
DEFAULT_PRIMARY = DATA_DIR / "roboflow_workflow_predictions_pose_drop_hybrid_100ms_50ms.csv"
DEFAULT_FALLBACK = DATA_DIR / "roboflow_workflow_predictions_v2_conf10.csv"
DEFAULT_DENSE = DATA_DIR / "roboflow_workflow_predictions_frames_31_1200_2700.csv"
DEFAULT_CALIBRATIONS = DATA_DIR / "frame_zone_calibrations.json"
DEFAULT_OUTPUT = DATA_DIR / "game_calibrated_statcast_zones_400.csv"
DEFAULT_FRAME_WIDTH = 1280.0
DEFAULT_FRAME_HEIGHT = 720.0
DEFAULT_ZONE_CALIBRATION = {
    "left": 43.0,
    "top": 22.0,
    "width": 14.0,
    "height": 31.0,
}

OUTPUT_COLUMNS = [
    "frame_uid",
    "pitch_uid",
    "image_path",
    "workflow_output_key",
    "zone_confidence",
    "zone_refinement_status",
    "image_width",
    "image_height",
    "zone_edge_score",
    "zone_edge_min_support",
    "zone_refinement_frame_support",
    "zone_refinement_reference_frame_uid",
    "zone_baseline_source",
    "glove_confidence",
    "prediction_count",
    "zone_source",
    "zone_anchor_source",
    "zone_registration_source",
    "zone_anchor_pitch_support",
    "zone_anchor_frame_support",
    "zone_anchor_center_mad_px",
    "zone_plate_appearance_key",
    "zone_calibration_reference_pitch_uid",
    "zone_calibration_reference_status",
    "glove_source_model",
    "statcast_sz_top",
    "statcast_sz_bot",
    "statcast_zone_height_ft",
    "zone_x",
    "zone_y",
    "zone_width",
    "zone_height",
    "glove_x",
    "glove_y",
    "glove_width",
    "glove_height",
    "vision_target_x_01",
    "vision_target_y_01",
    "raw_response_path",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--primary-predictions", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--fallback-predictions", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--dense-predictions", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--calibrations", type=Path, default=DEFAULT_CALIBRATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-primary-glove-confidence", type=float, default=0.50)
    parser.add_argument("--min-fallback-glove-confidence", type=float, default=0.10)
    parser.add_argument("--min-detected-zone-confidence", type=float, default=0.60)
    parser.add_argument("--zone-min-width", type=float, default=50.0)
    parser.add_argument("--zone-max-width", type=float, default=95.0)
    parser.add_argument("--zone-min-height", type=float, default=65.0)
    parser.add_argument("--zone-max-height", type=float, default=125.0)
    parser.add_argument("--zone-min-aspect", type=float, default=1.05)
    parser.add_argument("--zone-max-aspect", type=float, default=1.70)
    parser.add_argument("--registration-min-frames", type=int, default=4)
    parser.add_argument("--registration-max-center-mad-px", type=float, default=8.0)
    parser.add_argument("--registration-max-shift-px", type=float, default=24.0)
    parser.add_argument("--template-min-pitches", type=int, default=4)
    parser.add_argument("--template-max-center-delta-px", type=float, default=30.0)
    parser.add_argument("--template-max-size-delta-px", type=float, default=20.0)
    parser.add_argument("--template-pa-min-frames", type=int, default=4)
    parser.add_argument("--template-local-min-pitches", type=int, default=2)
    parser.add_argument("--template-pa-max-horizontal-shift-px", type=float, default=20.0)
    parser.add_argument("--template-pa-max-vertical-shift-px", type=float, default=30.0)
    parser.add_argument("--template-edge-frames-per-pitch", type=int, default=9)
    parser.add_argument("--template-edge-min-score", type=float, default=0.65)
    parser.add_argument("--template-edge-min-frames", type=int, default=3)
    parser.add_argument("--template-edge-max-center-mad-px", type=float, default=3.0)
    parser.add_argument("--template-edge-search-radius-px", type=float, default=35.0)
    return parser.parse_args()


def resolve(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


def normalized(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def number(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def latest_analysis_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing manifest: {path}")
    manifest = pd.read_csv(path)
    manifest = manifest.drop_duplicates("pitch_uid", keep="first").copy()
    manifest["game_pk_key"] = manifest["game_pk"].astype(str)
    manifest["at_bat_number_key"] = manifest["at_bat_number"].astype(str)
    manifest["plate_appearance_key"] = (
        manifest["game_pk_key"] + ":" + manifest["at_bat_number_key"]
    )
    if "batter" in manifest.columns:
        batter_ids = pd.to_numeric(manifest["batter"], errors="coerce").astype("Int64").astype(str)
    else:
        batter_ids = pd.Series("unknown", index=manifest.index)
    manifest["game_batter_key"] = manifest["game_pk_key"] + ":" + batter_ids
    manifest["sz_height_ft"] = pd.to_numeric(manifest["sz_top"], errors="coerce") - pd.to_numeric(
        manifest["sz_bot"], errors="coerce"
    )
    manifest["sz_center_ft"] = (
        pd.to_numeric(manifest["sz_top"], errors="coerce") + pd.to_numeric(manifest["sz_bot"], errors="coerce")
    ) / 2
    return manifest


def read_predictions(path: Optional[Path], source_label: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["pitch_uid"])
    df = pd.read_csv(path).drop_duplicates("pitch_uid", keep="first").copy()
    df["prediction_source_label"] = source_label
    return df


def read_dense_predictions(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["pitch_uid"])
    return pd.read_csv(path).copy()


def row_value(row: pd.Series, column: str) -> Optional[float]:
    return number(row.get(column))


def has_glove(row: Optional[pd.Series], min_confidence: float) -> bool:
    if row is None:
        return False
    glove_x = row_value(row, "glove_x")
    glove_y = row_value(row, "glove_y")
    confidence = row_value(row, "glove_confidence") or 0.0
    return glove_x is not None and glove_y is not None and confidence >= min_confidence


def choose_glove_row(
    primary_row: Optional[pd.Series],
    fallback_row: Optional[pd.Series],
    min_primary_confidence: float,
    min_fallback_confidence: float,
) -> tuple[Optional[pd.Series], str]:
    # Zone rebuilding must stay in the selected frame's coordinate system.
    if primary_row is not None:
        return primary_row, "primary" if has_glove(primary_row, min_primary_confidence) else "primary_low_confidence"
    if has_glove(fallback_row, min_fallback_confidence):
        return fallback_row, "fallback"
    if primary_row is not None:
        return primary_row, "primary_missing_glove"
    if fallback_row is not None:
        return fallback_row, "fallback_missing_glove"
    return None, "none"


def valid_zone_box(row: pd.Series, args: argparse.Namespace) -> bool:
    zone_x = row_value(row, "zone_x")
    zone_y = row_value(row, "zone_y")
    zone_width = row_value(row, "zone_width")
    zone_height = row_value(row, "zone_height")
    confidence = row_value(row, "zone_confidence") or 0.0
    if None in {zone_x, zone_y, zone_width, zone_height}:
        return False
    aspect = zone_height / zone_width if zone_width else None
    return (
        confidence >= args.min_detected_zone_confidence
        and args.zone_min_width <= zone_width <= args.zone_max_width
        and args.zone_min_height <= zone_height <= args.zone_max_height
        and aspect is not None
        and args.zone_min_aspect <= aspect <= args.zone_max_aspect
    )


def broadcast_box_edge_center(
    image_path: str,
    zone: dict,
    max_shift_px: float,
) -> tuple[Optional[float], float]:
    """Snap an approximate zone to the pair of visible vertical broadcast-box edges."""
    path = resolve(Path(image_path))
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path else None
    if image is None:
        return None, 0.0

    center_x = float(zone["zone_x"])
    center_y = float(zone["zone_y"])
    width = float(zone["zone_width"])
    height = float(zone["zone_height"])
    if width <= 0 or height <= 0:
        return None, 0.0

    horizontal_pad = max_shift_px + 5
    x0 = max(0, int(center_x - (width / 2) - horizontal_pad))
    x1 = min(image.shape[1], int(center_x + (width / 2) + horizontal_pad))
    y0 = max(0, int(center_y - (height / 2) - 12))
    y1 = min(image.shape[0], int(center_y + (height / 2) + 12))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, 0.0

    crop = cv2.GaussianBlur(image[y0:y1, x0:x1], (3, 3), 0)
    edges = cv2.Canny(crop, 35, 110)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(12, int(height * 0.20)),
        minLineLength=max(16, int(height * 0.25)),
        maxLineGap=8,
    )
    if lines is None:
        return None, 0.0

    vertical_lines = []
    for raw_line in lines[:, 0]:
        xa, ya, xb, yb = (int(value) for value in raw_line)
        if abs(xb - xa) > 4 or abs(yb - ya) < max(16, height * 0.25):
            continue
        vertical_lines.append(
            (
                ((xa + xb) / 2) + x0,
                min(ya, yb) + y0,
                max(ya, yb) + y0,
            )
        )

    zone_top = center_y - (height / 2)
    zone_bottom = center_y + (height / 2)
    candidates = []
    for left_index, line_a in enumerate(vertical_lines):
        for line_b in vertical_lines[left_index + 1 :]:
            left, right = sorted((line_a, line_b), key=lambda line: line[0])
            separation = right[0] - left[0]
            candidate_center = (left[0] + right[0]) / 2
            if not 0.72 * width <= separation <= 1.28 * width:
                continue
            if abs(candidate_center - center_x) > max_shift_px:
                continue

            left_overlap = max(
                0.0,
                min(zone_bottom, left[2]) - max(zone_top, left[1]),
            ) / height
            right_overlap = max(
                0.0,
                min(zone_bottom, right[2]) - max(zone_top, right[1]),
            ) / height
            pair_overlap = max(
                0.0,
                min(left[2], right[2]) - max(left[1], right[1]),
            ) / height
            width_score = max(0.0, 1 - abs(separation - width) / (0.28 * width))
            prior_score = max(0.0, 1 - abs(candidate_center - center_x) / max_shift_px)
            score = (
                (0.25 * min(1.0, left_overlap))
                + (0.25 * min(1.0, right_overlap))
                + (0.20 * min(1.0, pair_overlap))
                + (0.25 * width_score)
                + (0.05 * prior_score)
            )
            candidates.append((score, candidate_center))

    if not candidates:
        return None, 0.0
    score, candidate_center = max(candidates)
    return float(candidate_center), float(score)


def detected_game_anchors(predictions: pd.DataFrame, manifest: pd.DataFrame, args: argparse.Namespace) -> dict:
    if predictions.empty:
        return {}
    rows = predictions.merge(
        manifest[["pitch_uid", "game_pk_key", "sz_height_ft", "sz_center_ft"]],
        on="pitch_uid",
        how="left",
    )
    rows = rows[rows.apply(lambda row: valid_zone_box(row, args), axis=1)].copy()
    if rows.empty:
        return {}

    anchors = {}
    for game_pk, game_rows in rows.groupby("game_pk_key", dropna=True):
        anchors[str(game_pk)] = {
            "zone_x": game_rows["zone_x"].astype(float).median(),
            "zone_y": game_rows["zone_y"].astype(float).median(),
            "zone_width": game_rows["zone_width"].astype(float).median(),
            "zone_height": game_rows["zone_height"].astype(float).median(),
            "anchor_sz_height_ft": game_rows["sz_height_ft"].astype(float).median(),
            "anchor_sz_center_ft": game_rows["sz_center_ft"].astype(float).median(),
            "source": "detected_game_median",
        }
    return anchors


def dense_game_anchors(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict, dict, dict]:
    if predictions.empty:
        return {}, {}, {}
    metadata = manifest[
        [
            "pitch_uid",
            "game_pk_key",
            "plate_appearance_key",
            "sz_height_ft",
            "sz_center_ft",
        ]
    ].drop_duplicates("pitch_uid")
    all_rows = predictions.merge(metadata, on="pitch_uid", how="inner")
    rows = all_rows[all_rows.apply(lambda row: valid_zone_box(row, args), axis=1)].copy()
    if rows.empty:
        return {}, {}, {}

    for column in [
        "zone_x",
        "zone_y",
        "zone_width",
        "zone_height",
        "sz_height_ft",
        "sz_center_ft",
    ]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(
        subset=["zone_x", "zone_y", "zone_width", "zone_height", "sz_height_ft", "sz_center_ft"]
    )

    anchors = {}
    registrations = {}
    plate_appearance_anchors = {}
    for game_pk, game_rows in rows.groupby("game_pk_key", dropna=True):
        reference_height = float(game_rows["sz_height_ft"].median())
        reference_center = float(game_rows["sz_center_ft"].median())
        game_rows = game_rows.copy()
        game_rows["pixels_per_foot"] = game_rows["zone_height"] / game_rows["sz_height_ft"]
        game_rows["normalized_zone_y"] = game_rows["zone_y"] + (
            (game_rows["sz_center_ft"] - reference_center) * game_rows["pixels_per_foot"]
        )
        game_rows["normalized_zone_height"] = reference_height * game_rows["pixels_per_foot"]

        pitch_rows = []
        for pitch_uid, pitch_group in game_rows.groupby("pitch_uid"):
            center_x = float(pitch_group["zone_x"].median())
            center_y = float(pitch_group["zone_y"].median())
            center_distance = (
                (pitch_group["zone_x"] - center_x) ** 2
                + (pitch_group["zone_y"] - center_y) ** 2
            ) ** 0.5
            center_mad = float(center_distance.median())
            frame_support = int(len(pitch_group))
            pitch_rows.append(
                {
                    "pitch_uid": str(pitch_uid),
                    "plate_appearance_key": str(pitch_group["plate_appearance_key"].iloc[0]),
                    "zone_x": float(pitch_group["zone_x"].median()),
                    "zone_y": float(pitch_group["normalized_zone_y"].median()),
                    "zone_width": float(pitch_group["zone_width"].median()),
                    "zone_height": float(pitch_group["normalized_zone_height"].median()),
                    "detected_pitch_zone_x": center_x,
                    "detected_pitch_zone_y": center_y,
                    "detected_pitch_zone_width": float(pitch_group["zone_width"].median()),
                    "detected_pitch_zone_height": float(pitch_group["zone_height"].median()),
                    "sz_height_ft": float(pitch_group["sz_height_ft"].median()),
                    "sz_center_ft": float(pitch_group["sz_center_ft"].median()),
                    "frame_support": frame_support,
                    "center_mad_px": center_mad,
                }
            )
            registrations[str(pitch_uid)] = {
                "zone_x": center_x,
                "zone_y": center_y,
                "zone_width": float(pitch_group["zone_width"].median()),
                "zone_height": float(pitch_group["zone_height"].median()),
                "zone_confidence": float(pitch_group["zone_confidence"].median()),
                "anchor_sz_height_ft": float(pitch_group["sz_height_ft"].median()),
                "anchor_sz_center_ft": float(pitch_group["sz_center_ft"].median()),
                "pitch_support": 1,
                "frame_support": frame_support,
                "center_mad_px": center_mad,
                "source": "dense_pitch_consensus",
                "calibration_reference_pitch_uid": str(pitch_uid),
                "calibration_reference_status": "detected_pitch_reference",
                "plate_appearance_key": str(pitch_group["plate_appearance_key"].iloc[0]),
            }

        pitch_boxes = pd.DataFrame(pitch_rows)
        if pitch_boxes.empty:
            continue
        anchor_x = float(pitch_boxes["zone_x"].median())
        anchor_y = float(pitch_boxes["zone_y"].median())
        anchor_center_distance = (
            (pitch_boxes["zone_x"] - anchor_x) ** 2
            + (pitch_boxes["zone_y"] - anchor_y) ** 2
        ) ** 0.5
        anchors[str(game_pk)] = {
            "zone_x": anchor_x,
            "zone_y": anchor_y,
            "zone_width": float(pitch_boxes["zone_width"].median()),
            "zone_height": float(pitch_boxes["zone_height"].median()),
            "anchor_sz_height_ft": reference_height,
            "anchor_sz_center_ft": reference_center,
            "pitch_support": int(len(pitch_boxes)),
            "frame_support": int(pitch_boxes["frame_support"].sum()),
            "center_mad_px": float(anchor_center_distance.median()),
            "source": "dense_game_consensus",
        }

        for plate_appearance_key, pa_rows in pitch_boxes.groupby("plate_appearance_key"):
            zone_x = float(pa_rows["detected_pitch_zone_x"].median())
            zone_y = float(pa_rows["detected_pitch_zone_y"].median())
            pitch_center_distance = (
                (pa_rows["detected_pitch_zone_x"] - zone_x) ** 2
                + (pa_rows["detected_pitch_zone_y"] - zone_y) ** 2
            ) ** 0.5
            between_pitch_mad = float(pitch_center_distance.median())
            within_pitch_mad = float(pa_rows["center_mad_px"].median())
            center_mad = max(between_pitch_mad, within_pitch_mad)
            frame_support = int(pa_rows["frame_support"].sum())
            if (
                frame_support < args.registration_min_frames
                or center_mad > args.registration_max_center_mad_px
            ):
                continue
            plate_appearance_anchors[str(plate_appearance_key)] = {
                "zone_x": zone_x,
                "zone_y": zone_y,
                "zone_width": float(pa_rows["detected_pitch_zone_width"].median()),
                "zone_height": float(pa_rows["detected_pitch_zone_height"].median()),
                "anchor_sz_height_ft": float(pa_rows["sz_height_ft"].median()),
                "anchor_sz_center_ft": float(pa_rows["sz_center_ft"].median()),
                "pitch_support": int(len(pa_rows)),
                "frame_support": frame_support,
                "center_mad_px": center_mad,
                "source": "dense_plate_appearance_consensus",
                "calibration_reference_status": "detected_plate_appearance_reference",
                "plate_appearance_key": str(plate_appearance_key),
            }
    return anchors, registrations, plate_appearance_anchors


def broadcast_zone_templates(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict, dict, dict, dict]:
    """Build a robust screen-space template from multiple pitches per broadcast."""
    if predictions.empty:
        return {}, {}, {}, {}
    metadata = manifest[
        [
            "pitch_uid",
            "game_pk_key",
            "plate_appearance_key",
            "game_batter_key",
            "sz_height_ft",
            "sz_center_ft",
        ]
    ].drop_duplicates("pitch_uid")
    all_rows = predictions.merge(metadata, on="pitch_uid", how="inner")
    rows = all_rows[all_rows.apply(lambda row: valid_zone_box(row, args), axis=1)].copy()
    if rows.empty:
        return {}, {}, {}, {}
    numeric_columns = [
        "zone_x",
        "zone_y",
        "zone_width",
        "zone_height",
        "zone_confidence",
        "sz_height_ft",
        "sz_center_ft",
    ]
    for column in numeric_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(
        subset=["zone_x", "zone_y", "zone_width", "zone_height", "sz_height_ft", "sz_center_ft"]
    )

    game_templates = {}
    pa_templates = {}
    batter_templates = {}
    pitch_edge_templates = {}
    for game_pk, game_rows in rows.groupby("game_pk_key", dropna=True):
        game_frame_rows = all_rows[all_rows["game_pk_key"] == game_pk]
        pitch_records = []
        for pitch_uid, pitch_rows in game_rows.groupby("pitch_uid"):
            center_x = float(pitch_rows["zone_x"].median())
            center_y = float(pitch_rows["zone_y"].median())
            center_distance = (
                (pitch_rows["zone_x"] - center_x) ** 2
                + (pitch_rows["zone_y"] - center_y) ** 2
            ) ** 0.5
            pitch_records.append(
                {
                    "pitch_uid": str(pitch_uid),
                    "plate_appearance_key": str(pitch_rows["plate_appearance_key"].iloc[0]),
                    "game_batter_key": str(pitch_rows["game_batter_key"].iloc[0]),
                    "zone_x": center_x,
                    "zone_y": center_y,
                    "zone_width": float(pitch_rows["zone_width"].median()),
                    "zone_height": float(pitch_rows["zone_height"].median()),
                    "zone_confidence": float(pitch_rows["zone_confidence"].median()),
                    "sz_height_ft": float(pitch_rows["sz_height_ft"].median()),
                    "sz_center_ft": float(pitch_rows["sz_center_ft"].median()),
                    "frame_support": int(len(pitch_rows)),
                    "within_pitch_center_mad_px": float(center_distance.median()),
                }
            )
        pitch_boxes = pd.DataFrame(pitch_records)
        if len(pitch_boxes) < args.template_min_pitches:
            continue

        pitch_boxes["pixels_per_foot"] = pitch_boxes["zone_height"] / pitch_boxes["sz_height_ft"]
        reference_height = float(pitch_boxes["sz_height_ft"].median())
        reference_center = float(pitch_boxes["sz_center_ft"].median())
        pitch_boxes["normalized_zone_y"] = pitch_boxes["zone_y"] + (
            (pitch_boxes["sz_center_ft"] - reference_center) * pitch_boxes["pixels_per_foot"]
        )
        pitch_boxes["normalized_zone_height"] = reference_height * pitch_boxes["pixels_per_foot"]

        median_x = float(pitch_boxes["zone_x"].median())
        median_y = float(pitch_boxes["normalized_zone_y"].median())
        median_width = float(pitch_boxes["zone_width"].median())
        median_height = float(pitch_boxes["normalized_zone_height"].median())
        pitch_boxes["template_center_delta"] = (
            (pitch_boxes["zone_x"] - median_x) ** 2
            + (pitch_boxes["normalized_zone_y"] - median_y) ** 2
        ) ** 0.5
        pitch_boxes["template_size_delta"] = (
            (pitch_boxes["zone_width"] - median_width) ** 2
            + (pitch_boxes["normalized_zone_height"] - median_height) ** 2
        ) ** 0.5
        clean = pitch_boxes[
            pitch_boxes["template_center_delta"].le(args.template_max_center_delta_px)
            & pitch_boxes["template_size_delta"].le(args.template_max_size_delta_px)
        ].copy()
        if len(clean) < args.template_min_pitches:
            continue

        reference_height = float(clean["sz_height_ft"].median())
        reference_center = float(clean["sz_center_ft"].median())
        pixels_per_foot = float(clean["pixels_per_foot"].median())
        clean["normalized_zone_y"] = clean["zone_y"] + (
            (clean["sz_center_ft"] - reference_center) * clean["pixels_per_foot"]
        )
        template_x = float(clean["zone_x"].median())
        template_y = float(clean["normalized_zone_y"].median())
        template_width = float(clean["zone_width"].median())
        template_height = reference_height * pixels_per_foot
        template_center_delta = (
            (clean["zone_x"] - template_x) ** 2
            + (clean["normalized_zone_y"] - template_y) ** 2
        ) ** 0.5
        game_template = {
            "zone_x": template_x,
            "zone_y": template_y,
            "zone_width": template_width,
            "zone_height": template_height,
            "zone_confidence": float(clean["zone_confidence"].median()),
            "anchor_sz_height_ft": reference_height,
            "anchor_sz_center_ft": reference_center,
            "pixels_per_vertical_foot": pixels_per_foot,
            "pitch_support": int(len(clean)),
            "frame_support": int(clean["frame_support"].sum()),
            "center_mad_px": float(template_center_delta.median()),
            "rejected_pitch_count": int(len(pitch_boxes) - len(clean)),
            "source": "broadcast_game_template_statcast",
            "calibration_reference_status": "robust_multi_pitch_broadcast_template",
        }

        clean["edge_zone_x"] = np.nan
        clean["edge_zone_x_score"] = np.nan
        clean["edge_zone_x_frame_support"] = 0
        for clean_index, pitch_record in clean.iterrows():
            expected_zone = adjusted_zone(game_template, pitch_record)
            if expected_zone is None:
                continue
            pitch_frames = game_frame_rows[
                game_frame_rows["pitch_uid"].astype(str) == str(pitch_record["pitch_uid"])
            ].sort_values("frame_time_sec" if "frame_time_sec" in game_rows.columns else "image_path")
            if len(pitch_frames) > args.template_edge_frames_per_pitch:
                sample_indices = np.linspace(
                    0,
                    len(pitch_frames) - 1,
                    args.template_edge_frames_per_pitch,
                    dtype=int,
                )
                pitch_frames = pitch_frames.iloc[np.unique(sample_indices)]

            edge_centers = []
            edge_scores = []
            for _, frame_row in pitch_frames.iterrows():
                edge_center, edge_score = broadcast_box_edge_center(
                    normalized(frame_row.get("image_path")),
                    expected_zone,
                    args.template_edge_search_radius_px,
                )
                if edge_center is None or edge_score < args.template_edge_min_score:
                    continue
                edge_centers.append(edge_center)
                edge_scores.append(edge_score)
            if len(edge_centers) < args.template_edge_min_frames:
                continue
            edge_center = float(np.median(edge_centers))
            edge_center_mad = float(np.median(np.abs(np.asarray(edge_centers) - edge_center)))
            if edge_center_mad > args.template_edge_max_center_mad_px:
                continue
            clean.at[clean_index, "edge_zone_x"] = edge_center
            clean.at[clean_index, "edge_zone_x_score"] = float(np.median(edge_scores))
            clean.at[clean_index, "edge_zone_x_frame_support"] = len(edge_centers)
            pitch_edge_templates[str(pitch_record["pitch_uid"])] = {
                **game_template,
                "zone_x": edge_center,
                "zone_confidence": float(game_template["zone_confidence"]),
                "horizontal_edge_score": float(np.median(edge_scores)),
                "pitch_support": 1,
                "frame_support": len(edge_centers),
                "center_mad_px": edge_center_mad,
                "source": "broadcast_pitch_edge_template",
                "calibration_reference_pitch_uid": str(pitch_record["pitch_uid"]),
                "calibration_reference_status": "stable_multi_frame_broadcast_edge_track",
            }

        reliable_edge_rows = clean.dropna(subset=["edge_zone_x"])
        if len(reliable_edge_rows) >= args.template_min_pitches:
            edge_template_x = float(game_template["zone_x"])
            rejected_edge_rows = reliable_edge_rows[
                (reliable_edge_rows["edge_zone_x"] - edge_template_x).abs()
                > args.template_pa_max_horizontal_shift_px
            ]
            for rejected_index, rejected_row in rejected_edge_rows.iterrows():
                clean.at[rejected_index, "edge_zone_x"] = np.nan
                pitch_edge_templates.pop(str(rejected_row["pitch_uid"]), None)
            reliable_edge_rows = clean.dropna(subset=["edge_zone_x"])
        if len(reliable_edge_rows) >= args.template_min_pitches:
            template_x = float(reliable_edge_rows["edge_zone_x"].median())
            game_template["zone_x"] = template_x
            game_template["horizontal_edge_pitch_support"] = int(len(reliable_edge_rows))
            game_template["horizontal_edge_frame_support"] = int(
                reliable_edge_rows["edge_zone_x_frame_support"].sum()
            )
            game_template["calibration_reference_status"] = (
                "robust_multi_pitch_broadcast_template_with_edge_tracking"
            )
        game_templates[str(game_pk)] = game_template

        for plate_appearance_key, pa_rows in clean.groupby("plate_appearance_key"):
            frame_support = int(pa_rows["frame_support"].sum())
            if (
                frame_support < args.template_pa_min_frames
                or len(pa_rows) < args.template_local_min_pitches
            ):
                continue
            representative = pa_rows.iloc[0]
            expected = adjusted_zone(game_template, representative)
            if expected is None:
                continue
            detected_y = float(pa_rows["zone_y"].median())
            detected_height = float(pa_rows["zone_height"].median())
            pa_edge_rows = pa_rows.dropna(subset=["edge_zone_x"])
            has_local_horizontal = len(pa_edge_rows) >= args.template_local_min_pitches
            detected_x = (
                float(pa_edge_rows["edge_zone_x"].median())
                if has_local_horizontal
                else float(game_template["zone_x"])
            )
            horizontal_shift = abs(detected_x - expected["zone_x"])
            vertical_shift = abs(detected_y - expected["zone_y"])
            height_shift = abs(detected_height - expected["zone_height"])
            if (
                horizontal_shift > args.template_pa_max_horizontal_shift_px
                or vertical_shift > args.template_pa_max_vertical_shift_px
                or height_shift > args.template_max_size_delta_px
            ):
                continue
            pa_center_delta = (
                (pa_rows["zone_x"] - detected_x) ** 2
                + (pa_rows["zone_y"] - detected_y) ** 2
            ) ** 0.5
            pa_templates[str(plate_appearance_key)] = {
                **game_template,
                "zone_x": detected_x,
                "zone_y": detected_y,
                "zone_height": detected_height,
                "zone_confidence": float(pa_rows["zone_confidence"].median()),
                "anchor_sz_height_ft": float(pa_rows["sz_height_ft"].median()),
                "anchor_sz_center_ft": float(pa_rows["sz_center_ft"].median()),
                "pitch_support": int(len(pa_rows)),
                "frame_support": frame_support,
                "center_mad_px": float(pa_center_delta.median()),
                "source": "broadcast_plate_appearance_template",
                "calibration_reference_status": (
                    "robust_plate_appearance_xy_template"
                    if has_local_horizontal
                    else "robust_plate_appearance_vertical_template"
                ),
                "plate_appearance_key": str(plate_appearance_key),
            }

        for game_batter_key, batter_rows in clean.groupby("game_batter_key"):
            batter_rows = batter_rows.dropna(subset=["edge_zone_x"])
            frame_support = int(batter_rows["frame_support"].sum())
            if (
                frame_support < args.template_pa_min_frames
                or len(batter_rows) < args.template_local_min_pitches
            ):
                continue
            detected_x = float(batter_rows["edge_zone_x"].median())
            if abs(detected_x - game_template["zone_x"]) > args.template_pa_max_horizontal_shift_px:
                continue
            batter_templates[str(game_batter_key)] = {
                **game_template,
                "zone_x": detected_x,
                "zone_confidence": float(batter_rows["zone_confidence"].median()),
                "pitch_support": int(len(batter_rows)),
                "frame_support": frame_support,
                "center_mad_px": float((batter_rows["edge_zone_x"] - detected_x).abs().median()),
                "source": "broadcast_batter_horizontal_template",
                "calibration_reference_status": "robust_batter_horizontal_template",
                "game_batter_key": str(game_batter_key),
            }
    return game_templates, pa_templates, batter_templates, pitch_edge_templates


def global_anchor(anchors: dict, manifest: pd.DataFrame) -> Optional[dict]:
    if not anchors:
        return None
    values = pd.DataFrame(list(anchors.values()))
    return {
        "zone_x": values["zone_x"].median(),
        "zone_y": values["zone_y"].median(),
        "zone_width": values["zone_width"].median(),
        "zone_height": values["zone_height"].median(),
        "anchor_sz_height_ft": manifest["sz_height_ft"].median(),
        "anchor_sz_center_ft": manifest["sz_center_ft"].median(),
        "pitch_support": int(values.get("pitch_support", pd.Series(dtype=float)).sum()),
        "frame_support": int(values.get("frame_support", pd.Series(dtype=float)).sum()),
        "center_mad_px": float(values.get("center_mad_px", pd.Series([0.0])).median()),
        "source": "detected_global_median",
    }


def read_calibrations(path: Path) -> dict:
    if not path.exists():
        return {"default": DEFAULT_ZONE_CALIBRATION, "games": {}}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"default": DEFAULT_ZONE_CALIBRATION, "games": {}}
    payload.setdefault("default", DEFAULT_ZONE_CALIBRATION)
    payload.setdefault("games", {})
    return payload


def selected_frame_anchor(
    selected_row: Optional[pd.Series],
    dense_by_frame_uid: dict,
    pitch_anchor: Optional[dict],
    args: argparse.Namespace,
) -> Optional[dict]:
    if selected_row is None:
        return None
    frame_uid = normalized(selected_row.get("frame_uid"))
    dense_row = dense_by_frame_uid.get(frame_uid)
    if dense_row is None or not valid_zone_box(dense_row, args):
        return None

    zone_x = row_value(dense_row, "zone_x")
    zone_y = row_value(dense_row, "zone_y")
    zone_width = row_value(dense_row, "zone_width")
    zone_height = row_value(dense_row, "zone_height")
    if None in {zone_x, zone_y, zone_width, zone_height}:
        return None

    if pitch_anchor:
        center_delta = (
            (zone_x - pitch_anchor["zone_x"]) ** 2
            + (zone_y - pitch_anchor["zone_y"]) ** 2
        ) ** 0.5
        width_delta = abs(zone_width - pitch_anchor["zone_width"])
        height_delta = abs(zone_height - pitch_anchor["zone_height"])
        # A single-frame prediction must agree with the stable clip track. This
        # keeps an isolated false box from replacing a strong 31-frame consensus.
        if center_delta > 14.0 or width_delta > 14.0 or height_delta > 18.0:
            return None

    return {
        "zone_x": zone_x,
        "zone_y": zone_y,
        "zone_width": zone_width,
        "zone_height": zone_height,
        "zone_confidence": row_value(dense_row, "zone_confidence") or 0.0,
        "pitch_support": 1,
        "frame_support": 1,
        "center_mad_px": 0.0,
        "source": "dense_selected_frame_detection",
        "calibration_reference_pitch_uid": normalized(dense_row.get("pitch_uid")),
        "calibration_reference_status": "detected_selected_frame_reference",
    }


def selected_frame_center_with_manual_size(
    selected_row: Optional[pd.Series],
    dense_by_frame_uid: dict,
    manual_anchor: Optional[dict],
) -> Optional[dict]:
    if selected_row is None or manual_anchor is None:
        return None
    dense_row = dense_by_frame_uid.get(normalized(selected_row.get("frame_uid")))
    if dense_row is None:
        return None

    zone_x = row_value(dense_row, "zone_x")
    zone_y = row_value(dense_row, "zone_y")
    detected_width = row_value(dense_row, "zone_width")
    detected_height = row_value(dense_row, "zone_height")
    confidence = row_value(dense_row, "zone_confidence") or 0.0
    if None in {zone_x, zone_y, detected_width, detected_height} or confidence < 0.40:
        return None
    if not 35.0 <= detected_width <= 125.0 or not 45.0 <= detected_height <= 160.0:
        return None

    center_shift = (
        (zone_x - manual_anchor["zone_x"]) ** 2
        + (zone_y - manual_anchor["zone_y"]) ** 2
    ) ** 0.5
    if center_shift > 120.0:
        return None

    return {
        **manual_anchor,
        "zone_x": zone_x,
        "zone_y": zone_y,
        "zone_confidence": confidence,
        "pitch_support": 1,
        "frame_support": 1,
        "center_mad_px": 0.0,
        "source": "dense_selected_center_manual_size",
        "calibration_reference_pitch_uid": normalized(dense_row.get("pitch_uid")),
        "calibration_reference_status": "detected_center_manual_size_reference",
    }


def overlay_is_default(overlay: dict, default_overlay: dict) -> bool:
    for key in DEFAULT_ZONE_CALIBRATION:
        overlay_value = number(overlay.get(key)) or 0
        default_value = number(default_overlay.get(key)) or 0
        if abs(overlay_value - default_value) >= 0.001:
            return False
    return True


def manual_anchor_for_game(
    game_pk: str,
    calibrations: dict,
    manifest: pd.DataFrame,
    frame_width: float,
    frame_height: float,
    args: argparse.Namespace,
) -> Optional[dict]:
    overlay = calibrations.get("games", {}).get(str(game_pk))
    if not overlay or overlay_is_default(overlay, calibrations.get("default", DEFAULT_ZONE_CALIBRATION)):
        return None

    left = (number(overlay.get("left")) or 0.0) / 100 * frame_width
    top = (number(overlay.get("top")) or 0.0) / 100 * frame_height
    width = (number(overlay.get("width")) or 0.0) / 100 * frame_width
    height = (number(overlay.get("height")) or 0.0) / 100 * frame_height
    aspect = height / width if width else 0
    # A saved human calibration is authoritative. The tighter detector limits are
    # useful for rejecting model boxes, but should not silently discard a box the
    # reviewer placed and saved intentionally.
    if width <= 0 or height <= 0 or not 0.50 <= aspect <= 3.00:
        return None

    game_rows = manifest[manifest["game_pk_key"] == str(game_pk)]
    reference_top = number(overlay.get("reference_sz_top"))
    reference_bot = number(overlay.get("reference_sz_bot"))
    if (
        reference_top is not None
        and reference_bot is not None
        and reference_top > reference_bot
    ):
        reference_height = reference_top - reference_bot
        reference_center = (reference_top + reference_bot) / 2
        reference_status = "explicit_pitch_reference"
    else:
        reference_height = game_rows["sz_height_ft"].median()
        reference_center = game_rows["sz_center_ft"].median()
        reference_status = "legacy_game_median_reference"
    return {
        "zone_x": left + (width / 2),
        "zone_y": top + (height / 2),
        "zone_width": width,
        "zone_height": height,
        "anchor_sz_height_ft": reference_height,
        "anchor_sz_center_ft": reference_center,
        "pitch_support": int(len(game_rows)),
        "frame_support": 1,
        "center_mad_px": 0.0,
        "source": "manual_game_calibration",
        "calibration_reference_pitch_uid": normalized(overlay.get("reference_pitch_uid")),
        "calibration_reference_status": reference_status,
    }


def frame_size(image_path: str) -> tuple[float, float]:
    path = resolve(Path(image_path))
    image = cv2.imread(str(path)) if path and path.is_file() else None
    if image is not None:
        return float(image.shape[1]), float(image.shape[0])
    return DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT


def adjusted_zone(anchor: dict, pitch_row: pd.Series) -> Optional[dict]:
    sz_height = number(pitch_row.get("sz_height_ft"))
    sz_center = number(pitch_row.get("sz_center_ft"))
    anchor_height_ft = number(anchor.get("anchor_sz_height_ft"))
    anchor_center_ft = number(anchor.get("anchor_sz_center_ft"))
    anchor_px_height = number(anchor.get("zone_height"))
    if None in {sz_height, sz_center, anchor_height_ft, anchor_center_ft, anchor_px_height}:
        return None
    if anchor_height_ft <= 0 or anchor_px_height <= 0:
        return None

    pixels_per_vertical_foot = anchor_px_height / anchor_height_ft
    zone_height = sz_height * pixels_per_vertical_foot
    zone_y = anchor["zone_y"] - ((sz_center - anchor_center_ft) * pixels_per_vertical_foot)
    return {
        "zone_x": anchor["zone_x"],
        "zone_y": zone_y,
        "zone_width": anchor["zone_width"],
        "zone_height": zone_height,
    }


def registered_zone(
    zone: Optional[dict],
    registration: Optional[dict],
    args: argparse.Namespace,
) -> tuple[Optional[dict], str]:
    if not zone or not registration:
        return zone, "game_anchor_only"
    frame_support = int(number(registration.get("frame_support")) or 0)
    center_mad = number(registration.get("center_mad_px"))
    detected_x = number(registration.get("zone_x"))
    detected_y = number(registration.get("zone_y"))
    if (
        frame_support < args.registration_min_frames
        or center_mad is None
        or center_mad > args.registration_max_center_mad_px
        or detected_x is None
        or detected_y is None
    ):
        return zone, "game_anchor_only"

    shift_x = detected_x - zone["zone_x"]
    shift_y = detected_y - zone["zone_y"]
    shift = (shift_x**2 + shift_y**2) ** 0.5
    if shift > args.registration_max_shift_px:
        return zone, "game_anchor_only_large_shift_rejected"
    registered = dict(zone)
    registered["zone_x"] += shift_x
    registered["zone_y"] += shift_y
    return registered, "dense_clip_center_registration"


def target_from_zone_and_glove(zone: Optional[dict], glove_row: Optional[pd.Series]) -> tuple[Optional[float], Optional[float]]:
    if not zone or glove_row is None:
        return None, None
    glove_x = row_value(glove_row, "glove_x")
    glove_y = row_value(glove_row, "glove_y")
    if None in {glove_x, glove_y}:
        return None, None
    zone_left = zone["zone_x"] - (zone["zone_width"] / 2)
    zone_top = zone["zone_y"] - (zone["zone_height"] / 2)
    target_x = (glove_x - zone_left) / zone["zone_width"]
    target_y = 1 - ((glove_y - zone_top) / zone["zone_height"])
    return target_x, target_y


def zone_confidence_for_source(source: str) -> float:
    return {
        "dense_selected_frame_detection": 0.97,
        "dense_selected_center_manual_size": 0.82,
        "dense_pitch_consensus": 0.96,
        "dense_plate_appearance_consensus": 0.96,
        "broadcast_plate_appearance_template": 0.95,
        "broadcast_pitch_edge_template": 0.97,
        "broadcast_batter_horizontal_template": 0.92,
        "broadcast_game_template_statcast": 0.90,
        "manual_game_calibration": 0.98,
        "manual_game_fallback": 0.55,
        "dense_game_consensus": 0.92,
        "detected_game_median": 0.90,
        "detected_global_median": 0.70,
    }.get(source, 0.0)


def main() -> None:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    primary_path = resolve(args.primary_predictions)
    fallback_path = resolve(args.fallback_predictions)
    dense_path = resolve(args.dense_predictions)
    calibrations_path = resolve(args.calibrations)
    output_path = resolve(args.output)

    manifest = latest_analysis_manifest(manifest_path)
    primary = read_predictions(primary_path, "primary")
    fallback = read_predictions(fallback_path, "fallback")
    dense = read_dense_predictions(dense_path)
    dense_by_frame_uid = {
        normalized(row.get("frame_uid")): row
        for _, row in dense.drop_duplicates("frame_uid", keep="last").iterrows()
    }
    primary_by_pitch = {row.get("pitch_uid"): row for _, row in primary.iterrows()}
    fallback_by_pitch = {row.get("pitch_uid"): row for _, row in fallback.iterrows()}

    game_anchors, pitch_registrations, plate_appearance_anchors = dense_game_anchors(
        dense, manifest, args
    )
    (
        broadcast_game_templates,
        broadcast_pa_templates,
        broadcast_batter_templates,
        broadcast_pitch_edge_templates,
    ) = broadcast_zone_templates(dense, manifest, args)
    legacy_game_anchors = detected_game_anchors(primary, manifest, args)
    for game_pk, anchor in legacy_game_anchors.items():
        game_anchors.setdefault(game_pk, anchor)
    fallback_game_anchors = detected_game_anchors(fallback, manifest, args)
    for game_pk, anchor in fallback_game_anchors.items():
        game_anchors.setdefault(game_pk, anchor)
    fallback_anchor = global_anchor(game_anchors, manifest)
    calibrations = read_calibrations(calibrations_path)

    refiner = ZoneRefiner(dense, ROOT)
    output_rows = []
    source_counts = {}
    registration_counts = {}
    glove_counts = {}
    for _, pitch_row in manifest.sort_values(["sample_pitcher_name", "game_date", "pitch_uid"]).iterrows():
        pitch_uid = normalized(pitch_row.get("pitch_uid"))
        game_pk = normalized(pitch_row.get("game_pk"))
        plate_appearance_key = normalized(pitch_row.get("plate_appearance_key"))
        game_batter_key = normalized(pitch_row.get("game_batter_key"))
        primary_row = primary_by_pitch.get(pitch_uid)
        fallback_row = fallback_by_pitch.get(pitch_uid)
        glove_row, glove_source = choose_glove_row(
            primary_row,
            fallback_row,
            args.min_primary_glove_confidence,
            args.min_fallback_glove_confidence,
        )
        if glove_row is not None:
            base_row = glove_row
        elif primary_row is not None:
            base_row = primary_row
        else:
            base_row = fallback_row

        image_path = normalized(base_row.get("image_path") if base_row is not None else pitch_row.get("image_path"))
        frame_width, frame_height = frame_size(image_path)
        pitch_anchor = pitch_registrations.get(pitch_uid)
        if pitch_anchor:
            pitch_frame_support = int(number(pitch_anchor.get("frame_support")) or 0)
            pitch_center_mad = number(pitch_anchor.get("center_mad_px"))
            if (
                pitch_frame_support < args.registration_min_frames
                or pitch_center_mad is None
                or pitch_center_mad > args.registration_max_center_mad_px
            ):
                pitch_anchor = None
        manual_game_anchor = manual_anchor_for_game(
            game_pk, calibrations, manifest, frame_width, frame_height, args
        )
        explicit_manual_anchor = (
            manual_game_anchor
            and normalized(manual_game_anchor.get("calibration_reference_status"))
            == "explicit_pitch_reference"
            and normalized(calibrations.get("games", {}).get(game_pk, {}).get("reference_frame_uid"))
            == normalized(base_row.get("frame_uid") if base_row is not None else pitch_row.get("frame_uid"))
        )
        template_game_anchor = broadcast_game_templates.get(game_pk)
        template_pa_anchor = broadcast_pa_templates.get(plate_appearance_key)
        template_batter_anchor = broadcast_batter_templates.get(game_batter_key)
        template_pitch_edge_anchor = broadcast_pitch_edge_templates.get(pitch_uid)

        if explicit_manual_anchor:
            anchor = manual_game_anchor
            zone = {
                "zone_x": anchor["zone_x"],
                "zone_y": anchor["zone_y"],
                "zone_width": anchor["zone_width"],
                "zone_height": anchor["zone_height"],
            }
            registration_source = "explicit_manual_anchor"
        elif template_game_anchor:
            vertical_anchor = template_pa_anchor or template_game_anchor
            anchor = (
                template_pitch_edge_anchor
                or template_pa_anchor
                or template_batter_anchor
                or template_game_anchor
            )
            if template_pa_anchor:
                zone = {
                    "zone_x": (
                        template_pitch_edge_anchor["zone_x"]
                        if template_pitch_edge_anchor
                        else template_pa_anchor["zone_x"]
                    ),
                    "zone_y": template_pa_anchor["zone_y"],
                    "zone_width": template_game_anchor["zone_width"],
                    "zone_height": template_pa_anchor["zone_height"],
                }
                registration_source = (
                    "broadcast_pitch_edge_plate_appearance_vertical"
                    if template_pitch_edge_anchor
                    else "broadcast_plate_appearance_xy"
                )
            elif template_pitch_edge_anchor:
                zone = adjusted_zone(vertical_anchor, pitch_row)
                if zone:
                    zone["zone_x"] = template_pitch_edge_anchor["zone_x"]
                    zone["zone_width"] = template_game_anchor["zone_width"]
                registration_source = "broadcast_pitch_edge_statcast_vertical"
            elif template_batter_anchor:
                zone = adjusted_zone(template_game_anchor, pitch_row)
                if zone:
                    zone["zone_x"] = template_batter_anchor["zone_x"]
                    zone["zone_width"] = template_game_anchor["zone_width"]
                registration_source = "broadcast_batter_horizontal_statcast_vertical"
            else:
                zone = adjusted_zone(template_game_anchor, pitch_row)
                registration_source = "broadcast_game_template_statcast"
        else:
            anchor = selected_frame_anchor(primary_row, dense_by_frame_uid, pitch_anchor, args)
            if anchor is None:
                anchor = pitch_anchor
            if anchor is None:
                anchor = plate_appearance_anchors.get(plate_appearance_key)
            if anchor is None:
                anchor = selected_frame_center_with_manual_size(
                    primary_row, dense_by_frame_uid, manual_game_anchor
                )
            if anchor is None:
                anchor = manual_game_anchor
            if (
                anchor
                and normalized(anchor.get("source")) == "manual_game_calibration"
                and normalized(anchor.get("calibration_reference_pitch_uid")) != pitch_uid
            ):
                anchor = {
                    **anchor,
                    "source": "manual_game_fallback",
                    "calibration_reference_status": "manual_game_reference_other_pitch",
                }
            if anchor is None:
                anchor = game_anchors.get(game_pk) or fallback_anchor
            direct_broadcast_anchor = (
                anchor
                and normalized(anchor.get("source"))
                in {
                    "dense_selected_frame_detection",
                    "dense_selected_center_manual_size",
                    "dense_pitch_consensus",
                    "dense_plate_appearance_consensus",
                }
            )
            if direct_broadcast_anchor:
                zone = {
                    "zone_x": anchor["zone_x"],
                    "zone_y": anchor["zone_y"],
                    "zone_width": anchor["zone_width"],
                    "zone_height": anchor["zone_height"],
                }
                registration_source = {
                    "dense_selected_frame_detection": "selected_frame_detection",
                    "dense_selected_center_manual_size": "selected_center_manual_size",
                    "dense_pitch_consensus": "pitch_clip_consensus",
                    "dense_plate_appearance_consensus": "plate_appearance_consensus",
                }[normalized(anchor.get("source"))]
            else:
                zone = adjusted_zone(anchor, pitch_row) if anchor else None
                zone, registration_source = registered_zone(
                    zone,
                    pitch_registrations.get(pitch_uid),
                    args,
                )

        if explicit_manual_anchor:
            # The reviewer aligned this box to the broadcast graphic itself. Keep
            # that pixel geometry exact; Statcast sz_top/sz_bot belongs to the
            # later plate-location conversion, not to moving the video overlay.
            zone = {
                "zone_x": anchor["zone_x"],
                "zone_y": anchor["zone_y"],
                "zone_width": anchor["zone_width"],
                "zone_height": anchor["zone_height"],
            }
        selected_uid = (normalized(base_row.get("selected_frame_uid")) or normalized(base_row.get("frame_uid"))) if base_row is not None else normalized(pitch_row.get("frame_uid"))
        zone, refinement = refiner.resolve(pitch_uid, selected_uid, image_path, zone)
        if explicit_manual_anchor and calibrations.get("games", {}).get(game_pk, {}).get("frame_override") is True:
            zone = {key: anchor[key] for key in ["zone_x", "zone_y", "zone_width", "zone_height"]}
            refinement.update(zone_refinement_status="manual_frame_override", zone_baseline_source="manual_frame_override", zone_edge_score="", zone_edge_min_support="", zone_refinement_frame_support=1, zone_refinement_reference_frame_uid=selected_uid)
        if refinement["zone_refinement_status"] in {"visible_four_edge_fit", "registered_temporal_fit"}:
            registration_source = refinement["zone_refinement_status"]
        target_x, target_y = target_from_zone_and_glove(zone, glove_row)

        zone_source = "game_calibrated_statcast_zone" if zone else "missing_statcast_zone"
        anchor_source = normalized(anchor.get("source")) if anchor else ""
        status = "ok" if zone else "missing_required_detection"
        error = ""
        if zone and target_x is None:
            error = "missing_glove_detection"

        source_counts[anchor_source or "missing"] = source_counts.get(anchor_source or "missing", 0) + 1
        registration_counts[registration_source] = registration_counts.get(registration_source, 0) + 1
        glove_counts[glove_source] = glove_counts.get(glove_source, 0) + 1

        record = {
            "frame_uid": normalized(base_row.get("frame_uid") if base_row is not None else pitch_row.get("frame_uid")),
            "pitch_uid": pitch_uid,
            "image_path": image_path,
            "workflow_output_key": zone_source,
            "zone_confidence": (
                number(anchor.get("zone_confidence"))
                if anchor and number(anchor.get("zone_confidence")) is not None
                else zone_confidence_for_source(anchor_source)
            ),
            "glove_confidence": normalized(glove_row.get("glove_confidence")) if glove_row is not None else "",
            "prediction_count": normalized(glove_row.get("prediction_count")) if glove_row is not None else "",
            "zone_source": zone_source,
            "zone_anchor_source": anchor_source,
            "zone_registration_source": registration_source,
            "zone_anchor_pitch_support": normalized(anchor.get("pitch_support")) if anchor else "",
            "zone_anchor_frame_support": normalized(anchor.get("frame_support")) if anchor else "",
            "zone_anchor_center_mad_px": normalized(anchor.get("center_mad_px")) if anchor else "",
            "zone_plate_appearance_key": plate_appearance_key,
            "zone_calibration_reference_pitch_uid": normalized(
                anchor.get("calibration_reference_pitch_uid")
            ) if anchor else "",
            "zone_calibration_reference_status": normalized(
                anchor.get("calibration_reference_status")
            ) if anchor else "",
            "glove_source_model": glove_source,
            "statcast_sz_top": normalized(pitch_row.get("sz_top")),
            "statcast_sz_bot": normalized(pitch_row.get("sz_bot")),
            "statcast_zone_height_ft": normalized(pitch_row.get("sz_height_ft")),
            "zone_x": normalized(zone.get("zone_x")) if zone else "",
            "zone_y": normalized(zone.get("zone_y")) if zone else "",
            "zone_width": normalized(zone.get("zone_width")) if zone else "",
            "zone_height": normalized(zone.get("zone_height")) if zone else "",
            "glove_x": normalized(glove_row.get("glove_x")) if glove_row is not None else "",
            "glove_y": normalized(glove_row.get("glove_y")) if glove_row is not None else "",
            "glove_width": normalized(glove_row.get("glove_width")) if glove_row is not None else "",
            "glove_height": normalized(glove_row.get("glove_height")) if glove_row is not None else "",
            "vision_target_x_01": normalized(target_x),
            "vision_target_y_01": normalized(target_y),
            "raw_response_path": normalized(glove_row.get("raw_response_path")) if glove_row is not None else "",
            "status": status,
            "error": error,
        }
        record.update(refinement)
        record["zone_anchor_source"] = refinement["zone_baseline_source"]
        record["zone_registration_source"] = refinement["zone_refinement_status"]
        record["zone_source"] = "refined_broadcast_rectangle" if refinement["zone_refinement_status"] in {"visible_four_edge_fit", "registered_temporal_fit"} else "estimated_broadcast_rectangle" if zone else "missing_broadcast_rectangle"
        record["zone_calibration_reference_status"] = ""
        record["zone_calibration_reference_pitch_uid"] = ""
        record["image_width"] = frame_width
        record["image_height"] = frame_height
        # Evidence scores are not probabilities of geometric correctness.
        if refinement["zone_refinement_status"] == "manual_frame_override":
            record["zone_confidence"] = 0.98
            record["zone_source"] = "manual_frame_override"
            record["zone_calibration_reference_status"] = "explicit_pitch_reference"
            record["zone_calibration_reference_pitch_uid"] = pitch_uid
        elif refinement["zone_refinement_status"] == "visible_four_edge_fit":
            record["zone_confidence"] = refinement["zone_edge_score"]
        elif refinement["zone_refinement_status"] == "registered_temporal_fit":
            record["zone_confidence"] = min(0.85, refinement["zone_edge_score"])
        else:
            record["zone_confidence"] = min(0.55, number(record["zone_confidence"]) or 0.0)
        output_rows.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows)[OUTPUT_COLUMNS].fillna("").to_csv(output_path, index=False)

    print(f"Output: {output_path}")
    print(f"Rows: {len(output_rows)}")
    print(f"Zone baselines: {pd.Series([row['zone_baseline_source'] for row in output_rows]).value_counts().to_dict()}")
    print(f"Zone registrations: {pd.Series([row['zone_refinement_status'] for row in output_rows]).value_counts().to_dict()}")
    print(f"Glove sources: {glove_counts}")
    print(f"Rows with target: {sum(1 for row in output_rows if row['vision_target_x_01'] != '' and row['vision_target_y_01'] != '')}")


if __name__ == "__main__":
    main()

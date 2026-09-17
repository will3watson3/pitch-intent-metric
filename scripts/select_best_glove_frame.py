import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_MANIFEST = DATA_DIR / "setup_frame_manifest_15.csv"
DEFAULT_ALL_FRAME_PREDICTIONS = DATA_DIR / "roboflow_workflow_predictions_review_frames_15.csv"
DEFAULT_ZONE_PREDICTIONS = DATA_DIR / "game_calibrated_statcast_zones_400.csv"
DEFAULT_POSE_PREDICTIONS = DATA_DIR / "pitcher_pose_frames_1200_2700.csv"
DEFAULT_OUTPUT = DATA_DIR / "roboflow_workflow_predictions_best_glove_frame.csv"


OUTPUT_COLUMNS = [
    "frame_uid",
    "pitch_uid",
    "image_path",
    "frame_index",
    "frame_time_sec",
    "workflow_output_key",
    "zone_confidence",
    "glove_confidence",
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
    "prediction_count",
    "raw_response_path",
    "status",
    "error",
    "zone_source",
    "zone_anchor_source",
    "zone_registration_source",
    "zone_anchor_pitch_support",
    "zone_anchor_frame_support",
    "zone_anchor_center_mad_px",
    "glove_source_model",
    "statcast_sz_top",
    "statcast_sz_bot",
    "statcast_zone_height_ft",
    "vision_fallback_used",
    "vision_fallback_reason",
    "selected_frame_uid",
    "selected_frame_index",
    "selected_frame_time_sec",
    "glove_candidate_count",
    "plausible_glove_candidate_count",
    "glove_selection_score",
    "glove_selection_reason",
    "motion_gate_used",
    "delivery_signal_source",
    "delivery_motion_status",
    "delivery_start_time_sec",
    "delivery_motion_threshold",
    "delivery_motion_peak",
    "selected_delivery_motion_score",
    "selected_pose_visibility",
    "pose_delivery_status",
    "pose_coverage",
    "selected_after_delivery_start",
    "glove_candidate_source",
    "glove_candidate_rank",
    "glove_track_id",
    "glove_track_frame_count",
    "glove_track_stability",
    "glove_stable_run_count",
    "glove_drop_confirmed",
    "glove_hold_start_time_sec",
    "glove_hold_end_time_sec",
    "glove_selection_mode",
    "glove_spatial_status",
    "glove_temporal_stability",
    "detected_glove_drop_time_sec",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--all-frame-predictions", type=Path, default=DEFAULT_ALL_FRAME_PREDICTIONS)
    parser.add_argument("--zone-predictions", type=Path, default=DEFAULT_ZONE_PREDICTIONS)
    parser.add_argument("--pose-predictions", type=Path, default=DEFAULT_POSE_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-glove-confidence", type=float, default=0.25)
    parser.add_argument("--window-start-sec", type=float, default=1.00)
    parser.add_argument("--window-end-sec", type=float, default=2.40)
    parser.add_argument(
        "--min-preferred-time-sec",
        type=float,
        default=1.40,
        help="Frames before this are allowed but penalized because catchers may still be moving.",
    )
    parser.add_argument("--neighbor-gap-sec", type=float, default=0.16)
    parser.add_argument("--stable-distance-01", type=float, default=0.30)
    parser.add_argument("--drop-delta-y-01", type=float, default=0.14)
    parser.add_argument("--after-drop-penalty", type=float, default=75.0)
    parser.add_argument("--early-frame-penalty", type=float, default=45.0)
    parser.add_argument(
        "--allow-glove-only-frames",
        action="store_true",
        help="Allow frames where Roboflow found a glove but did not also find a zone. Usually leave off.",
    )
    parser.add_argument("--max-detected-zone-center-delta-px", type=float, default=120.0)
    parser.add_argument("--max-detected-zone-size-delta-px", type=float, default=90.0)
    parser.add_argument("--min-target-x", type=float, default=-1.50)
    parser.add_argument("--max-target-x", type=float, default=2.50)
    parser.add_argument("--min-target-y", type=float, default=-1.00)
    parser.add_argument("--max-target-y", type=float, default=2.00)
    parser.add_argument(
        "--motion-aware",
        action="store_true",
        help="Use pitcher pose, with pixel-motion fallback, to prefer mitts presented after delivery begins.",
    )
    parser.add_argument(
        "--require-pose",
        action="store_true",
        help="Do not fall back to generic pixel motion when no leg kick is detected.",
    )
    parser.add_argument(
        "--raw-glove-candidates",
        action="store_true",
        help="Score every glove detection in each saved raw response instead of only the top-confidence box.",
    )
    parser.add_argument("--motion-roi-left", type=float, default=0.18)
    parser.add_argument("--motion-roi-top", type=float, default=0.45)
    parser.add_argument("--motion-roi-right", type=float, default=0.52)
    parser.add_argument("--motion-roi-bottom", type=float, default=0.90)
    parser.add_argument("--motion-search-start-sec", type=float, default=1.40)
    parser.add_argument("--motion-min-rise", type=float, default=0.60)
    parser.add_argument("--motion-peak-fraction", type=float, default=0.25)
    parser.add_argument("--camera-cut-motion", type=float, default=12.0)
    parser.add_argument("--before-delivery-penalty", type=float, default=110.0)
    parser.add_argument("--delivery-pre-roll-sec", type=float, default=0.10)
    parser.add_argument("--delivery-window-sec", type=float, default=0.45)
    parser.add_argument("--delivery-window-bonus", type=float, default=35.0)
    parser.add_argument("--probable-mask-target-y", type=float, default=1.40)
    parser.add_argument("--probable-mask-penalty", type=float, default=90.0)
    parser.add_argument("--min-glove-width-zone-ratio", type=float, default=0.10)
    parser.add_argument("--max-glove-width-zone-ratio", type=float, default=1.25)
    parser.add_argument("--min-glove-height-zone-ratio", type=float, default=0.08)
    parser.add_argument("--max-glove-height-zone-ratio", type=float, default=1.10)
    parser.add_argument("--glove-track-max-distance-01", type=float, default=0.55)
    parser.add_argument("--glove-track-min-frames", type=int, default=3)
    parser.add_argument("--glove-stable-run-distance-01", type=float, default=0.10)
    parser.add_argument("--drop-persistence-fraction", type=float, default=0.65)
    parser.add_argument("--presentation-offset-sec", type=float, default=0.10)
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
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
    if pd.isna(parsed):
        return None
    return parsed


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    return pd.read_csv(path)


def is_glove_class(value) -> bool:
    label = normalized(value).strip().lower().replace("_", " ").replace("-", " ")
    return label in {
        "catcher glove",
        "catchers glove",
        "catcher's glove",
        "catcher mitt",
        "glove",
        "mitt",
    }


def iter_predictions(value) -> list[dict]:
    found = []
    if isinstance(value, dict):
        predictions = value.get("predictions")
        if isinstance(predictions, list):
            found.extend(prediction for prediction in predictions if isinstance(prediction, dict))
        for child in value.values():
            found.extend(iter_predictions(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(iter_predictions(child))
    return found


def raw_glove_rows(row: pd.Series, args: argparse.Namespace) -> list[tuple[pd.Series, str, int]]:
    if not args.raw_glove_candidates:
        return [(row, "csv_top_detection", 1)]

    raw_path = normalized(row.get("raw_response_path"))
    path = Path(raw_path) if raw_path else None
    if path is not None and not path.is_absolute():
        path = ROOT / path

    predictions = []
    if path is not None and path.exists():
        try:
            predictions = [
                prediction
                for prediction in iter_predictions(json.loads(path.read_text()))
                if is_glove_class(
                    prediction.get("class")
                    or prediction.get("class_name")
                    or prediction.get("label")
                )
            ]
        except (json.JSONDecodeError, OSError):
            predictions = []

    unique = {}
    for prediction in predictions:
        try:
            key = tuple(round(float(prediction.get(column)), 3) for column in ["x", "y", "width", "height"])
        except (TypeError, ValueError):
            continue
        confidence = number(prediction.get("confidence")) or number(prediction.get("score")) or 0.0
        if key not in unique or confidence > unique[key][0]:
            unique[key] = (confidence, prediction)

    ranked = sorted(unique.values(), key=lambda item: item[0], reverse=True)
    rows = []
    for rank, (confidence, prediction) in enumerate(ranked, start=1):
        candidate_row = row.copy()
        candidate_row["glove_confidence"] = confidence
        for source_column, output_column in [
            ("x", "glove_x"),
            ("y", "glove_y"),
            ("width", "glove_width"),
            ("height", "glove_height"),
        ]:
            candidate_row[output_column] = prediction.get(source_column)
        rows.append((candidate_row, "raw_response_detection", rank))

    if rows:
        return rows
    glove_x = number(row.get("glove_x"))
    glove_y = number(row.get("glove_y"))
    if glove_x is not None and glove_y is not None:
        return [(row, "csv_fallback_detection", 1)]
    return []


def image_array(image_path: str) -> Optional[np.ndarray]:
    path = Path(normalized(image_path))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            gray = image.convert("L").resize((160, 90), Image.Resampling.BILINEAR)
            return np.asarray(gray, dtype=np.float32)
    except (OSError, ValueError):
        return None


def pose_delivery_motion(pose_rows: pd.DataFrame, args: argparse.Namespace) -> Optional[dict]:
    if pose_rows.empty:
        return None

    pose_rows = pose_rows.copy()
    pose_rows["frame_uid"] = pose_rows["frame_uid"].astype(str)
    pose_rows["leg_kick_score_num"] = pd.to_numeric(pose_rows.get("leg_kick_score"), errors="coerce")
    pose_rows["pose_visibility_num"] = pd.to_numeric(pose_rows.get("pose_visibility"), errors="coerce")
    pose_rows["pose_coverage_num"] = pd.to_numeric(pose_rows.get("pose_coverage"), errors="coerce")
    pose_status = normalized(pose_rows.get("pose_delivery_status", pd.Series(dtype=str)).dropna().iloc[0]) if pose_rows.get("pose_delivery_status", pd.Series(dtype=str)).notna().any() else "pose_missing_sequence"
    start_values = pd.to_numeric(pose_rows.get("pose_delivery_start_time_sec"), errors="coerce").dropna()
    threshold_values = pd.to_numeric(pose_rows.get("leg_kick_threshold"), errors="coerce").dropna()
    score_by_uid = dict(zip(pose_rows["frame_uid"], pose_rows["leg_kick_score_num"]))
    visibility_by_uid = dict(zip(pose_rows["frame_uid"], pose_rows["pose_visibility_num"]))
    peak = number(pose_rows["leg_kick_score_num"].max())
    coverage = number(pose_rows["pose_coverage_num"].max())
    result = {
        "source": "mediapipe_pose",
        "status": pose_status,
        "pose_status": pose_status,
        "start_time_sec": number(start_values.iloc[0]) if not start_values.empty else None,
        "threshold": number(threshold_values.iloc[0]) if not threshold_values.empty else None,
        "peak": peak,
        "scores": score_by_uid,
        "pose_visibility": visibility_by_uid,
        "pose_coverage": coverage,
    }
    return result


def pixel_delivery_motion(rows: pd.DataFrame, args: argparse.Namespace) -> dict:
    fallback = {
        "source": "pixel_motion",
        "status": "missing_frames",
        "pose_status": "",
        "start_time_sec": args.min_preferred_time_sec,
        "threshold": None,
        "peak": None,
        "scores": {},
        "pose_visibility": {},
        "pose_coverage": None,
    }
    if rows.empty:
        return fallback

    frames = rows.drop_duplicates("frame_uid", keep="first").copy()
    frames["frame_time_sec_num"] = pd.to_numeric(frames.get("frame_time_sec"), errors="coerce")
    frames = frames.dropna(subset=["frame_time_sec_num"]).sort_values("frame_time_sec_num")
    if len(frames) < 4:
        return fallback

    loaded = []
    for _, frame in frames.iterrows():
        array = image_array(normalized(frame.get("image_path")))
        if array is not None:
            loaded.append((frame, array))
    if len(loaded) < 4:
        return fallback

    height, width = loaded[0][1].shape
    left = max(0, min(width - 1, int(args.motion_roi_left * width)))
    right = max(left + 1, min(width, int(args.motion_roi_right * width)))
    top = max(0, min(height - 1, int(args.motion_roi_top * height)))
    bottom = max(top + 1, min(height, int(args.motion_roi_bottom * height)))

    pairs = []
    score_by_uid = {}
    for (previous_row, previous), (current_row, current) in zip(loaded, loaded[1:]):
        difference = np.abs(current - previous)
        roi_score = float(np.mean(difference[top:bottom, left:right]))
        global_score = float(np.mean(difference))
        frame_uid = normalized(current_row.get("frame_uid"))
        score_by_uid[frame_uid] = roi_score
        pairs.append(
            {
                "frame_uid": frame_uid,
                "time": number(current_row.get("frame_time_sec")),
                "roi_score": roi_score,
                "global_score": global_score,
                "camera_cut": global_score >= args.camera_cut_motion,
            }
        )

    usable = [
        pair
        for pair in pairs
        if not pair["camera_cut"]
        and pair["time"] is not None
        and pair["time"] >= args.motion_search_start_sec
    ]
    if len(usable) < 3:
        fallback["scores"] = score_by_uid
        fallback["status"] = "camera_cut_or_missing_motion"
        return fallback

    baseline_values = [pair["roi_score"] for pair in usable[: min(4, len(usable))]]
    baseline = float(np.median(baseline_values))
    peak = max(pair["roi_score"] for pair in usable)
    threshold = baseline + max(
        args.motion_min_rise,
        (peak - baseline) * args.motion_peak_fraction,
    )
    fallback.update({"threshold": threshold, "peak": peak, "scores": score_by_uid})
    if peak < threshold:
        fallback["status"] = "no_clear_delivery_motion"
        return fallback

    persistence_floor = baseline + (args.motion_min_rise * 0.45)
    for index, pair in enumerate(usable):
        if pair["roi_score"] < threshold:
            continue
        following = usable[index + 1 : index + 3]
        persistent = not following or any(item["roi_score"] >= persistence_floor for item in following)
        if persistent:
            fallback["status"] = "delivery_motion_detected"
            fallback["start_time_sec"] = pair["time"]
            return fallback

    fallback["status"] = "no_persistent_delivery_motion"
    return fallback


def delivery_motion(
    rows: pd.DataFrame,
    args: argparse.Namespace,
    pose_rows: Optional[pd.DataFrame] = None,
) -> dict:
    if not args.motion_aware:
        return {
            "source": "disabled",
            "status": "disabled",
            "pose_status": "disabled",
            "start_time_sec": args.min_preferred_time_sec,
            "threshold": None,
            "peak": None,
            "scores": {},
            "pose_visibility": {},
            "pose_coverage": None,
        }

    pose_motion = pose_delivery_motion(pose_rows, args) if pose_rows is not None else None
    if pose_motion and pose_motion.get("status") == "pose_leg_kick_detected":
        return pose_motion
    if args.require_pose:
        return pose_motion or {
            "source": "mediapipe_pose",
            "status": "pose_predictions_missing",
            "pose_status": "pose_predictions_missing",
            "start_time_sec": args.min_preferred_time_sec,
            "threshold": None,
            "peak": None,
            "scores": {},
            "pose_visibility": {},
            "pose_coverage": None,
        }

    pixel_motion = pixel_delivery_motion(rows, args)
    pixel_motion["source"] = "pixel_motion_fallback" if pose_motion else "pixel_motion"
    if pose_motion:
        pixel_motion["pose_status"] = pose_motion.get("status")
        pixel_motion["pose_coverage"] = pose_motion.get("pose_coverage")
        pixel_motion["pose_visibility"] = pose_motion.get("pose_visibility", {})
    return pixel_motion


def merge_manifest(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        "frame_uid",
        "pitch_uid",
        "image_path",
        "frame_index",
        "frame_time_sec",
        "sample_pitcher_name",
        "game_date",
        "pitch_type",
    ]
    metadata = manifest[[column for column in keep_columns if column in manifest.columns]].drop_duplicates("frame_uid")
    merged = predictions.merge(metadata, on="frame_uid", how="left", suffixes=("", "_manifest"))
    for column in keep_columns:
        manifest_column = f"{column}_manifest"
        if manifest_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[manifest_column]
        else:
            merged[column] = merged[column].where(
                merged[column].notna() & merged[column].astype(str).ne(""),
                merged[manifest_column],
            )
        merged = merged.drop(columns=[manifest_column])
    return merged


def target_from_zone_and_glove(zone_row: pd.Series, glove_row: pd.Series) -> tuple[Optional[float], Optional[float]]:
    zone_x = number(zone_row.get("zone_x"))
    zone_y = number(zone_row.get("zone_y"))
    zone_width = number(zone_row.get("zone_width"))
    zone_height = number(zone_row.get("zone_height"))
    glove_x = number(glove_row.get("glove_x"))
    glove_y = number(glove_row.get("glove_y"))
    if None in {zone_x, zone_y, zone_width, zone_height, glove_x, glove_y}:
        return None, None
    if zone_width <= 0 or zone_height <= 0:
        return None, None

    zone_left = zone_x - (zone_width / 2)
    zone_top = zone_y - (zone_height / 2)
    target_x = (glove_x - zone_left) / zone_width
    target_y = 1 - ((glove_y - zone_top) / zone_height)
    return target_x, target_y


def distance_01(a: dict, b: dict) -> Optional[float]:
    if a.get("target_x") is None or a.get("target_y") is None:
        return None
    if b.get("target_x") is None or b.get("target_y") is None:
        return None
    return ((a["target_x"] - b["target_x"]) ** 2 + (a["target_y"] - b["target_y"]) ** 2) ** 0.5


def is_plausible(candidate: dict, args: argparse.Namespace) -> bool:
    target_x = candidate.get("target_x")
    target_y = candidate.get("target_y")
    if target_x is None or target_y is None:
        return False
    target_plausible = (
        args.min_target_x <= target_x <= args.max_target_x
        and args.min_target_y <= target_y <= args.max_target_y
    )
    width_ratio = candidate.get("glove_width_zone_ratio")
    height_ratio = candidate.get("glove_height_zone_ratio")
    size_plausible = (
        width_ratio is not None
        and height_ratio is not None
        and args.min_glove_width_zone_ratio <= width_ratio <= args.max_glove_width_zone_ratio
        and args.min_glove_height_zone_ratio <= height_ratio <= args.max_glove_height_zone_ratio
    )
    return target_plausible and size_plausible


def detected_zone_matches_statcast(zone_row: pd.Series, glove_row: pd.Series, args: argparse.Namespace) -> tuple[bool, str]:
    if args.allow_glove_only_frames:
        return True, "glove_only_allowed"

    if normalized(glove_row.get("status")) != "ok":
        return False, "missing_detected_zone_or_glove"

    detected_zone = {
        "zone_x": number(glove_row.get("zone_x")),
        "zone_y": number(glove_row.get("zone_y")),
        "zone_width": number(glove_row.get("zone_width")),
        "zone_height": number(glove_row.get("zone_height")),
    }
    statcast_zone = {
        "zone_x": number(zone_row.get("zone_x")),
        "zone_y": number(zone_row.get("zone_y")),
        "zone_width": number(zone_row.get("zone_width")),
        "zone_height": number(zone_row.get("zone_height")),
    }
    if any(value is None for value in detected_zone.values()) or any(value is None for value in statcast_zone.values()):
        return False, "missing_zone_geometry"

    center_delta = (
        (detected_zone["zone_x"] - statcast_zone["zone_x"]) ** 2
        + (detected_zone["zone_y"] - statcast_zone["zone_y"]) ** 2
    ) ** 0.5
    size_delta = (
        (detected_zone["zone_width"] - statcast_zone["zone_width"]) ** 2
        + (detected_zone["zone_height"] - statcast_zone["zone_height"]) ** 2
    ) ** 0.5
    if center_delta > args.max_detected_zone_center_delta_px:
        return False, "detected_zone_center_mismatch"
    if size_delta > args.max_detected_zone_size_delta_px:
        return False, "detected_zone_size_mismatch"
    return True, "detected_zone_matches"


def first_drop_time(candidates: list[dict], args: argparse.Namespace) -> Optional[float]:
    valid = [candidate for candidate in candidates if candidate.get("target_y") is not None]
    by_time = {}
    for candidate in valid:
        by_time.setdefault(candidate["frame_time_sec"], []).append(candidate)
    times = sorted(by_time)
    for previous_time, current_time in zip(times, times[1:]):
        if current_time - previous_time > args.neighbor_gap_sec:
            continue
        previous_candidates = by_time[previous_time]
        current_candidates = by_time[current_time]
        for previous in previous_candidates:
            current = min(
                current_candidates,
                key=lambda item: distance_01(previous, item) if distance_01(previous, item) is not None else 999,
            )
            if abs(previous["target_x"] - current["target_x"]) > 0.45:
                continue
            if previous["target_y"] - current["target_y"] >= args.drop_delta_y_01:
                return current_time
    return None


def stability_for_candidate(candidate: dict, candidates: list[dict], args: argparse.Namespace) -> float:
    distances = []
    for other in candidates:
        if other is candidate:
            continue
        if abs(other["frame_time_sec"] - candidate["frame_time_sec"]) < 0.001:
            continue
        if abs(other["frame_time_sec"] - candidate["frame_time_sec"]) > args.neighbor_gap_sec:
            continue
        distance = distance_01(candidate, other)
        if distance is not None:
            distances.append(distance)
    if not distances:
        return 0.0
    nearest = min(distances)
    return max(0.0, min(1.0, 1 - (nearest / args.stable_distance_01)))


def candidate_rows(
    rows: pd.DataFrame,
    zone_row: pd.Series,
    args: argparse.Namespace,
    motion: dict,
) -> list[dict]:
    candidates = []
    for _, row in rows.iterrows():
        frame_time = number(row.get("frame_time_sec"))
        if frame_time is None:
            continue
        if frame_time < args.window_start_sec or frame_time > args.window_end_sec:
            continue
        for glove_row, candidate_source, candidate_rank in raw_glove_rows(row, args):
            glove_x = number(glove_row.get("glove_x"))
            glove_y = number(glove_row.get("glove_y"))
            if glove_x is None or glove_y is None:
                continue
            frame_valid, frame_valid_reason = detected_zone_matches_statcast(zone_row, glove_row, args)
            if not frame_valid:
                continue

            target_x, target_y = target_from_zone_and_glove(zone_row, glove_row)
            confidence = number(glove_row.get("glove_confidence")) or 0.0
            zone_width = number(zone_row.get("zone_width"))
            zone_height = number(zone_row.get("zone_height"))
            glove_width = number(glove_row.get("glove_width"))
            glove_height = number(glove_row.get("glove_height"))
            width_ratio = glove_width / zone_width if glove_width and zone_width else None
            height_ratio = glove_height / zone_height if glove_height and zone_height else None
            candidate = {
                "row": glove_row,
                "frame_time_sec": frame_time,
                "target_x": target_x,
                "target_y": target_y,
                "glove_confidence": confidence,
                "glove_width_zone_ratio": width_ratio,
                "glove_height_zone_ratio": height_ratio,
                "frame_valid_reason": frame_valid_reason,
                "candidate_source": candidate_source,
                "candidate_rank": candidate_rank,
                "motion_score": motion.get("scores", {}).get(normalized(row.get("frame_uid"))),
                "pose_visibility": motion.get("pose_visibility", {}).get(normalized(row.get("frame_uid"))),
                "plausible": False,
                "score": 0.0,
                "reason": [],
            }
            candidate["plausible"] = is_plausible(candidate, args)
            candidates.append(candidate)
    return candidates


def score_candidates(candidates: list[dict], args: argparse.Namespace, motion: dict) -> None:
    if not candidates:
        return

    plausible_candidates = [candidate for candidate in candidates if candidate.get("plausible")]
    drop_candidates = plausible_candidates or candidates
    delivery_start = number(motion.get("start_time_sec"))
    if args.motion_aware and delivery_start is not None:
        post_delivery = [
            candidate
            for candidate in drop_candidates
            if candidate["frame_time_sec"] >= delivery_start
        ]
        if post_delivery:
            drop_candidates = post_delivery
    drop_time = first_drop_time(drop_candidates, args)
    for candidate in candidates:
        candidate["drop_time_sec"] = drop_time
        confidence = candidate["glove_confidence"]
        time_span = max(args.window_end_sec - args.window_start_sec, 0.001)
        time_norm = (candidate["frame_time_sec"] - args.window_start_sec) / time_span
        stability = stability_for_candidate(candidate, candidates, args)
        candidate["stability"] = stability

        late_weight = 5 if args.motion_aware else 18
        score = (confidence * 100) + (stability * 35) + (time_norm * late_weight)
        reason = ["confidence", "late_window"]
        if stability > 0:
            reason.append("stable_neighbor")
        if confidence < args.min_glove_confidence:
            score -= 45
            reason.append("low_confidence")
        if candidate["frame_time_sec"] < args.min_preferred_time_sec:
            score -= args.early_frame_penalty
            reason.append("early_setup_window")
        if not candidate["plausible"]:
            score -= 75
            reason.append("implausible_glove_geometry")
        if candidate.get("target_y") is not None and candidate["target_y"] > args.probable_mask_target_y:
            score -= args.probable_mask_penalty
            reason.append("probable_mask_or_head")

        delivery_start = number(motion.get("start_time_sec"))
        if args.motion_aware and delivery_start is not None:
            if candidate["frame_time_sec"] < delivery_start - args.delivery_pre_roll_sec:
                score -= args.before_delivery_penalty
                reason.append("before_delivery_start")
            elif candidate["frame_time_sec"] <= delivery_start + args.delivery_window_sec:
                score += args.delivery_window_bonus
                reason.append("delivery_presentation_window")
            else:
                reason.append("late_after_delivery_window")

            peak = number(motion.get("peak"))
            motion_score = number(candidate.get("motion_score"))
            if peak and motion_score is not None:
                score += min(10.0, max(0.0, (motion_score / peak) * 10.0))
                reason.append("pitcher_motion_signal")
        if drop_time is not None:
            if candidate["frame_time_sec"] >= drop_time:
                score -= args.after_drop_penalty
                reason.append("after_detected_drop")
            elif drop_time - candidate["frame_time_sec"] <= args.neighbor_gap_sec:
                score += 25
                reason.append("last_frame_before_drop")

        if candidate.get("frame_valid_reason"):
            reason.append(candidate["frame_valid_reason"])
        candidate["score"] = round(score, 4)
        candidate["reason"] = reason


def build_glove_tracks(candidates: list[dict], args: argparse.Namespace) -> list[list[dict]]:
    tracks = []
    by_time = {}
    for candidate in candidates:
        by_time.setdefault(candidate["frame_time_sec"], []).append(candidate)

    for frame_time in sorted(by_time):
        assigned_track_ids = set()
        frame_candidates = sorted(
            by_time[frame_time],
            key=lambda candidate: candidate["glove_confidence"],
            reverse=True,
        )
        for candidate in frame_candidates:
            choices = []
            for track_id, track in enumerate(tracks):
                if track_id in assigned_track_ids:
                    continue
                previous = track[-1]
                gap = frame_time - previous["frame_time_sec"]
                if gap <= 0 or gap > args.neighbor_gap_sec:
                    continue
                distance = distance_01(candidate, previous)
                if distance is None or distance > args.glove_track_max_distance_01:
                    continue
                choices.append((distance, track_id))

            if choices:
                _, track_id = min(choices)
                tracks[track_id].append(candidate)
                assigned_track_ids.add(track_id)
            else:
                tracks.append([candidate])
                assigned_track_ids.add(len(tracks) - 1)
    return tracks


def stable_runs(track: list[dict], args: argparse.Namespace) -> list[list[dict]]:
    if not track:
        return []
    ordered = sorted(track, key=lambda candidate: candidate["frame_time_sec"])
    runs = [[ordered[0]]]
    for candidate in ordered[1:]:
        previous = runs[-1][-1]
        gap = candidate["frame_time_sec"] - previous["frame_time_sec"]
        distance = distance_01(candidate, previous)
        if (
            gap <= args.neighbor_gap_sec
            and distance is not None
            and distance <= args.glove_stable_run_distance_01
        ):
            runs[-1].append(candidate)
        else:
            runs.append([candidate])
    return runs


def setup_drop_evidence(
    track: list[dict],
    args: argparse.Namespace,
    motion: dict,
) -> dict:
    delivery_start = number(motion.get("start_time_sec"))
    ordered = sorted(
        [
            candidate
            for candidate in track
            if delivery_start is None or candidate["frame_time_sec"] >= delivery_start
        ],
        key=lambda candidate: candidate["frame_time_sec"],
    )
    fallback = {
        "confirmed": False,
        "setup_candidate": None,
        "hold_run": [],
        "drop_time_sec": None,
        "mode": "no_post_leg_kick_track",
    }
    if not ordered:
        return fallback

    runs = stable_runs(ordered, args)
    stable = [run for run in runs if len(run) >= 2]
    confirmed = []
    for run in stable:
        run_end_time = run[-1]["frame_time_sec"]
        baseline_y = float(np.median([candidate["target_y"] for candidate in run]))
        future = [candidate for candidate in ordered if candidate["frame_time_sec"] > run_end_time]
        for index, candidate in enumerate(future):
            total_drop = baseline_y - candidate["target_y"]
            if total_drop < args.drop_delta_y_01:
                continue
            following = future[index + 1 : index + 3]
            sustained = not following or any(
                baseline_y - item["target_y"]
                >= args.drop_delta_y_01 * args.drop_persistence_fraction
                for item in following
            )
            if not sustained:
                continue
            confirmed.append(
                {
                    "confirmed": True,
                    "setup_candidate": run[-1],
                    "hold_run": run,
                    "drop_time_sec": candidate["frame_time_sec"],
                    "mode": "last_frame_before_confirmed_glove_drop",
                    "evidence_score": (len(run) * 20) + (total_drop * 100),
                }
            )
            break

    if confirmed:
        return max(
            confirmed,
            key=lambda evidence: (
                evidence["evidence_score"],
                -evidence["drop_time_sec"],
            ),
        )

    if stable:
        if delivery_start is not None:
            early_stable = [
                run
                for run in stable
                if run[0]["frame_time_sec"] <= delivery_start + args.delivery_window_sec
            ]
        else:
            early_stable = stable
        run = max(early_stable or stable, key=lambda values: (len(values), -values[0]["frame_time_sec"]))
        return {
            "confirmed": False,
            "setup_candidate": run[-1],
            "hold_run": run,
            "drop_time_sec": None,
            "mode": "stable_hold_without_confirmed_drop",
        }

    ideal_time = (
        delivery_start + args.presentation_offset_sec
        if delivery_start is not None
        else ordered[0]["frame_time_sec"]
    )
    return {
        "confirmed": False,
        "setup_candidate": min(
            ordered,
            key=lambda candidate: abs(candidate["frame_time_sec"] - ideal_time),
        ),
        "hold_run": [],
        "drop_time_sec": None,
        "mode": "single_frame_without_hold_or_drop",
    }


def track_score(track: list[dict], args: argparse.Namespace) -> tuple[float, float, int]:
    if not track:
        return -999.0, 0.0, 0
    steps = [
        distance_01(previous, current)
        for previous, current in zip(track, track[1:])
    ]
    steps = [distance for distance in steps if distance is not None]
    median_step = float(np.median(steps)) if steps else args.glove_track_max_distance_01
    stability = max(0.0, min(1.0, 1 - median_step / args.glove_track_max_distance_01))
    frame_count = len({candidate["frame_time_sec"] for candidate in track})
    confidence = float(np.median([candidate["glove_confidence"] for candidate in track]))
    target_x = float(np.median([candidate["target_x"] for candidate in track]))
    target_y = float(np.median([candidate["target_y"] for candidate in track]))
    prior_distance = (((target_x - 0.42) / 0.70) ** 2 + ((target_y - 0.61) / 0.55) ** 2) ** 0.5

    score = (frame_count * 22) + (stability * 45) + (confidence * 30)
    if frame_count < args.glove_track_min_frames:
        score -= (args.glove_track_min_frames - frame_count) * 35
    if prior_distance > 1.0:
        score -= min(45.0, (prior_distance - 1.0) * 22)
    if target_y > 1.20:
        score -= 40
    return score, stability, frame_count


def tracked_candidate(
    candidates: list[dict],
    args: argparse.Namespace,
    motion: dict,
) -> Optional[dict]:
    tracks = build_glove_tracks(candidates, args)
    if not tracks:
        return None

    scored_tracks = []
    for track_id, track in enumerate(tracks, start=1):
        score, stability, frame_count = track_score(track, args)
        evidence = setup_drop_evidence(track, args, motion)
        if evidence["confirmed"]:
            score += 100 + number(evidence.get("evidence_score"))
        elif evidence["hold_run"]:
            score += len(evidence["hold_run"]) * 8
        scored_tracks.append((score, stability, frame_count, track_id, track, evidence))
    persistent = [item for item in scored_tracks if item[2] >= args.glove_track_min_frames]
    score, stability, frame_count, track_id, track, evidence = max(
        persistent or scored_tracks,
        key=lambda item: item[0],
    )
    selected_source = evidence.get("setup_candidate")
    if selected_source is None:
        return None
    selected = selected_source.copy()
    run = evidence.get("hold_run", [])
    selected["track_id"] = track_id
    selected["track_frame_count"] = frame_count
    selected["track_stability"] = stability
    selected["stable_run_count"] = len(run)
    selected["drop_confirmed"] = evidence["confirmed"]
    selected["hold_start_time_sec"] = run[0]["frame_time_sec"] if run else None
    selected["hold_end_time_sec"] = run[-1]["frame_time_sec"] if run else None
    selected["drop_time_sec"] = evidence.get("drop_time_sec")
    selected["selection_mode"] = evidence.get("mode")
    selected["score"] = round(selected.get("score", 0.0) + score, 4)
    selected["reason"] = [
        *selected.get("reason", []),
        "persistent_glove_track" if frame_count >= args.glove_track_min_frames else "short_glove_track",
        "stable_glove_plateau" if len(run) >= 2 else "single_frame_glove_fallback",
        "confirmed_glove_drop" if evidence["confirmed"] else "unconfirmed_glove_drop",
    ]
    return selected


def choose_candidate(candidates: list[dict], args: argparse.Namespace, motion: dict) -> Optional[dict]:
    score_candidates(candidates, args, motion)
    if not candidates:
        return None
    plausible = [candidate for candidate in candidates if candidate["plausible"]]
    if args.motion_aware and not plausible:
        return None
    pool = plausible or candidates
    if args.motion_aware:
        delivery_start = number(motion.get("start_time_sec"))
        after_delivery = [
            candidate
            for candidate in pool
            if delivery_start is None
            or candidate["frame_time_sec"] >= delivery_start
        ]
        if after_delivery:
            pool = after_delivery
        elif delivery_start is not None:
            return None

        tracked = tracked_candidate(pool, args, motion)
        if tracked is not None:
            return tracked
    return max(pool, key=lambda candidate: (candidate["score"], -abs(candidate["frame_time_sec"] - args.min_preferred_time_sec)))


def output_record(
    pitch_uid: str,
    zone_row: pd.Series,
    selected: Optional[dict],
    candidate_count: int,
    plausible_count: int,
    motion: dict,
    args: argparse.Namespace,
) -> dict:
    zone_present = all(normalized(zone_row.get(column)) for column in ["zone_x", "zone_y", "zone_width", "zone_height"])
    if selected:
        glove_row = selected["row"]
        target_x = selected.get("target_x")
        target_y = selected.get("target_y")
        frame_uid = normalized(glove_row.get("frame_uid"))
        image_path = normalized(glove_row.get("image_path"))
        frame_index = normalized(glove_row.get("frame_index"))
        frame_time_sec = normalized(glove_row.get("frame_time_sec"))
        status = "ok" if zone_present else "missing_required_detection"
        error = "" if target_x is not None and target_y is not None else "missing_vision_target"
    else:
        glove_row = pd.Series(dtype="object")
        target_x = None
        target_y = None
        frame_uid = normalized(zone_row.get("frame_uid"))
        image_path = normalized(zone_row.get("image_path"))
        frame_index = normalized(zone_row.get("frame_index"))
        frame_time_sec = normalized(zone_row.get("frame_time_sec"))
        status = "ok" if zone_present else "missing_required_detection"
        error = "missing_glove_detection"

    return {
        "frame_uid": frame_uid,
        "pitch_uid": pitch_uid,
        "image_path": image_path,
        "frame_index": frame_index,
        "frame_time_sec": frame_time_sec,
        "workflow_output_key": "statcast_zone_best_glove_frame",
        "zone_confidence": normalized(zone_row.get("zone_confidence")),
        "glove_confidence": normalized(glove_row.get("glove_confidence")),
        "zone_x": normalized(zone_row.get("zone_x")),
        "zone_y": normalized(zone_row.get("zone_y")),
        "zone_width": normalized(zone_row.get("zone_width")),
        "zone_height": normalized(zone_row.get("zone_height")),
        "glove_x": normalized(glove_row.get("glove_x")),
        "glove_y": normalized(glove_row.get("glove_y")),
        "glove_width": normalized(glove_row.get("glove_width")),
        "glove_height": normalized(glove_row.get("glove_height")),
        "vision_target_x_01": normalized(target_x),
        "vision_target_y_01": normalized(target_y),
        "prediction_count": normalized(glove_row.get("prediction_count")),
        "raw_response_path": normalized(glove_row.get("raw_response_path")),
        "status": status,
        "error": error,
        "zone_source": normalized(zone_row.get("zone_source")),
        "zone_anchor_source": normalized(zone_row.get("zone_anchor_source")),
        "zone_registration_source": normalized(zone_row.get("zone_registration_source")),
        "zone_anchor_pitch_support": normalized(zone_row.get("zone_anchor_pitch_support")),
        "zone_anchor_frame_support": normalized(zone_row.get("zone_anchor_frame_support")),
        "zone_anchor_center_mad_px": normalized(zone_row.get("zone_anchor_center_mad_px")),
        "glove_source_model": (
            "motion_aware_dense15_raw"
            if selected and args.motion_aware and args.raw_glove_candidates
            else "motion_aware_dense15"
            if selected and args.motion_aware
            else "dense15_best_frame"
            if selected
            else "missing_dense15_glove"
        ),
        "statcast_sz_top": normalized(zone_row.get("statcast_sz_top")),
        "statcast_sz_bot": normalized(zone_row.get("statcast_sz_bot")),
        "statcast_zone_height_ft": normalized(zone_row.get("statcast_zone_height_ft")),
        "vision_fallback_used": normalized(zone_row.get("vision_fallback_used")),
        "vision_fallback_reason": normalized(zone_row.get("vision_fallback_reason")),
        "selected_frame_uid": frame_uid if selected else "",
        "selected_frame_index": frame_index if selected else "",
        "selected_frame_time_sec": frame_time_sec if selected else "",
        "glove_candidate_count": candidate_count,
        "plausible_glove_candidate_count": plausible_count,
        "glove_selection_score": normalized(selected.get("score")) if selected else "",
        "glove_selection_reason": ";".join(selected.get("reason", [])) if selected else "no_glove_candidates",
        "motion_gate_used": "yes" if args.motion_aware else "no",
        "delivery_signal_source": normalized(motion.get("source")),
        "delivery_motion_status": normalized(motion.get("status")),
        "delivery_start_time_sec": normalized(motion.get("start_time_sec")),
        "delivery_motion_threshold": normalized(motion.get("threshold")),
        "delivery_motion_peak": normalized(motion.get("peak")),
        "selected_delivery_motion_score": normalized(selected.get("motion_score")) if selected else "",
        "selected_pose_visibility": normalized(selected.get("pose_visibility")) if selected else "",
        "pose_delivery_status": normalized(motion.get("pose_status")),
        "pose_coverage": normalized(motion.get("pose_coverage")),
        "selected_after_delivery_start": (
            "yes"
            if selected
            and number(motion.get("start_time_sec")) is not None
            and selected["frame_time_sec"] >= number(motion.get("start_time_sec"))
            else "no"
            if selected
            else ""
        ),
        "glove_candidate_source": normalized(selected.get("candidate_source")) if selected else "",
        "glove_candidate_rank": normalized(selected.get("candidate_rank")) if selected else "",
        "glove_track_id": normalized(selected.get("track_id")) if selected else "",
        "glove_track_frame_count": normalized(selected.get("track_frame_count")) if selected else "",
        "glove_track_stability": normalized(selected.get("track_stability")) if selected else "",
        "glove_stable_run_count": normalized(selected.get("stable_run_count")) if selected else "",
        "glove_drop_confirmed": (
            "yes" if selected and selected.get("drop_confirmed") else "no" if selected else ""
        ),
        "glove_hold_start_time_sec": normalized(selected.get("hold_start_time_sec")) if selected else "",
        "glove_hold_end_time_sec": normalized(selected.get("hold_end_time_sec")) if selected else "",
        "glove_selection_mode": normalized(selected.get("selection_mode")) if selected else "",
        "glove_spatial_status": "plausible" if selected and selected.get("plausible") else "implausible" if selected else "",
        "glove_temporal_stability": normalized(selected.get("stability")) if selected else "",
        "detected_glove_drop_time_sec": normalized(selected.get("drop_time_sec")) if selected else "",
    }


def main() -> None:
    args = parse_args()
    manifest = read_csv(resolve(args.manifest))
    all_frame_predictions = merge_manifest(read_csv(resolve(args.all_frame_predictions)), manifest)
    zone_predictions = read_csv(resolve(args.zone_predictions)).drop_duplicates("pitch_uid", keep="first")
    pose_path = resolve(args.pose_predictions)
    pose_predictions = read_csv(pose_path) if pose_path.exists() else pd.DataFrame()
    output_path = resolve(args.output)

    all_frame_predictions["pitch_uid"] = all_frame_predictions["pitch_uid"].astype(str)
    zone_predictions["pitch_uid"] = zone_predictions["pitch_uid"].astype(str)
    rows_by_pitch = dict(tuple(all_frame_predictions.groupby("pitch_uid", dropna=False)))
    if not pose_predictions.empty:
        pose_predictions["pitch_uid"] = pose_predictions["pitch_uid"].astype(str)
        pose_by_pitch = dict(tuple(pose_predictions.groupby("pitch_uid", dropna=False)))
    else:
        pose_by_pitch = {}

    output_rows = []
    for _, zone_row in zone_predictions.sort_values("pitch_uid").iterrows():
        pitch_uid = normalized(zone_row.get("pitch_uid"))
        rows = rows_by_pitch.get(pitch_uid, pd.DataFrame())
        pose_rows = pose_by_pitch.get(pitch_uid, pd.DataFrame())
        motion = delivery_motion(rows, args, pose_rows)
        candidates = candidate_rows(rows, zone_row, args, motion) if not rows.empty else []
        selected = choose_candidate(candidates, args, motion)
        plausible_count = sum(1 for candidate in candidates if candidate["plausible"])
        output_rows.append(
            output_record(
                pitch_uid,
                zone_row,
                selected,
                len(candidates),
                plausible_count,
                motion,
                args,
            )
        )

    output = pd.DataFrame(output_rows)
    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output[OUTPUT_COLUMNS].fillna("").to_csv(output_path, index=False)

    print(f"Output: {output_path}")
    print(f"Rows: {len(output)}")
    print(f"Rows with selected glove: {(output['selected_frame_uid'].astype(str) != '').sum()}")
    print(f"Rows with target: {(output['vision_target_x_01'].astype(str) != '').sum()}")
    if args.motion_aware:
        print("Delivery motion statuses:")
        print(output["delivery_motion_status"].value_counts().to_string())
        print("Delivery signal sources:")
        print(output["delivery_signal_source"].value_counts().to_string())
        print("Selected after delivery start:")
        print(output["selected_after_delivery_start"].value_counts(dropna=False).to_string())
    if args.raw_glove_candidates:
        print("Selected glove candidate ranks:")
        print(output["glove_candidate_rank"].value_counts(dropna=False).head(10).to_string())
    print("Selection reasons:")
    print(output["glove_selection_reason"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()

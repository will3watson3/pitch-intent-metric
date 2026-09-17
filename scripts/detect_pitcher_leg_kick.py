import argparse
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pitch-intent-matplotlib")

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_MANIFEST = DATA_DIR / "setup_frame_manifest_16_1200_2700.csv"
DEFAULT_OUTPUT = DATA_DIR / "pitcher_pose_frames_1200_2700.csv"

LANDMARK_COLUMNS = [
    "left_shoulder_x",
    "left_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_y",
    "left_hip_x",
    "left_hip_y",
    "right_hip_x",
    "right_hip_y",
    "left_knee_x",
    "left_knee_y",
    "right_knee_x",
    "right_knee_y",
    "left_ankle_x",
    "left_ankle_y",
    "right_ankle_x",
    "right_ankle_y",
]

OUTPUT_COLUMNS = [
    "frame_uid",
    "pitch_uid",
    "image_path",
    "frame_index",
    "frame_time_sec",
    "pose_status",
    "pose_visibility",
    "pose_center_x",
    "pose_center_y",
    "torso_length",
    *LANDMARK_COLUMNS,
    "left_knee_height_torso",
    "right_knee_height_torso",
    "leg_kick_score",
    "lead_leg",
    "leg_kick_threshold",
    "pose_delivery_status",
    "pose_delivery_start_time_sec",
    "pose_frame_count",
    "pose_coverage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect pitcher pose landmarks and infer leg-kick timing for each pitch."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-pitches", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--crop-left", type=float, default=0.12)
    parser.add_argument("--crop-top", type=float, default=0.28)
    parser.add_argument("--crop-right", type=float, default=0.54)
    parser.add_argument("--crop-bottom", type=float, default=0.98)
    parser.add_argument("--min-pose-visibility", type=float, default=0.35)
    parser.add_argument("--min-hip-x", type=float, default=0.18)
    parser.add_argument("--max-hip-x", type=float, default=0.52)
    parser.add_argument("--min-hip-y", type=float, default=0.42)
    parser.add_argument("--max-hip-y", type=float, default=0.90)
    parser.add_argument("--min-torso-length", type=float, default=0.035)
    parser.add_argument("--max-torso-length", type=float, default=0.24)
    parser.add_argument("--search-start-sec", type=float, default=1.30)
    parser.add_argument("--baseline-frame-count", type=int, default=3)
    parser.add_argument("--leg-kick-threshold", type=float, default=0.18)
    parser.add_argument(
        "--absolute-knee-height-threshold",
        type=float,
        default=0.30,
        help="Detect a kick when either knee rises within this many torso lengths below the hips.",
    )
    parser.add_argument(
        "--relative-knee-height-ceiling",
        type=float,
        default=0.65,
        help="A relative lift is only valid once the lifted knee is this close to the hips.",
    )
    parser.add_argument("--persistence-fraction", type=float, default=0.70)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def number(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def image_path(value) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_pose_runtime(min_detection_confidence: float):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "mediapipe is not installed. Run: "
            "./.venv/bin/python -m pip install mediapipe==0.10.21"
        ) from exc

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
    )
    return mp, pose


def mean_point(a, b) -> tuple[float, float]:
    return (float(a.x + b.x) / 2, float(a.y + b.y) / 2)


def full_frame_point(landmark, crop: tuple[float, float, float, float]) -> tuple[float, float]:
    left, top, right, bottom = crop
    return (
        left + (float(landmark.x) * (right - left)),
        top + (float(landmark.y) * (bottom - top)),
    )


def empty_pose_record(row: pd.Series, status: str) -> dict:
    record = {
        "frame_uid": str(row.get("frame_uid", "")),
        "pitch_uid": str(row.get("pitch_uid", "")),
        "image_path": str(row.get("image_path", "")),
        "frame_index": row.get("frame_index", ""),
        "frame_time_sec": row.get("frame_time_sec", ""),
        "pose_status": status,
    }
    for column in OUTPUT_COLUMNS:
        record.setdefault(column, "")
    return record


def detect_frame(row: pd.Series, pose, mp, args: argparse.Namespace) -> dict:
    path = image_path(row.get("image_path", ""))
    if not path.exists():
        return empty_pose_record(row, "missing_image")

    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
    except (OSError, ValueError):
        return empty_pose_record(row, "unreadable_image")

    height, width = rgb.shape[:2]
    left_px = max(0, min(width - 1, int(args.crop_left * width)))
    right_px = max(left_px + 1, min(width, int(args.crop_right * width)))
    top_px = max(0, min(height - 1, int(args.crop_top * height)))
    bottom_px = max(top_px + 1, min(height, int(args.crop_bottom * height)))
    crop_image = np.ascontiguousarray(rgb[top_px:bottom_px, left_px:right_px])
    result = pose.process(crop_image)
    if not result.pose_landmarks:
        return empty_pose_record(row, "pose_missing")

    landmarks = result.pose_landmarks.landmark
    pose_landmark = mp.solutions.pose.PoseLandmark
    selected = {
        "left_shoulder": landmarks[pose_landmark.LEFT_SHOULDER.value],
        "right_shoulder": landmarks[pose_landmark.RIGHT_SHOULDER.value],
        "left_hip": landmarks[pose_landmark.LEFT_HIP.value],
        "right_hip": landmarks[pose_landmark.RIGHT_HIP.value],
        "left_knee": landmarks[pose_landmark.LEFT_KNEE.value],
        "right_knee": landmarks[pose_landmark.RIGHT_KNEE.value],
        "left_ankle": landmarks[pose_landmark.LEFT_ANKLE.value],
        "right_ankle": landmarks[pose_landmark.RIGHT_ANKLE.value],
    }
    crop = (args.crop_left, args.crop_top, args.crop_right, args.crop_bottom)
    points = {name: full_frame_point(landmark, crop) for name, landmark in selected.items()}
    visibility = float(np.mean([landmark.visibility for landmark in selected.values()]))
    shoulder_center = (
        (points["left_shoulder"][0] + points["right_shoulder"][0]) / 2,
        (points["left_shoulder"][1] + points["right_shoulder"][1]) / 2,
    )
    hip_center = (
        (points["left_hip"][0] + points["right_hip"][0]) / 2,
        (points["left_hip"][1] + points["right_hip"][1]) / 2,
    )
    torso_length = float(
        ((shoulder_center[0] - hip_center[0]) ** 2 + (shoulder_center[1] - hip_center[1]) ** 2) ** 0.5
    )

    status = "pose_detected"
    if visibility < args.min_pose_visibility:
        status = "low_pose_visibility"
    elif not (args.min_hip_x <= hip_center[0] <= args.max_hip_x):
        status = "pose_outside_pitcher_roi"
    elif not (args.min_hip_y <= hip_center[1] <= args.max_hip_y):
        status = "pose_outside_pitcher_roi"
    elif not (args.min_torso_length <= torso_length <= args.max_torso_length):
        status = "implausible_pose_scale"

    record = empty_pose_record(row, status)
    record.update(
        {
            "pose_visibility": visibility,
            "pose_center_x": hip_center[0],
            "pose_center_y": hip_center[1],
            "torso_length": torso_length,
        }
    )
    for name, (x, y) in points.items():
        record[f"{name}_x"] = x
        record[f"{name}_y"] = y
    return record


def apply_leg_kick_metrics(group: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    group = group.copy().sort_values("frame_time_sec_num")
    valid = group[group["pose_status"] == "pose_detected"].copy()
    if len(valid) >= 3:
        centers = valid[["pose_center_x", "pose_center_y"]].apply(pd.to_numeric, errors="coerce")
        support = {}
        for index, center in centers.iterrows():
            nearby = (
                (centers["pose_center_x"] - center["pose_center_x"]).abs() <= 0.10
            ) & ((centers["pose_center_y"] - center["pose_center_y"]).abs() <= 0.15)
            support[index] = int(nearby.sum())
        track_index = max(
            support,
            key=lambda index: (
                support[index],
                -(
                    number(centers.loc[index, "pose_center_x"])
                    if number(centers.loc[index, "pose_center_x"]) is not None
                    else 999.0
                ),
            ),
        )
        track_center = centers.loc[track_index]
        track_mask = (
            (centers["pose_center_x"] - track_center["pose_center_x"]).abs() <= 0.10
        ) & ((centers["pose_center_y"] - track_center["pose_center_y"]).abs() <= 0.15)
        outlier_indexes = valid.index[~track_mask]
        group.loc[outlier_indexes, "pose_status"] = "pose_track_outlier"
        valid = valid.loc[track_mask].copy()
    pose_frame_count = len(valid)
    coverage = pose_frame_count / len(group) if len(group) else 0.0

    group["pose_frame_count"] = pose_frame_count
    group["pose_coverage"] = coverage
    group["leg_kick_threshold"] = args.leg_kick_threshold
    group["pose_delivery_status"] = "pose_missing_sequence"
    group["pose_delivery_start_time_sec"] = ""
    if pose_frame_count < 3:
        return group

    valid["torso_length_num"] = pd.to_numeric(valid["torso_length"], errors="coerce")
    for side in ["left", "right"]:
        hip = pd.to_numeric(valid[f"{side}_hip_y"], errors="coerce")
        knee = pd.to_numeric(valid[f"{side}_knee_y"], errors="coerce")
        valid[f"{side}_knee_height_torso"] = (knee - hip) / valid["torso_length_num"]

    baseline_rows = valid.head(max(1, args.baseline_frame_count))
    baseline_left = baseline_rows["left_knee_height_torso"].median()
    baseline_right = baseline_rows["right_knee_height_torso"].median()
    valid["left_leg_lift"] = baseline_left - valid["left_knee_height_torso"]
    valid["right_leg_lift"] = baseline_right - valid["right_knee_height_torso"]
    valid["leg_kick_score"] = valid[["left_leg_lift", "right_leg_lift"]].max(axis=1).clip(lower=0)
    valid["lead_leg"] = np.where(valid["left_leg_lift"] >= valid["right_leg_lift"], "left", "right")

    for column in ["left_knee_height_torso", "right_knee_height_torso", "leg_kick_score", "lead_leg"]:
        group.loc[valid.index, column] = valid[column]

    search = valid[valid["frame_time_sec_num"] >= args.search_start_sec]
    delivery_start = None
    for position, (_, row) in enumerate(search.iterrows()):
        score = number(row.get("leg_kick_score")) or 0.0
        left_height = number(row.get("left_knee_height_torso"))
        right_height = number(row.get("right_knee_height_torso"))
        knee_height = min(value for value in [left_height, right_height] if value is not None)
        relative_trigger = (
            score >= args.leg_kick_threshold
            and knee_height <= args.relative_knee_height_ceiling
        )
        absolute_trigger = knee_height <= args.absolute_knee_height_threshold
        if not relative_trigger and not absolute_trigger:
            continue
        following = search.iloc[position + 1 : position + 3]
        following_scores = pd.to_numeric(following["leg_kick_score"], errors="coerce")
        following_knees = pd.concat(
            [
                pd.to_numeric(following["left_knee_height_torso"], errors="coerce"),
                pd.to_numeric(following["right_knee_height_torso"], errors="coerce"),
            ],
            axis=1,
        ).min(axis=1)
        persistent = (
            following.empty
            or (following_scores >= args.leg_kick_threshold * args.persistence_fraction).any()
            or (following_knees <= args.relative_knee_height_ceiling).any()
            or score >= args.leg_kick_threshold * 1.25
        )
        if persistent:
            delivery_start = number(row.get("frame_time_sec_num"))
            break

    if delivery_start is not None:
        status = "pose_leg_kick_detected"
    elif number(search["leg_kick_score"].max()) is not None:
        status = "pose_no_clear_leg_kick"
    else:
        status = "pose_missing_leg_geometry"
    group["pose_delivery_status"] = status
    group["pose_delivery_start_time_sec"] = delivery_start if delivery_start is not None else ""
    return group


def sequence_metrics(frame_rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame_rows = frame_rows.copy()
    frame_rows["frame_time_sec_num"] = pd.to_numeric(frame_rows["frame_time_sec"], errors="coerce")
    groups = [apply_leg_kick_metrics(group, args) for _, group in frame_rows.groupby("pitch_uid", sort=False)]
    if not groups:
        return frame_rows
    return pd.concat(groups, ignore_index=True)


def main() -> None:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    output_path = resolve(args.output)
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    manifest["pitch_uid"] = manifest["pitch_uid"].astype(str)
    if args.limit_pitches:
        pitch_ids = manifest["pitch_uid"].drop_duplicates().head(args.limit_pitches)
        manifest = manifest[manifest["pitch_uid"].isin(pitch_ids)].copy()

    existing = pd.DataFrame()
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        existing["frame_uid"] = existing["frame_uid"].astype(str)
    processed = set(existing.get("frame_uid", pd.Series(dtype=str)).astype(str))
    pending = manifest[~manifest["frame_uid"].astype(str).isin(processed)].copy()

    new_records = []
    if not pending.empty:
        mp, pose = load_pose_runtime(args.min_pose_visibility)
        try:
            for count, (_, row) in enumerate(pending.iterrows(), start=1):
                new_records.append(detect_frame(row, pose, mp, args))
                if args.progress_every and count % args.progress_every == 0:
                    print(f"Pose progress: {count}/{len(pending)}")
        finally:
            pose.close()

    raw_columns = [
        "frame_uid",
        "pitch_uid",
        "image_path",
        "frame_index",
        "frame_time_sec",
        "pose_status",
        "pose_visibility",
        "pose_center_x",
        "pose_center_y",
        "torso_length",
        *LANDMARK_COLUMNS,
    ]
    new_df = pd.DataFrame(new_records)
    combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("frame_uid", keep="last")
    combined = combined[combined["frame_uid"].astype(str).isin(manifest["frame_uid"].astype(str))]
    for column in raw_columns:
        if column not in combined.columns:
            combined[column] = ""
    output = sequence_metrics(combined[raw_columns], args)
    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output[OUTPUT_COLUMNS].fillna("").to_csv(output_path, index=False)
    print(f"Output: {output_path}")
    print(f"Rows: {len(output)}")
    print("Frame pose statuses:")
    print(output["pose_status"].value_counts(dropna=False).to_string())
    pitch_statuses = output.drop_duplicates("pitch_uid")["pose_delivery_status"].value_counts(dropna=False)
    print("Pitch delivery statuses:")
    print(pitch_statuses.to_string())


if __name__ == "__main__":
    main()

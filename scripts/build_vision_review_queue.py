import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANALYSIS_PREFIX = "labeled_target_analysis_sample_"
DEFAULT_PREDICTIONS = DATA_DIR / "roboflow_workflow_predictions.csv"
DEFAULT_QUALITY_OUTPUT = DATA_DIR / "roboflow_vision_quality.csv"
DEFAULT_REVIEW_OUTPUT = DATA_DIR / "roboflow_vision_review_queue.csv"
DEFAULT_REVIEW_LABELS = DATA_DIR / "vision_review_labels.csv"
EXCLUDED_REVIEW_LABELS = {"unusable_video"}
BAD_ZONE_REVIEW_LABELS = {"wrong_zone"}

METADATA_COLUMNS = [
    "pitch_uid",
    "sample_pitcher_name",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "pitch_name",
    "pitch_family",
    "stand",
    "balls",
    "strikes",
    "inning",
    "inning_topbot",
    "description",
    "events",
    "target_x_01",
    "target_y_01",
    "target_confidence_1_to_5",
    "setup_visible",
    "label_notes",
    "direct_mp4_url",
    "savant_video_url",
]

QUALITY_COLUMNS = [
    "frame_uid",
    "pitch_uid",
    "image_path",
    "sample_pitcher_name",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "pitch_name",
    "pitch_family",
    "stand",
    "balls",
    "strikes",
    "inning",
    "inning_topbot",
    "description",
    "events",
    "status",
    "vision_filter_status",
    "vision_trusted",
    "vision_excluded",
    "vision_exclusion_reason",
    "vision_filter_flags",
    "vision_in_review_queue",
    "vision_review_priority",
    "vision_review_reason",
    "vision_review_rank",
    "zone_confidence",
    "zone_refinement_status",
    "image_width",
    "image_height",
    "zone_edge_score",
    "zone_edge_min_support",
    "zone_refinement_frame_support",
    "zone_refinement_reference_frame_uid",
    "recovery_reason",
    "recovery_original_frame_uid",
    "recovery_frame_changed",
    "frame_view_status",
    "fox_logo_score",
    "fox_frame_support",
    "fox_candidate_count",
    "fox_temporal_error_px",
    "fox_score",
    "fox_reference_frames",
    "zone_baseline_source",
    "glove_confidence",
    "prediction_count",
    "zone_source",
    "zone_anchor_source",
    "zone_registration_source",
    "zone_anchor_pitch_support",
    "zone_anchor_frame_support",
    "zone_anchor_center_mad_px",
    "zone_calibration_reference_pitch_uid",
    "zone_calibration_reference_status",
    "glove_source_model",
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
    "glove_setup_evidence",
    "strong_stable_glove_hold",
    "glove_hold_start_time_sec",
    "glove_hold_end_time_sec",
    "glove_selection_mode",
    "glove_spatial_status",
    "glove_temporal_stability",
    "detected_glove_drop_time_sec",
    "dense_glove_used",
    "dense_glove_reason",
    "dense_glove_reason_category",
    "vision_fallback_used",
    "vision_fallback_reason",
    "statcast_sz_top",
    "statcast_sz_bot",
    "statcast_zone_height_ft",
    "zone_quality_status",
    "zone_quality_score",
    "zone_quality_flags",
    "zone_aspect_h_w",
    "zone_area",
    "zone_game_sample_size",
    "zone_center_game_delta",
    "zone_size_game_delta",
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
    "target_x_01",
    "target_y_01",
    "vision_manual_dx_01",
    "vision_manual_dy_01",
    "vision_manual_delta_01",
    "target_confidence_1_to_5",
    "setup_visible",
    "label_notes",
    "manual_review_label",
    "manual_review_notes",
    "manual_reviewed_at",
    "manual_review_frame_uid",
    "direct_mp4_url",
    "savant_video_url",
    "raw_response_path",
    "error",
]

REVIEW_COLUMNS = [
    "vision_review_rank",
    "vision_review_priority",
    "vision_review_reason",
    "frame_uid",
    "pitch_uid",
    "sample_pitcher_name",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "pitch_name",
    "pitch_family",
    "stand",
    "balls",
    "strikes",
    "status",
    "vision_filter_status",
    "vision_excluded",
    "vision_exclusion_reason",
    "vision_filter_flags",
    "zone_quality_status",
    "zone_quality_score",
    "zone_quality_flags",
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
    "prediction_count",
    "zone_source",
    "zone_anchor_source",
    "zone_registration_source",
    "zone_anchor_pitch_support",
    "zone_anchor_frame_support",
    "zone_anchor_center_mad_px",
    "zone_calibration_reference_pitch_uid",
    "zone_calibration_reference_status",
    "glove_source_model",
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
    "glove_setup_evidence",
    "strong_stable_glove_hold",
    "glove_hold_start_time_sec",
    "glove_hold_end_time_sec",
    "glove_selection_mode",
    "glove_spatial_status",
    "glove_temporal_stability",
    "detected_glove_drop_time_sec",
    "dense_glove_used",
    "dense_glove_reason",
    "dense_glove_reason_category",
    "vision_fallback_used",
    "vision_fallback_reason",
    "vision_target_x_01",
    "vision_target_y_01",
    "target_x_01",
    "target_y_01",
    "vision_manual_delta_01",
    "manual_review_label",
    "setup_visible",
    "label_notes",
    "image_path",
    "direct_mp4_url",
    "savant_video_url",
    "raw_response_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--analysis", type=Path, default=None)
    parser.add_argument("--quality-output", type=Path, default=DEFAULT_QUALITY_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--min-zone-confidence", type=float, default=0.60)
    parser.add_argument("--min-glove-confidence", type=float, default=0.50)
    parser.add_argument("--min-target-x", type=float, default=-1.50)
    parser.add_argument("--max-target-x", type=float, default=2.50)
    parser.add_argument("--min-target-y", type=float, default=-1.00)
    parser.add_argument("--max-target-y", type=float, default=2.00)
    parser.add_argument("--manual-outlier-threshold", type=float, default=0.75)
    parser.add_argument("--top-outliers", type=int, default=40)
    parser.add_argument("--review-labels", type=Path, default=DEFAULT_REVIEW_LABELS)
    parser.add_argument(
        "--review-label-mode",
        choices=["all", "exclusions-only", "none"],
        default="all",
        help="Use all manual review labels, only exclusion labels, or no review labels.",
    )
    parser.add_argument("--zone-min-width", type=float, default=50.0)
    parser.add_argument("--zone-max-width", type=float, default=95.0)
    parser.add_argument("--zone-min-height", type=float, default=65.0)
    parser.add_argument("--zone-max-height", type=float, default=125.0)
    parser.add_argument("--zone-min-aspect", type=float, default=1.05)
    parser.add_argument("--zone-max-aspect", type=float, default=1.70)
    parser.add_argument("--zone-min-game-sample-size", type=int, default=4)
    parser.add_argument("--max-zone-center-game-delta", type=float, default=45.0)
    parser.add_argument("--max-zone-size-game-delta", type=float, default=35.0)
    parser.add_argument("--min-glove-track-frames", type=int, default=3)
    parser.add_argument("--min-glove-track-stability", type=float, default=0.45)
    parser.add_argument("--min-glove-stable-run", type=int, default=2)
    parser.add_argument("--strong-hold-min-track-frames", type=int, default=3)
    parser.add_argument("--strong-hold-min-stability", type=float, default=0.80)
    parser.add_argument("--strong-hold-min-stable-frames", type=int, default=3)
    return parser.parse_args()


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


def latest_analysis_file() -> Optional[Path]:
    files = sorted(
        DATA_DIR.glob(f"{ANALYSIS_PREFIX}*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    return pd.read_csv(path)


def read_review_labels(path: Optional[Path], mode: str = "all") -> pd.DataFrame:
    if mode == "none" or path is None or not path.exists():
        return pd.DataFrame(columns=["pitch_uid"])
    labels = pd.read_csv(path)
    if labels.empty or "pitch_uid" not in labels.columns:
        return pd.DataFrame(columns=["pitch_uid"])
    labels = labels.drop_duplicates("pitch_uid", keep="last")
    rename_map = {
        "review_label": "manual_review_label",
        "review_notes": "manual_review_notes",
        "reviewed_at": "manual_reviewed_at",
        "frame_uid": "manual_review_frame_uid",
    }
    labels = labels.rename(columns=rename_map)
    keep_columns = [
        "pitch_uid",
        "manual_review_label",
        "manual_review_notes",
        "manual_reviewed_at",
        "manual_review_frame_uid",
    ]
    for column in keep_columns:
        if column not in labels.columns:
            labels[column] = ""
    if mode == "exclusions-only":
        labels = labels[labels["manual_review_label"].isin(EXCLUDED_REVIEW_LABELS)]
    return labels[keep_columns]


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce")


def add_zone_quality(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.copy()
    for column in ["zone_x", "zone_y", "zone_width", "zone_height"]:
        df[f"{column}_num"] = numeric(df, column)

    df["zone_aspect_h_w"] = df["zone_height_num"] / df["zone_width_num"]
    df["zone_area"] = df["zone_width_num"] * df["zone_height_num"]

    valid_zone = df["zone_x_num"].notna() & df["zone_y_num"].notna()
    grouped = df[valid_zone].groupby("game_pk", dropna=False)
    game_medians = grouped[["zone_x_num", "zone_y_num", "zone_width_num", "zone_height_num"]].transform("median")
    game_counts = grouped["zone_x_num"].transform("count")

    df["zone_game_sample_size"] = pd.NA
    df.loc[valid_zone, "zone_game_sample_size"] = game_counts
    for column in ["zone_x_num", "zone_y_num", "zone_width_num", "zone_height_num"]:
        df[f"{column}_game_median"] = pd.NA
        df.loc[valid_zone, f"{column}_game_median"] = game_medians[column]

    df["zone_center_game_delta"] = (
        (df["zone_x_num"] - df["zone_x_num_game_median"]).pow(2)
        + (df["zone_y_num"] - df["zone_y_num_game_median"]).pow(2)
    ).pow(0.5)
    df["zone_size_game_delta"] = (
        (df["zone_width_num"] - df["zone_width_num_game_median"]).pow(2)
        + (df["zone_height_num"] - df["zone_height_num_game_median"]).pow(2)
    ).pow(0.5)

    flag_lists = []
    scores = []
    statuses = []
    for _, row in df.iterrows():
        flags = []
        score = 100
        status = str(row.get("status", "") or "")
        review_label = str(row.get("manual_review_label", "") or "")

        zone_confidence = row.get("zone_confidence_num")
        zone_x = row.get("zone_x_num")
        zone_y = row.get("zone_y_num")
        zone_width = row.get("zone_width_num")
        zone_height = row.get("zone_height_num")
        zone_aspect = row.get("zone_aspect_h_w")
        zone_game_sample_size = row.get("zone_game_sample_size")
        zone_center_delta = row.get("zone_center_game_delta")
        zone_size_delta = row.get("zone_size_game_delta")
        explicit_manual_zone = (
            str(row.get("zone_calibration_reference_status", "") or "")
            == "explicit_pitch_reference"
        )
        statcast_adjusted_template = str(row.get("zone_anchor_source", "") or "") in {
            "broadcast_game_template_statcast",
            "broadcast_plate_appearance_template",
            "broadcast_pitch_edge_template",
            "broadcast_batter_horizontal_template",
        }
        if pd.isna(zone_x) or pd.isna(zone_y) or pd.isna(zone_width) or pd.isna(zone_height):
            flags.append("missing_zone_detection")
            score = 0
        else:
            if pd.isna(zone_confidence) or zone_confidence < args.min_zone_confidence:
                flags.append("low_zone_confidence")
                score -= 35
            elif zone_confidence < 0.80:
                flags.append("medium_zone_confidence")
                score -= 12

            pixel_aligned = str(row.get("zone_refinement_status", "")) in {
                "visible_four_edge_fit", "registered_temporal_fit",
                "fox_temporal_alpha_fit", "recovered_four_edge_frame",
                "fox_at_bat_consensus_fit",
            }
            if not explicit_manual_zone and not pixel_aligned:
                if zone_width < args.zone_min_width:
                    flags.append("zone_too_narrow")
                    score -= 25
                if zone_width > args.zone_max_width:
                    flags.append("zone_too_wide")
                    score -= 25
                if not statcast_adjusted_template and zone_height < args.zone_min_height:
                    flags.append("zone_too_short")
                    score -= 25
                if not statcast_adjusted_template and zone_height > args.zone_max_height:
                    flags.append("zone_too_tall")
                    score -= 25
                if pd.isna(zone_aspect) or zone_aspect < args.zone_min_aspect:
                    flags.append("zone_bad_aspect_low")
                    score -= 20
                if not pd.isna(zone_aspect) and zone_aspect > args.zone_max_aspect:
                    flags.append("zone_bad_aspect_high")
                    score -= 20

                if (
                    not pd.isna(zone_game_sample_size)
                    and zone_game_sample_size >= args.zone_min_game_sample_size
                ):
                    if not pd.isna(zone_center_delta) and zone_center_delta > args.max_zone_center_game_delta:
                        flags.append("zone_center_game_outlier")
                        score -= 25
                    if not pd.isna(zone_size_delta) and zone_size_delta > args.max_zone_size_game_delta:
                        flags.append("zone_size_game_outlier")
                        score -= 25

        refinement_status = str(row.get("zone_refinement_status", ""))
        if refinement_status in {"unverified_fallback", "missing_image", "no_zone_evidence", "recovery_rejected"}:
            flags.append("zone_geometry_unverified")
            score = min(score, 55)

        if review_label in BAD_ZONE_REVIEW_LABELS:
            flags.append("review_wrong_zone")
            score = min(score, 20)

        if "review_wrong_zone" in flags:
            zone_status = "bad_reviewed"
        elif "missing_zone_detection" in flags:
            zone_status = "missing"
        elif score < 60 or any(flag.startswith("zone_") for flag in flags):
            zone_status = "poor"
        elif flags:
            zone_status = "warn"
        else:
            zone_status = "ok"

        flag_lists.append(list(dict.fromkeys(flags)))
        scores.append(max(0, min(100, round(score, 1))))
        statuses.append(zone_status)

    df["zone_quality_flags_list"] = flag_lists
    df["zone_quality_flags"] = df["zone_quality_flags_list"].map(lambda flags: ";".join(flags))
    df["zone_quality_score"] = scores
    df["zone_quality_status"] = statuses
    return df


def quality_flags(row: pd.Series, args: argparse.Namespace) -> list[str]:
    flags = []
    status = str(row.get("status", "") or "")
    if status != "ok":
        flags.append(status or "missing_prediction")
        return flags

    zone_confidence = row.get("zone_confidence_num")
    glove_confidence = row.get("glove_confidence_num")
    target_x = row.get("vision_target_x_num")
    target_y = row.get("vision_target_y_num")
    track_frames = row.get("glove_track_frame_count_num")
    track_stability = row.get("glove_track_stability_num")
    stable_run = row.get("glove_stable_run_count_num")
    drop_confirmed = str(row.get("glove_drop_confirmed", "") or "").lower()

    for flag in row.get("zone_quality_flags_list", []) or []:
        flags.append(flag)
    if pd.isna(glove_confidence) or glove_confidence < args.min_glove_confidence:
        flags.append("low_glove_confidence")
    if not pd.isna(track_frames) and track_frames < args.min_glove_track_frames:
        flags.append("short_glove_track")
    if not pd.isna(track_stability) and track_stability < args.min_glove_track_stability:
        flags.append("unstable_glove_track")
    if not pd.isna(stable_run) and stable_run < args.min_glove_stable_run:
        flags.append("no_stable_glove_plateau")
    strong_hold = str(row.get("strong_stable_glove_hold", "") or "").lower() == "yes"
    if drop_confirmed == "no" and not strong_hold:
        flags.append("no_confirmed_setup_evidence")
    if pd.isna(target_x) or pd.isna(target_y):
        flags.append("missing_vision_target")
    elif (
        target_x < args.min_target_x
        or target_x > args.max_target_x
        or target_y < args.min_target_y
        or target_y > args.max_target_y
    ):
        flags.append("implausible_vision_target")

    return flags


def review_priority(row: pd.Series) -> str:
    reason = str(row.get("vision_review_reason", "") or "")
    status = str(row.get("status", "") or "")
    delta = row.get("vision_manual_delta_01_num")
    if (
        status != "ok"
        or "missing_zone_detection" in reason
        or "implausible_vision_target" in reason
        or "review_wrong_zone" in reason
    ):
        return "high"
    if not pd.isna(delta) and delta >= 1.0:
        return "high"
    if (
        "low_" in reason
        or "manual_outlier" in reason
        or "zone_center_game_outlier" in reason
        or "zone_size_game_outlier" in reason
        or "zone_bad_aspect" in reason
        or "zone_too_" in reason
        or "short_glove_track" in reason
        or "unstable_glove_track" in reason
        or "no_stable_glove_plateau" in reason
        or "no_confirmed_setup_evidence" in reason
    ):
        return "medium"
    if reason:
        return "low"
    return ""


def priority_sort_value(value: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(value), 9)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df[columns]


def build_quality(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_path = resolve_path(args.predictions)
    analysis_path = resolve_path(args.analysis) or latest_analysis_file()
    review_labels_path = resolve_path(args.review_labels)
    quality_output = resolve_path(args.quality_output)
    review_output = resolve_path(args.review_output)

    if analysis_path is None:
        raise SystemExit("No labeled target analysis file found.")

    predictions = read_csv(predictions_path)
    analysis = read_csv(analysis_path)
    review_labels = read_review_labels(review_labels_path, args.review_label_mode)
    metadata_columns = [column for column in METADATA_COLUMNS if column in analysis.columns]
    merged = predictions.merge(analysis[metadata_columns], on="pitch_uid", how="left")
    merged = merged.merge(review_labels, on="pitch_uid", how="left")
    for column in [
        "manual_review_label",
        "manual_review_notes",
        "manual_reviewed_at",
        "manual_review_frame_uid",
    ]:
        if column not in merged.columns:
            merged[column] = ""
        merged[column] = merged[column].fillna("")

    merged["zone_confidence_num"] = numeric(merged, "zone_confidence")
    merged["glove_confidence_num"] = numeric(merged, "glove_confidence")
    merged["vision_target_x_num"] = numeric(merged, "vision_target_x_01")
    merged["vision_target_y_num"] = numeric(merged, "vision_target_y_01")
    merged["glove_track_frame_count_num"] = numeric(merged, "glove_track_frame_count")
    merged["glove_track_stability_num"] = numeric(merged, "glove_track_stability")
    merged["glove_stable_run_count_num"] = numeric(merged, "glove_stable_run_count")
    merged["manual_target_x_num"] = numeric(merged, "target_x_01")
    merged["manual_target_y_num"] = numeric(merged, "target_y_01")

    merged["vision_manual_dx_01"] = merged["vision_target_x_num"] - merged["manual_target_x_num"]
    merged["vision_manual_dy_01"] = merged["vision_target_y_num"] - merged["manual_target_y_num"]
    merged["vision_manual_delta_01"] = (
        merged["vision_manual_dx_01"].pow(2) + merged["vision_manual_dy_01"].pow(2)
    ).pow(0.5)
    merged["vision_manual_delta_01_num"] = merged["vision_manual_delta_01"]

    drop_confirmed = merged["glove_drop_confirmed"].fillna("").astype(str).str.lower().eq("yes")
    stable_hold = (
        merged["glove_track_frame_count_num"].ge(args.strong_hold_min_track_frames)
        & merged["glove_track_stability_num"].ge(args.strong_hold_min_stability)
        & merged["glove_stable_run_count_num"].ge(args.strong_hold_min_stable_frames)
        & merged["glove_confidence_num"].ge(args.min_glove_confidence)
    )
    merged["strong_stable_glove_hold"] = stable_hold.map(lambda value: "yes" if value else "no")
    merged["glove_setup_evidence"] = "insufficient"
    merged.loc[stable_hold, "glove_setup_evidence"] = "stable_hold"
    merged.loc[drop_confirmed, "glove_setup_evidence"] = "confirmed_drop"

    merged = add_zone_quality(merged, args)

    flag_lists = merged.apply(lambda row: quality_flags(row, args), axis=1)
    merged["vision_filter_flags"] = flag_lists.map(lambda flags: ";".join(flags))
    merged["vision_filter_status"] = flag_lists.map(lambda flags: "needs_review" if flags else "trusted")
    merged["vision_trusted"] = flag_lists.map(lambda flags: "no" if flags else "yes")

    bad_zone_reviewed = merged["manual_review_label"].isin(BAD_ZONE_REVIEW_LABELS)
    merged.loc[bad_zone_reviewed, "vision_filter_status"] = "bad_zone"
    merged.loc[bad_zone_reviewed, "vision_trusted"] = "no"

    excluded = merged["manual_review_label"].isin(EXCLUDED_REVIEW_LABELS)
    merged["vision_excluded"] = excluded.map(lambda value: "yes" if value else "no")
    merged["vision_exclusion_reason"] = ""
    merged.loc[excluded, "vision_exclusion_reason"] = "manual_unusable_video"
    merged.loc[excluded, "vision_filter_status"] = "excluded"
    merged.loc[excluded, "vision_trusted"] = "no"

    review_reasons = flag_lists.map(list)
    manual_delta = merged["vision_manual_delta_01_num"]
    manual_outlier = manual_delta.notna() & (manual_delta >= args.manual_outlier_threshold)
    for index in merged.index[manual_outlier]:
        review_reasons.at[index].append("manual_outlier")

    top_outliers = set()
    if args.top_outliers > 0:
        top_outliers = set(manual_delta.dropna().sort_values(ascending=False).head(args.top_outliers).index)
        for index in top_outliers:
            if "top_manual_outlier" not in review_reasons.at[index]:
                review_reasons.at[index].append("top_manual_outlier")

    merged["vision_review_reason"] = review_reasons.map(lambda flags: ";".join(dict.fromkeys(flags)))
    merged["vision_in_review_queue"] = merged["vision_review_reason"].map(lambda value: "yes" if value else "no")
    merged.loc[excluded, "vision_review_reason"] = "excluded_unusable_video"
    merged.loc[excluded, "vision_in_review_queue"] = "no"
    reviewed = merged["manual_review_label"].astype(str).str.len().gt(0)
    merged.loc[reviewed, "vision_in_review_queue"] = "no"
    merged["vision_review_priority"] = merged.apply(review_priority, axis=1)
    merged.loc[reviewed, "vision_review_priority"] = ""

    review_mask = merged["vision_in_review_queue"].eq("yes") & merged["vision_excluded"].ne("yes")
    review = merged[review_mask].copy()
    review["_priority_sort"] = review["vision_review_priority"].map(priority_sort_value)
    review["_delta_sort"] = pd.to_numeric(review["vision_manual_delta_01"], errors="coerce").fillna(-1)
    review = review.sort_values(
        ["_priority_sort", "_delta_sort", "sample_pitcher_name", "pitch_uid"],
        ascending=[True, False, True, True],
    ).drop(columns=["_priority_sort", "_delta_sort"])
    review["vision_review_rank"] = range(1, len(review) + 1)

    rank_by_pitch = review.set_index("pitch_uid")["vision_review_rank"].to_dict()
    merged["vision_review_rank"] = merged["pitch_uid"].map(rank_by_pitch).fillna("")

    quality = ensure_columns(merged.copy(), QUALITY_COLUMNS)
    review = ensure_columns(review.copy(), REVIEW_COLUMNS)

    quality_output.parent.mkdir(parents=True, exist_ok=True)
    review_output.parent.mkdir(parents=True, exist_ok=True)
    quality.fillna("").to_csv(quality_output, index=False)
    review.fillna("").to_csv(review_output, index=False)

    return quality, review


def main() -> None:
    args = parse_args()
    quality, review = build_quality(args)

    print(f"Quality output: {resolve_path(args.quality_output)}")
    print(f"Review queue: {resolve_path(args.review_output)}")
    print(f"Rows: {len(quality)}")
    print(f"Trusted after quality filters: {(quality['vision_trusted'] == 'yes').sum()}")
    print(f"Excluded unusable videos: {(quality['vision_excluded'] == 'yes').sum()}")
    print(f"Not trusted after quality filters: {(quality['vision_trusted'] == 'no').sum()}")
    print(f"Active review queue rows: {len(review)}")
    print("Vision statuses:")
    print(quality["vision_filter_status"].value_counts().to_string())
    print("Zone quality statuses:")
    print(quality["zone_quality_status"].value_counts().to_string())
    if len(review):
        print("Review priorities:")
        print(review["vision_review_priority"].value_counts().to_string())
        print("Top review reasons:")
        print(review["vision_review_reason"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()

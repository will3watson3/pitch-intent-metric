from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
PLATE_EDGE_FT = 17 / 12 / 2
MIN_MOVEMENT_OFFSET_GROUP_SIZE = 4
FULL_CONFIDENCE_GROUP_SIZE = 20
FASTBALL_PITCHES = {"FF", "SI", "FC", "FA"}
BREAKING_PITCHES = {"SL", "ST", "CU", "KC", "SV", "CS"}
OFFSPEED_PITCHES = {"CH", "FS", "FO", "SC"}

PITCHER_NAME_BY_SLUG = {
    "cease": "Dylan Cease",
    "fried": "Max Fried",
    "misiorowski": "Jacob Misiorowski",
    "ohtani": "Shohei Ohtani",
    "skubal": "Tarik Skubal",
    "webb": "Logan Webb",
    "wheeler": "Zack Wheeler",
    "woo": "Bryan Woo",
}

# Coordinates are in pitcher POV, normalized to the strike-zone rectangle:
# x: 0 = left edge, 0.5 = middle, 1 = right edge
# y: 0 = bottom of zone, 0.5 = middle, 1 = top of zone
# Values outside [0, 1] are allowed for chase targets.
TARGET_CENTERS = {
    "low_in": (0.20, 0.25),
    "low_middle": (0.50, 0.25),
    "low_away": (0.80, 0.25),
    "middle_in": (0.20, 0.50),
    "middle_middle": (0.50, 0.50),
    "middle_away": (0.80, 0.50),
    "up_in": (0.20, 0.80),
    "up_middle": (0.50, 0.80),
    "up_away": (0.80, 0.80),
    "chase_away": (1.20, 0.50),
    "chase_up_away": (1.20, 1.15),
    "chase_middle_away": (1.20, 0.50),
    "chase_up_middle": (0.50, 1.15),
}


def pitch_family(pitch_type) -> str:
    pitch = str(pitch_type or "").upper()
    if pitch in FASTBALL_PITCHES:
        return "fastball"
    if pitch in BREAKING_PITCHES:
        return "breaking"
    if pitch in OFFSPEED_PITCHES:
        return "offspeed"
    return "other"


def is_away_setup(row) -> bool:
    stand = str(row.get("stand") or "").upper()
    target_x = row.get("target_x")
    if pd.isna(target_x):
        return False
    if stand == "R":
        return target_x <= 0.15
    if stand == "L":
        return target_x >= 0.85
    return target_x <= 0.05 or target_x >= 0.95


def is_far_outside(row) -> bool:
    target_x = row.get("target_x")
    if pd.isna(target_x):
        return False
    return target_x <= -0.15 or target_x >= 1.15


def infer_intent_label(row) -> str:
    balls = pd.to_numeric(row.get("balls"), errors="coerce")
    strikes = pd.to_numeric(row.get("strikes"), errors="coerce")
    target_x = row.get("target_x")
    target_y = row.get("target_y")
    family = row.get("pitch_family")
    two_strikes = pd.notna(strikes) and strikes >= 2
    ahead = pd.notna(strikes) and pd.notna(balls) and strikes > balls

    if pd.isna(target_x) or pd.isna(target_y):
        return "unknown"
    if two_strikes and (target_y > 1.18 or target_y < -0.12 or target_x < -0.35 or target_x > 1.35):
        return "waste"
    if target_y >= 1.02 and (two_strikes or family == "fastball"):
        return "chase_up"
    if family in {"breaking", "offspeed"} and is_away_setup(row) and (two_strikes or ahead or is_far_outside(row)):
        return "chase_away"
    if family == "fastball" and -0.05 <= target_x <= 1.05 and -0.05 <= target_y <= 1.05:
        return "zone_attack"
    if family == "breaking" and is_far_outside(row):
        return "frontdoor_backdoor"
    if target_y < 0:
        return "chase_down"
    if 0 <= target_x <= 1 and 0 <= target_y <= 1:
        return "zone_attack"
    return "unknown"


def clipped_score(series: pd.Series, lower: float, upper: float, invert: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if upper == lower:
        scaled = values * 0
    else:
        scaled = (values - lower) / (upper - lower)
    scaled = scaled.clip(0, 1)
    return 1 - scaled if invert else scaled


def visibility_score(value) -> float:
    text = str(value or "").strip().lower()
    if text == "yes":
        return 1.0
    if text == "partial":
        return 0.6
    if text == "no":
        return 0.2
    return 0.45


def confidence_tier(score) -> str:
    if pd.isna(score):
        return "unavailable"
    if score >= 75:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def add_movement_intent_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_cols = ["sample_pitcher_name", "pitch_type"]
    grouped = df.groupby(group_cols, dropna=False)

    df["intent_model_group"] = (
        df["sample_pitcher_name"].astype(str) + " | " + df["pitch_type"].astype(str)
    )
    df["intent_model_group_size"] = grouped["pitch_uid"].transform("count")
    df["intent_model_training_size"] = df["intent_model_group_size"] - 1
    df["group_offset_x_std"] = grouped["miss_x"].transform("std").fillna(0)
    df["group_offset_y_std"] = grouped["miss_y"].transform("std").fillna(0)
    df["group_offset_x_ft_std"] = grouped["miss_x_ft"].transform("std").fillna(0)
    df["group_offset_z_ft_std"] = grouped["miss_z_ft"].transform("std").fillna(0)
    df["group_offset_spread"] = (
        (df["group_offset_x_std"] ** 2 + df["group_offset_y_std"] ** 2) ** 0.5
    )
    df["group_offset_spread_ft"] = (
        (df["group_offset_x_ft_std"] ** 2 + df["group_offset_z_ft_std"] ** 2) ** 0.5
    )

    offset_columns = {
        "miss_x": "movement_offset_x",
        "miss_y": "movement_offset_y",
        "miss_x_ft": "movement_offset_x_ft",
        "miss_z_ft": "movement_offset_z_ft",
    }
    training_size = df["intent_model_training_size"].where(df["intent_model_training_size"] > 0)

    for source_column, offset_column in offset_columns.items():
        leave_one_out_sum = grouped[source_column].transform("sum") - df[source_column]
        df[f"loo_{offset_column}"] = leave_one_out_sum / training_size
        df[offset_column] = pd.NA

    is_fastball = df["pitch_family"] == "fastball"
    can_use_movement_model = (
        df["pitch_family"].isin(["breaking", "offspeed"])
        & (df["intent_model_group_size"] >= MIN_MOVEMENT_OFFSET_GROUP_SIZE)
    )
    has_intent_model = is_fastball | can_use_movement_model

    for offset_column in offset_columns.values():
        df.loc[is_fastball, offset_column] = 0.0
        df.loc[can_use_movement_model, offset_column] = df.loc[
            can_use_movement_model, f"loo_{offset_column}"
        ]
        df[offset_column] = pd.to_numeric(df[offset_column], errors="coerce")

    df["inferred_intent_x"] = pd.NA
    df["inferred_intent_y"] = pd.NA
    df["inferred_intent_plate_x_ft"] = pd.NA
    df["inferred_intent_plate_z_ft"] = pd.NA

    df.loc[has_intent_model, "inferred_intent_x"] = (
        df.loc[has_intent_model, "target_x"] + df.loc[has_intent_model, "movement_offset_x"]
    )
    df.loc[has_intent_model, "inferred_intent_y"] = (
        df.loc[has_intent_model, "target_y"] + df.loc[has_intent_model, "movement_offset_y"]
    )
    df.loc[has_intent_model, "inferred_intent_plate_x_ft"] = (
        df.loc[has_intent_model, "target_plate_x_ft"]
        + df.loc[has_intent_model, "movement_offset_x_ft"]
    )
    df.loc[has_intent_model, "inferred_intent_plate_z_ft"] = (
        df.loc[has_intent_model, "target_plate_z_ft"]
        + df.loc[has_intent_model, "movement_offset_z_ft"]
    )

    df["intent_adjusted_miss_x"] = df["actual_x_01"] - pd.to_numeric(
        df["inferred_intent_x"], errors="coerce"
    )
    df["intent_adjusted_miss_y"] = df["actual_y_01"] - pd.to_numeric(
        df["inferred_intent_y"], errors="coerce"
    )
    df["intent_adjusted_miss"] = (
        (df["intent_adjusted_miss_x"] ** 2 + df["intent_adjusted_miss_y"] ** 2) ** 0.5
    )
    df["intent_adjusted_miss_x_ft"] = df["plate_x"] - pd.to_numeric(
        df["inferred_intent_plate_x_ft"], errors="coerce"
    )
    df["intent_adjusted_miss_z_ft"] = df["plate_z"] - pd.to_numeric(
        df["inferred_intent_plate_z_ft"], errors="coerce"
    )
    df["intent_adjusted_miss_ft"] = (
        (df["intent_adjusted_miss_x_ft"] ** 2 + df["intent_adjusted_miss_z_ft"] ** 2)
        ** 0.5
    )

    df["intent_model_status"] = "insufficient_pitcher_pitch_type_samples"
    df.loc[is_fastball, "intent_model_status"] = "ready_fastball_setup_equals_intent"
    df.loc[can_use_movement_model, "intent_model_status"] = (
        "ready_leave_one_out_movement_offset"
    )
    df["intent_model_kind"] = "none"
    df.loc[is_fastball, "intent_model_kind"] = "setup_equals_intent"
    df.loc[can_use_movement_model, "intent_model_kind"] = "pitcher_pitch_type_loo_offset"

    sample_score = clipped_score(df["intent_model_training_size"], 2, FULL_CONFIDENCE_GROUP_SIZE)
    consistency_score = clipped_score(df["group_offset_spread_ft"], 0.45, 2.25, invert=True)
    visible_score = df["setup_visible"].map(visibility_score)
    df["intent_model_confidence"] = (
        100 * ((0.45 * sample_score) + (0.35 * consistency_score) + (0.20 * visible_score))
    ).round(1)
    df.loc[~has_intent_model, "intent_model_confidence"] = pd.NA
    df["intent_model_confidence_tier"] = df["intent_model_confidence"].map(confidence_tier)
    df["intent_model_warning"] = ""
    df.loc[df["intent_model_group_size"] < MIN_MOVEMENT_OFFSET_GROUP_SIZE, "intent_model_warning"] = (
        "insufficient pitcher-pitch samples"
    )
    df.loc[
        has_intent_model & (df["group_offset_spread_ft"] >= 1.75),
        "intent_model_warning",
    ] = "high offset spread"
    df.loc[
        has_intent_model & (df["setup_visible"].astype(str).str.lower() != "yes"),
        "intent_model_warning",
    ] = df.loc[
        has_intent_model & (df["setup_visible"].astype(str).str.lower() != "yes"),
        "intent_model_warning",
    ].map(lambda warning: f"{warning}; limited setup visibility".strip("; "))

    return df.drop(columns=[column for column in df.columns if column.startswith("loo_")])


def pitcher_name_from_path(path: Path) -> str:
    slug = path.name.split("_labeling_queue", 1)[0]
    name_key = slug.replace("_2025", "")
    return PITCHER_NAME_BY_SLUG.get(name_key, slug.replace("_", " ").title())


def discover_labeled_files() -> list[tuple[str, Path]]:
    files = sorted(DATA_DIR.glob("*_2025_labeling_queue*.csv"))
    if not files:
        raise FileNotFoundError(f"No labeling queue files found in {DATA_DIR}")
    return [(pitcher_name_from_path(path), path) for path in files]


def load_labeled_data() -> pd.DataFrame:
    frames = []
    for pitcher_name, path in discover_labeled_files():
        df = pd.read_csv(path)
        df["sample_pitcher_name"] = pitcher_name
        df["source_file"] = path.name
        for column in [
            "target_x_01",
            "target_y_01",
            "catcher_target_bucket",
            "target_confidence_1_to_5",
            "setup_visible",
            "label_notes",
        ]:
            if column not in df.columns:
                df[column] = pd.NA

        has_continuous_target = (
            pd.to_numeric(df["target_x_01"], errors="coerce").notna()
            & pd.to_numeric(df["target_y_01"], errors="coerce").notna()
        )
        has_bucket_target = df["catcher_target_bucket"].notna()
        df = df[has_continuous_target | has_bucket_target].copy()
        frames.append(df)

    if not frames:
        raise ValueError("No labeled rows found.")
    labeled = pd.concat(frames, ignore_index=True)
    return labeled


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Baseball Savant plate_x is from the catcher's perspective. The target
    # coordinates are from pitcher POV, so invert plate_x before normalizing.
    pitcher_pov_plate_x = -df["plate_x"]
    df["actual_x_01"] = (pitcher_pov_plate_x + PLATE_EDGE_FT) / (2 * PLATE_EDGE_FT)
    df["actual_y_01"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])

    bucket_target_x = df["catcher_target_bucket"].map(
        lambda bucket: TARGET_CENTERS.get(bucket, (pd.NA, pd.NA))[0]
    )
    bucket_target_y = df["catcher_target_bucket"].map(
        lambda bucket: TARGET_CENTERS.get(bucket, (pd.NA, pd.NA))[1]
    )
    continuous_target_x = pd.to_numeric(df["target_x_01"], errors="coerce")
    continuous_target_y = pd.to_numeric(df["target_y_01"], errors="coerce")
    has_continuous_target = continuous_target_x.notna() & continuous_target_y.notna()

    df["target_x"] = continuous_target_x.where(has_continuous_target, bucket_target_x)
    df["target_y"] = continuous_target_y.where(has_continuous_target, bucket_target_y)
    df["target_source"] = has_continuous_target.map(
        {True: "continuous_xy", False: "bucket_fallback"}
    )

    missing_targets = sorted(
        set(df.loc[df["target_x"].isna(), "catcher_target_bucket"].dropna())
    )
    if missing_targets:
        raise ValueError(f"Missing target center mapping for: {missing_targets}")

    df["target_x"] = df["target_x"].astype(float)
    df["target_y"] = df["target_y"].astype(float)
    df["target_plate_x_ft"] = PLATE_EDGE_FT * (1 - (2 * df["target_x"]))
    df["target_plate_z_ft"] = df["sz_bot"] + (df["target_y"] * (df["sz_top"] - df["sz_bot"]))

    df["miss_x"] = df["actual_x_01"] - df["target_x"]
    df["miss_y"] = df["actual_y_01"] - df["target_y"]
    df["miss_distance"] = (df["miss_x"] ** 2 + df["miss_y"] ** 2) ** 0.5
    df["miss_x_ft"] = df["plate_x"] - df["target_plate_x_ft"]
    df["miss_z_ft"] = df["plate_z"] - df["target_plate_z_ft"]
    df["miss_distance_ft"] = (df["miss_x_ft"] ** 2 + df["miss_z_ft"] ** 2) ** 0.5
    df["pitch_family"] = df["pitch_type"].map(pitch_family)
    df["setup_miss"] = df["miss_distance"]
    df["setup_miss_ft"] = df["miss_distance_ft"]
    df["intent_label"] = df.apply(infer_intent_label, axis=1)
    df = add_movement_intent_model(df)
    df["intent_adjustment_delta_ft"] = df["setup_miss_ft"] - df["intent_adjusted_miss_ft"]
    df["intent_adjustment_effect"] = df["intent_adjustment_delta_ft"].map(
        lambda value: "improved"
        if pd.notna(value) and value > 0.15
        else "worse"
        if pd.notna(value) and value < -0.15
        else "neutral"
        if pd.notna(value)
        else "unavailable"
    )
    df["hit_side"] = df["miss_x"].map(
        lambda value: "more_right" if value > 0.20 else "more_left" if value < -0.20 else "near_x"
    )
    df["hit_height"] = df["miss_y"].map(
        lambda value: "higher" if value > 0.20 else "lower" if value < -0.20 else "near_y"
    )
    return df


def write_summary_tables(df: pd.DataFrame) -> None:
    combined_out = DATA_DIR / f"labeled_target_analysis_sample_{len(df)}.csv"
    df.to_csv(combined_out, index=False)

    summary = (
        df.groupby("sample_pitcher_name")
        .agg(
            pitches=("pitch_uid", "count"),
            avg_confidence=("target_confidence_1_to_5", "mean"),
            visible_rate=("setup_visible", lambda values: (values == "yes").mean()),
            avg_miss_distance=("miss_distance", "mean"),
            median_miss_distance=("miss_distance", "median"),
            avg_miss_distance_ft=("miss_distance_ft", "mean"),
            median_miss_distance_ft=("miss_distance_ft", "median"),
            avg_setup_miss_ft=("setup_miss_ft", "mean"),
            median_setup_miss_ft=("setup_miss_ft", "median"),
            avg_intent_adjusted_miss_ft=("intent_adjusted_miss_ft", "mean"),
            median_intent_adjusted_miss_ft=("intent_adjusted_miss_ft", "median"),
            intent_model_ready=("intent_adjusted_miss_ft", lambda values: values.notna().sum()),
            pending_intent_model=("intent_adjusted_miss_ft", lambda values: values.isna().sum()),
            avg_model_confidence=("intent_model_confidence", "mean"),
            low_confidence_models=("intent_model_confidence_tier", lambda values: (values == "low").sum()),
            avg_abs_x_miss=("miss_x", lambda values: values.abs().mean()),
            avg_abs_y_miss=("miss_y", lambda values: values.abs().mean()),
            avg_abs_x_miss_ft=("miss_x_ft", lambda values: values.abs().mean()),
            avg_abs_z_miss_ft=("miss_z_ft", lambda values: values.abs().mean()),
            continuous_targets=("target_source", lambda values: (values == "continuous_xy").sum()),
            bucket_fallback_targets=("target_source", lambda values: (values == "bucket_fallback").sum()),
        )
        .round(3)
        .reset_index()
    )
    summary.to_csv(DATA_DIR / "labeled_target_summary_by_pitcher.csv", index=False)

    bucket_summary = (
        df[df["catcher_target_bucket"].notna()]
        .groupby(["sample_pitcher_name", "catcher_target_bucket"])
        .agg(
            pitches=("pitch_uid", "count"),
            avg_miss_distance=("miss_distance", "mean"),
            avg_miss_distance_ft=("miss_distance_ft", "mean"),
            avg_miss_x=("miss_x", "mean"),
            avg_miss_y=("miss_y", "mean"),
            avg_miss_x_ft=("miss_x_ft", "mean"),
            avg_miss_z_ft=("miss_z_ft", "mean"),
            continuous_targets=("target_source", lambda values: (values == "continuous_xy").sum()),
        )
        .round(3)
        .reset_index()
        .sort_values(["sample_pitcher_name", "pitches"], ascending=[True, False])
    )
    bucket_summary.to_csv(DATA_DIR / "labeled_target_summary_by_bucket.csv", index=False)

    pitch_type_summary = (
        df.groupby(["sample_pitcher_name", "pitch_family", "pitch_type"], dropna=False)
        .agg(
            pitches=("pitch_uid", "count"),
            avg_miss_distance=("miss_distance", "mean"),
            median_miss_distance=("miss_distance", "median"),
            avg_miss_distance_ft=("miss_distance_ft", "mean"),
            median_miss_distance_ft=("miss_distance_ft", "median"),
            avg_setup_miss_ft=("setup_miss_ft", "mean"),
            median_setup_miss_ft=("setup_miss_ft", "median"),
            avg_intent_adjusted_miss_ft=("intent_adjusted_miss_ft", "mean"),
            median_intent_adjusted_miss_ft=("intent_adjusted_miss_ft", "median"),
            intent_model_ready=("intent_adjusted_miss_ft", lambda values: values.notna().sum()),
            pending_intent_model=("intent_adjusted_miss_ft", lambda values: values.isna().sum()),
            avg_model_confidence=("intent_model_confidence", "mean"),
            low_confidence_models=("intent_model_confidence_tier", lambda values: (values == "low").sum()),
            avg_movement_offset_x=("movement_offset_x", "mean"),
            avg_movement_offset_y=("movement_offset_y", "mean"),
            avg_movement_offset_x_ft=("movement_offset_x_ft", "mean"),
            avg_movement_offset_z_ft=("movement_offset_z_ft", "mean"),
            avg_abs_x_miss=("miss_x", lambda values: values.abs().mean()),
            avg_abs_y_miss=("miss_y", lambda values: values.abs().mean()),
            avg_abs_x_miss_ft=("miss_x_ft", lambda values: values.abs().mean()),
            avg_abs_z_miss_ft=("miss_z_ft", lambda values: values.abs().mean()),
        )
        .round(3)
        .reset_index()
        .sort_values(["sample_pitcher_name", "pitches"], ascending=[True, False])
    )
    pitch_type_summary.to_csv(DATA_DIR / "labeled_target_summary_by_pitch_type.csv", index=False)

    diagnostics = (
        df.groupby(["sample_pitcher_name", "pitch_type", "pitch_family"], dropna=False)
        .agg(
            pitches=("pitch_uid", "count"),
            model_ready=("intent_adjusted_miss_ft", lambda values: values.notna().sum()),
            avg_setup_miss_ft=("setup_miss_ft", "mean"),
            avg_intent_adjusted_miss_ft=("intent_adjusted_miss_ft", "mean"),
            avg_improvement_ft=("intent_adjustment_delta_ft", "mean"),
            improved=("intent_adjustment_effect", lambda values: (values == "improved").sum()),
            worsened=("intent_adjustment_effect", lambda values: (values == "worse").sum()),
            neutral=("intent_adjustment_effect", lambda values: (values == "neutral").sum()),
            avg_movement_offset_x=("movement_offset_x", "mean"),
            avg_movement_offset_y=("movement_offset_y", "mean"),
            avg_movement_offset_x_ft=("movement_offset_x_ft", "mean"),
            avg_movement_offset_z_ft=("movement_offset_z_ft", "mean"),
            offset_spread_ft=("group_offset_spread_ft", "mean"),
            avg_model_confidence=("intent_model_confidence", "mean"),
            low_confidence_models=("intent_model_confidence_tier", lambda values: (values == "low").sum()),
            high_confidence_models=("intent_model_confidence_tier", lambda values: (values == "high").sum()),
        )
        .round(3)
        .reset_index()
    )
    diagnostics = diagnostics.sort_values(
        ["avg_model_confidence", "model_ready", "pitches"], ascending=[True, False, False]
    )
    diagnostics.to_csv(DATA_DIR / "intent_model_diagnostics_by_pitch_type.csv", index=False)

    outliers = (
        df[df["intent_adjusted_miss_ft"].notna()]
        .copy()
        .assign(abs_adjustment_delta=lambda frame: frame["intent_adjustment_delta_ft"].abs())
        .sort_values(["intent_model_group", "abs_adjustment_delta"], ascending=[True, False])
        .groupby("intent_model_group", group_keys=False)
        .head(3)
    )
    outlier_columns = [
        "intent_model_group",
        "pitch_uid",
        "sample_pitcher_name",
        "pitch_type",
        "pitch_family",
        "game_date",
        "balls",
        "strikes",
        "description",
        "events",
        "setup_miss_ft",
        "intent_adjusted_miss_ft",
        "intent_adjustment_delta_ft",
        "intent_model_confidence",
        "intent_model_confidence_tier",
        "group_offset_spread_ft",
        "target_x",
        "target_y",
        "inferred_intent_x",
        "inferred_intent_y",
        "actual_x_01",
        "actual_y_01",
        "direct_mp4_url",
    ]
    outliers[outlier_columns].to_csv(DATA_DIR / "intent_model_biggest_outliers.csv", index=False)

    family_summary = (
        df.groupby(["sample_pitcher_name", "pitch_family"], dropna=False)
        .agg(
            pitches=("pitch_uid", "count"),
            avg_setup_miss_ft=("setup_miss_ft", "mean"),
            median_setup_miss_ft=("setup_miss_ft", "median"),
            avg_intent_adjusted_miss_ft=("intent_adjusted_miss_ft", "mean"),
            median_intent_adjusted_miss_ft=("intent_adjusted_miss_ft", "median"),
            intent_model_ready=("intent_adjusted_miss_ft", lambda values: values.notna().sum()),
            pending_intent_model=("intent_adjusted_miss_ft", lambda values: values.isna().sum()),
            avg_model_confidence=("intent_model_confidence", "mean"),
            low_confidence_models=("intent_model_confidence_tier", lambda values: (values == "low").sum()),
        )
        .round(3)
        .reset_index()
        .sort_values(["sample_pitcher_name", "pitches"], ascending=[True, False])
    )
    family_summary.to_csv(DATA_DIR / "labeled_target_summary_by_pitch_family.csv", index=False)

    intent_summary = (
        df.groupby(["pitch_family", "intent_label"], dropna=False)
        .agg(
            pitches=("pitch_uid", "count"),
            avg_setup_miss_ft=("setup_miss_ft", "mean"),
            median_setup_miss_ft=("setup_miss_ft", "median"),
            avg_intent_adjusted_miss_ft=("intent_adjusted_miss_ft", "mean"),
            median_intent_adjusted_miss_ft=("intent_adjusted_miss_ft", "median"),
            intent_model_ready=("intent_adjusted_miss_ft", lambda values: values.notna().sum()),
            pending_intent_model=("intent_adjusted_miss_ft", lambda values: values.isna().sum()),
            avg_model_confidence=("intent_model_confidence", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values(["pitch_family", "pitches"], ascending=[True, False])
    )
    intent_summary.to_csv(DATA_DIR / "labeled_target_summary_by_intent.csv", index=False)

    print(f"Wrote {combined_out}")
    print(summary.to_string(index=False))
    print()
    if not bucket_summary.empty:
        print(bucket_summary.to_string(index=False))
        print()
    print(pitch_type_summary.to_string(index=False))
    print()
    print(family_summary.to_string(index=False))
    print()
    print(intent_summary.to_string(index=False))
    print()
    print(diagnostics.to_string(index=False))


def plot_locations(df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(exist_ok=True)

    colors = {
        "Bryan Woo": "#146C7C",
        "Dylan Cease": "#744FC6",
        "Jacob Misiorowski": "#D38B26",
        "Logan Webb": "#2F7D4E",
        "Max Fried": "#2D6FA3",
        "Shohei Ohtani": "#C44536",
        "Tarik Skubal": "#8B5E3C",
        "Zack Wheeler": "#5B6C77",
    }
    pitcher_groups = list(df.groupby("sample_pitcher_name"))
    cols = 4 if len(pitcher_groups) > 4 else max(1, len(pitcher_groups))
    rows = (len(pitcher_groups) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.7 * cols, 4.6 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, (pitcher_name, pitcher_df) in zip(axes_flat, pitcher_groups):
        ax.axhspan(0, 1, color="#ECF2F0", zorder=0)
        ax.axvspan(0, 1, color="#F8FAFA", zorder=0)
        ax.axhline(0, color="#9AA7AD", linewidth=1)
        ax.axhline(1, color="#9AA7AD", linewidth=1)
        ax.axvline(0, color="#9AA7AD", linewidth=1)
        ax.axvline(1, color="#9AA7AD", linewidth=1)
        ax.axvline(0.5, color="#9AA7AD", linewidth=1, alpha=0.55)
        ax.axhline(0.5, color="#9AA7AD", linewidth=1, alpha=0.55)
        ax.scatter(
            pitcher_df["target_x"],
            pitcher_df["target_y"],
            marker="x",
            s=70,
            linewidths=2,
            color="#222222",
            label="target",
        )
        ax.scatter(
            pitcher_df["actual_x_01"],
            pitcher_df["actual_y_01"],
            s=42,
            alpha=0.8,
            color=colors.get(pitcher_name, "#555555"),
            label="actual pitch",
        )

        for _, row in pitcher_df.iterrows():
            ax.plot(
                [row["target_x"], row["actual_x_01"]],
                [row["target_y"], row["actual_y_01"]],
                color="#AEB8BD",
                linewidth=0.8,
                alpha=0.55,
            )

        ax.set_title(pitcher_name)
        ax.set_xlabel("Pitcher POV Horizontal Location\n0 = left edge, 1 = right edge")
        ax.grid(True, alpha=0.18)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(-0.25, 1.35)

    for ax in axes_flat[len(pitcher_groups):]:
        ax.axis("off")

    for ax in axes[:, 0]:
        ax.set_ylabel("Vertical Location\n0 = bottom, 1 = top")
    axes_flat[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Catcher Target vs Actual Pitch Location", fontsize=15)
    fig.tight_layout()
    out = REPORT_DIR / "target_vs_actual_locations.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")

    fig, ax = plt.subplots(figsize=(8, 5))
    pitcher_names = [pitcher_name for pitcher_name, _ in pitcher_groups]
    plot_data = [
        df.loc[df["sample_pitcher_name"] == pitcher_name, "miss_distance"]
        for pitcher_name in pitcher_names
    ]
    ax.boxplot(plot_data, tick_labels=pitcher_names, patch_artist=True)
    ax.set_ylabel("Distance From Approx. Target Center")
    ax.set_title("Early Target Miss Distribution")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out = REPORT_DIR / "target_miss_distance_boxplot.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    labeled = load_labeled_data()
    analyzed = add_location_features(labeled)
    write_summary_tables(analyzed)
    plot_locations(analyzed)


if __name__ == "__main__":
    main()

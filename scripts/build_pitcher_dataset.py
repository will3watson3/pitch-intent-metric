import argparse
from pathlib import Path

import pandas as pd
from pybaseball import statcast_pitcher


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CORE_COLUMNS = [
    "pitch_uid",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "pitch_name",
    "player_name",
    "pitcher",
    "batter",
    "fielder_2",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "on_1b",
    "on_2b",
    "on_3b",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "zone",
    "description",
    "events",
    "des",
    "type",
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "pfx_x",
    "pfx_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "api_break_x_arm",
    "api_break_x_batter_in",
    "api_break_z_with_gravity",
    "arm_angle",
    "spin_axis",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "bb_type",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "delta_run_exp",
    "delta_pitcher_run_exp",
    "delta_home_win_exp",
]


def add_pitch_uid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    game_pk = df["game_pk"].astype("Int64").astype(str)
    at_bat_number = df["at_bat_number"].astype("Int64").astype(str)
    pitch_number = df["pitch_number"].astype("Int64").astype(str)
    df.insert(0, "pitch_uid", game_pk + "_" + at_bat_number + "_" + pitch_number)
    return df


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitcher-id", type=int, required=True)
    parser.add_argument("--pitcher-name", required=True)
    parser.add_argument("--slug", required=True, help="File prefix, e.g. wheeler_2025")
    parser.add_argument("--start-date", default="2025-03-01")
    parser.add_argument("--end-date", default="2025-11-30")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    raw_out = DATA_DIR / f"{args.slug}_statcast_raw.csv"
    clean_out = DATA_DIR / f"{args.slug}_pitch_level.csv"
    no_runner_2b_out = DATA_DIR / f"{args.slug}_pitch_level_no_runner_on_2b.csv"

    print(
        f"Pulling {args.pitcher_name} ({args.pitcher_id}) "
        f"from {args.start_date} to {args.end_date}..."
    )
    df = statcast_pitcher(args.start_date, args.end_date, args.pitcher_id)

    if df.empty:
        raise RuntimeError("No rows returned from pybaseball.")

    sort_columns = ["game_pk", "at_bat_number", "pitch_number"]
    df = df.sort_values(sort_columns).reset_index(drop=True)
    df = add_pitch_uid(df)
    df.to_csv(raw_out, index=False)

    clean = select_existing_columns(df, CORE_COLUMNS)
    clean.to_csv(clean_out, index=False)

    no_runner_2b = clean[clean["on_2b"].isna()].copy()
    no_runner_2b.to_csv(no_runner_2b_out, index=False)

    print(f"Raw rows: {len(df):,}")
    print(f"Clean columns: {len(clean.columns):,}")
    print(f"No runner on 2B rows: {len(no_runner_2b):,}")
    print(f"Wrote {raw_out}")
    print(f"Wrote {clean_out}")
    print(f"Wrote {no_runner_2b_out}")


if __name__ == "__main__":
    main()

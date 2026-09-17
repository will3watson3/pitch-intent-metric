import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DEFAULT_COLUMNS = [
    "pitch_uid",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "pitch_name",
    "batter",
    "fielder_2",
    "stand",
    "balls",
    "strikes",
    "inning",
    "inning_topbot",
    "description",
    "events",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "savant_video_url",
    "direct_mp4_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    df = df[df["direct_mp4_url"].notna()].head(args.limit).copy()

    columns = [column for column in DEFAULT_COLUMNS if column in df.columns]
    queue = df[columns].copy()
    queue["catcher_target_bucket"] = ""
    queue["target_confidence_1_to_5"] = ""
    queue["setup_visible"] = ""
    queue["label_notes"] = ""

    args.output.parent.mkdir(exist_ok=True)
    queue.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(queue):,} rows)")


if __name__ == "__main__":
    main()

import argparse
import html
import re
from pathlib import Path
from time import sleep
from typing import Optional

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAVANT_VIDEO_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
MP4_RE = re.compile(r'<source[^>]+src="([^"]+\.mp4)"', re.IGNORECASE)
FASTBALL_PITCHES = {"FF", "SI", "FC", "FA"}

QUEUE_COLUMNS = [
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
    "target_x_01",
    "target_y_01",
    "catcher_target_bucket",
    "target_confidence_1_to_5",
    "setup_visible",
    "label_notes",
]


def extract_direct_mp4(play_id: str):
    if not play_id:
        return None, None

    response = requests.get(SAVANT_VIDEO_URL.format(play_id=play_id), timeout=30)
    status_code = response.status_code
    if not response.ok:
        return status_code, None

    match = MP4_RE.search(response.text)
    if not match:
        return status_code, None

    return status_code, html.unescape(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-per-game", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument(
        "--exclude-fastballs",
        action="store_true",
        help="Exclude four-seamers, sinkers, cutters, and generic fastballs.",
    )
    parser.add_argument(
        "--fetch-missing-mp4",
        action="store_true",
        help="Fetch direct MP4 links for selected rows that only have MLB play IDs.",
    )
    parser.add_argument(
        "--slugs",
        nargs="*",
        default=None,
        help="Optional slugs, e.g. woo_2025 webb_2025. Defaults to every existing sample queue.",
    )
    parser.add_argument("--suffix", default="extra_25")
    return parser.parse_args()


def source_slugs(requested_slugs: Optional[list[str]]) -> list[str]:
    if requested_slugs:
        return requested_slugs
    slugs = []
    for path in sorted(DATA_DIR.glob("*_labeling_queue_sample_25.csv")):
        slugs.append(path.name.replace("_labeling_queue_sample_25.csv", ""))
    return slugs


def selected_rows(
    matches: pd.DataFrame,
    original_queue: pd.DataFrame,
    limit: int,
    max_per_game: int,
    exclude_fastballs: bool,
) -> pd.DataFrame:
    original_game_pks = set(pd.to_numeric(original_queue["game_pk"], errors="coerce").dropna().astype(int))
    original_pitch_uids = set(original_queue["pitch_uid"].dropna().astype(str))

    candidates = matches.copy()
    candidates = candidates[candidates["mlb_play_id"].notna()]
    if exclude_fastballs:
        candidates = candidates[~candidates["pitch_type"].astype(str).str.upper().isin(FASTBALL_PITCHES)]
    candidates = candidates[~candidates["pitch_uid"].astype(str).isin(original_pitch_uids)]
    candidates = candidates[~pd.to_numeric(candidates["game_pk"], errors="coerce").isin(original_game_pks)]
    candidates = candidates.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        ascending=[False, False, True, True],
    )

    picks = []
    for _, game_df in candidates.groupby("game_pk", sort=False):
        picks.append(game_df.head(max_per_game))

    if not picks:
        raise ValueError("No candidate pitches left after excluding original games.")

    queue = pd.concat(picks, ignore_index=True).head(limit).copy()
    if len(queue) < limit:
        raise ValueError(f"Only found {len(queue)} candidate pitches; requested {limit}.")
    return queue


def ensure_mp4_urls(queue: pd.DataFrame, matches: pd.DataFrame, matches_path: Path, sleep_seconds: float) -> pd.DataFrame:
    queue = queue.copy()
    updated_matches = matches.copy()

    for index, row in queue.iterrows():
        if pd.notna(row.get("direct_mp4_url")) and str(row.get("direct_mp4_url")):
            continue

        play_id = row.get("mlb_play_id")
        pitch_uid = row.get("pitch_uid")
        status_code, mp4_url = extract_direct_mp4(play_id)
        savant_url = SAVANT_VIDEO_URL.format(play_id=play_id)

        queue.at[index, "savant_video_status"] = status_code
        queue.at[index, "savant_video_url"] = savant_url
        queue.at[index, "direct_mp4_url"] = mp4_url

        match_mask = updated_matches["pitch_uid"].astype(str) == str(pitch_uid)
        updated_matches.loc[match_mask, "savant_video_status"] = status_code
        updated_matches.loc[match_mask, "savant_video_url"] = savant_url
        updated_matches.loc[match_mask, "direct_mp4_url"] = mp4_url

        print(f"  {pitch_uid}: status={status_code}, mp4={'yes' if mp4_url else 'no'}")
        sleep(sleep_seconds)

    updated_matches.to_csv(matches_path, index=False)
    return queue


def build_queue(
    slug: str,
    limit: int,
    max_per_game: int,
    suffix: str,
    fetch_missing_mp4: bool,
    sleep_seconds: float,
    exclude_fastballs: bool,
) -> Path:
    original_path = DATA_DIR / f"{slug}_labeling_queue_sample_25.csv"
    matches_path = DATA_DIR / f"{slug}_pitch_video_matches.csv"
    output_path = DATA_DIR / f"{slug}_labeling_queue_{suffix}.csv"

    if not original_path.exists():
        raise FileNotFoundError(f"Missing original queue: {original_path}")
    if not matches_path.exists():
        raise FileNotFoundError(f"Missing video matches: {matches_path}")

    original_queue = pd.read_csv(original_path)
    matches = pd.read_csv(matches_path)
    queue = selected_rows(matches, original_queue, limit, max_per_game, exclude_fastballs)
    if fetch_missing_mp4:
        queue = ensure_mp4_urls(queue, matches, matches_path, sleep_seconds)

    for column in QUEUE_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    for column in ["target_x_01", "target_y_01", "catcher_target_bucket", "target_confidence_1_to_5", "setup_visible", "label_notes"]:
        queue[column] = ""

    queue = queue[[column for column in QUEUE_COLUMNS if column in queue.columns]]
    queue.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    for slug in source_slugs(args.slugs):
        print(f"\n=== {slug} ===")
        output_path = build_queue(
            slug,
            args.limit,
            args.max_per_game,
            args.suffix,
            args.fetch_missing_mp4,
            args.sleep,
            args.exclude_fastballs,
        )
        df = pd.read_csv(output_path)
        games = ", ".join(str(game_pk) for game_pk in sorted(df["game_pk"].unique()))
        dates = ", ".join(str(date) for date in sorted(df["game_date"].unique()))
        print(f"Wrote {output_path} ({len(df)} rows, {df['game_pk'].nunique()} games)")
        print(f"  games: {games}")
        print(f"  dates: {dates}")


if __name__ == "__main__":
    main()

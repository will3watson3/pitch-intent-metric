import argparse
import html
import json
import re
from pathlib import Path
from time import sleep
from typing import Optional, Tuple

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "mlb_gamefeeds"
DEFAULT_INPUT = DATA_DIR / "woo_2025_pitch_level_no_runner_on_2b.csv"
DEFAULT_OUTPUT = DATA_DIR / "woo_2025_pitch_video_matches.csv"

GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
SAVANT_VIDEO_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
MP4_RE = re.compile(r'<source[^>]+src="([^"]+\.mp4)"', re.IGNORECASE)


def fetch_json(url: str, cache_path: Optional[Path] = None) -> dict:
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text())

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data))

    return data


def build_pitch_event_map(game_pk: int) -> dict[tuple[int, int, int], dict]:
    cache_path = CACHE_DIR / f"{game_pk}.json"
    data = fetch_json(GAME_FEED_URL.format(game_pk=game_pk), cache_path)
    event_map = {}

    for play in data["liveData"]["plays"]["allPlays"]:
        at_bat_index = play["about"].get("atBatIndex")
        if at_bat_index is None:
            continue

        savant_at_bat_number = int(at_bat_index) + 1
        batter_id = play["matchup"]["batter"]["id"]

        for event in play.get("playEvents", []):
            if not event.get("isPitch"):
                continue

            pitch_number = event.get("pitchNumber")
            if pitch_number is None:
                continue

            details = event.get("details", {})
            pitch_type = details.get("type", {})
            pitch_data = event.get("pitchData", {})
            coordinates = pitch_data.get("coordinates", {})

            key = (int(game_pk), savant_at_bat_number, int(pitch_number))
            event_map[key] = {
                "mlb_play_id": event.get("playId"),
                "mlb_at_bat_index": at_bat_index,
                "mlb_batter": batter_id,
                "mlb_pitch_type": pitch_type.get("code"),
                "mlb_pitch_name": pitch_type.get("description"),
                "mlb_pitch_description": details.get("description"),
                "mlb_start_speed": pitch_data.get("startSpeed"),
                "mlb_plate_x": coordinates.get("pX"),
                "mlb_plate_z": coordinates.get("pZ"),
            }

    return event_map


def extract_direct_mp4(play_id: str) -> Tuple[Optional[int], Optional[str]]:
    if not play_id:
        return None, None

    url = SAVANT_VIDEO_URL.format(play_id=play_id)
    response = requests.get(url, timeout=30)
    status_code = response.status_code
    if not response.ok:
        return status_code, None

    match = MP4_RE.search(response.text)
    if not match:
        return status_code, None

    return status_code, html.unescape(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mp4-limit",
        type=int,
        default=25,
        help="Fetch direct .mp4 links for the first N matched rows. Use 0 to skip.",
    )
    parser.add_argument(
        "--all-mp4",
        action="store_true",
        help="Fetch direct .mp4 links for every matched pitch.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Seconds to sleep between Savant video-page requests.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"game_pk", "at_bat_number", "pitch_number"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    unique_games = sorted(df["game_pk"].dropna().astype(int).unique())
    print(f"Loading game feeds for {len(unique_games):,} games...")

    event_maps = {}
    for game_pk in unique_games:
        event_maps[game_pk] = build_pitch_event_map(game_pk)

    match_rows = []
    for row in df.to_dict("records"):
        game_pk = int(row["game_pk"])
        key = (
            game_pk,
            int(row["at_bat_number"]),
            int(row["pitch_number"]),
        )
        event = event_maps.get(game_pk, {}).get(key, {})
        play_id = event.get("mlb_play_id")

        match_rows.append(
            {
                **row,
                **event,
                "savant_video_url": SAVANT_VIDEO_URL.format(play_id=play_id)
                if play_id
                else None,
                "savant_video_status": None,
                "direct_mp4_url": None,
            }
        )

    matched = pd.DataFrame(match_rows)
    with_play_id = matched["mlb_play_id"].notna().sum()
    print(f"Matched playIds: {with_play_id:,} / {len(matched):,}")

    if args.all_mp4 or args.mp4_limit > 0:
        matched_indexes = matched.index[matched["mlb_play_id"].notna()].tolist()
        indexes_to_fetch = matched_indexes if args.all_mp4 else matched_indexes[: args.mp4_limit]
        for index in indexes_to_fetch:
            play_id = matched.at[index, "mlb_play_id"]
            status_code, mp4_url = extract_direct_mp4(play_id)
            matched.at[index, "savant_video_status"] = status_code
            matched.at[index, "direct_mp4_url"] = mp4_url
            print(
                f"{matched.at[index, 'pitch_uid']}: "
                f"status={status_code}, mp4={'yes' if mp4_url else 'no'}"
            )
            sleep(args.sleep)

    args.output.parent.mkdir(exist_ok=True)
    matched.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "mlb_gamefeeds"
STAGES = ["dataset", "videos", "frames", "inference", "pose", "selection", "quality", "dashboard"]
PLATE_EDGE_FT = 17 / 12 / 2
FASTBALL_PITCHES = {"FF", "SI", "FC", "FA"}
BREAKING_PITCHES = {"SL", "ST", "CU", "KC", "SV", "CS"}
OFFSPEED_PITCHES = {"CH", "FS", "FO", "SC"}


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    pitch_data: Path
    video_matches: Path
    frames_dir: Path
    frame_manifest: Path
    detections: Path
    raw_responses: Path
    pose: Path
    provisional_zones: Path
    selected_provisional: Path
    final_zones: Path
    final_predictions: Path
    quality: Path
    review_queue: Path
    final_dataset: Path
    summary: Path
    dashboard: Path
    recovered_frames: Path
    fox_recovery_audit: Path
    run_manifest: Path
    no_fallback_predictions: Path
    no_calibrations: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Statcast-to-dashboard autonomous pitch-intent vision for a pitcher/game."
    )
    parser.add_argument("--pitcher-id", type=int)
    parser.add_argument("--pitcher-name")
    parser.add_argument("--game-pk", type=int)
    parser.add_argument("--game-date", help="YYYY-MM-DD. Inferred from --game-pk when omitted.")
    parser.add_argument("--start-date", help="YYYY-MM-DD for a multi-game run.")
    parser.add_argument("--end-date", help="YYYY-MM-DD for a multi-game run.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Use an existing Statcast/video CSV instead of querying pybaseball.",
    )
    parser.add_argument("--run-name", help="Folder name under data/runs.")
    parser.add_argument("--limit-pitches", type=int)
    parser.add_argument("--include-runner-on-second", action="store_true")
    parser.add_argument("--preserve-manual-targets", action="store_true")
    parser.add_argument("--model-id", default="baseball-catcher-setups/1")
    parser.add_argument("--model-confidence", type=int, default=25)
    parser.add_argument("--model-overlap", type=int, default=30)
    parser.add_argument("--frame-preset", default="delivery31", choices=["delivery31"])
    parser.add_argument("--sleep-sec", type=float, default=0.03)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-videos", action="store_true")
    parser.add_argument("--refresh-inference", action="store_true")
    parser.add_argument("--start-at", choices=STAGES, default="dataset")
    parser.add_argument("--stop-after", choices=STAGES, default="dashboard")
    parser.add_argument("--plan", action="store_true", help="Print run paths without executing stages.")
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "pitch-intent-run"


def build_paths(run_name: str) -> RunPaths:
    run_dir = DATA_DIR / "runs" / run_name
    return RunPaths(
        run_dir=run_dir,
        pitch_data=run_dir / "01_statcast_pitches.csv",
        video_matches=run_dir / "02_pitch_videos.csv",
        frames_dir=run_dir / "frames",
        frame_manifest=run_dir / "03_frame_manifest.csv",
        detections=run_dir / "04_frame_detections.csv",
        raw_responses=run_dir / "roboflow_raw_responses",
        pose=run_dir / "05_pitcher_pose.csv",
        provisional_zones=run_dir / "06_provisional_zones.csv",
        selected_provisional=run_dir / "07_selected_setup_frames.csv",
        final_zones=run_dir / "08_final_broadcast_zones.csv",
        final_predictions=run_dir / "09_final_vision_predictions.csv",
        quality=run_dir / "10_vision_quality.csv",
        review_queue=run_dir / "11_review_queue.csv",
        final_dataset=run_dir / "pitch_intent.csv",
        summary=run_dir / "summary.json",
        dashboard=run_dir / "dashboard.html",
        recovered_frames=run_dir / "12_recovered_frame_zones.csv",
        fox_recovery_audit=run_dir / "13_fox_zone_candidates.csv",
        run_manifest=run_dir / "run.json",
        no_fallback_predictions=run_dir / "no_fallback_predictions.csv",
        no_calibrations=run_dir / "no_manual_calibrations.json",
    )


def normalized_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def stage_active(args: argparse.Namespace, stage: str) -> bool:
    return STAGES.index(args.start_at) <= STAGES.index(stage) <= STAGES.index(args.stop_after)


def require_file(path: Path, stage: str) -> None:
    if not path.exists():
        raise SystemExit(f"Stage '{stage}' requires missing file: {path}")


def pitch_family(value) -> str:
    pitch_type = str(value or "").upper()
    if pitch_type in FASTBALL_PITCHES:
        return "fastball"
    if pitch_type in BREAKING_PITCHES:
        return "breaking"
    if pitch_type in OFFSPEED_PITCHES:
        return "offspeed"
    return "other"


def game_date_from_feed(game_pk: int) -> str:
    cache_path = CACHE_DIR / f"{game_pk}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
    else:
        import requests

        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))
    game_data = payload.get("gameData", {})
    date = game_data.get("datetime", {}).get("originalDate") or game_data.get("officialDate")
    if not date:
        raise RuntimeError(f"Could not infer date for game {game_pk}")
    return str(date)


def resolve_dates(args: argparse.Namespace) -> tuple[str, str]:
    if args.game_date:
        return args.game_date, args.game_date
    if args.start_date or args.end_date:
        start = args.start_date or args.end_date
        end = args.end_date or args.start_date
        return str(start), str(end)
    if args.game_pk:
        date = game_date_from_feed(args.game_pk)
        return date, date
    raise SystemExit("Provide --game-pk, --game-date, --start-date/--end-date, or --input-csv.")


def add_pitch_uid(df: pd.DataFrame) -> pd.DataFrame:
    if "pitch_uid" in df.columns:
        df["pitch_uid"] = df["pitch_uid"].astype(str)
        return df
    required = ["game_pk", "at_bat_number", "pitch_number"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot create pitch_uid; missing columns: {missing}")
    keys = [pd.to_numeric(df[column], errors="raise").astype("Int64").astype(str) for column in required]
    df.insert(0, "pitch_uid", keys[0] + "_" + keys[1] + "_" + keys[2])
    return df


def prepare_pitch_data(args: argparse.Namespace, paths: RunPaths) -> None:
    if paths.pitch_data.exists() and not args.refresh_data:
        print(f"Reusing Statcast data: {paths.pitch_data}")
        return

    if args.input_csv:
        input_path = normalized_path(args.input_csv)
        if not input_path.exists():
            raise SystemExit(f"Missing --input-csv: {input_path}")
        df = pd.read_csv(input_path)
        source_file = input_path.name
    else:
        if not args.pitcher_id:
            raise SystemExit("--pitcher-id is required when --input-csv is not provided.")
        start_date, end_date = resolve_dates(args)
        from pybaseball import statcast_pitcher

        print(f"Pulling Statcast: pitcher={args.pitcher_id}, {start_date} to {end_date}")
        df = statcast_pitcher(start_date, end_date, args.pitcher_id)
        source_file = f"pybaseball_statcast_pitcher_{args.pitcher_id}.csv"

    if df.empty:
        raise RuntimeError("No Statcast pitches were returned.")
    df = add_pitch_uid(df.copy())
    if args.game_pk:
        df = df[pd.to_numeric(df["game_pk"], errors="coerce") == args.game_pk].copy()
    if not args.include_runner_on_second and "on_2b" in df.columns:
        df = df[df["on_2b"].isna()].copy()
    if df.empty:
        raise RuntimeError("No pitches remained after game and runner-on-second filters.")

    sort_columns = [column for column in ["game_pk", "at_bat_number", "pitch_number"] if column in df]
    if sort_columns:
        df = df.sort_values(sort_columns)
    if args.limit_pitches:
        df = df.head(args.limit_pitches).copy()

    if args.pitcher_name:
        df["sample_pitcher_name"] = args.pitcher_name
    elif "sample_pitcher_name" not in df.columns:
        player_name = str(df.get("player_name", pd.Series(["Unknown Pitcher"])).iloc[0])
        if "," in player_name:
            last, first = [part.strip() for part in player_name.split(",", 1)]
            player_name = f"{first} {last}"
        df["sample_pitcher_name"] = player_name
    df["source_file"] = source_file
    df["pitch_family"] = df.get("pitch_type", pd.Series(index=df.index, dtype=str)).map(pitch_family)

    for column in ["plate_x", "plate_z", "sz_top", "sz_bot"]:
        df[column] = pd.to_numeric(df.get(column), errors="coerce")
    df["actual_x_01"] = (-df["plate_x"] + PLATE_EDGE_FT) / (2 * PLATE_EDGE_FT)
    df["actual_y_01"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])
    for column in [
        "target_x_01",
        "target_y_01",
        "setup_visible",
        "label_notes",
        "savant_video_url",
        "direct_mp4_url",
    ]:
        if column not in df.columns:
            df[column] = ""
    if not args.preserve_manual_targets:
        df["target_x_01"] = ""
        df["target_y_01"] = ""
        df["setup_visible"] = ""
        df["label_notes"] = ""
        derived_prefixes = (
            "target_plate_",
            "miss_",
            "setup_miss",
            "intent_",
            "movement_offset_",
            "inferred_intent_",
            "group_offset_",
        )
        derived_columns = [
            column
            for column in df.columns
            if column in {
                "target_x",
                "target_y",
                "target_source",
                "catcher_target_bucket",
                "target_confidence_1_to_5",
                "hit_side",
                "hit_height",
            }
            or column.startswith(derived_prefixes)
        ]
        df = df.drop(columns=derived_columns, errors="ignore")

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    df.fillna("").to_csv(paths.pitch_data, index=False)
    print(f"Statcast pitches: {len(df)}")
    print(f"Output: {paths.pitch_data}")


def match_videos(args: argparse.Namespace, paths: RunPaths) -> None:
    require_file(paths.pitch_data, "videos")
    if paths.video_matches.exists() and not args.refresh_videos:
        print(f"Reusing video matches: {paths.video_matches}")
        return
    pitches = pd.read_csv(paths.pitch_data)
    has_all_urls = (
        "direct_mp4_url" in pitches
        and len(pitches)
        and pitches["direct_mp4_url"].fillna("").astype(str).str.len().gt(0).all()
    )
    if has_all_urls:
        pitches.to_csv(paths.video_matches, index=False)
        print(f"Reused {len(pitches)} direct video URLs from the input CSV.")
        return
    run_command(
        [
            sys.executable,
            "scripts/match_woo_2025_videos.py",
            "--input",
            str(paths.pitch_data),
            "--output",
            str(paths.video_matches),
            "--all-mp4",
            "--sleep",
            "0.10",
        ]
    )
    matches = pd.read_csv(paths.video_matches)
    video_count = int(matches.get("direct_mp4_url", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0).sum())
    print(f"Direct videos: {video_count}/{len(matches)}")
    if video_count == 0:
        raise RuntimeError("No direct pitch videos were found.")


def extract_frames(paths: RunPaths) -> None:
    require_file(paths.video_matches, "frames")
    run_command(
        [
            sys.executable,
            "scripts/extract_setup_frames.py",
            "--input",
            str(paths.video_matches),
            "--output-dir",
            str(paths.frames_dir),
            "--manifest",
            str(paths.frame_manifest),
            "--frame-preset",
            "delivery31",
            "--extract",
        ]
    )
    manifest = pd.read_csv(paths.frame_manifest)
    usable = manifest["extraction_status"].isin(["extracted", "exists"])
    if not usable.any():
        raise RuntimeError("Frame extraction produced no usable images. Check ffmpeg and video URLs.")
    print(f"Usable frames: {int(usable.sum())}/{len(manifest)}")


def run_inference(args: argparse.Namespace, paths: RunPaths) -> None:
    require_file(paths.frame_manifest, "inference")
    if not os.environ.get("ROBOFLOW_API_KEY"):
        raise SystemExit("Missing ROBOFLOW_API_KEY. Export a private Roboflow inference key and rerun.")
    command = [
        sys.executable,
        "scripts/run_roboflow_workflow.py",
        "--manifest",
        str(paths.frame_manifest),
        "--all-frames",
        "--direct-model-id",
        args.model_id,
        "--confidence",
        str(args.model_confidence),
        "--overlap",
        str(args.model_overlap),
        "--output",
        str(paths.detections),
        "--raw-dir",
        str(paths.raw_responses),
        "--sleep-sec",
        str(args.sleep_sec),
        "--progress-every",
        "50",
        "--use-cache",
    ]
    if not args.refresh_inference:
        command.append("--resume-raw")
    run_command(command)
    detections = pd.read_csv(paths.detections)
    usable = detections["status"].isin(["ok", "missing_required_detection"])
    if not usable.any():
        errors = detections.get("error", pd.Series(dtype=str)).dropna().astype(str).head(3).tolist()
        raise RuntimeError("Roboflow inference produced no usable rows: " + "; ".join(errors))


def detect_pose(paths: RunPaths) -> None:
    require_file(paths.frame_manifest, "pose")
    run_command(
        [
            sys.executable,
            "scripts/detect_pitcher_leg_kick.py",
            "--manifest",
            str(paths.frame_manifest),
            "--output",
            str(paths.pose),
            "--resume",
            "--progress-every",
            "400",
        ]
    )


def build_selection(paths: RunPaths) -> None:
    for path in [paths.frame_manifest, paths.detections, paths.pose]:
        require_file(path, "selection")
    common_zone = [
        sys.executable,
        "scripts/build_statcast_zone_predictions.py",
        "--manifest",
        str(paths.frame_manifest),
        "--fallback-predictions",
        str(paths.no_fallback_predictions),
        "--dense-predictions",
        str(paths.detections),
        "--calibrations",
        str(paths.no_calibrations),
    ]
    run_command(
        [
            *common_zone,
            "--primary-predictions",
            str(paths.detections),
            "--output",
            str(paths.provisional_zones),
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/select_best_glove_frame.py",
            "--manifest",
            str(paths.frame_manifest),
            "--all-frame-predictions",
            str(paths.detections),
            "--zone-predictions",
            str(paths.provisional_zones),
            "--pose-predictions",
            str(paths.pose),
            "--output",
            str(paths.selected_provisional),
            "--motion-aware",
            "--raw-glove-candidates",
            "--allow-glove-only-frames",
            "--window-start-sec",
            "1.20",
            "--window-end-sec",
            "2.70",
            "--min-target-x",
            "-0.75",
            "--max-target-x",
            "1.65",
            "--min-target-y",
            "-0.60",
            "--max-target-y",
            "1.35",
            "--probable-mask-target-y",
            "1.35",
        ]
    )
    run_command(
        [
            *common_zone,
            "--primary-predictions",
            str(paths.selected_provisional),
            "--output",
            str(paths.final_zones),
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/apply_calibrated_zones_to_predictions.py",
            "--predictions",
            str(paths.selected_provisional),
            "--zones",
            str(paths.final_zones),
            "--output",
            str(paths.final_predictions),
        ]
    )
    run_command([
        sys.executable,"scripts/cache_recovery_videos.py","--predictions",str(paths.final_predictions),
        "--matches",str(paths.video_matches),"--output-dir",str(ROOT/'.cache/zone_recovery_videos'),
    ])
    run_command([
        sys.executable,"scripts/recover_unverified_zones.py",
        "--predictions",str(paths.final_predictions),"--dense",str(paths.detections),
        "--poses",str(paths.pose),"--output",str(paths.final_predictions),
        "--frame-output",str(paths.recovered_frames),"--audit-output",str(paths.fox_recovery_audit),
    ])


def build_quality(paths: RunPaths) -> None:
    for path in [paths.final_predictions, paths.video_matches]:
        require_file(path, "quality")
    run_command(
        [
            sys.executable,
            "scripts/build_vision_review_queue.py",
            "--predictions",
            str(paths.final_predictions),
            "--analysis",
            str(paths.video_matches),
            "--quality-output",
            str(paths.quality),
            "--review-output",
            str(paths.review_queue),
            "--review-label-mode",
            "none",
            "--top-outliers",
            "0",
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/build_autonomous_pitch_dataset.py",
            "--quality",
            str(paths.quality),
            "--pitch-data",
            str(paths.video_matches),
            "--output",
            str(paths.final_dataset),
            "--summary",
            str(paths.summary),
        ]
    )


def build_dashboard(paths: RunPaths) -> None:
    for path in [paths.final_dataset, paths.summary]:
        require_file(path, "dashboard")
    run_command(
        [
            sys.executable,
            "scripts/build_autonomous_dashboard.py",
            "--input",
            str(paths.final_dataset),
            "--summary",
            str(paths.summary),
            "--output",
            str(paths.dashboard),
        ]
    )


def default_run_name(args: argparse.Namespace) -> str:
    pitcher = args.pitcher_name or (f"pitcher-{args.pitcher_id}" if args.pitcher_id else "pitcher")
    scope = f"game-{args.game_pk}" if args.game_pk else args.game_date or args.start_date or "input"
    return slugify(f"{pitcher}-{scope}")


def write_run_manifest(args: argparse.Namespace, paths: RunPaths) -> None:
    payload = {
        "pipeline": "autonomous_pitch_intent_v1",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "paths": {key: str(value) for key, value in asdict(paths).items()},
    }
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.run_manifest.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    if STAGES.index(args.start_at) > STAGES.index(args.stop_after):
        raise SystemExit("--start-at must come before or equal to --stop-after.")
    if not args.input_csv and not args.pitcher_id:
        raise SystemExit("Provide --pitcher-id or --input-csv.")

    run_name = slugify(args.run_name) if args.run_name else default_run_name(args)
    paths = build_paths(run_name)
    write_run_manifest(args, paths)
    print(f"Run: {run_name}")
    print(f"Directory: {paths.run_dir}")
    if args.plan:
        for key, value in asdict(paths).items():
            print(f"{key}: {value}")
        return

    if stage_active(args, "dataset"):
        prepare_pitch_data(args, paths)
    if stage_active(args, "videos"):
        match_videos(args, paths)
    if stage_active(args, "frames"):
        extract_frames(paths)
    if stage_active(args, "inference"):
        run_inference(args, paths)
    if stage_active(args, "pose"):
        detect_pose(paths)
    if stage_active(args, "selection"):
        build_selection(paths)
    if stage_active(args, "quality"):
        build_quality(paths)
    if stage_active(args, "dashboard"):
        build_dashboard(paths)

    print("\nAutonomous pitch-intent pipeline finished.")
    if paths.final_dataset.exists():
        print(f"CSV: {paths.final_dataset}")
    if paths.dashboard.exists():
        print(f"Dashboard: {paths.dashboard}")


if __name__ == "__main__":
    main()

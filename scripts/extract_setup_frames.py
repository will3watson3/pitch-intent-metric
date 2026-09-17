import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "setup_frames"
DEFAULT_MANIFEST = DATA_DIR / "setup_frame_manifest.csv"
STANDARD_FRAME_TIMES = "1.00,1.20,1.40,1.60,1.80,2.00,2.25,2.50"
DENSE15_FRAME_TIMES = "1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.00,2.10,2.20,2.30,2.40"
DELIVERY21_FRAME_TIMES = "1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.00,2.10,2.20,2.30,2.40,2.50,2.60,2.70,2.80,2.90,3.00"
DELIVERY16_FRAME_TIMES = "1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.00,2.10,2.20,2.30,2.40,2.50,2.60,2.70"
DELIVERY31_FRAME_TIMES = ",".join(f"{value / 1000:.2f}" for value in range(1200, 2701, 50))
FRAME_PRESETS = {
    "standard8": STANDARD_FRAME_TIMES,
    "dense15": DENSE15_FRAME_TIMES,
    "delivery16": DELIVERY16_FRAME_TIMES,
    "delivery31": DELIVERY31_FRAME_TIMES,
    "delivery21": DELIVERY21_FRAME_TIMES,
}


MANIFEST_COLUMNS = [
    "frame_uid",
    "pitch_uid",
    "sample_pitcher_name",
    "source_file",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "pitch_name",
    "pitch_family",
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
    "actual_x_01",
    "actual_y_01",
    "target_x_01",
    "target_y_01",
    "setup_visible",
    "label_notes",
    "savant_video_url",
    "direct_mp4_url",
    "frame_index",
    "frame_label",
    "frame_time_sec",
    "image_path",
    "extraction_status",
]


def latest_analysis_file() -> Path:
    files = sorted(
        DATA_DIR.glob("labeled_target_analysis_sample_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No labeled_target_analysis_sample_*.csv file found.")
    return files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pitch-uid", default=None)
    parser.add_argument(
        "--pitch-uids-file",
        type=Path,
        default=None,
        help="CSV/text file of pitch_uid values to include. CSV files may include a pitch_uid column.",
    )
    parser.add_argument(
        "--frame-times-sec",
        default=None,
        help="Comma-separated seconds to sample from the start of each MP4.",
    )
    parser.add_argument(
        "--frame-preset",
        choices=sorted(FRAME_PRESETS),
        default="standard8",
        help="Named frame-time preset. Use delivery21 for motion-aware delivery and glove-drop selection.",
    )
    parser.add_argument(
        "--frame-time-sec",
        type=float,
        default=None,
        help="Legacy single-frame option. Overrides --frame-times-sec when provided.",
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quality", type=int, default=2)
    parser.add_argument(
        "--only-visible",
        action="store_true",
        help="Only include rows where setup_visible is yes.",
    )
    return parser.parse_args()


def read_pitch_uid_filter(path: Path) -> set[str]:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise SystemExit(f"Missing pitch uid file: {resolved}")

    if resolved.suffix.lower() == ".csv":
        df = pd.read_csv(resolved)
        if "pitch_uid" in df.columns:
            values = df["pitch_uid"]
        elif len(df.columns):
            values = df[df.columns[0]]
        else:
            values = []
        return {normalized(value) for value in values if normalized(value)}

    return {
        normalized(line)
        for line in resolved.read_text().splitlines()
        if normalized(line) and not normalized(line).startswith("#")
    }


def parse_frame_times(args: argparse.Namespace) -> list[float]:
    if args.frame_time_sec is not None:
        return [args.frame_time_sec]

    raw_times = args.frame_times_sec or FRAME_PRESETS[args.frame_preset]
    times = []
    for raw_value in str(raw_times).split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        try:
            frame_time = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid frame time: {raw_value}") from exc
        if frame_time < 0:
            raise ValueError("Frame times must be non-negative.")
        times.append(frame_time)

    if not times:
        raise ValueError("At least one frame time is required.")
    return times


def ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except Exception:
        return ""

    try:
        ffmpeg = get_ffmpeg_exe()
    except Exception:
        return ""

    return ffmpeg if ffmpeg and Path(ffmpeg).exists() else ""


def normalized(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def load_rows(path: Path, only_visible: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "source_file" not in df.columns:
        df["source_file"] = path.name
    if "target_x_01" not in df.columns and "target_x" in df.columns:
        df["target_x_01"] = df["target_x"]
    if "target_y_01" not in df.columns and "target_y" in df.columns:
        df["target_y_01"] = df["target_y"]
    for column in MANIFEST_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[df["direct_mp4_url"].notna() & (df["direct_mp4_url"].astype(str) != "")]
    if only_visible:
        df = df[df["setup_visible"].astype(str).str.lower() == "yes"]
    return df.copy()


def frame_path(output_dir: Path, row: pd.Series, frame_index: int, frame_time_sec: float) -> Path:
    pitcher = normalized(row.get("sample_pitcher_name")) or "unknown_pitcher"
    pitcher_slug = pitcher.lower().replace(" ", "_").replace(".", "")
    pitch_type = normalized(row.get("pitch_type")) or "unknown"
    time_ms = int(round(frame_time_sec * 1000))
    pitch_uid = normalized(row.get("pitch_uid")) or "unknown_pitch"
    return output_dir / pitcher_slug / pitch_type / f"{pitch_uid}_setup{frame_index}_{time_ms:04d}ms.jpg"


def extract_frame(
    row: pd.Series,
    image_path: Path,
    frame_time_sec: float,
    overwrite: bool,
    quality: int,
    ffmpeg: str,
) -> str:
    if not ffmpeg:
        return "missing_ffmpeg"
    if image_path.exists() and not overwrite:
        return "exists"

    image_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{frame_time_sec:.3f}",
        "-i",
        normalized(row.get("direct_mp4_url")),
        "-frames:v",
        "1",
        "-q:v",
        str(quality),
        str(image_path),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return f"failed: {result.stderr.strip()[:180]}"
    if not image_path.exists():
        return "failed: no output image"
    return "extracted"


def extract_frame_set(
    row: pd.Series,
    frame_jobs: list[tuple[Path, float]],
    overwrite: bool,
    quality: int,
    ffmpeg: str,
) -> list[str]:
    if not ffmpeg:
        return ["missing_ffmpeg" for _ in frame_jobs]

    statuses = []
    pending_jobs = []
    for image_path, frame_time_sec in frame_jobs:
        if image_path.exists() and not overwrite:
            statuses.append("exists")
        else:
            statuses.append("")
            pending_jobs.append((image_path, frame_time_sec))

    if not pending_jobs:
        return statuses

    for image_path, _ in pending_jobs:
        image_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        normalized(row.get("direct_mp4_url")),
    ]
    for image_path, frame_time_sec in pending_jobs:
        command.extend(
            [
                "-ss",
                f"{frame_time_sec:.3f}",
                "-frames:v",
                "1",
                "-q:v",
                str(quality),
                str(image_path),
            ]
        )

    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        failure = f"failed: {result.stderr.strip()[:180]}"
        return [status or failure for status in statuses]

    pending_index = 0
    final_statuses = []
    for status, (image_path, _) in zip(statuses, frame_jobs):
        if status:
            final_statuses.append(status)
            continue
        pending_image_path = pending_jobs[pending_index][0]
        pending_index += 1
        final_statuses.append("extracted" if pending_image_path.exists() else "failed: no output image")
    return final_statuses


def write_manifest(rows: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = args.input or latest_analysis_file()
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    df = load_rows(input_path, args.only_visible)
    if args.pitch_uid:
        df = df[df["pitch_uid"].astype(str) == str(args.pitch_uid)]
    if args.pitch_uids_file:
        pitch_uid_filter = read_pitch_uid_filter(args.pitch_uids_file)
        df = df[df["pitch_uid"].astype(str).isin(pitch_uid_filter)]
    if args.limit is not None:
        df = df.head(args.limit)
    frame_times = parse_frame_times(args)
    ffmpeg = ffmpeg_executable() if args.extract else ""

    rows = []
    for _, row in df.iterrows():
        pitch_uid = normalized(row.get("pitch_uid")) or "unknown_pitch"
        records = []
        frame_jobs = []
        for frame_index, frame_time_sec in enumerate(frame_times, start=1):
            image_path = frame_path(output_dir, row, frame_index, frame_time_sec)
            record = {column: normalized(row.get(column)) for column in MANIFEST_COLUMNS}
            record["frame_uid"] = f"{pitch_uid}_setup{frame_index}"
            record["frame_index"] = str(frame_index)
            record["frame_label"] = f"setup {frame_index}"
            record["frame_time_sec"] = f"{frame_time_sec:.3f}"
            record["image_path"] = str(image_path.relative_to(ROOT))
            record["extraction_status"] = "planned"
            records.append(record)
            frame_jobs.append((image_path, frame_time_sec))

        if args.extract:
            statuses = extract_frame_set(row, frame_jobs, args.overwrite, args.quality, ffmpeg)
            for record, status in zip(records, statuses):
                record["extraction_status"] = status

        rows.extend(records)

    write_manifest(rows, manifest_path)
    status_counts = pd.Series([row["extraction_status"] for row in rows]).value_counts().to_dict()
    print(f"Input: {input_path}")
    print(f"Wrote {manifest_path} ({len(rows)} rows)")
    print(f"Statuses: {status_counts}")


if __name__ == "__main__":
    main()

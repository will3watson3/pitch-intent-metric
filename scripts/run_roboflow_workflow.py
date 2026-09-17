import argparse
import base64
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_MANIFEST = DATA_DIR / "setup_frame_manifest.csv"
DEFAULT_OUTPUT = DATA_DIR / "roboflow_workflow_predictions.csv"
DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_DETECT_API_URL = "https://detect.roboflow.com"
WORKSPACE_NAME = "wills-workspace-0l72m"
WORKFLOW_ID = "baseball-catcher-setups-vbaseball-catcher-setups-1-rfdetr-small-t1-logic"
WORKFLOW_ENDPOINT = f"{DEFAULT_API_URL}/{WORKSPACE_NAME}/workflows/{WORKFLOW_ID}"
ZONE_CLASS_ALIASES = {"strike_zone", "strike zone", "strikezone", "zone", "tv strike zone"}
GLOVE_CLASS_ALIASES = {
    "catcher_glove",
    "catcher glove",
    "catchers glove",
    "catcher's glove",
    "glove",
    "catcher mitt",
    "mitt",
}


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
]


class RoboflowWorkflowError(RuntimeError):
    pass


class RoboflowAuthError(RoboflowWorkflowError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=None, help="Run one local image path.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DATA_DIR / "roboflow_raw_responses")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pitch-uid", default=None)
    parser.add_argument(
        "--pitch-uids-file",
        type=Path,
        default=None,
        help="CSV/text file of pitch_uid values to run. CSV files may include a pitch_uid column.",
    )
    parser.add_argument("--preferred-time-sec", type=float, default=2.0)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument(
        "--direct-model-id",
        default=None,
        help="Bypass the Roboflow Workflow and call a model directly, e.g. baseball-catcher-setups/1.",
    )
    parser.add_argument("--confidence", type=int, default=40, help="Direct model confidence threshold, 0-100.")
    parser.add_argument("--overlap", type=int, default=30, help="Direct model NMS overlap threshold, 0-100.")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N frames. Use 0 to disable progress prints.",
    )
    parser.add_argument(
        "--resume-raw",
        action="store_true",
        help="Reuse existing raw response JSON files instead of calling Roboflow again.",
    )
    return parser.parse_args()


def normalized(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def class_key(value) -> str:
    return normalized(value).strip().lower().replace("-", "_").replace(" ", "_")


def is_zone_class(value) -> bool:
    text = normalized(value).strip().lower().replace("_", " ")
    return text in ZONE_CLASS_ALIASES


def is_glove_class(value) -> bool:
    text = normalized(value).strip().lower().replace("_", " ")
    return text in GLOVE_CLASS_ALIASES


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def run_workflow_with_sdk(
    image_path: Path,
    api_key: str,
    use_cache: bool,
    timeout_sec: float,
) -> Any:
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        raise RoboflowWorkflowError(
            "inference-sdk is not installed. Run: ./.venv/bin/python -m pip install inference-sdk"
        ) from exc

    client = InferenceHTTPClient(api_url=DEFAULT_API_URL, api_key=api_key)
    # The official SDK currently handles request execution internally; the REST fallback below
    # exists so this script can still enforce a timeout when needed.
    return client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": str(image_path)},
        use_cache=use_cache,
    )


def run_direct_model_with_sdk(
    image_path: Path,
    api_key: str,
    model_id: str,
    confidence: int,
    overlap: int,
) -> Any:
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        raise RoboflowWorkflowError(
            "inference-sdk is not installed. Run: ./.venv/bin/python -m pip install inference-sdk"
        ) from exc

    client = InferenceHTTPClient(api_url=DEFAULT_DETECT_API_URL, api_key=api_key)
    return client.infer(
        str(image_path),
        model_id=model_id,
        confidence=confidence,
        overlap=overlap,
    )


def run_workflow_with_rest(
    image_path: Path,
    api_key: str,
    use_cache: bool,
    timeout_sec: float,
) -> Any:
    import requests

    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_to_base64(image_path),
            }
        },
    }
    if use_cache:
        payload["use_cache"] = True

    response = requests.post(WORKFLOW_ENDPOINT, json=payload, timeout=timeout_sec)
    if response.status_code >= 400:
        message = response.text[:400]
        if response.status_code in {401, 403}:
            raise RoboflowAuthError(
                f"Roboflow HTTP {response.status_code}: {message}. "
                "Check that ROBOFLOW_API_KEY is a private key authorized for serverless inference."
            )
        raise RoboflowWorkflowError(f"Roboflow HTTP {response.status_code}: {message}")
    return response.json()


def run_direct_model_with_rest(
    image_path: Path,
    api_key: str,
    model_id: str,
    confidence: int,
    overlap: int,
    timeout_sec: float,
) -> Any:
    import requests

    endpoint = f"{DEFAULT_DETECT_API_URL}/{model_id}"
    params = {
        "api_key": api_key,
        "confidence": confidence,
        "overlap": overlap,
    }
    response = requests.post(
        endpoint,
        params=params,
        data=image_to_base64(image_path),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout_sec,
    )
    if response.status_code >= 400:
        message = response.text[:400]
        if response.status_code in {401, 403}:
            raise RoboflowAuthError(
                f"Roboflow HTTP {response.status_code}: {message}. "
                "Check that ROBOFLOW_API_KEY can run this direct model endpoint."
            )
        raise RoboflowWorkflowError(f"Roboflow HTTP {response.status_code}: {message}")
    return response.json()


def run_with_retries(
    image_path: Path,
    api_key: str,
    use_cache: bool,
    timeout_sec: float,
    retries: int,
    direct_model_id: Optional[str] = None,
    confidence: int = 40,
    overlap: int = 30,
) -> Any:
    errors = []
    for attempt in range(retries + 1):
        try:
            try:
                if direct_model_id:
                    return run_direct_model_with_sdk(
                        image_path,
                        api_key,
                        direct_model_id,
                        confidence,
                        overlap,
                    )
                return run_workflow_with_sdk(image_path, api_key, use_cache, timeout_sec)
            except RoboflowWorkflowError as sdk_error:
                # Use REST when the SDK is not installed yet; keep other errors visible.
                if "not installed" not in str(sdk_error):
                    raise
                if direct_model_id:
                    return run_direct_model_with_rest(
                        image_path,
                        api_key,
                        direct_model_id,
                        confidence,
                        overlap,
                        timeout_sec,
                    )
                return run_workflow_with_rest(image_path, api_key, use_cache, timeout_sec)
        except Exception as exc:
            if isinstance(exc, RoboflowAuthError):
                raise
            errors.append(str(exc))
            if attempt >= retries:
                raise RoboflowWorkflowError("; ".join(errors[-3:])) from exc
            time.sleep(0.75 * (2**attempt))
    raise RoboflowWorkflowError("; ".join(errors[-3:]))


def iter_prediction_lists(value: Any, output_key: str = "") -> list[tuple[str, list[dict]]]:
    found = []
    if isinstance(value, dict):
        predictions = value.get("predictions")
        if isinstance(predictions, list) and all(isinstance(item, dict) for item in predictions):
            found.append((output_key, predictions))
        for key, child in value.items():
            child_key = key if not output_key else f"{output_key}.{key}"
            found.extend(iter_prediction_lists(child, child_key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_key = f"{output_key}[{index}]" if output_key else f"[{index}]"
            found.extend(iter_prediction_lists(child, child_key))
    return found


def prediction_confidence(prediction: dict) -> float:
    for key in ["confidence", "score"]:
        try:
            return float(prediction.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def best_prediction(predictions: list[dict], class_matcher) -> Optional[dict]:
    matches = [
        prediction
        for prediction in predictions
        if class_matcher(prediction.get("class") or prediction.get("class_name") or prediction.get("label"))
    ]
    if not matches:
        return None
    return max(matches, key=prediction_confidence)


def box_value(prediction: Optional[dict], key: str) -> Optional[float]:
    if not prediction:
        return None
    try:
        return float(prediction.get(key))
    except (TypeError, ValueError):
        return None


def target_from_boxes(zone: Optional[dict], glove: Optional[dict]) -> tuple[Optional[float], Optional[float]]:
    zone_x = box_value(zone, "x")
    zone_y = box_value(zone, "y")
    zone_width = box_value(zone, "width")
    zone_height = box_value(zone, "height")
    glove_x = box_value(glove, "x")
    glove_y = box_value(glove, "y")
    if None in {zone_x, zone_y, zone_width, zone_height, glove_x, glove_y}:
        return None, None
    if zone_width <= 0 or zone_height <= 0:
        return None, None

    zone_left = zone_x - (zone_width / 2)
    zone_top = zone_y - (zone_height / 2)
    target_x = (glove_x - zone_left) / zone_width
    target_y = 1 - ((glove_y - zone_top) / zone_height)
    return target_x, target_y


def parse_workflow_result(result: Any) -> dict:
    result_item = result[0] if isinstance(result, list) and result else result
    prediction_lists = iter_prediction_lists(result_item)
    if not prediction_lists:
        return {
            "workflow_output_key": "",
            "zone": None,
            "glove": None,
            "prediction_count": 0,
            "target_x": None,
            "target_y": None,
        }

    best = None
    for output_key, predictions in prediction_lists:
        zone = best_prediction(predictions, is_zone_class)
        glove = best_prediction(predictions, is_glove_class)
        score = (prediction_confidence(zone or {}) + prediction_confidence(glove or {}), len(predictions))
        if best is None or score > best[0]:
            best = (score, output_key, predictions, zone, glove)

    _, output_key, predictions, zone, glove = best
    target_x, target_y = target_from_boxes(zone, glove)
    return {
        "workflow_output_key": output_key,
        "zone": zone,
        "glove": glove,
        "prediction_count": len(predictions),
        "target_x": target_x,
        "target_y": target_y,
    }


def read_pitch_uid_filter(path: Optional[Path]) -> set[str]:
    if path is None:
        return set()
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


def manifest_rows(
    path: Path,
    pitch_uid: Optional[str],
    pitch_uids_file: Optional[Path],
    preferred_time_sec: float,
    all_frames: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["image_path"].notna() & (df["image_path"].astype(str) != "")]
    df = df[df["extraction_status"].isin(["extracted", "exists"])]
    if pitch_uid:
        df = df[df["pitch_uid"].astype(str) == str(pitch_uid)]
    pitch_uid_filter = read_pitch_uid_filter(pitch_uids_file)
    if pitch_uid_filter:
        df = df[df["pitch_uid"].astype(str).isin(pitch_uid_filter)]
    if all_frames:
        return df.sort_values(["sample_pitcher_name", "game_date", "pitch_uid", "frame_time_sec"])

    df = df.copy()
    df["frame_time_delta"] = (pd.to_numeric(df["frame_time_sec"], errors="coerce") - preferred_time_sec).abs()
    return (
        df.sort_values(["pitch_uid", "frame_time_delta", "frame_time_sec"])
        .groupby("pitch_uid", as_index=False)
        .first()
        .sort_values(["sample_pitcher_name", "game_date", "pitch_uid"])
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_raw_response(raw_dir: Path, frame_uid: str, result: Any) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{frame_uid}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def raw_response_path(raw_dir: Path, frame_uid: str) -> Path:
    return raw_dir / f"{frame_uid}.json"


def load_raw_response(raw_dir: Path, frame_uid: str) -> Optional[tuple[Path, Any]]:
    path = raw_response_path(raw_dir, frame_uid)
    if not path.exists():
        return None
    try:
        return path, json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def relative_path_text(path: Optional[Path]) -> str:
    if not path:
        return ""
    try:
        resolved = path.resolve()
    except (FileNotFoundError, RuntimeError):
        return normalized(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def result_record(row: dict, parsed: dict, raw_path: Optional[Path], status: str, error: str = "") -> dict:
    zone = parsed.get("zone") or {}
    glove = parsed.get("glove") or {}
    return {
        "frame_uid": normalized(row.get("frame_uid")),
        "pitch_uid": normalized(row.get("pitch_uid")),
        "image_path": normalized(row.get("image_path")),
        "frame_index": normalized(row.get("frame_index")),
        "frame_time_sec": normalized(row.get("frame_time_sec")),
        "workflow_output_key": normalized(parsed.get("workflow_output_key")),
        "zone_confidence": prediction_confidence(zone),
        "glove_confidence": prediction_confidence(glove),
        "zone_x": normalized(zone.get("x")),
        "zone_y": normalized(zone.get("y")),
        "zone_width": normalized(zone.get("width")),
        "zone_height": normalized(zone.get("height")),
        "glove_x": normalized(glove.get("x")),
        "glove_y": normalized(glove.get("y")),
        "glove_width": normalized(glove.get("width")),
        "glove_height": normalized(glove.get("height")),
        "vision_target_x_01": normalized(parsed.get("target_x")),
        "vision_target_y_01": normalized(parsed.get("target_y")),
        "prediction_count": normalized(parsed.get("prediction_count")),
        "raw_response_path": relative_path_text(raw_path),
        "status": status,
        "error": error,
    }


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY environment variable.")

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else ROOT / args.raw_dir

    if args.image:
        image_path = args.image if args.image.is_absolute() else ROOT / args.image
        frame_uid = image_path.stem
        rows = [
            {
                "frame_uid": frame_uid,
                "pitch_uid": "",
                "image_path": str(image_path.relative_to(ROOT)),
            }
        ]
    else:
        rows_df = manifest_rows(
            manifest_path,
            args.pitch_uid,
            args.pitch_uids_file,
            args.preferred_time_sec,
            args.all_frames,
        )
        if args.limit is not None:
            rows_df = rows_df.head(args.limit)
        rows = rows_df.to_dict("records")

    output_rows = []
    status_counts: dict[str, int] = {}
    last_error = ""
    for index, row in enumerate(rows, start=1):
        image_path = ROOT / normalized(row.get("image_path"))
        frame_uid = normalized(row.get("frame_uid")) or image_path.stem
        if not image_path.exists():
            record = result_record(row, {}, None, "error", "image_path does not exist")
            output_rows.append(record)
            status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
            if args.progress_every and index % args.progress_every == 0:
                print(f"Progress: {index}/{len(rows)} | statuses: {status_counts}", flush=True)
            continue
        requested_inference = False
        try:
            cached = load_raw_response(raw_dir, frame_uid) if args.resume_raw else None
            requested_inference = cached is None
            if cached:
                raw_path, result = cached
            else:
                result = run_with_retries(
                    image_path=image_path,
                    api_key=api_key,
                    use_cache=args.use_cache,
                    timeout_sec=args.timeout_sec,
                    retries=args.retries,
                    direct_model_id=args.direct_model_id,
                    confidence=args.confidence,
                    overlap=args.overlap,
                )
                raw_path = save_raw_response(raw_dir, frame_uid, result)
            parsed = parse_workflow_result(result)
            status = "ok" if parsed.get("zone") and parsed.get("glove") else "missing_required_detection"
            record = result_record(row, parsed, raw_path, status)
            output_rows.append(record)
        except Exception as exc:
            last_error = str(exc)
            record = result_record(row, {}, None, "error", str(exc))
            output_rows.append(record)
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            suffix = f" | last_error: {last_error[:240]}" if last_error else ""
            print(f"Progress: {index}/{len(rows)} | statuses: {status_counts}{suffix}", flush=True)
        if args.sleep_sec and requested_inference and index < len(rows):
            time.sleep(args.sleep_sec)

    write_csv(output_path, output_rows)
    counts = pd.Series([row["status"] for row in output_rows]).value_counts().to_dict()
    print(f"Output: {output_path}")
    print(f"Rows: {len(output_rows)}")
    print(f"Statuses: {counts}")


if __name__ == "__main__":
    main()

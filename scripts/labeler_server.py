import argparse
import csv
import json
import mimetypes
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LABELER_DIR = ROOT / "labeler"
DASHBOARD_DIR = ROOT / "dashboard"
FRAME_REVIEW_DIR = ROOT / "frame-review"
TARGET_AUDIT_DIR = ROOT / "target-audit"
INTENT_AUDIT_DIR = ROOT / "intent-audit"
SETUP_FRAMES_DIR = DATA_DIR / "setup_frames"
ZONE_CALIBRATION_PATH = DATA_DIR / "frame_zone_calibrations.json"
VISION_QUALITY_PATH = DATA_DIR / "roboflow_vision_quality_pose_drop_hybrid_zone_calibrated.csv"
VISION_REVIEW_QUEUE_PATH = DATA_DIR / "roboflow_vision_review_queue_pose_drop_hybrid_zone_calibrated.csv"
VISION_REVIEW_LABELS_PATH = DATA_DIR / "vision_review_labels.csv"
TARGET_AUDIT_LABELS_PATH = DATA_DIR / "target_audit_labels.csv"
INTENT_MODEL_AUDIT_LABELS_PATH = DATA_DIR / "intent_model_audit_labels.csv"
INTENT_PREDICTIONS_PATH = DATA_DIR / "audited_intent_pilot_predictions.csv"
INTENT_VARIANTS_PATH = ROOT / "reports" / "intent_pilot" / "variant_predictions.csv"
TARGET_AUDIT_GROUPS = [
    ("Dylan Cease", "SL", "Slider"),
    ("Tarik Skubal", "CH", "Changeup"),
]
DEFAULT_ZONE_CALIBRATION = {
    "left": 43.0,
    "top": 22.0,
    "width": 14.0,
    "height": 31.0,
}
LABEL_COLUMNS = [
    "target_x_01",
    "target_y_01",
    "target_confidence_1_to_5",
    "setup_visible",
    "label_notes",
]
ANALYSIS_PREFIX = "labeled_target_analysis_sample_"
VISION_REVIEW_LABELS = {
    "good",
    "close_not_perfect",
    "bad_blue_target",
    "wrong_glove",
    "wrong_zone",
    "wrong_frame",
    "unusable_video",
    "manual_label_questionable",
}
VISION_REVIEW_LABEL_COLUMNS = [
    "pitch_uid",
    "frame_uid",
    "review_label",
    "review_notes",
    "reviewed_at",
    "sample_pitcher_name",
    "game_date",
    "pitch_type",
    "vision_filter_status",
    "vision_review_reason",
]
TARGET_AUDIT_LABEL_COLUMNS = [
    "pitch_uid",
    "frame_uid",
    "audit_status",
    "original_target_x_01",
    "original_target_y_01",
    "audited_target_x_01",
    "audited_target_y_01",
    "audit_notes",
    "audited_at",
    "sample_pitcher_name",
    "pitch_type",
]
TARGET_AUDIT_STATUSES = {"confirmed", "corrected", "needs_review", "unusable"}
INTENT_MODEL_AUDIT_COLUMNS = [
    "pitch_uid",
    "model",
    "review_label",
    "review_notes",
    "reviewed_at",
    "sample_pitcher_name",
    "pitch_type",
]
INTENT_MODEL_REVIEW_LABELS = {"plausible", "uncertain", "wrong"}


def configured_path(env_name: str, default: Path) -> Path:
    raw_path = os.environ.get(env_name)
    if not raw_path:
        return default
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def setup_frame_dirs() -> list[Path]:
    dirs = [path.resolve() for path in DATA_DIR.glob("setup_frames*") if path.is_dir()]
    default = SETUP_FRAMES_DIR.resolve()
    if default not in dirs:
        dirs.append(default)
    return dirs


def dataset_files() -> dict:
    files = {}
    for path in sorted(DATA_DIR.glob("*labeling_queue*.csv")):
        dataset_id = path.stem
        files[dataset_id] = path
    return files


def display_name(dataset_id: str) -> str:
    return (
        dataset_id.replace("_labeling_queue_sample_25", " sample 25")
        .replace("_labeling_queue_extra_25", " extra 25")
        .replace("_", " ")
        .title()
    )


def normalize_value(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def read_rows(path: Path) -> tuple[list, list]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: normalize_value(value) for key, value in row.items()} for row in reader]
        fieldnames = list(reader.fieldnames or [])

    insert_at = fieldnames.index("catcher_target_bucket") if "catcher_target_bucket" in fieldnames else len(fieldnames)
    for column in reversed(LABEL_COLUMNS):
        if column not in fieldnames:
            fieldnames.insert(insert_at, column)
            for row in rows:
                row[column] = ""

    for row in rows:
        for column in fieldnames:
            row.setdefault(column, "")

    return rows, fieldnames


def read_csv_records(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: normalize_value(value) for key, value in row.items()} for row in reader]


def indexed_records(path: Path, key: str) -> dict:
    return {row.get(key): row for row in read_csv_records(path) if row.get(key)}


def read_vision_review_labels() -> list:
    rows = read_csv_records(VISION_REVIEW_LABELS_PATH)
    for row in rows:
        for column in VISION_REVIEW_LABEL_COLUMNS:
            row.setdefault(column, "")
    return rows


def write_vision_review_labels(rows: list) -> None:
    VISION_REVIEW_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = VISION_REVIEW_LABELS_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VISION_REVIEW_LABEL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: normalize_value(row.get(column)) for column in VISION_REVIEW_LABEL_COLUMNS})
    os.replace(temp_path, VISION_REVIEW_LABELS_PATH)


def read_target_audit_labels() -> list:
    rows = read_csv_records(TARGET_AUDIT_LABELS_PATH)
    for row in rows:
        for column in TARGET_AUDIT_LABEL_COLUMNS:
            row.setdefault(column, "")
    return rows


def write_target_audit_labels(rows: list) -> None:
    TARGET_AUDIT_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = TARGET_AUDIT_LABELS_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_AUDIT_LABEL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: normalize_value(row.get(column)) for column in TARGET_AUDIT_LABEL_COLUMNS}
            )
    os.replace(temp_path, TARGET_AUDIT_LABELS_PATH)


def read_intent_model_audit_labels() -> list:
    rows = read_csv_records(INTENT_MODEL_AUDIT_LABELS_PATH)
    for row in rows:
        for column in INTENT_MODEL_AUDIT_COLUMNS:
            row.setdefault(column, "")
    return rows


def write_intent_model_audit_labels(rows: list) -> None:
    temp_path = INTENT_MODEL_AUDIT_LABELS_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INTENT_MODEL_AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: normalize_value(row.get(column)) for column in INTENT_MODEL_AUDIT_COLUMNS}
            for row in rows
        )
    os.replace(temp_path, INTENT_MODEL_AUDIT_LABELS_PATH)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_overlay(raw_overlay) -> dict:
    overlay = {}
    if isinstance(raw_overlay, dict):
        for key in ["left", "top", "width", "height"]:
            try:
                overlay[key] = float(raw_overlay.get(key, DEFAULT_ZONE_CALIBRATION[key]))
            except (TypeError, ValueError):
                overlay[key] = DEFAULT_ZONE_CALIBRATION[key]
    else:
        overlay = DEFAULT_ZONE_CALIBRATION.copy()

    overlay["width"] = min(max(overlay["width"], 0.1), 80.0)
    overlay["height"] = min(max(overlay["height"], 0.1), 90.0)
    overlay["left"] = min(max(overlay["left"], 0.0), 100.0 - overlay["width"])
    overlay["top"] = min(max(overlay["top"], 0.0), 100.0 - overlay["height"])
    return overlay


def normalize_calibration_entry(raw_entry) -> dict:
    entry = normalize_overlay(raw_entry)
    entry["frame_override"] = isinstance(raw_entry, dict) and raw_entry.get("frame_override") is True
    if not isinstance(raw_entry, dict):
        return entry

    for key in [
        "reference_pitch_uid",
        "reference_frame_uid",
        "reference_at_bat_number",
        "reference_batter",
    ]:
        value = normalize_value(raw_entry.get(key))
        if value:
            entry[key] = value
    for key in ["reference_sz_top", "reference_sz_bot"]:
        try:
            value = float(raw_entry.get(key))
        except (TypeError, ValueError):
            continue
        entry[key] = value
    return entry


def read_zone_calibrations() -> dict:
    if not ZONE_CALIBRATION_PATH.exists():
        return {"default": DEFAULT_ZONE_CALIBRATION.copy(), "games": {}}
    try:
        payload = json.loads(ZONE_CALIBRATION_PATH.read_text())
    except json.JSONDecodeError:
        return {"default": DEFAULT_ZONE_CALIBRATION.copy(), "games": {}}

    default_overlay = normalize_overlay(payload.get("default", DEFAULT_ZONE_CALIBRATION))
    games = {}
    for game_pk, overlay in (payload.get("games") or {}).items():
        games[str(game_pk)] = normalize_calibration_entry(overlay)
    return {"default": default_overlay, "games": games}


def write_zone_calibrations(calibrations: dict) -> None:
    ZONE_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = ZONE_CALIBRATION_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(calibrations, indent=2, sort_keys=True))
    os.replace(temp_path, ZONE_CALIBRATION_PATH)


def latest_analysis_file():
    files = sorted(
        DATA_DIR.glob(f"{ANALYSIS_PREFIX}*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def dashboard_payload() -> dict:
    analysis_path = latest_analysis_file()
    pitches = read_csv_records(analysis_path) if analysis_path else []
    return {
        "analysis_file": analysis_path.name if analysis_path else "",
        "pitcher_summary": read_csv_records(DATA_DIR / "labeled_target_summary_by_pitcher.csv"),
        "pitch_type_summary": read_csv_records(DATA_DIR / "labeled_target_summary_by_pitch_type.csv"),
        "model_diagnostics": read_csv_records(DATA_DIR / "intent_model_diagnostics_by_pitch_type.csv"),
        "model_outliers": read_csv_records(DATA_DIR / "intent_model_biggest_outliers.csv"),
        "pitches": pitches,
    }


def frame_review_payload() -> dict:
    manifest_path = configured_path(
        "FRAME_REVIEW_MANIFEST",
        DATA_DIR / "setup_frame_manifest_31_1200_2700.csv",
    )
    vision_quality_path = configured_path("FRAME_REVIEW_VISION_QUALITY", VISION_QUALITY_PATH)
    vision_review_queue_path = configured_path("FRAME_REVIEW_VISION_REVIEW_QUEUE", VISION_REVIEW_QUEUE_PATH)
    frames = read_csv_records(manifest_path)
    vision_quality_by_pitch = indexed_records(vision_quality_path, "pitch_uid")
    frame_zones_path = configured_path("FRAME_REVIEW_ZONE_PREDICTIONS", DATA_DIR / "refined_broadcast_frame_zones.csv")
    zones_by_frame = indexed_records(frame_zones_path, "frame_uid")
    review_labels_by_pitch = indexed_records(VISION_REVIEW_LABELS_PATH, "pitch_uid")
    review_queue_count = sum(
        1 for row in vision_quality_by_pitch.values() if row.get("vision_in_review_queue") == "yes"
    )
    reviewed_count = len(
        {
            pitch_uid
            for pitch_uid, row in review_labels_by_pitch.items()
            if row.get("review_label")
        }
    )
    for row in frames:
        image_path = row.get("image_path", "")
        image_exists = False
        if image_path:
            resolved = (ROOT / image_path).resolve()
            if any((resolved == setup_root or setup_root in resolved.parents) for setup_root in setup_frame_dirs()) and resolved.exists():
                image_exists = True
        row["image_exists"] = "yes" if image_exists else "no"
        row["image_url"] = f"/{image_path}" if image_exists else ""

        vision = vision_quality_by_pitch.get(row.get("pitch_uid"), {})
        row["vision_prediction_frame_uid"] = vision.get("frame_uid", "")
        row["vision_anchor_frame"] = "yes" if vision.get("frame_uid") == row.get("frame_uid") else "no"
        for column in [
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
            "glove_confidence",
            "prediction_count",
            "zone_source",
            "zone_anchor_source",
            "zone_registration_source",
            "zone_x",
            "zone_y",
            "zone_width",
            "zone_height",
            "glove_source_model",
            "selected_frame_uid",
            "selected_frame_index",
            "selected_frame_time_sec",
            "glove_candidate_count",
            "plausible_glove_candidate_count",
            "glove_selection_score",
            "glove_selection_reason",
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
            "vision_target_x_01",
            "vision_target_y_01",
            "vision_manual_delta_01",
            "vision_manual_dx_01",
            "vision_manual_dy_01",
        ]:
            row[column] = vision.get(column, "")

        # A selected pitch box cannot be rendered on a different video frame.
        geometry = vision if row["vision_anchor_frame"] == "yes" else zones_by_frame.get(row.get("frame_uid"), {})
        for column in ["zone_x", "zone_y", "zone_width", "zone_height", "zone_refinement_status", "zone_edge_score", "image_width", "image_height"]:
            row[column] = geometry.get(column, "")
        if row["vision_anchor_frame"] != "yes":
            row["vision_target_x_01"] = ""
            row["vision_target_y_01"] = ""

        review_label = review_labels_by_pitch.get(row.get("pitch_uid"), {})
        row["manual_review_label"] = review_label.get("review_label", "")
        row["manual_review_notes"] = review_label.get("review_notes", "")
        row["manual_reviewed_at"] = review_label.get("reviewed_at", "")
        row["manual_review_frame_uid"] = review_label.get("frame_uid", "")

    analysis_path = latest_analysis_file()
    return {
        "manifest_file": manifest_path.name if manifest_path.exists() else "",
        "analysis_file": analysis_path.name if analysis_path else "",
        "vision_quality_file": vision_quality_path.name if vision_quality_path.exists() else "",
        "vision_review_queue_file": vision_review_queue_path.name if vision_review_queue_path.exists() else "",
        "vision_review_labels_file": VISION_REVIEW_LABELS_PATH.name if VISION_REVIEW_LABELS_PATH.exists() else "",
        "vision_review_queue_count": review_queue_count,
        "vision_reviewed_count": reviewed_count,
        "calibrations_file": ZONE_CALIBRATION_PATH.name,
        "calibrations": read_zone_calibrations(),
        "frame_count": len(frames),
        "pitch_count": len({row.get("pitch_uid") for row in frames if row.get("pitch_uid")}),
        "frames": frames,
    }


def target_audit_payload() -> dict:
    frame_payload = frame_review_payload()
    analysis_path = latest_analysis_file()
    analysis_by_pitch = indexed_records(analysis_path, "pitch_uid") if analysis_path else {}
    audit_by_pitch = {
        row.get("pitch_uid"): row for row in read_target_audit_labels() if row.get("pitch_uid")
    }

    frames = []
    group_keys = {(pitcher, pitch_type) for pitcher, pitch_type, _ in TARGET_AUDIT_GROUPS}
    for row in frame_payload["frames"]:
        if (row.get("sample_pitcher_name"), row.get("pitch_type")) not in group_keys:
            continue
        analysis = analysis_by_pitch.get(row.get("pitch_uid"), {})
        for column in ["target_confidence_1_to_5", "setup_visible", "label_notes"]:
            if column in analysis:
                row[column] = analysis.get(column, "")
        audit = audit_by_pitch.get(row.get("pitch_uid"), {})
        for column in TARGET_AUDIT_LABEL_COLUMNS:
            row[f"saved_{column}"] = audit.get(column, "")
        frames.append(row)

    pitch_uids = {row.get("pitch_uid") for row in frames if row.get("pitch_uid")}
    reviewed = {
        row.get("pitch_uid")
        for row in audit_by_pitch.values()
        if row.get("audit_status") and row.get("pitch_uid") in pitch_uids
    }
    usable = {
        row.get("pitch_uid")
        for row in frames
        if row.get("setup_visible", "").lower() == "yes"
        and float(row.get("target_confidence_1_to_5") or 0) >= 4
    }
    return {
        "cohort_label": " + ".join(
            f"{pitcher} {pitch_name}" for pitcher, _, pitch_name in TARGET_AUDIT_GROUPS
        ),
        "groups": [
            {"pitcher": pitcher, "pitch_type": pitch_type, "pitch_name": pitch_name}
            for pitcher, pitch_type, pitch_name in TARGET_AUDIT_GROUPS
        ],
        "pitch_count": len(pitch_uids),
        "usable_pitch_count": len(usable),
        "reviewed_pitch_count": len(reviewed),
        "labels_file": TARGET_AUDIT_LABELS_PATH.name,
        "calibrations": frame_payload["calibrations"],
        "frames": frames,
    }


def numeric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def intent_audit_payload() -> dict:
    pitches = read_csv_records(INTENT_PREDICTIONS_PATH)
    variants = read_csv_records(INTENT_VARIANTS_PATH)
    vision_by_pitch = indexed_records(VISION_QUALITY_PATH, "pitch_uid")
    reviews = {
        (row.get("pitch_uid"), row.get("model")): row
        for row in read_intent_model_audit_labels()
        if row.get("pitch_uid") and row.get("model")
    }
    variant_by_pitch = {}
    for row in variants:
        variant_by_pitch.setdefault(row.get("pitch_uid"), {})[row.get("variant")] = row

    ordered = sorted(
        pitches,
        key=lambda row: (
            row.get("sample_pitcher_name", ""),
            row.get("game_date", ""),
            numeric(row.get("at_bat_number")) or 0,
            numeric(row.get("pitch_number")) or 0,
        ),
    )
    output = []
    for pitch in ordered:
        pitch_uid = pitch.get("pitch_uid")
        vision = vision_by_pitch.get(pitch_uid, {})
        models = {}
        for model in ["comparables", "no_outcome_weight", "no_shape_weight"]:
            estimate = dict(variant_by_pitch.get(pitch_uid, {}).get(model, {}))
            estimate["review"] = reviews.get((pitch_uid, model), {})
            models[model] = estimate

        models["glove_only"] = {
            "variant": "glove_only",
            "status": "baseline",
            "intent_x_01": pitch.get("model_target_x_01", ""),
            "intent_y_01": pitch.get("model_target_y_01", ""),
            "review": reviews.get((pitch_uid, "glove_only"), {}),
        }

        earlier = [
            row for row in ordered
            if row.get("sample_pitcher_name") == pitch.get("sample_pitcher_name")
            and row.get("pitch_type") == pitch.get("pitch_type")
            and row.get("game_date", "") < pitch.get("game_date", "")
        ]
        offsets_x = [numeric(row.get("model_miss_x_01")) for row in earlier]
        offsets_y = [numeric(row.get("model_miss_y_01")) for row in earlier]
        offsets_x = [value for value in offsets_x if value is not None]
        offsets_y = [value for value in offsets_y if value is not None]
        if offsets_x and len(offsets_x) == len(offsets_y):
            models["past_average_offset"] = {
                "variant": "past_average_offset",
                "status": "baseline",
                "intent_x_01": (numeric(pitch.get("model_target_x_01")) or 0)
                + sum(offsets_x) / len(offsets_x),
                "intent_y_01": (numeric(pitch.get("model_target_y_01")) or 0)
                + sum(offsets_y) / len(offsets_y),
                "prior_group_n": len(offsets_x),
                "review": reviews.get((pitch_uid, "past_average_offset"), {}),
            }
        else:
            models["past_average_offset"] = {
                "variant": "past_average_offset",
                "status": "insufficient_history",
                "review": reviews.get((pitch_uid, "past_average_offset"), {}),
            }

        base_columns = [
            "pitch_uid", "sample_pitcher_name", "pitch_type", "pitch_name", "game_date",
            "game_pk", "at_bat_number", "pitch_number", "stand", "balls", "strikes",
            "inning", "inning_topbot", "description", "events", "direct_mp4_url",
            "savant_video_url", "model_target_x_01", "model_target_y_01", "actual_x_01",
            "actual_y_01", "plate_x", "plate_z", "sz_top", "sz_bot", "release_speed",
            "release_spin_rate", "pfx_x", "pfx_z", "audit_status", "audit_frame_uid",
        ]
        result = {column: pitch.get(column, "") for column in base_columns}
        result.update(
            detector_target_x_01=vision.get("vision_target_x_01", ""),
            detector_target_y_01=vision.get("vision_target_y_01", ""),
            detector_status=vision.get("vision_filter_status", ""),
            models=models,
        )
        output.append(result)

    reviewed_keys = {
        (row.get("pitch_uid"), row.get("model"))
        for row in reviews.values()
        if row.get("review_label")
    }
    return {
        "pitch_count": len(output),
        "review_count": len(reviewed_keys),
        "models": [
            {"id": "no_outcome_weight", "label": "No outcome weighting"},
            {"id": "comparables", "label": "Full comparables"},
            {"id": "no_shape_weight", "label": "No shape weighting"},
            {"id": "glove_only", "label": "Glove only"},
            {"id": "past_average_offset", "label": "Past average offset"},
        ],
        "pitches": output,
    }


def write_rows(path: Path, rows: list, fieldnames: list) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def dataset_summary(dataset_id: str, path: Path) -> dict:
    rows, _ = read_rows(path)
    labeled = sum(1 for row in rows if row.get("target_x_01") and row.get("target_y_01"))
    with_mp4 = sum(1 for row in rows if row.get("direct_mp4_url"))
    return {
        "id": dataset_id,
        "name": display_name(dataset_id),
        "total": len(rows),
        "labeled_continuous": labeled,
        "with_mp4": with_mp4,
    }


class LabelerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text, status=400):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"ok": True})
            return
        if parsed.path == "/api/datasets":
            files = dataset_files()
            self.send_json({"datasets": [dataset_summary(key, value) for key, value in files.items()]})
            return
        if parsed.path == "/api/pitches":
            params = parse_qs(parsed.query)
            dataset_id = params.get("dataset", [""])[0]
            files = dataset_files()
            path = files.get(dataset_id)
            if path is None:
                self.send_text("Unknown dataset", 404)
                return
            rows, fieldnames = read_rows(path)
            if any(column in LABEL_COLUMNS for column in fieldnames):
                write_rows(path, rows, fieldnames)
            self.send_json({"dataset": dataset_summary(dataset_id, path), "pitches": rows})
            return
        if parsed.path == "/api/dashboard":
            self.send_json(dashboard_payload())
            return
        if parsed.path == "/api/frame-review":
            self.send_json(frame_review_payload())
            return
        if parsed.path == "/api/target-audit":
            self.send_json(target_audit_payload())
            return
        if parsed.path == "/api/intent-audit":
            self.send_json(intent_audit_payload())
            return
        self.serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        request_path = parsed.path.rstrip("/") or "/"
        if request_path != "/api/label":
            if request_path == "/api/frame-calibration":
                self.save_frame_calibration()
                return
            if request_path == "/api/vision-review-label":
                self.save_vision_review_label()
                return
            if request_path == "/api/target-audit-label":
                self.save_target_audit_label()
                return
            if request_path == "/api/intent-audit-label":
                self.save_intent_audit_label()
                return
            self.send_text("Not found", 404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(content_length) or b"{}")
        except json.JSONDecodeError:
            self.send_text("Invalid JSON", 400)
            return

        dataset_id = payload.get("dataset")
        pitch_uid = payload.get("pitch_uid")
        files = dataset_files()
        path = files.get(dataset_id)
        if path is None:
            self.send_text("Unknown dataset", 404)
            return
        if not pitch_uid:
            self.send_text("Missing pitch_uid", 400)
            return

        rows, fieldnames = read_rows(path)
        updated = None
        for row in rows:
            if row.get("pitch_uid") == pitch_uid:
                for column in LABEL_COLUMNS:
                    if column in payload:
                        row[column] = normalize_value(payload[column])
                updated = row
                break

        if updated is None:
            self.send_text("Pitch not found", 404)
            return

        write_rows(path, rows, fieldnames)
        self.send_json({"ok": True, "pitch": updated, "dataset": dataset_summary(dataset_id, path)})

    def save_frame_calibration(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(content_length) or b"{}")
        except json.JSONDecodeError:
            self.send_text("Invalid JSON", 400)
            return

        scope = payload.get("scope")
        overlay = normalize_overlay(payload.get("overlay"))
        calibrations = read_zone_calibrations()
        if scope == "default":
            calibrations["default"] = overlay
        elif scope == "game":
            game_pk = normalize_value(payload.get("game_pk"))
            if not game_pk:
                self.send_text("Missing game_pk", 400)
                return
            reference = payload.get("reference") if isinstance(payload.get("reference"), dict) else {}
            calibration = dict(overlay)
            calibration["frame_override"] = True
            for source_key, output_key in [
                ("pitch_uid", "reference_pitch_uid"),
                ("frame_uid", "reference_frame_uid"),
                ("at_bat_number", "reference_at_bat_number"),
                ("batter", "reference_batter"),
            ]:
                value = normalize_value(reference.get(source_key))
                if value:
                    calibration[output_key] = value
            for source_key, output_key in [
                ("sz_top", "reference_sz_top"),
                ("sz_bot", "reference_sz_bot"),
            ]:
                try:
                    value = float(reference.get(source_key))
                except (TypeError, ValueError):
                    continue
                calibration[output_key] = value
            calibrations.setdefault("games", {})[game_pk] = calibration
        else:
            self.send_text("Unknown calibration scope", 400)
            return

        write_zone_calibrations(calibrations)
        self.send_json({"ok": True, "scope": scope, "calibrations": calibrations})

    def save_vision_review_label(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(content_length) or b"{}")
        except json.JSONDecodeError:
            self.send_text("Invalid JSON", 400)
            return

        pitch_uid = normalize_value(payload.get("pitch_uid"))
        review_label = normalize_value(payload.get("review_label"))
        if not pitch_uid:
            self.send_text("Missing pitch_uid", 400)
            return
        if review_label not in VISION_REVIEW_LABELS:
            self.send_text("Unknown review_label", 400)
            return

        rows = read_vision_review_labels()
        updated = None
        for row in rows:
            if row.get("pitch_uid") == pitch_uid:
                updated = row
                break

        if updated is None:
            updated = {column: "" for column in VISION_REVIEW_LABEL_COLUMNS}
            updated["pitch_uid"] = pitch_uid
            rows.append(updated)

        for column in [
            "frame_uid",
            "review_notes",
            "sample_pitcher_name",
            "game_date",
            "pitch_type",
            "vision_filter_status",
            "vision_review_reason",
        ]:
            if column in payload:
                updated[column] = normalize_value(payload.get(column))
        updated["review_label"] = review_label
        updated["reviewed_at"] = utc_now_iso()

        write_vision_review_labels(rows)
        self.send_json({"ok": True, "review": updated})

    def save_target_audit_label(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(content_length) or b"{}")
        except json.JSONDecodeError:
            self.send_text("Invalid JSON", 400)
            return

        pitch_uid = normalize_value(payload.get("pitch_uid"))
        audit_status = normalize_value(payload.get("audit_status"))
        if not pitch_uid:
            self.send_text("Missing pitch_uid", 400)
            return
        if audit_status not in TARGET_AUDIT_STATUSES:
            self.send_text("Unknown audit_status", 400)
            return

        rows = read_target_audit_labels()
        updated = next((row for row in rows if row.get("pitch_uid") == pitch_uid), None)
        if updated is None:
            updated = {column: "" for column in TARGET_AUDIT_LABEL_COLUMNS}
            updated["pitch_uid"] = pitch_uid
            rows.append(updated)

        for column in TARGET_AUDIT_LABEL_COLUMNS:
            if column in payload and column not in {"pitch_uid", "audited_at"}:
                updated[column] = normalize_value(payload.get(column))
        updated["audit_status"] = audit_status
        updated["audited_at"] = utc_now_iso()
        write_target_audit_labels(rows)
        self.send_json({"ok": True, "audit": updated})

    def save_intent_audit_label(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(content_length) or b"{}")
        except json.JSONDecodeError:
            self.send_text("Invalid JSON", 400)
            return
        pitch_uid = normalize_value(payload.get("pitch_uid"))
        model = normalize_value(payload.get("model"))
        review_label = normalize_value(payload.get("review_label"))
        if not pitch_uid or not model:
            self.send_text("Missing pitch_uid or model", 400)
            return
        if review_label not in INTENT_MODEL_REVIEW_LABELS:
            self.send_text("Unknown review_label", 400)
            return
        rows = read_intent_model_audit_labels()
        updated = next(
            (
                row for row in rows
                if row.get("pitch_uid") == pitch_uid and row.get("model") == model
            ),
            None,
        )
        if updated is None:
            updated = {column: "" for column in INTENT_MODEL_AUDIT_COLUMNS}
            updated.update(pitch_uid=pitch_uid, model=model)
            rows.append(updated)
        for column in ["review_notes", "sample_pitcher_name", "pitch_type"]:
            if column in payload:
                updated[column] = normalize_value(payload.get(column))
        updated["review_label"] = review_label
        updated["reviewed_at"] = utc_now_iso()
        write_intent_model_audit_labels(rows)
        self.send_json({"ok": True, "review": updated})

    def serve_static(self, request_path: str):
        if request_path in {"", "/"}:
            file_path = LABELER_DIR / "index.html"
        elif request_path.startswith("/labeler/"):
            file_path = ROOT / request_path.lstrip("/")
        elif request_path in {"/dashboard", "/dashboard/"}:
            file_path = DASHBOARD_DIR / "index.html"
        elif request_path.startswith("/dashboard/"):
            file_path = ROOT / request_path.lstrip("/")
        elif request_path in {"/frame-review", "/frame-review/"}:
            file_path = FRAME_REVIEW_DIR / "index.html"
        elif request_path.startswith("/frame-review/"):
            file_path = ROOT / request_path.lstrip("/")
        elif request_path in {"/target-audit", "/target-audit/"}:
            file_path = TARGET_AUDIT_DIR / "index.html"
        elif request_path.startswith("/target-audit/"):
            file_path = ROOT / request_path.lstrip("/")
        elif request_path in {"/intent-audit", "/intent-audit/"}:
            file_path = INTENT_AUDIT_DIR / "index.html"
        elif request_path.startswith("/intent-audit/"):
            file_path = ROOT / request_path.lstrip("/")
        elif request_path.startswith("/data/setup_frames"):
            file_path = ROOT / request_path.lstrip("/")
        else:
            self.send_text("Not found", 404)
            return

        try:
            resolved = file_path.resolve()
        except FileNotFoundError:
            self.send_text("Not found", 404)
            return

        allowed_roots = [
            LABELER_DIR.resolve(),
            DASHBOARD_DIR.resolve(),
            FRAME_REVIEW_DIR.resolve(),
            TARGET_AUDIT_DIR.resolve(),
            INTENT_AUDIT_DIR.resolve(),
        ] + setup_frame_dirs()
        if not any(root in resolved.parents for root in allowed_roots):
            self.send_text("Not found", 404)
            return
        if not resolved.exists() or not resolved.is_file():
            self.send_text("Not found", 404)
            return

        content_type, _ = mimetypes.guess_type(str(resolved))
        data = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), LabelerHandler)
    print(f"Pitch Intent Labeler running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()

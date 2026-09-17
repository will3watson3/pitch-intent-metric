import argparse
import html
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained dashboard for an autonomous pitch-intent run."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Autonomous Pitch Intent")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def relative_image_path(value, output_dir: Path) -> str:
    text = str(value or "")
    if not text or text.lower() == "nan":
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return os.path.relpath(path, output_dir)


def json_records(df: pd.DataFrame) -> str:
    clean = df.astype(object).where(pd.notna(df), None)
    return json.dumps(clean.to_dict("records"), separators=(",", ":")).replace("</", "<\\/")


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    summary_path = resolve(args.summary)
    output_path = resolve(args.output)
    if not input_path.exists():
        raise SystemExit(f"Missing autonomous dataset: {input_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing run summary: {summary_path}")

    pitches = pd.read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "image_path" in pitches.columns:
        pitches["dashboard_image_path"] = pitches["image_path"].map(
            lambda value: relative_image_path(value, output_path.parent)
        )
    else:
        pitches["dashboard_image_path"] = ""
    summary = json.loads(summary_path.read_text())
    data_json = json_records(pitches)
    summary_json = json.dumps(summary, separators=(",", ":")).replace("</", "<\\/")

    document = (
        TEMPLATE.replace("__PITCH_DATA__", data_json)
        .replace("__RUN_SUMMARY__", summary_json)
        .replace("__DASHBOARD_TITLE__", html.escape(args.title))
    )
    output_path.write_text(document)
    print(f"Dashboard: {output_path}")
    print(f"Rows: {len(pitches)}")


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__DASHBOARD_TITLE__</title>
  <style>
    :root {
      --ink: #182026;
      --muted: #66727b;
      --line: #d8dee2;
      --panel: #ffffff;
      --surface: #f4f6f7;
      --cyan: #087f8c;
      --red: #c84630;
      --amber: #d89118;
      --green: #2d7a4b;
      --blue: #2b67a3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--surface);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select { font: inherit; letter-spacing: 0; }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      color: white;
      background: #17232b;
      border-bottom: 3px solid var(--cyan);
    }
    h1 { margin: 0; font-size: 23px; }
    header p { margin: 4px 0 0; color: #b8c4ca; font-size: 13px; }
    .header-status { text-align: right; font-size: 13px; color: #d9e1e5; }
    .controls {
      display: grid;
      grid-template-columns: repeat(5, minmax(135px, 1fr)) minmax(210px, 1.35fr);
      gap: 10px;
      padding: 14px 24px;
      background: white;
      border-bottom: 1px solid var(--line);
    }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; }
    select, input {
      width: 100%;
      min-height: 38px;
      padding: 7px 9px;
      color: var(--ink);
      background: white;
      border: 1px solid #bdc7cd;
      border-radius: 4px;
    }
    .kpis {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      background: white;
      border-bottom: 1px solid var(--line);
    }
    .kpi { padding: 14px 18px; border-right: 1px solid var(--line); }
    .kpi:last-child { border-right: 0; }
    .kpi span { display: block; color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; }
    .kpi strong { display: block; margin-top: 4px; font-size: 22px; }
    .kpi small { color: var(--muted); }
    main { display: grid; gap: 14px; padding: 16px 24px 28px; }
    .overview { display: grid; grid-template-columns: minmax(360px, 0.9fr) minmax(500px, 1.4fr); gap: 14px; }
    .panel { min-width: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
    .panel-head { display: flex; align-items: center; justify-content: space-between; min-height: 44px; padding: 10px 14px; border-bottom: 1px solid var(--line); }
    .panel-head h2 { margin: 0; font-size: 15px; }
    .panel-head span { color: var(--muted); font-size: 12px; }
    .panel-actions { display: flex; align-items: center; gap: 7px; }
    .nav-button { min-height: 32px; padding: 5px 10px; color: var(--ink); background: white; border: 1px solid #bdc7cd; border-radius: 4px; cursor: pointer; }
    .nav-button:disabled { cursor: default; opacity: .45; }
    .plot-wrap { height: 430px; padding: 12px; }
    #locationPlot { width: 100%; height: 100%; display: block; }
    .selected-grid { display: grid; grid-template-columns: minmax(340px, 1.2fr) minmax(240px, 0.8fr); min-height: 430px; }
    .frame-stage { position: relative; align-self: center; margin: 12px; background: #11191e; aspect-ratio: 16 / 9; overflow: hidden; }
    .frame-stage img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .frame-stage .box { position: absolute; pointer-events: none; }
    .frame-stage .zone { border: 2px solid #f2b531; }
    .frame-stage .glove { border: 2px solid #20b7d6; }
    .frame-stage .actual-point { width: 13px; height: 13px; border: 3px solid var(--red); border-radius: 50%; transform: translate(-50%, -50%); }
    .empty-frame { display: grid; place-items: center; height: 100%; color: #aab5bb; }
    .selected-meta { padding: 14px; border-left: 1px solid var(--line); }
    .selected-meta h3 { margin: 0 0 4px; font-size: 18px; }
    .selected-meta > p { margin: 0 0 14px; color: var(--muted); font-size: 13px; }
    .meta-list { margin: 0; display: grid; grid-template-columns: 1fr auto; gap: 0; }
    .meta-list dt, .meta-list dd { margin: 0; padding: 7px 0; border-bottom: 1px solid #edf0f2; font-size: 12px; }
    .meta-list dt { color: var(--muted); }
    .meta-list dd { font-weight: 700; text-align: right; }
    .link-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    .link-row a { display: inline-flex; align-items: center; min-height: 34px; padding: 6px 10px; color: white; background: var(--blue); border-radius: 4px; text-decoration: none; font-size: 12px; font-weight: 700; }
    .table-wrap { max-height: 560px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th { position: sticky; top: 0; z-index: 1; padding: 9px 10px; color: #50606a; background: #edf1f3; border-bottom: 1px solid #cbd4d9; text-align: left; white-space: nowrap; }
    td { padding: 9px 10px; border-bottom: 1px solid #e5eaed; white-space: nowrap; }
    tbody tr { cursor: pointer; }
    tbody tr:hover, tbody tr.selected { background: #e8f3f4; }
    .tier { display: inline-block; min-width: 62px; padding: 3px 7px; border-radius: 10px; color: white; text-align: center; font-size: 11px; font-weight: 800; }
    .tier.high { background: var(--green); }
    .tier.medium { background: var(--amber); }
    .tier.low { background: var(--red); }
    .muted { color: var(--muted); }
    .target-key { color: var(--cyan); font-weight: 800; }
    .actual-key { color: var(--red); font-weight: 800; }
    @media (max-width: 1000px) {
      .controls { grid-template-columns: repeat(2, 1fr); }
      .kpis { grid-template-columns: repeat(2, 1fr); }
      .overview, .selected-grid { grid-template-columns: 1fr; }
      .selected-meta { border-left: 0; border-top: 1px solid var(--line); }
    }
    @media (max-width: 620px) {
      header { align-items: start; flex-direction: column; padding: 16px; }
      .header-status { text-align: left; }
      .controls, main { padding-left: 12px; padding-right: 12px; }
      .controls, .kpis { grid-template-columns: 1fr; }
      .overview { grid-template-columns: minmax(0, 1fr); }
      .plot-wrap { height: 350px; }
      .kpi { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <div><h1>__DASHBOARD_TITLE__</h1><p id="runMeta"></p></div>
    <div class="header-status" id="headerStatus"></div>
  </header>
  <section class="controls">
    <label>Pitcher<select id="pitcherFilter"></select></label>
    <label>Pitch Type<select id="typeFilter"></select></label>
    <label>Confidence<select id="confidenceFilter"><option>All</option><option>High</option><option>Medium</option><option>Low</option></select></label>
    <label>Vision Status<select id="statusFilter"></select></label>
    <label>Review Focus<select id="focusFilter"><option>All</option><option>Review Queue</option><option>Zone Issues</option><option>Target Issues</option><option>Missing Target</option><option>Autonomous Ready</option></select></label>
    <label>Search<input id="searchInput" type="search" placeholder="Pitch, count, result, intent"></label>
  </section>
  <section class="kpis" id="kpis"></section>
  <main>
    <section class="overview">
      <div class="panel">
        <div class="panel-head"><h2>Target vs. Actual</h2><span><b class="target-key">Target</b> / <b class="actual-key">Actual</b></span></div>
        <div class="plot-wrap"><canvas id="locationPlot"></canvas></div>
      </div>
      <div class="panel">
        <div class="panel-head"><h2>Selected Pitch</h2><div class="panel-actions"><button class="nav-button" id="previousPitch" type="button">Previous</button><span id="selectedId">-</span><button class="nav-button" id="nextPitch" type="button">Next</button></div></div>
        <div class="selected-grid">
          <div class="frame-stage" id="frameStage"></div>
          <aside class="selected-meta" id="selectedMeta"></aside>
        </div>
      </div>
    </section>
    <section class="panel">
      <div class="panel-head"><h2>Pitch Log</h2><span id="rowCount"></span></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Date</th><th>Pitch</th><th>Count</th><th>Target X/Y</th><th>Actual X/Y</th><th>Miss</th><th>Target Conf.</th><th>Zone Conf.</th><th>Overall</th><th>Vision</th><th>Result</th></tr></thead>
          <tbody id="pitchRows"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const allPitches = __PITCH_DATA__;
    const runSummary = __RUN_SUMMARY__;
    let selectedPitchUid = allPitches[0]?.pitch_uid || null;
    const $ = (selector) => document.querySelector(selector);
    const num = (value) => value === null || value === undefined || value === "" ? null : Number(value);
    const fmt = (value, digits = 2) => Number.isFinite(num(value)) ? num(value).toFixed(digits) : "-";
    const text = (value) => value === null || value === undefined || value === "" ? "-" : String(value);
    const pretty = (value) => text(value).replaceAll("_", " ");
    const unique = (values) => [...new Set(values.filter(Boolean))].sort();

    function fillSelect(node, values) {
      node.innerHTML = values.map((value) => `<option>${value}</option>`).join("");
    }
    fillSelect($("#pitcherFilter"), ["All", ...unique(allPitches.map((row) => row.sample_pitcher_name))]);
    fillSelect($("#typeFilter"), ["All", ...unique(allPitches.map((row) => row.pitch_type))]);
    fillSelect($("#statusFilter"), ["All", ...unique(allPitches.map((row) => row.vision_filter_status))]);
    $("#runMeta").textContent = `${allPitches.length} pitches | ${unique(allPitches.map((row) => row.game_pk)).length} games`;
    $("#headerStatus").textContent = `${runSummary.autonomous_ready || 0} autonomous-ready | ${runSummary.targets_created || 0} targets`;

    function filteredRows() {
      const pitcher = $("#pitcherFilter").value;
      const type = $("#typeFilter").value;
      const confidence = $("#confidenceFilter").value.toLowerCase();
      const status = $("#statusFilter").value;
      const focus = $("#focusFilter").value;
      const query = $("#searchInput").value.trim().toLowerCase();
      return allPitches.filter((row) => {
        if (pitcher !== "All" && row.sample_pitcher_name !== pitcher) return false;
        if (type !== "All" && row.pitch_type !== type) return false;
        if (confidence !== "all" && row.vision_confidence_tier !== confidence) return false;
        if (status !== "All" && row.vision_filter_status !== status) return false;
        if (focus === "Review Queue" && row.vision_in_review_queue !== "yes") return false;
        if (focus === "Zone Issues" && !["warn", "poor", "missing", "bad_reviewed"].includes(row.zone_quality_status)) return false;
        if (focus === "Target Issues" && row.target_confidence_tier !== "low") return false;
        if (focus === "Missing Target" && Number.isFinite(num(row.target_x)) && Number.isFinite(num(row.target_y))) return false;
        if (focus === "Autonomous Ready" && row.autonomous_ready !== "yes") return false;
        if (!query) return true;
        return [row.pitch_uid, row.pitch_name, row.description, row.events, row.intent_label, row.vision_confidence_reasons]
          .join(" ").toLowerCase().includes(query);
      });
    }

    function average(values) {
      const usable = values.map(num).filter(Number.isFinite);
      return usable.length ? usable.reduce((sum, value) => sum + value, 0) / usable.length : null;
    }

    function renderKpis(rows) {
      const ready = rows.filter((row) => row.autonomous_ready === "yes").length;
      const targets = rows.filter((row) => Number.isFinite(num(row.target_x)) && Number.isFinite(num(row.target_y))).length;
      const high = rows.filter((row) => row.vision_confidence_tier === "high").length;
      const avgConfidence = average(rows.map((row) => row.vision_confidence_score));
      const avgTargetConfidence = average(rows.map((row) => row.target_confidence_score));
      const avgZoneConfidence = average(rows.map((row) => row.zone_confidence_score));
      const values = [
        ["Pitches", rows.length, `${targets} targets`],
        ["Autonomous Ready", ready, `${rows.length ? Math.round(100 * ready / rows.length) : 0}% of pitches`],
        ["Target Confidence", fmt(avgTargetConfidence, 1), "mitt and setup frame"],
        ["Zone Confidence", fmt(avgZoneConfidence, 1), "broadcast box quality"],
        ["Overall Confidence", fmt(avgConfidence, 1), `${high} high-confidence pitches`],
      ];
      $("#kpis").innerHTML = values.map(([label, value, detail]) => `<div class="kpi"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join("");
    }

    function drawPlot(rows) {
      const canvas = $("#locationPlot");
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      const context = canvas.getContext("2d");
      context.scale(ratio, ratio);
      const width = rect.width;
      const height = rect.height;
      const pad = { left: 46, right: 18, top: 18, bottom: 34 };
      const domain = { x0: -0.5, x1: 1.5, y0: -0.4, y1: 1.4 };
      const x = (value) => pad.left + ((value - domain.x0) / (domain.x1 - domain.x0)) * (width - pad.left - pad.right);
      const y = (value) => height - pad.bottom - ((value - domain.y0) / (domain.y1 - domain.y0)) * (height - pad.top - pad.bottom);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#fafbfb";
      context.fillRect(0, 0, width, height);
      context.strokeStyle = "#dce2e5";
      context.lineWidth = 1;
      for (const tick of [-0.5, 0, 0.5, 1, 1.5]) {
        context.beginPath(); context.moveTo(x(tick), pad.top); context.lineTo(x(tick), height - pad.bottom); context.stroke();
      }
      for (const tick of [-0.4, 0, 0.5, 1, 1.4]) {
        context.beginPath(); context.moveTo(pad.left, y(tick)); context.lineTo(width - pad.right, y(tick)); context.stroke();
      }
      context.fillStyle = "rgba(8,127,140,.06)";
      context.fillRect(x(0), y(1), x(1) - x(0), y(0) - y(1));
      context.strokeStyle = "#34434c";
      context.lineWidth = 2;
      context.strokeRect(x(0), y(1), x(1) - x(0), y(0) - y(1));
      for (const row of rows) {
        const tx = num(row.target_x), ty = num(row.target_y), ax = num(row.actual_x_01), ay = num(row.actual_y_01);
        if (![tx, ty, ax, ay].every(Number.isFinite)) continue;
        const selected = row.pitch_uid === selectedPitchUid;
        context.strokeStyle = selected ? "rgba(24,32,38,.75)" : "rgba(90,105,114,.18)";
        context.lineWidth = selected ? 2 : 1;
        context.beginPath(); context.moveTo(x(tx), y(ty)); context.lineTo(x(ax), y(ay)); context.stroke();
        context.fillStyle = selected ? "#045d66" : "rgba(8,127,140,.62)";
        context.beginPath(); context.arc(x(tx), y(ty), selected ? 6 : 3.5, 0, Math.PI * 2); context.fill();
        context.strokeStyle = selected ? "#8f2d1d" : "rgba(200,70,48,.72)";
        context.lineWidth = selected ? 2.5 : 1.5;
        context.beginPath(); context.moveTo(x(ax) - 4, y(ay) - 4); context.lineTo(x(ax) + 4, y(ay) + 4); context.moveTo(x(ax) + 4, y(ay) - 4); context.lineTo(x(ax) - 4, y(ay) + 4); context.stroke();
      }
      context.fillStyle = "#66727b";
      context.font = "11px system-ui";
      context.textAlign = "center";
      context.fillText("Pitcher POV horizontal", (pad.left + width - pad.right) / 2, height - 8);
      context.save(); context.translate(12, (pad.top + height - pad.bottom) / 2); context.rotate(-Math.PI / 2); context.fillText("Vertical", 0, 0); context.restore();
    }

    function boxStyle(row, prefix) {
      const cx = num(row[`${prefix}_x`]), cy = num(row[`${prefix}_y`]), width = num(row[`${prefix}_width`]), height = num(row[`${prefix}_height`]);
      if (![cx, cy, width, height].every(Number.isFinite)) return "display:none";
      const imageWidth = num(row.image_width) || 1280, imageHeight = num(row.image_height) || 720;
      return `left:${100 * (cx - width / 2) / imageWidth}%;top:${100 * (cy - height / 2) / imageHeight}%;width:${100 * width / imageWidth}%;height:${100 * height / imageHeight}%`;
    }

    function actualPointStyle(row) {
      const zoneX = num(row.zone_x), zoneY = num(row.zone_y), zoneWidth = num(row.zone_width), zoneHeight = num(row.zone_height);
      const actualX = num(row.actual_x_01), actualY = num(row.actual_y_01);
      if (![zoneX, zoneY, zoneWidth, zoneHeight, actualX, actualY].every(Number.isFinite)) return "display:none";
      const imageX = zoneX - zoneWidth / 2 + actualX * zoneWidth;
      const imageY = zoneY + zoneHeight / 2 - actualY * zoneHeight;
      const imageWidth = num(row.image_width) || 1280, imageHeight = num(row.image_height) || 720;
      return `left:${100 * imageX / imageWidth}%;top:${100 * imageY / imageHeight}%`;
    }

    function renderSelected(rows) {
      let row = rows.find((item) => item.pitch_uid === selectedPitchUid);
      if (!row) { row = rows[0]; selectedPitchUid = row?.pitch_uid || null; }
      if (!row) {
        $("#selectedId").textContent = "-";
        $("#frameStage").innerHTML = `<div class="empty-frame">No pitch selected</div>`;
        $("#selectedMeta").innerHTML = "";
        return;
      }
      $("#selectedId").textContent = row.pitch_uid;
      $("#frameStage").innerHTML = row.dashboard_image_path
        ? `<img src="${row.dashboard_image_path}" alt="Selected setup frame"><div class="box zone" style="${boxStyle(row, "zone")}"></div><div class="box glove" style="${boxStyle(row, "glove")}"></div><div class="box actual-point" title="Actual pitch location" style="${actualPointStyle(row)}"></div>`
        : `<div class="empty-frame">No setup frame</div>`;
      const links = [
        row.direct_mp4_url ? `<a href="${row.direct_mp4_url}" target="_blank" rel="noreferrer">Pitch Video</a>` : "",
        row.savant_video_url ? `<a href="${row.savant_video_url}" target="_blank" rel="noreferrer">Baseball Savant</a>` : "",
      ].join("");
      $("#selectedMeta").innerHTML = `
        <h3>${text(row.sample_pitcher_name)} | ${text(row.pitch_type)}</h3>
        <p>${text(row.game_date)} | ${text(row.inning_topbot)} ${text(row.inning)} | ${text(row.balls)}-${text(row.strikes)}</p>
        <dl class="meta-list">
          <dt>Target</dt><dd>${fmt(row.target_x)}, ${fmt(row.target_y)}</dd>
          <dt>Actual</dt><dd>${fmt(row.actual_x_01)}, ${fmt(row.actual_y_01)}</dd>
          <dt>Setup miss</dt><dd>${fmt(row.setup_miss_ft)} ft</dd>
          <dt>Target confidence</dt><dd>${fmt(row.target_confidence_score, 1)} / ${text(row.target_confidence_tier)}</dd>
          <dt>Zone confidence</dt><dd>${fmt(row.zone_confidence_score, 1)} / ${text(row.zone_confidence_tier)}</dd>
          <dt>Overall confidence</dt><dd>${fmt(row.vision_confidence_score, 1)} / ${text(row.vision_confidence_tier)}</dd>
          <dt>Review reason</dt><dd>${pretty(row.vision_review_reason)}</dd>
          <dt>Setup evidence</dt><dd>${pretty(row.glove_setup_evidence)}</dd>
          <dt>Zone source</dt><dd>${pretty(row.zone_anchor_source)}</dd>
          <dt>Frame time</dt><dd>${fmt(row.selected_frame_time_sec)}s</dd>
          <dt>Intent label</dt><dd>${pretty(row.intent_label)}</dd>
          <dt>Result</dt><dd>${text(row.description || row.events)}</dd>
        </dl>
        <div class="link-row">${links}</div>`;
    }

    function renderTable(rows) {
      $("#rowCount").textContent = `${rows.length} pitches`;
      $("#pitchRows").innerHTML = rows.map((row) => `
        <tr data-pitch="${row.pitch_uid}" class="${row.pitch_uid === selectedPitchUid ? "selected" : ""}">
          <td>${text(row.game_date)}</td><td><b>${text(row.pitch_type)}</b> <span class="muted">${text(row.pitch_uid)}</span></td>
          <td>${text(row.balls)}-${text(row.strikes)}</td><td>${fmt(row.target_x)}, ${fmt(row.target_y)}</td><td>${fmt(row.actual_x_01)}, ${fmt(row.actual_y_01)}</td>
          <td>${fmt(row.setup_miss_ft)} ft</td>
          <td><span class="tier ${row.target_confidence_tier}">${fmt(row.target_confidence_score, 0)} ${text(row.target_confidence_tier)}</span></td>
          <td><span class="tier ${row.zone_confidence_tier}">${fmt(row.zone_confidence_score, 0)} ${text(row.zone_confidence_tier)}</span></td>
          <td><span class="tier ${row.vision_confidence_tier}">${fmt(row.vision_confidence_score, 0)} ${text(row.vision_confidence_tier)}</span></td>
          <td>${pretty(row.vision_filter_status)}</td><td>${text(row.description || row.events)}</td>
        </tr>`).join("");
      $("#pitchRows").querySelectorAll("tr").forEach((node) => node.addEventListener("click", () => {
        selectedPitchUid = node.dataset.pitch; render();
      }));
    }

    function render() {
      const rows = filteredRows();
      if (!rows.some((row) => row.pitch_uid === selectedPitchUid)) selectedPitchUid = rows[0]?.pitch_uid || null;
      const selectedIndex = rows.findIndex((row) => row.pitch_uid === selectedPitchUid);
      $("#previousPitch").disabled = selectedIndex <= 0;
      $("#nextPitch").disabled = selectedIndex < 0 || selectedIndex >= rows.length - 1;
      renderKpis(rows); drawPlot(rows); renderSelected(rows); renderTable(rows);
    }
    function moveSelection(direction) {
      const rows = filteredRows();
      const index = rows.findIndex((row) => row.pitch_uid === selectedPitchUid);
      const nextIndex = Math.max(0, Math.min(rows.length - 1, index + direction));
      if (rows[nextIndex]) selectedPitchUid = rows[nextIndex].pitch_uid;
      render();
    }
    ["#pitcherFilter", "#typeFilter", "#confidenceFilter", "#statusFilter", "#focusFilter"].forEach((selector) => $(selector).addEventListener("change", render));
    $("#searchInput").addEventListener("input", render);
    $("#previousPitch").addEventListener("click", () => moveSelection(-1));
    $("#nextPitch").addEventListener("click", () => moveSelection(1));
    window.addEventListener("resize", () => drawPlot(filteredRows()));
    render();
  </script>
</body>
</html>'''


if __name__ == "__main__":
    main()

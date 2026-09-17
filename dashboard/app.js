const state = {
  pitches: [],
  pitcherSummary: [],
  pitchTypeSummary: [],
  modelDiagnostics: [],
  modelOutliers: [],
  selectedPitchUid: null,
};

const els = {
  datasetMeta: document.querySelector("#datasetMeta"),
  pitcherFilter: document.querySelector("#pitcherFilter"),
  pitchTypeFilter: document.querySelector("#pitchTypeFilter"),
  familyFilter: document.querySelector("#familyFilter"),
  targetModeSelect: document.querySelector("#targetModeSelect"),
  searchInput: document.querySelector("#searchInput"),
  sortSelect: document.querySelector("#sortSelect"),
  kpiStrip: document.querySelector("#kpiStrip"),
  leaderboardTable: document.querySelector("#leaderboardTable"),
  plotGrid: document.querySelector("#plotGrid"),
  pitchVideo: document.querySelector("#pitchVideo"),
  selectedPitchTitle: document.querySelector("#selectedPitchTitle"),
  selectedCanvas: document.querySelector("#selectedCanvas"),
  selectedMeta: document.querySelector("#selectedMeta"),
  savantLink: document.querySelector("#savantLink"),
  splitTable: document.querySelector("#splitTable"),
  diagnosticsTable: document.querySelector("#diagnosticsTable"),
  outlierTable: document.querySelector("#outlierTable"),
  tableCount: document.querySelector("#tableCount"),
  pitchTableBody: document.querySelector("#pitchTableBody"),
};

const PLATE_EDGE_FT = 17 / 12 / 2;
const domain = { xMin: -0.5, xMax: 1.5, yMin: -0.25, yMax: 1.25 };
const colors = {
  "Bryan Woo": "#146c7c",
  "Dylan Cease": "#744fc6",
  "Jacob Misiorowski": "#d38b26",
  "Logan Webb": "#2f7d4e",
  "Max Fried": "#2d6fa3",
  "Shohei Ohtani": "#c44536",
  "Tarik Skubal": "#8b5e3c",
  "Zack Wheeler": "#5b6c77",
};

function num(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function fmt(value, digits = 2) {
  const parsed = num(value);
  return parsed === null ? "-" : parsed.toFixed(digits);
}

function text(value) {
  if (value === null || value === undefined || value === "" || String(value).toLowerCase() === "nan") {
    return "-";
  }
  return String(value);
}

function pretty(value) {
  return text(value).replaceAll("_", " ");
}

function tier(value) {
  return text(value).toLowerCase();
}

function deltaClass(value) {
  const parsed = num(value);
  if (parsed === null) return "delta-neutral";
  if (parsed > 0.15) return "delta-good";
  if (parsed < -0.15) return "delta-bad";
  return "delta-neutral";
}

function unique(values) {
  return [...new Set(values.filter((value) => value && value !== "-"))].sort();
}

function average(values) {
  const parsed = values.map(num).filter((value) => value !== null);
  if (!parsed.length) return null;
  return parsed.reduce((total, value) => total + value, 0) / parsed.length;
}

function median(values) {
  const parsed = values.map(num).filter((value) => value !== null).sort((a, b) => a - b);
  if (!parsed.length) return null;
  const mid = Math.floor(parsed.length / 2);
  return parsed.length % 2 ? parsed[mid] : (parsed[mid - 1] + parsed[mid]) / 2;
}

function byPitcher(rows) {
  return rows.reduce((groups, row) => {
    const pitcher = row.sample_pitcher_name || "Unknown";
    if (!groups[pitcher]) groups[pitcher] = [];
    groups[pitcher].push(row);
    return groups;
  }, {});
}

function normalizePitch(row) {
  return {
    ...row,
    miss_distance_ft_num: num(row.miss_distance_ft),
    miss_distance_num: num(row.miss_distance),
    setup_miss_ft_num: num(row.setup_miss_ft || row.miss_distance_ft),
    intent_adjusted_miss_ft_num: num(row.intent_adjusted_miss_ft),
    movement_offset_x_num: num(row.movement_offset_x),
    movement_offset_y_num: num(row.movement_offset_y),
    movement_offset_x_ft_num: num(row.movement_offset_x_ft),
    movement_offset_z_ft_num: num(row.movement_offset_z_ft),
    inferred_intent_x_num: num(row.inferred_intent_x),
    inferred_intent_y_num: num(row.inferred_intent_y),
    inferred_intent_plate_x_ft_num: num(row.inferred_intent_plate_x_ft),
    inferred_intent_plate_z_ft_num: num(row.inferred_intent_plate_z_ft),
    intent_model_group_size_num: num(row.intent_model_group_size),
    intent_model_training_size_num: num(row.intent_model_training_size),
    intent_model_confidence_num: num(row.intent_model_confidence),
    group_offset_spread_ft_num: num(row.group_offset_spread_ft),
    intent_adjustment_delta_ft_num: num(row.intent_adjustment_delta_ft),
    target_x_num: num(row.target_x),
    target_y_num: num(row.target_y),
    actual_x_num: num(row.actual_x_01),
    actual_y_num: num(row.actual_y_01),
  };
}

async function loadDashboard() {
  const response = await fetch("/api/dashboard");
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  state.pitches = data.pitches.map(normalizePitch);
  state.pitcherSummary = data.pitcher_summary;
  state.pitchTypeSummary = data.pitch_type_summary;
  state.modelDiagnostics = (data.model_diagnostics || []).map(normalizePitch);
  state.modelOutliers = (data.model_outliers || []).map(normalizePitch);
  els.datasetMeta.textContent = `${state.pitches.length} labeled pitches | ${unique(state.pitches.map((row) => row.sample_pitcher_name)).length} pitchers | ${data.analysis_file}`;
  buildFilters();
  state.selectedPitchUid = state.pitches[0]?.pitch_uid || null;
  render();
}

function buildFilters() {
  const pitchers = unique(state.pitches.map((row) => row.sample_pitcher_name));
  const pitchTypes = unique(state.pitches.map((row) => row.pitch_type));
  const families = unique(state.pitches.map((row) => row.pitch_family));
  fillSelect(els.pitcherFilter, ["All Pitchers", ...pitchers]);
  fillSelect(els.pitchTypeFilter, ["All Types", ...pitchTypes]);
  fillSelect(els.familyFilter, ["All Families", ...families]);
}

function fillSelect(select, values) {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
}

function filteredRows() {
  const pitcher = els.pitcherFilter.value;
  const pitchType = els.pitchTypeFilter.value;
  const family = els.familyFilter.value;
  const query = els.searchInput.value.trim().toLowerCase();
  let rows = state.pitches.filter((row) => {
    const pitcherOk = pitcher === "All Pitchers" || row.sample_pitcher_name === pitcher;
    const typeOk = pitchType === "All Types" || row.pitch_type === pitchType;
    const familyOk = family === "All Families" || row.pitch_family === family;
    if (!pitcherOk || !typeOk || !familyOk) return false;
    if (!query) return true;
    const haystack = [
      row.pitch_uid,
      row.sample_pitcher_name,
      row.pitch_type,
      row.pitch_family,
      row.pitch_name,
      row.intent_label,
      row.intent_model_status,
      row.intent_model_kind,
      row.intent_model_group,
      row.intent_model_confidence_tier,
      row.intent_model_warning,
      row.description,
      row.events,
      row.label_notes,
      row.setup_visible,
    ].join(" ").toLowerCase();
    return haystack.includes(query);
  });

  const sortValue = els.sortSelect.value;
  rows = [...rows].sort((a, b) => {
    if (sortValue === "setup_asc") return (a.setup_miss_ft_num ?? 999) - (b.setup_miss_ft_num ?? 999);
    if (sortValue === "intent_desc") return (b.intent_adjusted_miss_ft_num ?? -1) - (a.intent_adjusted_miss_ft_num ?? -1);
    if (sortValue === "intent_asc") return (a.intent_adjusted_miss_ft_num ?? 999) - (b.intent_adjusted_miss_ft_num ?? 999);
    if (sortValue === "date_desc") return String(b.game_date).localeCompare(String(a.game_date));
    if (sortValue === "pitcher") {
      return `${a.sample_pitcher_name}${a.game_date}${a.pitch_uid}`.localeCompare(`${b.sample_pitcher_name}${b.game_date}${b.pitch_uid}`);
    }
    return (b.setup_miss_ft_num ?? -1) - (a.setup_miss_ft_num ?? -1);
  });
  return rows;
}

function render() {
  const rows = filteredRows();
  if (!rows.some((row) => row.pitch_uid === state.selectedPitchUid)) {
    state.selectedPitchUid = rows[0]?.pitch_uid || null;
  }
  renderKpis(rows);
  renderLeaderboard(rows);
  renderPlots(rows);
  renderSelectedPitch(rows);
  renderPitchTypeSplits(rows);
  renderModelDiagnostics(rows);
  renderOutliers(rows);
  renderPitchTable(rows);
}

function renderKpis(rows) {
  const pitchers = unique(rows.map((row) => row.sample_pitcher_name));
  const pitchTypes = unique(rows.map((row) => row.pitch_type));
  const families = unique(rows.map((row) => row.pitch_family));
  const avgSetupMiss = average(rows.map((row) => row.setup_miss_ft_num));
  const medSetupMiss = median(rows.map((row) => row.setup_miss_ft_num));
  const adjustedRows = rows.filter((row) => row.intent_adjusted_miss_ft_num !== null);
  const avgAdjustedMiss = average(adjustedRows.map((row) => row.intent_adjusted_miss_ft_num));
  const pendingIntent = rows.length - adjustedRows.length;
  const avgModelConfidence = average(rows.map((row) => row.intent_model_confidence_num));
  const lowConfidence = rows.filter((row) => row.intent_model_confidence_tier === "low").length;
  const visibleRows = rows.filter((row) => row.setup_visible);
  const visibleRate = visibleRows.length
    ? rows.filter((row) => row.setup_visible === "yes").length / visibleRows.length
    : null;
  const metrics = [
    { label: "Pitches", value: rows.length, detail: `${pitchers.length} pitchers` },
    { label: "Families", value: families.length, detail: families.join(", ") || "-" },
    { label: "Setup Avg", value: `${fmt(avgSetupMiss)} ft`, detail: `${fmt(medSetupMiss)} ft median` },
    { label: "Intent Adj", value: `${fmt(avgAdjustedMiss)} ft`, detail: `${adjustedRows.length} model-ready pitches` },
    { label: "Model Conf", value: fmt(avgModelConfidence, 0), detail: `${lowConfidence} low confidence` },
    { label: "Pending Intent", value: pendingIntent, detail: "insufficient groups" },
  ];

  els.kpiStrip.innerHTML = "";
  for (const metric of metrics) {
    const node = document.createElement("div");
    node.className = "metric";
    node.innerHTML = `<span>${metric.label}</span><strong>${metric.value}</strong><small>${metric.detail}</small>`;
    els.kpiStrip.append(node);
  }
}

function leaderboardRows(rows) {
  return Object.entries(byPitcher(rows))
    .map(([pitcher, pitcherRows]) => ({
      pitcher,
      pitches: pitcherRows.length,
      avgMiss: average(pitcherRows.map((row) => row.setup_miss_ft_num)),
      medianMiss: median(pitcherRows.map((row) => row.setup_miss_ft_num)),
      avgAdjustedMiss: average(pitcherRows.map((row) => row.intent_adjusted_miss_ft_num)),
      avgConfidence: average(pitcherRows.map((row) => row.intent_model_confidence_num)),
    }))
    .sort((a, b) => (a.avgMiss ?? 999) - (b.avgMiss ?? 999));
}

function renderLeaderboard(rows) {
  const currentPitcher = els.pitcherFilter.value;
  els.leaderboardTable.innerHTML = "";
  const leaders = leaderboardRows(rows);
  if (!leaders.length) {
    els.leaderboardTable.innerHTML = `<div class="empty-state">No pitches</div>`;
    return;
  }
  leaders.forEach((entry, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `rank-row${currentPitcher === entry.pitcher ? " active" : ""}`;
    row.innerHTML = `
      <span class="rank-num">${index + 1}</span>
      <span class="rank-main"><strong>${entry.pitcher}</strong><span>${entry.pitches} pitches | ${fmt(entry.medianMiss)} ft med | ${fmt(entry.avgAdjustedMiss)} ft adj | ${fmt(entry.avgConfidence, 0)} conf</span></span>
      <span class="rank-value"><strong>${fmt(entry.avgMiss)}</strong><span>ft avg</span></span>
    `;
    row.addEventListener("click", () => {
      els.pitcherFilter.value = entry.pitcher;
      render();
    });
    els.leaderboardTable.append(row);
  });
}

function renderPlots(rows) {
  els.plotGrid.innerHTML = "";
  const groups = Object.entries(byPitcher(rows)).sort(([a], [b]) => a.localeCompare(b));
  if (!groups.length) {
    els.plotGrid.innerHTML = `<div class="empty-state">No pitches</div>`;
    return;
  }
  for (const [pitcher, pitcherRows] of groups) {
    const card = document.createElement("div");
    card.className = "plot-card";
    const title = document.createElement("h3");
    title.textContent = pitcher;
    const canvas = document.createElement("canvas");
    canvas.width = 420;
    canvas.height = 315;
    card.append(title, canvas);
    els.plotGrid.append(card);
    drawLocationPlot(canvas, pitcherRows, colors[pitcher] || "#146c7c", false);
  }
}

function drawLocationPlot(canvas, rows, color, selectedMode) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const cssWidth = Math.max(rect.width || canvas.width, 280);
  const cssHeight = selectedMode ? Math.max(rect.height || canvas.height, 190) : cssWidth * 0.75;
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  ctx.scale(dpr, dpr);

  const width = cssWidth;
  const height = cssHeight;
  const plot = { left: 30, top: 14, right: 12, bottom: 24 };
  const xToPx = (x) => plot.left + ((x - domain.xMin) / (domain.xMax - domain.xMin)) * (width - plot.left - plot.right);
  const yToPx = (y) => plot.top + ((domain.yMax - y) / (domain.yMax - domain.yMin)) * (height - plot.top - plot.bottom);
  const targetMode = els.targetModeSelect.value;
  const showSetup = targetMode === "setup" || targetMode === "both";
  const showIntent = targetMode === "intent" || targetMode === "both";

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfc";
  ctx.fillRect(0, 0, width, height);

  const zoneLeft = xToPx(0);
  const zoneRight = xToPx(1);
  const zoneTop = yToPx(1);
  const zoneBottom = yToPx(0);
  ctx.fillStyle = "#eaf1ef";
  ctx.fillRect(zoneLeft, zoneTop, zoneRight - zoneLeft, zoneBottom - zoneTop);
  ctx.strokeStyle = "#9facb3";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(zoneLeft, zoneTop, zoneRight - zoneLeft, zoneBottom - zoneTop);

  ctx.strokeStyle = "#d8e0e3";
  ctx.lineWidth = 1;
  [-0.5, 0, 0.5, 1, 1.5].forEach((x) => {
    ctx.beginPath();
    ctx.moveTo(xToPx(x), plot.top);
    ctx.lineTo(xToPx(x), height - plot.bottom);
    ctx.stroke();
  });
  [-0.25, 0, 0.5, 1, 1.25].forEach((y) => {
    ctx.beginPath();
    ctx.moveTo(plot.left, yToPx(y));
    ctx.lineTo(width - plot.right, yToPx(y));
    ctx.stroke();
  });

  if (showSetup) rows.forEach((row) => {
    if ([row.target_x_num, row.target_y_num, row.actual_x_num, row.actual_y_num].some((value) => value === null)) return;
    ctx.strokeStyle = "rgba(105, 117, 124, 0.28)";
    ctx.lineWidth = selectedMode ? 1.4 : 0.8;
    ctx.beginPath();
    ctx.moveTo(xToPx(row.target_x_num), yToPx(row.target_y_num));
    ctx.lineTo(xToPx(row.actual_x_num), yToPx(row.actual_y_num));
    ctx.stroke();
  });

  if (showIntent) rows.forEach((row) => {
    if ([row.inferred_intent_x_num, row.inferred_intent_y_num, row.actual_x_num, row.actual_y_num].some((value) => value === null)) return;
    ctx.strokeStyle = "rgba(211, 139, 38, 0.48)";
    ctx.lineWidth = selectedMode ? 1.7 : 1;
    ctx.beginPath();
    ctx.moveTo(xToPx(row.inferred_intent_x_num), yToPx(row.inferred_intent_y_num));
    ctx.lineTo(xToPx(row.actual_x_num), yToPx(row.actual_y_num));
    ctx.stroke();
  });

  rows.forEach((row) => {
    if (row.actual_x_num === null || row.actual_y_num === null) return;
    ctx.beginPath();
    ctx.arc(xToPx(row.actual_x_num), yToPx(row.actual_y_num), selectedMode ? 5 : 3.4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = selectedMode ? 0.9 : 0.76;
    ctx.fill();
    ctx.globalAlpha = 1;
  });

  if (showSetup) rows.forEach((row) => {
    if (row.target_x_num === null || row.target_y_num === null) return;
    const x = xToPx(row.target_x_num);
    const y = yToPx(row.target_y_num);
    const size = selectedMode ? 8 : 5;
    ctx.strokeStyle = "#172027";
    ctx.lineWidth = selectedMode ? 2.6 : 2;
    ctx.beginPath();
    ctx.moveTo(x - size, y - size);
    ctx.lineTo(x + size, y + size);
    ctx.moveTo(x + size, y - size);
    ctx.lineTo(x - size, y + size);
    ctx.stroke();
  });

  if (showIntent) rows.forEach((row) => {
    if (row.inferred_intent_x_num === null || row.inferred_intent_y_num === null) return;
    const x = xToPx(row.inferred_intent_x_num);
    const y = yToPx(row.inferred_intent_y_num);
    const size = selectedMode ? 8 : 5;
    ctx.strokeStyle = "#d38b26";
    ctx.fillStyle = "rgba(211, 139, 38, 0.16)";
    ctx.lineWidth = selectedMode ? 2.4 : 1.8;
    ctx.beginPath();
    ctx.moveTo(x, y - size);
    ctx.lineTo(x + size, y);
    ctx.lineTo(x, y + size);
    ctx.lineTo(x - size, y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  });

  ctx.fillStyle = "#65737d";
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "center";
  [0, 0.5, 1].forEach((x) => ctx.fillText(String(x), xToPx(x), height - 7));
  ctx.textAlign = "right";
  [0, 0.5, 1].forEach((y) => ctx.fillText(String(y), plot.left - 7, yToPx(y) + 4));

  ctx.textAlign = "left";
  ctx.font = "10px Inter, sans-serif";
  let legendX = plot.left + 2;
  const legendY = plot.top + 10;
  if (showSetup) {
    ctx.strokeStyle = "#172027";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY - 4);
    ctx.lineTo(legendX + 8, legendY + 4);
    ctx.moveTo(legendX + 8, legendY - 4);
    ctx.lineTo(legendX, legendY + 4);
    ctx.stroke();
    ctx.fillStyle = "#65737d";
    ctx.fillText("setup", legendX + 13, legendY + 4);
    legendX += 55;
  }
  if (showIntent) {
    ctx.strokeStyle = "#d38b26";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(legendX + 4, legendY - 5);
    ctx.lineTo(legendX + 9, legendY);
    ctx.lineTo(legendX + 4, legendY + 5);
    ctx.lineTo(legendX - 1, legendY);
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "#65737d";
    ctx.fillText("intent", legendX + 14, legendY + 4);
  }
}

function selectedPitch(rows) {
  return state.pitches.find((row) => row.pitch_uid === state.selectedPitchUid) || rows[0] || null;
}

function renderSelectedPitch(rows) {
  const pitch = selectedPitch(rows);
  if (!pitch) {
    els.selectedPitchTitle.textContent = "No pitch selected";
    els.pitchVideo.removeAttribute("src");
    els.selectedMeta.innerHTML = `<div class="empty-state">No pitch</div>`;
    drawLocationPlot(els.selectedCanvas, [], "#146c7c", true);
    return;
  }
  state.selectedPitchUid = pitch.pitch_uid;
  els.selectedPitchTitle.textContent = `${pitch.pitch_uid} | ${pitch.sample_pitcher_name}`;
  if (els.pitchVideo.src !== pitch.direct_mp4_url) {
    els.pitchVideo.src = pitch.direct_mp4_url || "";
  }
  els.savantLink.href = pitch.savant_video_url || "#";
  els.savantLink.style.visibility = pitch.savant_video_url ? "visible" : "hidden";
  drawLocationPlot(els.selectedCanvas, [pitch], colors[pitch.sample_pitcher_name] || "#146c7c", true);
  const count = `${text(pitch.balls)}-${text(pitch.strikes)}`;
  const result = pitch.events && pitch.events !== "-" ? pitch.events : pitch.description;
  els.selectedMeta.innerHTML = [
    ["Pitch", `${text(pitch.pitch_type)} | ${text(pitch.pitch_name)}`],
    ["Family", text(pitch.pitch_family)],
    ["Intent", text(pitch.intent_label)],
    ["Count", count],
    ["Result", text(result)],
    ["Game", `${text(pitch.game_date)} | ${text(pitch.inning_topbot)} ${text(pitch.inning)}`],
    ["Setup", `${fmt(pitch.target_x_num)}, ${fmt(pitch.target_y_num)}`],
    ["Inferred", `${fmt(pitch.inferred_intent_x_num)}, ${fmt(pitch.inferred_intent_y_num)}`],
    ["Offset", `${fmt(pitch.movement_offset_x_num)}, ${fmt(pitch.movement_offset_y_num)}`],
    ["Actual", `${fmt(pitch.actual_x_num)}, ${fmt(pitch.actual_y_num)}`],
    ["Setup Miss", `${fmt(pitch.setup_miss_ft_num)} ft`],
    ["Intent Adj", `${fmt(pitch.intent_adjusted_miss_ft_num)} ft`],
    ["Adjustment", `${fmt(pitch.intent_adjustment_delta_ft_num)} ft`],
    ["Confidence", `${fmt(pitch.intent_model_confidence_num, 0)} | ${pretty(pitch.intent_model_confidence_tier)}`],
    ["Offset Spread", `${fmt(pitch.group_offset_spread_ft_num)} ft`],
    ["Model Group", `${text(pitch.intent_model_group_size)} total | ${text(pitch.intent_model_training_size)} LOO`],
    ["Model", pretty(pitch.intent_model_kind)],
    ["Warning", pretty(pitch.intent_model_warning)],
    ["Visible", text(pitch.setup_visible)],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function renderPitchTypeSplits(rows) {
  const groups = rows.reduce((acc, row) => {
    const type = row.pitch_type || "-";
    if (!acc[type]) acc[type] = [];
    acc[type].push(row);
    return acc;
  }, {});
  const splits = Object.entries(groups)
    .map(([type, typeRows]) => ({
      type,
      pitches: typeRows.length,
      family: unique(typeRows.map((row) => row.pitch_family)).join(", "),
      avgMiss: average(typeRows.map((row) => row.setup_miss_ft_num)),
      medMiss: median(typeRows.map((row) => row.setup_miss_ft_num)),
      avgAdjustedMiss: average(typeRows.map((row) => row.intent_adjusted_miss_ft_num)),
      pendingIntent: typeRows.filter((row) => row.intent_adjusted_miss_ft_num === null).length,
      avgConfidence: average(typeRows.map((row) => row.intent_model_confidence_num)),
    }))
    .sort((a, b) => ((b.avgMiss ?? -1) - (a.avgMiss ?? -1)) || (b.pitches - a.pitches));
  const maxAvg = Math.max(...splits.map((split) => split.avgMiss || 0), 1);
  els.splitTable.innerHTML = "";
  if (!splits.length) {
    els.splitTable.innerHTML = `<div class="empty-state">No pitches</div>`;
    return;
  }
  for (const split of splits) {
    const width = Math.max(4, ((split.avgMiss || 0) / maxAvg) * 100);
    const row = document.createElement("div");
    row.className = "split-row";
    row.innerHTML = `
      <span class="split-main"><strong>${split.type}</strong><span>${split.pitches} pitches | ${fmt(split.medMiss)} ft med</span><span class="bar-track"><span class="bar-fill" style="width: ${width}%"></span></span></span>
      <span class="rank-value"><strong>${fmt(split.avgMiss)}</strong><span>setup ft</span><span>${split.family}</span><span>${split.pendingIntent ? `${split.pendingIntent} pending` : `${fmt(split.avgAdjustedMiss)} adj`}</span><span>${fmt(split.avgConfidence, 0)} conf</span></span>
    `;
    els.splitTable.append(row);
  }
}

function currentFilterAllows(row) {
  const pitcher = els.pitcherFilter.value;
  const pitchType = els.pitchTypeFilter.value;
  const family = els.familyFilter.value;
  const pitcherOk = pitcher === "All Pitchers" || row.sample_pitcher_name === pitcher;
  const typeOk = pitchType === "All Types" || row.pitch_type === pitchType;
  const familyOk = family === "All Families" || row.pitch_family === family;
  const query = els.searchInput.value.trim().toLowerCase();
  if (!pitcherOk || !typeOk || !familyOk) return false;
  if (!query) return true;
  return [
    row.pitch_uid,
    row.sample_pitcher_name,
    row.pitch_type,
    row.pitch_family,
    row.intent_model_group,
    row.intent_model_confidence_tier,
    row.intent_model_warning,
    row.description,
    row.events,
  ].join(" ").toLowerCase().includes(query);
}

function renderModelDiagnostics() {
  const rows = state.modelDiagnostics.filter(currentFilterAllows);
  if (!rows.length) {
    els.diagnosticsTable.innerHTML = `<div class="empty-state">No diagnostics</div>`;
    return;
  }
  const shown = rows
    .sort((a, b) => (a.avg_model_confidence_num ?? 999) - (b.avg_model_confidence_num ?? 999))
    .slice(0, 80);
  els.diagnosticsTable.innerHTML = `
    <table class="mini-table">
      <thead>
        <tr>
          <th>Group</th>
          <th>P</th>
          <th>Ready</th>
          <th>Setup</th>
          <th>Adj</th>
          <th>Delta</th>
          <th>Offset</th>
          <th>Spread</th>
          <th>Conf</th>
          <th>Low</th>
        </tr>
      </thead>
      <tbody>
        ${shown.map((row) => {
          const confidence = num(row.avg_model_confidence);
          const delta = num(row.avg_improvement_ft);
          return `
            <tr>
              <td><strong>${text(row.sample_pitcher_name)}</strong><br><span class="muted">${text(row.pitch_type)} | ${text(row.pitch_family)}</span></td>
              <td>${text(row.pitches)}</td>
              <td>${text(row.model_ready)}</td>
              <td>${fmt(row.avg_setup_miss_ft)}</td>
              <td>${fmt(row.avg_intent_adjusted_miss_ft)}</td>
              <td class="${deltaClass(delta)}">${fmt(delta)}</td>
              <td>${fmt(row.avg_movement_offset_x)}, ${fmt(row.avg_movement_offset_y)}</td>
              <td>${fmt(row.offset_spread_ft)}</td>
              <td><span class="confidence-pill confidence-${confidenceTierFromScore(confidence)}">${fmt(confidence, 0)}</span></td>
              <td>${text(row.low_confidence_models)}</td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;
}

function confidenceTierFromScore(score) {
  const parsed = num(score);
  if (parsed === null) return "unavailable";
  if (parsed >= 75) return "high";
  if (parsed >= 50) return "medium";
  return "low";
}

function renderOutliers() {
  const rows = state.modelOutliers.filter(currentFilterAllows);
  if (!rows.length) {
    els.outlierTable.innerHTML = `<div class="empty-state">No outliers</div>`;
    return;
  }
  const shown = rows
    .sort((a, b) => Math.abs(b.intent_adjustment_delta_ft_num ?? 0) - Math.abs(a.intent_adjustment_delta_ft_num ?? 0))
    .slice(0, 40);
  els.outlierTable.innerHTML = `
    <table class="mini-table">
      <thead>
        <tr>
          <th>Pitch</th>
          <th>Type</th>
          <th>Setup</th>
          <th>Adj</th>
          <th>Delta</th>
          <th>Conf</th>
        </tr>
      </thead>
      <tbody>
        ${shown.map((pitch) => `
          <tr tabindex="0" data-pitch-uid="${text(pitch.pitch_uid)}">
            <td><strong>${text(pitch.pitch_uid)}</strong><br><span class="muted">${text(pitch.sample_pitcher_name)}</span></td>
            <td>${text(pitch.pitch_type)}</td>
            <td>${fmt(pitch.setup_miss_ft)}</td>
            <td>${fmt(pitch.intent_adjusted_miss_ft)}</td>
            <td class="${deltaClass(pitch.intent_adjustment_delta_ft_num)}">${fmt(pitch.intent_adjustment_delta_ft_num)}</td>
            <td><span class="confidence-pill confidence-${tier(pitch.intent_model_confidence_tier)}">${fmt(pitch.intent_model_confidence, 0)}</span></td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
  els.outlierTable.querySelectorAll("tr[data-pitch-uid]").forEach((row) => {
    row.addEventListener("click", () => selectPitch(row.dataset.pitchUid));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectPitch(row.dataset.pitchUid);
      }
    });
  });
}

function renderPitchTable(rows) {
  els.tableCount.textContent = `${rows.length} pitches`;
  els.pitchTableBody.innerHTML = "";
  const shown = rows.slice(0, 200);
  for (const pitch of shown) {
    const row = document.createElement("tr");
    row.tabIndex = 0;
    row.className = pitch.pitch_uid === state.selectedPitchUid ? "selected" : "";
    const result = pitch.events && pitch.events !== "-" ? pitch.events : pitch.description;
    row.innerHTML = `
      <td><strong>${text(pitch.pitch_uid)}</strong><br><span class="muted">${text(pitch.game_date)}</span></td>
      <td>${text(pitch.sample_pitcher_name)}</td>
      <td><span class="pill">${text(pitch.pitch_type)}</span></td>
      <td><span class="pill family-${text(pitch.pitch_family)}">${text(pitch.pitch_family)}</span></td>
      <td><span class="intent-pill">${text(pitch.intent_label)}</span></td>
      <td>${text(pitch.balls)}-${text(pitch.strikes)}</td>
      <td>${text(result)}</td>
      <td>${fmt(pitch.target_x_num)}, ${fmt(pitch.target_y_num)}</td>
      <td>${fmt(pitch.inferred_intent_x_num)}, ${fmt(pitch.inferred_intent_y_num)}</td>
      <td>${fmt(pitch.actual_x_num)}, ${fmt(pitch.actual_y_num)}</td>
      <td><strong>${fmt(pitch.setup_miss_ft_num)}</strong></td>
      <td><strong>${fmt(pitch.intent_adjusted_miss_ft_num)}</strong></td>
      <td><span class="confidence-pill confidence-${tier(pitch.intent_model_confidence_tier)}">${fmt(pitch.intent_model_confidence_num, 0)}</span></td>
      <td>${text(pitch.intent_model_group_size)}</td>
      <td>${text(pitch.setup_visible)}</td>
    `;
    row.addEventListener("click", () => selectPitch(pitch.pitch_uid));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectPitch(pitch.pitch_uid);
      }
    });
    els.pitchTableBody.append(row);
  }
}

function selectPitch(pitchUid) {
  state.selectedPitchUid = pitchUid;
  render();
  document.querySelector(".video-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

["change", "input"].forEach((eventName) => {
  els.pitcherFilter.addEventListener(eventName, render);
  els.pitchTypeFilter.addEventListener(eventName, render);
  els.familyFilter.addEventListener(eventName, render);
  els.targetModeSelect.addEventListener(eventName, render);
  els.searchInput.addEventListener(eventName, render);
  els.sortSelect.addEventListener(eventName, render);
});

window.addEventListener("resize", () => renderPlots(filteredRows()));

loadDashboard().catch((error) => {
  els.datasetMeta.textContent = "Dashboard failed to load";
  els.kpiStrip.innerHTML = `<div class="empty-state">${error.message}</div>`;
});

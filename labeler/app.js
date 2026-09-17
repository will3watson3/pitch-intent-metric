const state = {
  datasets: [],
  datasetId: null,
  pitches: [],
  index: 0,
  selected: {
    x: null,
    y: null,
    confidence: "",
    visible: "",
    notes: "",
  },
  dirty: false,
};

const els = {
  datasetSelect: document.querySelector("#datasetSelect"),
  statusText: document.querySelector("#statusText"),
  progressText: document.querySelector("#progressText"),
  pitchList: document.querySelector("#pitchList"),
  prevButton: document.querySelector("#prevButton"),
  nextButton: document.querySelector("#nextButton"),
  saveButton: document.querySelector("#saveButton"),
  clearTargetButton: document.querySelector("#clearTargetButton"),
  video: document.querySelector("#pitchVideo"),
  savantLink: document.querySelector("#savantLink"),
  canvas: document.querySelector("#targetCanvas"),
  coordReadout: document.querySelector("#coordReadout"),
  targetXInput: document.querySelector("#targetXInput"),
  targetYInput: document.querySelector("#targetYInput"),
  notesInput: document.querySelector("#notesInput"),
  metaPitch: document.querySelector("#metaPitch"),
  metaCount: document.querySelector("#metaCount"),
  metaResult: document.querySelector("#metaResult"),
  metaActual: document.querySelector("#metaActual"),
  metaGame: document.querySelector("#metaGame"),
  metaIds: document.querySelector("#metaIds"),
};

const ctx = els.canvas.getContext("2d");
const PLATE_EDGE_FT = 17 / 12 / 2;
const MITT_DIAMETER_IN = 7;
const domain = { xMin: -0.5, xMax: 1.5, yMin: -0.25, yMax: 1.25 };
const plot = { left: 54, top: 28, right: 28, bottom: 46 };

function num(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function fmt(value, digits = 2) {
  const parsed = num(value);
  return parsed === null ? "-" : parsed.toFixed(digits);
}

function currentPitch() {
  return state.pitches[state.index] || null;
}

function actualCoords(pitch) {
  const plateX = num(pitch.plate_x);
  const plateZ = num(pitch.plate_z);
  const top = num(pitch.sz_top);
  const bottom = num(pitch.sz_bot);
  if ([plateX, plateZ, top, bottom].some((value) => value === null)) {
    return { x: null, y: null };
  }
  return {
    x: ((-plateX) + PLATE_EDGE_FT) / (2 * PLATE_EDGE_FT),
    y: (plateZ - bottom) / (top - bottom),
  };
}

function xToCanvas(x) {
  const width = els.canvas.width - plot.left - plot.right;
  return plot.left + ((x - domain.xMin) / (domain.xMax - domain.xMin)) * width;
}

function yToCanvas(y) {
  const height = els.canvas.height - plot.top - plot.bottom;
  return plot.top + ((domain.yMax - y) / (domain.yMax - domain.yMin)) * height;
}

function canvasToX(pixelX) {
  const width = els.canvas.width - plot.left - plot.right;
  return domain.xMin + ((pixelX - plot.left) / width) * (domain.xMax - domain.xMin);
}

function canvasToY(pixelY) {
  const height = els.canvas.height - plot.top - plot.bottom;
  return domain.yMax - ((pixelY - plot.top) / height) * (domain.yMax - domain.yMin);
}

function drawGrid() {
  const { width, height } = els.canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfc";
  ctx.fillRect(0, 0, width, height);

  const zoneLeft = xToCanvas(0);
  const zoneRight = xToCanvas(1);
  const zoneTop = yToCanvas(1);
  const zoneBottom = yToCanvas(0);

  ctx.fillStyle = "#eaf1ef";
  ctx.fillRect(zoneLeft, zoneTop, zoneRight - zoneLeft, zoneBottom - zoneTop);

  ctx.strokeStyle = "#a8b4ba";
  ctx.lineWidth = 2;
  ctx.strokeRect(zoneLeft, zoneTop, zoneRight - zoneLeft, zoneBottom - zoneTop);

  ctx.strokeStyle = "#d5dde2";
  ctx.lineWidth = 1;
  for (const x of [-0.5, 0, 0.5, 1, 1.5]) {
    ctx.beginPath();
    ctx.moveTo(xToCanvas(x), plot.top);
    ctx.lineTo(xToCanvas(x), height - plot.bottom);
    ctx.stroke();
  }
  for (const y of [-0.25, 0, 0.5, 1, 1.25]) {
    ctx.beginPath();
    ctx.moveTo(plot.left, yToCanvas(y));
    ctx.lineTo(width - plot.right, yToCanvas(y));
    ctx.stroke();
  }

  ctx.strokeStyle = "#8f9ca3";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(xToCanvas(0.5), zoneTop);
  ctx.lineTo(xToCanvas(0.5), zoneBottom);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(zoneLeft, yToCanvas(0.5));
  ctx.lineTo(zoneRight, yToCanvas(0.5));
  ctx.stroke();

  ctx.fillStyle = "#53616b";
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "center";
  for (const x of [0, 0.5, 1]) {
    ctx.fillText(String(x), xToCanvas(x), height - 18);
  }
  ctx.save();
  ctx.textAlign = "right";
  for (const y of [0, 0.5, 1]) {
    ctx.fillText(String(y), plot.left - 10, yToCanvas(y) + 4);
  }
  ctx.restore();

  const pitch = currentPitch();
  if (pitch) {
    const actual = actualCoords(pitch);
    if (actual.x !== null && actual.y !== null) {
      drawPoint(actual.x, actual.y, "#c44536", "actual", 5);
    }
  }
  if (state.selected.x !== null && state.selected.y !== null) {
    drawMittTarget(state.selected.x, state.selected.y);
  }
}

function drawPoint(x, y, color, label, radius) {
  const px = xToCanvas(x);
  const py = yToCanvas(y);
  ctx.beginPath();
  ctx.arc(px, py, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.fillStyle = color;
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(label, px + 8, py + 4);
}

function currentMittRadii() {
  const pitch = currentPitch();
  const top = num(pitch?.sz_top);
  const bottom = num(pitch?.sz_bot);
  const zoneHeightIn = top !== null && bottom !== null ? (top - bottom) * 12 : 22;
  return {
    x: MITT_DIAMETER_IN / (2 * 17),
    y: MITT_DIAMETER_IN / (2 * zoneHeightIn),
  };
}

function drawMittTarget(x, y) {
  const radii = currentMittRadii();
  const px = xToCanvas(x);
  const py = yToCanvas(y);
  const rx = Math.abs(xToCanvas(x + radii.x) - px);
  const ry = Math.abs(yToCanvas(y + radii.y) - py);

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(px - rx * 0.18, py - ry * 0.98);
  ctx.bezierCurveTo(px + rx * 0.45, py - ry * 1.05, px + rx * 0.98, py - ry * 0.55, px + rx * 0.96, py + ry * 0.08);
  ctx.bezierCurveTo(px + rx * 0.94, py + ry * 0.62, px + rx * 0.46, py + ry * 1.02, px - rx * 0.12, py + ry * 0.92);
  ctx.bezierCurveTo(px - rx * 0.76, py + ry * 0.80, px - rx * 1.04, py + ry * 0.30, px - rx * 0.88, py - ry * 0.28);
  ctx.bezierCurveTo(px - rx * 0.75, py - ry * 0.76, px - rx * 0.48, py - ry * 0.94, px - rx * 0.18, py - ry * 0.98);
  ctx.closePath();
  ctx.fillStyle = "rgba(17, 24, 32, 0.16)";
  ctx.fill();
  ctx.strokeStyle = "#111820";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  ctx.beginPath();
  ctx.ellipse(px + rx * 0.14, py - ry * 0.02, rx * 0.42, ry * 0.48, -0.18, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(17, 24, 32, 0.45)";
  ctx.lineWidth = 1.4;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(px - rx * 0.66, py + ry * 0.42);
  ctx.bezierCurveTo(px - rx * 0.38, py + ry * 0.62, px - rx * 0.02, py + ry * 0.70, px + rx * 0.34, py + ry * 0.58);
  ctx.strokeStyle = "rgba(17, 24, 32, 0.32)";
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.restore();

  drawCross(x, y, "#111820", 6, 2);
}

function drawCross(x, y, color, size = 9, lineWidth = 3) {
  const px = xToCanvas(x);
  const py = yToCanvas(y);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.moveTo(px - size, py - size);
  ctx.lineTo(px + size, py + size);
  ctx.moveTo(px + size, py - size);
  ctx.lineTo(px - size, py + size);
  ctx.stroke();
}

function setStatus(message) {
  els.statusText.textContent = message;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

async function loadDatasets() {
  const data = await api("/api/datasets");
  state.datasets = data.datasets;
  els.datasetSelect.innerHTML = "";
  for (const dataset of state.datasets) {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = `${dataset.name} (${dataset.labeled_continuous}/${dataset.total})`;
    els.datasetSelect.appendChild(option);
  }
  state.datasetId = state.datasets[0]?.id || null;
  if (state.datasetId) {
    els.datasetSelect.value = state.datasetId;
    await loadPitches(state.datasetId);
  } else {
    setStatus("No labeling queues found");
  }
}

async function loadPitches(datasetId) {
  const data = await api(`/api/pitches?dataset=${encodeURIComponent(datasetId)}`);
  state.datasetId = datasetId;
  state.pitches = data.pitches;
  state.index = firstUnlabeledIndex(state.pitches);
  state.dirty = false;
  renderAll();
}

function firstUnlabeledIndex(pitches) {
  const index = pitches.findIndex((pitch) => !hasContinuousLabel(pitch));
  return index === -1 ? 0 : index;
}

function hasContinuousLabel(pitch) {
  return num(pitch.target_x_01) !== null && num(pitch.target_y_01) !== null;
}

function renderAll() {
  renderPitchList();
  renderPitch();
  renderProgress();
}

function renderProgress() {
  const labeled = state.pitches.filter(hasContinuousLabel).length;
  els.progressText.textContent = `${labeled}/${state.pitches.length}`;
  const dataset = state.datasets.find((item) => item.id === state.datasetId);
  const name = dataset ? dataset.name : "Dataset";
  setStatus(`${name} - ${labeled} labeled`);
}

function renderPitchList() {
  els.pitchList.innerHTML = "";
  state.pitches.forEach((pitch, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "pitch-row";
    if (index === state.index) row.classList.add("current");
    if (hasContinuousLabel(pitch)) row.classList.add("done");
    row.innerHTML = `
      <span class="row-main">
        <strong>${pitch.at_bat_number}_${pitch.pitch_number} ${pitch.pitch_type || ""}</strong>
        <span>${pitch.balls}-${pitch.strikes} ${pitch.description || ""}</span>
      </span>
      <span class="row-mark" aria-hidden="true"></span>
    `;
    row.addEventListener("click", async () => {
      if (!(await maybeSaveBeforeMove())) return;
      state.index = index;
      state.dirty = false;
      renderAll();
    });
    els.pitchList.appendChild(row);
  });
}

function renderPitch() {
  const pitch = currentPitch();
  if (!pitch) {
    els.video.removeAttribute("src");
    return;
  }

  els.video.src = pitch.direct_mp4_url || "";
  els.savantLink.href = pitch.savant_video_url || "#";
  els.metaPitch.textContent = `${pitch.pitch_type || "-"} - ${pitch.pitch_name || "-"}`;
  els.metaCount.textContent = `${pitch.balls}-${pitch.strikes}`;
  els.metaResult.textContent = `${pitch.description || ""} ${pitch.events || ""}`.trim() || "-";
  const actual = actualCoords(pitch);
  els.metaActual.textContent = `x ${fmt(actual.x)}, y ${fmt(actual.y)}`;
  els.metaGame.textContent = `${pitch.game_date || "-"} ${pitch.inning_topbot || ""} ${pitch.inning || ""}`.trim();
  els.metaIds.textContent = `${pitch.pitch_uid || "-"} / ${pitch.batter || "-"} / ${pitch.fielder_2 || "-"}`;

  state.selected.x = num(pitch.target_x_01);
  state.selected.y = num(pitch.target_y_01);
  state.selected.confidence = String(pitch.target_confidence_1_to_5 || "");
  state.selected.visible = String(pitch.setup_visible || "");
  state.selected.notes = String(pitch.label_notes || "");

  els.targetXInput.value = state.selected.x === null ? "" : state.selected.x.toFixed(2);
  els.targetYInput.value = state.selected.y === null ? "" : state.selected.y.toFixed(2);
  els.notesInput.value = state.selected.notes;
  updateButtonGroups();
  updateReadout();
  drawGrid();

  els.prevButton.disabled = state.index === 0;
  els.nextButton.disabled = state.index >= state.pitches.length - 1;
}

function updateButtonGroups() {
  document.querySelectorAll("[data-confidence]").forEach((button) => {
    button.classList.toggle("active", button.dataset.confidence === state.selected.confidence);
  });
  document.querySelectorAll("[data-visible]").forEach((button) => {
    button.classList.toggle("active", button.dataset.visible === state.selected.visible);
  });
}

function updateReadout() {
  els.coordReadout.textContent = `x ${fmt(state.selected.x)}, y ${fmt(state.selected.y)}`;
}

function setTarget(x, y) {
  state.selected.x = Math.round(x * 100) / 100;
  state.selected.y = Math.round(y * 100) / 100;
  els.targetXInput.value = state.selected.x.toFixed(2);
  els.targetYInput.value = state.selected.y.toFixed(2);
  state.dirty = true;
  updateReadout();
  drawGrid();
}

async function saveCurrent() {
  const pitch = currentPitch();
  if (!pitch) return false;
  const x = num(els.targetXInput.value);
  const y = num(els.targetYInput.value);
  if (x === null || y === null) {
    setStatus("Pick a target before saving");
    return false;
  }

  const payload = {
    dataset: state.datasetId,
    pitch_uid: pitch.pitch_uid,
    target_x_01: x.toFixed(2),
    target_y_01: y.toFixed(2),
    target_confidence_1_to_5: state.selected.confidence || pitch.target_confidence_1_to_5 || "",
    setup_visible: state.selected.visible || pitch.setup_visible || "",
    label_notes: els.notesInput.value,
  };
  const result = await api("/api/label", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  Object.assign(pitch, result.pitch);
  state.dirty = false;
  renderAll();
  setStatus(`Saved ${pitch.pitch_uid}`);
  return true;
}

async function maybeSaveBeforeMove() {
  if (!state.dirty) return true;
  return saveCurrent();
}

async function move(delta) {
  if (!(await maybeSaveBeforeMove())) return;
  state.index = Math.max(0, Math.min(state.pitches.length - 1, state.index + delta));
  state.dirty = false;
  renderAll();
}

els.canvas.addEventListener("click", (event) => {
  const rect = els.canvas.getBoundingClientRect();
  const scaleX = els.canvas.width / rect.width;
  const scaleY = els.canvas.height / rect.height;
  const pixelX = (event.clientX - rect.left) * scaleX;
  const pixelY = (event.clientY - rect.top) * scaleY;
  setTarget(canvasToX(pixelX), canvasToY(pixelY));
});

els.targetXInput.addEventListener("input", () => {
  state.selected.x = num(els.targetXInput.value);
  state.dirty = true;
  updateReadout();
  drawGrid();
});

els.targetYInput.addEventListener("input", () => {
  state.selected.y = num(els.targetYInput.value);
  state.dirty = true;
  updateReadout();
  drawGrid();
});

els.notesInput.addEventListener("input", () => {
  state.selected.notes = els.notesInput.value;
  state.dirty = true;
});

document.querySelectorAll("[data-confidence]").forEach((button) => {
  button.addEventListener("click", () => {
    state.selected.confidence = button.dataset.confidence;
    state.dirty = true;
    updateButtonGroups();
  });
});

document.querySelectorAll("[data-visible]").forEach((button) => {
  button.addEventListener("click", () => {
    state.selected.visible = button.dataset.visible;
    state.dirty = true;
    updateButtonGroups();
  });
});

els.clearTargetButton.addEventListener("click", () => {
  state.selected.x = null;
  state.selected.y = null;
  els.targetXInput.value = "";
  els.targetYInput.value = "";
  state.dirty = true;
  updateReadout();
  drawGrid();
});

els.saveButton.addEventListener("click", saveCurrent);
els.prevButton.addEventListener("click", () => move(-1));
els.nextButton.addEventListener("click", () => move(1));
els.datasetSelect.addEventListener("change", async () => {
  if (!(await maybeSaveBeforeMove())) {
    els.datasetSelect.value = state.datasetId;
    return;
  }
  await loadPitches(els.datasetSelect.value);
});

window.addEventListener("keydown", async (event) => {
  if (event.target.matches("input, textarea, select")) return;
  if (event.key === "ArrowRight") {
    event.preventDefault();
    await move(1);
  }
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    await move(-1);
  }
  if (event.key === "s" || event.key === "S") {
    event.preventDefault();
    await saveCurrent();
  }
});

loadDatasets().catch((error) => {
  console.error(error);
  setStatus(`Error: ${error.message}`);
});

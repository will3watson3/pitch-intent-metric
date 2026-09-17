const state = {
  frames: [],
  pitchOrder: [],
  pitchIndex: 0,
  selectedFrameUid: null,
  showTarget: true,
  editTarget: false,
  draftTarget: null,
  calibrations: { default: { left: 43, top: 22, width: 14, height: 31 }, games: {} },
};

const els = Object.fromEntries([
  "subtitle", "progressText", "saveStatus", "progressBar", "statusFilter", "prevPitch",
  "nextPitch", "pitchPosition", "pitchIdentity", "stage", "frameStrip", "toggleTarget",
  "editTarget", "instruction", "metadata", "notes", "confirmTarget", "saveCorrection",
  "needsReview", "unusable",
].map((id) => [id, document.querySelector(`#${id}`)]));

function num(value) {
  if (value === "" || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function esc(value) {
  return String(value ?? "—")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function framesForPitch(pitchUid) {
  return state.frames.filter((row) => row.pitch_uid === pitchUid)
    .sort((a, b) => (num(a.frame_time_sec) ?? 99) - (num(b.frame_time_sec) ?? 99));
}

function auditStatus(row) { return row?.saved_audit_status || ""; }
function currentPitchUid() { return state.pitchOrder[state.pitchIndex] || null; }
function activeFrames() { return framesForPitch(currentPitchUid()); }
function activeFrame() { return activeFrames().find((row) => row.frame_uid === state.selectedFrameUid) || activeFrames()[0] || null; }

function representativeFrame(rows) {
  const audited = rows.find((row) => row.frame_uid === row.saved_frame_uid && row.image_exists === "yes");
  const selected = rows.find((row) => row.frame_uid === row.selected_frame_uid && row.image_exists === "yes");
  const available = rows.filter((row) => row.image_exists === "yes");
  let preDrop = null;
  if (selected) {
    const targetTime = (num(selected.frame_time_sec) ?? 2.2) - 0.2;
    preDrop = [...available].sort((a, b) =>
      Math.abs((num(a.frame_time_sec) ?? 9) - targetTime)
      - Math.abs((num(b.frame_time_sec) ?? 9) - targetTime)
    )[0] || null;
  }
  return audited || preDrop || selected || available
    .sort((a, b) => Math.abs((num(a.frame_time_sec) ?? 9) - 2.0) - Math.abs((num(b.frame_time_sec) ?? 9) - 2.0))[0] || rows[0];
}

function rebuildOrder(keepPitchUid = currentPitchUid()) {
  const grouped = new Map();
  for (const row of state.frames) if (!grouped.has(row.pitch_uid)) grouped.set(row.pitch_uid, row);
  let rows = [...grouped.values()];
  const filter = els.statusFilter.value;
  if (filter === "reviewed") rows = rows.filter((row) => auditStatus(row));
  if (filter === "high_quality") rows = rows.filter((row) => row.setup_visible === "yes" && (num(row.target_confidence_1_to_5) ?? 0) >= 4);
  rows.sort((a, b) => {
    if (filter === "pending") return Number(Boolean(auditStatus(a))) - Number(Boolean(auditStatus(b))) || String(a.game_date).localeCompare(String(b.game_date));
    return String(a.game_date).localeCompare(String(b.game_date)) || String(a.pitch_uid).localeCompare(String(b.pitch_uid));
  });
  state.pitchOrder = rows.map((row) => row.pitch_uid);
  state.pitchIndex = Math.max(0, state.pitchOrder.indexOf(keepPitchUid));
  if (!state.pitchOrder.includes(keepPitchUid)) state.pitchIndex = 0;
  const next = representativeFrame(activeFrames());
  state.selectedFrameUid = next?.frame_uid || null;
  state.draftTarget = null;
}

function zoneRect(frame) {
  const imageWidth = num(frame.image_width);
  const imageHeight = num(frame.image_height);
  const x = num(frame.zone_x), y = num(frame.zone_y), width = num(frame.zone_width), height = num(frame.zone_height);
  if ([imageWidth, imageHeight, x, y, width, height].every((value) => value !== null) && imageWidth > 0 && imageHeight > 0) {
    return {
      left: 100 * (x - width / 2) / imageWidth,
      top: 100 * (y - height / 2) / imageHeight,
      width: 100 * width / imageWidth,
      height: 100 * height / imageHeight,
    };
  }
  return state.calibrations.games?.[String(frame.game_pk)] || state.calibrations.default;
}

function targetFor(frame) {
  if (state.draftTarget) return state.draftTarget;
  if (frame.saved_audit_status === "corrected") {
    return { x: num(frame.saved_audited_target_x_01), y: num(frame.saved_audited_target_y_01) };
  }
  return { x: num(frame.target_x_01), y: num(frame.target_y_01) };
}

function imageContentRect(stage, image) {
  const stageBox = stage.getBoundingClientRect();
  const imageBox = image.getBoundingClientRect();
  if (!image.naturalWidth || !image.naturalHeight) {
    return {
      left: imageBox.left - stageBox.left,
      top: imageBox.top - stageBox.top,
      width: imageBox.width,
      height: imageBox.height,
    };
  }
  const scale = Math.min(
    imageBox.width / image.naturalWidth,
    imageBox.height / image.naturalHeight,
  );
  const width = image.naturalWidth * scale, height = image.naturalHeight * scale;
  return {
    left: imageBox.left - stageBox.left + (imageBox.width - width) / 2,
    top: imageBox.top - stageBox.top + (imageBox.height - height) / 2,
    width,
    height,
  };
}

function renderStage(frame) {
  els.stage.innerHTML = "";
  els.stage.classList.toggle("editing", state.editTarget);
  if (!frame?.image_url) {
    els.stage.innerHTML = '<div class="stage-message">This frame image is unavailable. Choose another frame.</div>';
    return;
  }
  const image = document.createElement("img");
  image.src = frame.image_url;
  image.alt = `Setup frame for ${frame.pitch_uid}`;
  const layer = document.createElement("div");
  layer.className = "media-layer";
  const renderOverlay = () => {
    const content = imageContentRect(els.stage, image);
    Object.assign(layer.style, { left: `${content.left}px`, top: `${content.top}px`, width: `${content.width}px`, height: `${content.height}px` });
    layer.innerHTML = "";
    const rect = zoneRect(frame);
    const zone = document.createElement("div");
    zone.className = "zone";
    Object.assign(zone.style, { left: `${rect.left}%`, top: `${rect.top}%`, width: `${rect.width}%`, height: `${rect.height}%` });
    const target = targetFor(frame);
    if (target.x !== null && target.y !== null) {
      const marker = document.createElement("div");
      marker.className = `target${state.showTarget ? "" : " hidden"}`;
      marker.style.left = `${target.x * 100}%`;
      marker.style.top = `${(1 - target.y) * 100}%`;
      marker.title = `Target ${target.x.toFixed(2)}, ${target.y.toFixed(2)}`;
      zone.append(marker);
    }
    layer.append(zone);
  };
  image.addEventListener("load", renderOverlay);
  els.stage.append(image, layer);
  requestAnimationFrame(renderOverlay);
}

function render() {
  const frame = activeFrame();
  const rows = activeFrames();
  const uniquePitches = [...new Map(state.frames.map((row) => [row.pitch_uid, row])).values()];
  const reviewed = uniquePitches.filter((row) => auditStatus(row)).length;
  els.progressText.textContent = `${reviewed} of ${uniquePitches.length} reviewed`;
  els.progressBar.style.width = `${uniquePitches.length ? 100 * reviewed / uniquePitches.length : 0}%`;
  els.pitchPosition.textContent = `Pitch ${state.pitchOrder.length ? state.pitchIndex + 1 : 0} of ${state.pitchOrder.length}`;
  els.prevPitch.disabled = state.pitchIndex <= 0;
  els.nextPitch.disabled = state.pitchIndex >= state.pitchOrder.length - 1;
  if (!frame) {
    els.pitchIdentity.textContent = "No pitches match this filter";
    renderStage(null);
    return;
  }
  const savedStatus = auditStatus(frame);
  els.pitchIdentity.innerHTML = `${esc(frame.game_date)} · ${esc(frame.balls)}-${esc(frame.strikes)} · <span class="status-${esc(savedStatus)}">${esc(savedStatus || "not reviewed")}</span>`;
  renderStage(frame);
  els.frameStrip.innerHTML = "";
  for (const row of rows) {
    if (row.image_exists !== "yes") continue;
    const button = document.createElement("button");
    button.type = "button";
    button.className = row.frame_uid === frame.frame_uid ? "active" : "";
    button.textContent = `${(num(row.frame_time_sec) ?? 0).toFixed(1)}s`;
    button.title = row.frame_uid;
    button.addEventListener("click", () => { state.selectedFrameUid = row.frame_uid; state.draftTarget = null; render(); });
    els.frameStrip.append(button);
  }
  const target = targetFor(frame);
  els.metadata.innerHTML = [
    ["Pitch", `${frame.pitch_name || frame.pitch_type} (${frame.pitch_type})`],
    ["Quality", `${frame.setup_visible || "—"} visibility · confidence ${frame.target_confidence_1_to_5 || "—"}/5`],
    ["Target", target.x === null ? "—" : `${target.x.toFixed(2)}, ${target.y.toFixed(2)}`],
    ["Result", frame.events || frame.description || "—"],
    ["Batter side", frame.stand || "—"],
    ["Frame", `${(num(frame.frame_time_sec) ?? 0).toFixed(2)} seconds`],
    ["Previous audit", savedStatus || "not reviewed"],
  ].map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join("");
  els.notes.value = frame.saved_audit_notes || "";
  els.toggleTarget.textContent = `${state.showTarget ? "Hide" : "Show"} target (T)`;
  els.editTarget.textContent = state.editTarget ? "Stop adjusting (E)" : "Adjust target (E)";
  els.saveCorrection.disabled = !state.draftTarget;
  els.instruction.textContent = state.editTarget
    ? "Click the center of the catcher’s presented glove. The corrected marker can fall outside the strike zone."
    : "Toggle the marker off to find the glove yourself, then turn it on and compare.";
}

function navigatePitch(delta) {
  const next = state.pitchIndex + delta;
  if (next < 0 || next >= state.pitchOrder.length) return;
  state.pitchIndex = next;
  state.selectedFrameUid = representativeFrame(activeFrames())?.frame_uid || null;
  state.draftTarget = null;
  state.editTarget = false;
  render();
}

function navigateFrame(delta) {
  const rows = activeFrames().filter((row) => row.image_exists === "yes");
  const index = rows.findIndex((row) => row.frame_uid === state.selectedFrameUid);
  const next = rows[index + delta];
  if (!next) return;
  state.selectedFrameUid = next.frame_uid;
  state.draftTarget = null;
  render();
}

async function saveAudit(status) {
  const frame = activeFrame();
  if (!frame) return;
  const target = state.draftTarget || targetFor(frame);
  const payload = {
    pitch_uid: frame.pitch_uid,
    frame_uid: frame.frame_uid,
    audit_status: status,
    original_target_x_01: frame.target_x_01 || "",
    original_target_y_01: frame.target_y_01 || "",
    audited_target_x_01: target.x ?? "",
    audited_target_y_01: target.y ?? "",
    audit_notes: els.notes.value.trim(),
    sample_pitcher_name: frame.sample_pitcher_name || "",
    pitch_type: frame.pitch_type || "",
  };
  els.saveStatus.textContent = "Saving…";
  for (const button of document.querySelectorAll("button")) button.disabled = true;
  try {
    const response = await fetch("/api/target-audit-label", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    for (const row of state.frames.filter((row) => row.pitch_uid === frame.pitch_uid)) {
      for (const [key, value] of Object.entries(data.audit || {})) row[`saved_${key}`] = value;
    }
    els.saveStatus.textContent = `${status.replaceAll("_", " ")} saved`;
    state.draftTarget = null;
    state.editTarget = false;
    rebuildOrder(frame.pitch_uid);
    if (els.statusFilter.value === "pending") {
      const nextPending = state.pitchOrder.findIndex((pitchUid) => !auditStatus(framesForPitch(pitchUid)[0]));
      if (nextPending >= 0) state.pitchIndex = nextPending;
    }
    state.selectedFrameUid = representativeFrame(activeFrames())?.frame_uid || null;
  } catch (error) {
    els.saveStatus.textContent = `Save failed: ${error.message}`;
  } finally {
    for (const button of document.querySelectorAll("button")) button.disabled = false;
    render();
  }
}

els.stage.addEventListener("click", (event) => {
  if (!state.editTarget) return;
  const frame = activeFrame();
  const image = els.stage.querySelector("img");
  if (!frame || !image) return;
  const stageBox = els.stage.getBoundingClientRect();
  const content = imageContentRect(els.stage, image);
  const zone = zoneRect(frame);
  const clickX = event.clientX - stageBox.left - content.left;
  const clickY = event.clientY - stageBox.top - content.top;
  const xPct = 100 * clickX / content.width;
  const yPct = 100 * clickY / content.height;
  state.draftTarget = {
    x: Math.max(-2, Math.min(3, (xPct - zone.left) / zone.width)),
    y: Math.max(-2, Math.min(3, 1 - (yPct - zone.top) / zone.height)),
  };
  state.showTarget = true;
  render();
});

els.prevPitch.addEventListener("click", () => navigatePitch(-1));
els.nextPitch.addEventListener("click", () => navigatePitch(1));
els.toggleTarget.addEventListener("click", () => { state.showTarget = !state.showTarget; render(); });
els.editTarget.addEventListener("click", () => { state.editTarget = !state.editTarget; state.showTarget = true; render(); });
els.confirmTarget.addEventListener("click", () => saveAudit("confirmed"));
els.saveCorrection.addEventListener("click", () => saveAudit("corrected"));
els.needsReview.addEventListener("click", () => saveAudit("needs_review"));
els.unusable.addEventListener("click", () => saveAudit("unusable"));
els.statusFilter.addEventListener("change", () => { rebuildOrder(); render(); });

window.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  if (event.key.toLowerCase() === "t") { state.showTarget = !state.showTarget; render(); }
  else if (event.key.toLowerCase() === "e") { state.editTarget = !state.editTarget; state.showTarget = true; render(); }
  else if (event.key === "ArrowLeft") navigatePitch(-1);
  else if (event.key === "ArrowRight") navigatePitch(1);
  else if (event.key === "[") navigateFrame(-1);
  else if (event.key === "]") navigateFrame(1);
  else if (event.key === "Enter" && !state.draftTarget) saveAudit("confirmed");
});

async function load() {
  const response = await fetch("/api/target-audit");
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  state.frames = data.frames || [];
  state.calibrations = data.calibrations || state.calibrations;
  els.subtitle.textContent = `${data.cohort_label} · ${data.usable_pitch_count} high-quality targets from ${data.pitch_count} pitches`;
  rebuildOrder(null);
  render();
}

load().catch((error) => {
  els.subtitle.textContent = `Could not load audit: ${error.message}`;
  els.stage.innerHTML = `<div class="stage-message">${esc(error.message)}</div>`;
});

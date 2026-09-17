const state = {
  frames: [],
  selectedPitchUid: null,
  selectedFrameUid: null,
  galleryLimit: 48,
  showZone: true,
  editZone: false,
  showTarget: false,
  overlayDirty: false,
  calibrations: {
    default: {
      left: 43,
      top: 22,
      width: 14,
      height: 31,
    },
    games: {},
  },
  overlay: {
    left: 43,
    top: 22,
    width: 14,
    height: 31,
  },
};

const els = {
  statusText: document.querySelector("#statusText"),
  pitcherFilter: document.querySelector("#pitcherFilter"),
  pitchTypeFilter: document.querySelector("#pitchTypeFilter"),
  familyFilter: document.querySelector("#familyFilter"),
  visibilityFilter: document.querySelector("#visibilityFilter"),
  statusFilter: document.querySelector("#statusFilter"),
  visionReviewFilter: document.querySelector("#visionReviewFilter"),
  visionStatusFilter: document.querySelector("#visionStatusFilter"),
  searchInput: document.querySelector("#searchInput"),
  sortSelect: document.querySelector("#sortSelect"),
  overlayReadout: document.querySelector("#overlayReadout"),
  overlayLeft: document.querySelector("#overlayLeft"),
  overlayTop: document.querySelector("#overlayTop"),
  overlayWidth: document.querySelector("#overlayWidth"),
  overlayHeight: document.querySelector("#overlayHeight"),
  showZoneToggle: document.querySelector("#showZoneToggle"),
  editZoneToggle: document.querySelector("#editZoneToggle"),
  showTargetToggle: document.querySelector("#showTargetToggle"),
  overlayScopeSelect: document.querySelector("#overlayScopeSelect"),
  resetOverlayButton: document.querySelector("#resetOverlayButton"),
  saveOverlayButton: document.querySelector("#saveOverlayButton"),
  nextCalibrationButton: document.querySelector("#nextCalibrationButton"),
  kpiStrip: document.querySelector("#kpiStrip"),
  selectedTitle: document.querySelector("#selectedTitle"),
  prevPitchButton: document.querySelector("#prevPitchButton"),
  nextPitchButton: document.querySelector("#nextPitchButton"),
  nextReviewButton: document.querySelector("#nextReviewButton"),
  selectedStage: document.querySelector("#selectedStage"),
  frameButtons: document.querySelector("#frameButtons"),
  selectedMeta: document.querySelector("#selectedMeta"),
  reviewButtons: document.querySelector("#reviewButtons"),
  reviewLabelStatus: document.querySelector("#reviewLabelStatus"),
  reviewNotesInput: document.querySelector("#reviewNotesInput"),
  videoLink: document.querySelector("#videoLink"),
  savantLink: document.querySelector("#savantLink"),
  frameCount: document.querySelector("#frameCount"),
  frameGrid: document.querySelector("#frameGrid"),
  loadMoreButton: document.querySelector("#loadMoreButton"),
};

const overlayStorageKey = "pitchIntentFrameOverlay";
const viewStorageKey = "pitchIntentFrameReviewView";
const galleryPageSize = 48;
const defaultOverlay = {
  left: 43,
  top: 22,
  width: 14,
  height: 31,
};
const reviewOptions = [
  { value: "good", label: "Good" },
  { value: "close_not_perfect", label: "Close" },
  { value: "bad_blue_target", label: "Bad Target" },
  { value: "wrong_glove", label: "Wrong Glove" },
  { value: "wrong_zone", label: "Wrong Zone" },
  { value: "wrong_frame", label: "Wrong Frame" },
  { value: "unusable_video", label: "Unusable Video" },
  { value: "manual_label_questionable", label: "Manual Label?" },
];

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

function escapeHtml(value) {
  return text(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function unique(values) {
  return [...new Set(values.filter((value) => value && value !== "-"))].sort();
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function normalizeOverlay(rawOverlay) {
  const overlay = {
    left: num(rawOverlay?.left) ?? defaultOverlay.left,
    top: num(rawOverlay?.top) ?? defaultOverlay.top,
    width: num(rawOverlay?.width) ?? defaultOverlay.width,
    height: num(rawOverlay?.height) ?? defaultOverlay.height,
  };
  overlay.width = clamp(overlay.width, 0.1, 80);
  overlay.height = clamp(overlay.height, 0.1, 90);
  overlay.left = clamp(overlay.left, 0, 100 - overlay.width);
  overlay.top = clamp(overlay.top, 0, 100 - overlay.height);
  return overlay;
}

function normalizeCalibrationEntry(rawCalibration) {
  const calibration = normalizeOverlay(rawCalibration);
  calibration.frame_override = rawCalibration?.frame_override === true;
  for (const key of [
    "reference_pitch_uid",
    "reference_frame_uid",
    "reference_at_bat_number",
    "reference_batter",
  ]) {
    if (rawCalibration?.[key] !== null && rawCalibration?.[key] !== undefined && rawCalibration?.[key] !== "") {
      calibration[key] = String(rawCalibration[key]);
    }
  }
  for (const key of ["reference_sz_top", "reference_sz_bot"]) {
    const value = num(rawCalibration?.[key]);
    if (value !== null) calibration[key] = value;
  }
  return calibration;
}

function normalizeCalibrations(rawCalibrations) {
  const games = {};
  for (const [gamePk, overlay] of Object.entries(rawCalibrations?.games || {})) {
    games[String(gamePk)] = normalizeCalibrationEntry(overlay);
  }
  return {
    default: normalizeOverlay(rawCalibrations?.default || defaultOverlay),
    games,
  };
}

function gameKey(frame) {
  const value = text(frame?.game_pk);
  return value === "-" ? "" : value;
}

function automatedOverlayForFrame(frame) {
  const zoneX = num(frame?.zone_x);
  const zoneY = num(frame?.zone_y);
  const zoneWidth = num(frame?.zone_width);
  const zoneHeight = num(frame?.zone_height);
  if ([zoneX, zoneY, zoneWidth, zoneHeight].some((value) => value === null) || zoneWidth <= 0 || zoneHeight <= 0) return null;
  const imageWidth = num(frame?.image_width) || 1280;
  const imageHeight = num(frame?.image_height) || 720;
  return normalizeOverlay({
    left: ((zoneX - (zoneWidth / 2)) / imageWidth) * 100,
    top: ((zoneY - (zoneHeight / 2)) / imageHeight) * 100,
    width: (zoneWidth / imageWidth) * 100,
    height: (zoneHeight / imageHeight) * 100,
  });
}

function savedOverlayForFrame(frame) {
  const key = gameKey(frame);
  const calibration = key ? state.calibrations.games[key] : null;
  const exactReference = calibration?.reference_frame_uid === frame?.frame_uid;
  if (exactReference && calibration.frame_override) return normalizeOverlay(calibration);
  const automated = automatedOverlayForFrame(frame);
  if (automated) return automated;
  if (exactReference) return normalizeOverlay(calibration);
  return normalizeOverlay(state.calibrations.default);
}

function calibrationHasReference(calibration) {
  return Boolean(
    calibration?.reference_pitch_uid
    && num(calibration?.reference_sz_top) !== null
    && num(calibration?.reference_sz_bot) !== null
  );
}

function updateCalibrationProgress() {
  const games = unique(state.frames.map(gameKey));
  const referencedGames = games.filter((gamePk) => calibrationHasReference(state.calibrations.games[gamePk])).length;
  els.statusText.textContent = `${state.frames.length} setup frames | ${unique(state.frames.map((row) => row.pitch_uid)).length} pitches | ${referencedGames}/${games.length} zone games calibrated`;
}

function activeFrame() {
  return state.frames.find((row) => row.frame_uid === state.selectedFrameUid) || null;
}

function activateOverlayForFrame(frame) {
  state.overlay = savedOverlayForFrame(frame);
  state.overlayDirty = false;
}

function markOverlayDirty() {
  state.overlayDirty = true;
  saveOverlay();
  syncOverlayControls();
}

function statusKey(frame) {
  if (frame.image_exists === "yes") return "extracted";
  const rawStatus = text(frame.extraction_status);
  if (rawStatus === "-") return "planned";
  if (rawStatus.startsWith("failed")) return "failed";
  if (rawStatus === "extracted") return "missing_image";
  return rawStatus;
}

function normalizeFrame(frame) {
  return {
    ...frame,
    target_x_num: num(frame.target_x_01),
    target_y_num: num(frame.target_y_01),
    actual_x_num: num(frame.actual_x_01),
    actual_y_num: num(frame.actual_y_01),
    vision_target_x_num: num(frame.vision_target_x_01),
    vision_target_y_num: num(frame.vision_target_y_01),
    vision_manual_delta_num: num(frame.vision_manual_delta_01),
    zone_confidence_num: num(frame.zone_confidence),
    zone_x_num: num(frame.zone_x),
    zone_y_num: num(frame.zone_y),
    zone_width_num: num(frame.zone_width),
    zone_height_num: num(frame.zone_height),
    glove_confidence_num: num(frame.glove_confidence),
    zone_quality_score_num: num(frame.zone_quality_score),
    zone_center_game_delta_num: num(frame.zone_center_game_delta),
    zone_size_game_delta_num: num(frame.zone_size_game_delta),
    frame_time_num: num(frame.frame_time_sec),
    frame_index_num: num(frame.frame_index),
    status_key: statusKey(frame),
    vision_review_key: frame.vision_in_review_queue === "yes" ? "Review Queue" : "Not Queued",
    vision_status_key: frame.vision_filter_status ? pretty(frame.vision_filter_status) : "No Vision",
    manual_review_key: frame.manual_review_label ? pretty(frame.manual_review_label) : "Not Reviewed",
  };
}

function loadOverlay() {
  try {
    const saved = JSON.parse(localStorage.getItem(overlayStorageKey) || "{}");
    for (const key of ["left", "top", "width", "height"]) {
      if (Number.isFinite(Number(saved[key]))) {
        state.overlay[key] = Number(saved[key]);
      }
    }
  } catch {
    localStorage.removeItem(overlayStorageKey);
  }

  try {
    const savedView = JSON.parse(localStorage.getItem(viewStorageKey) || "{}");
    if (typeof savedView.showZone === "boolean") state.showZone = savedView.showZone;
    if (typeof savedView.editZone === "boolean") state.editZone = savedView.editZone;
    if (typeof savedView.showTarget === "boolean") state.showTarget = savedView.showTarget;
  } catch {
    localStorage.removeItem(viewStorageKey);
  }

  if (!state.showZone) {
    state.editZone = false;
  }
}

function saveOverlay() {
  localStorage.setItem(overlayStorageKey, JSON.stringify(state.overlay));
}

function saveView() {
  localStorage.setItem(viewStorageKey, JSON.stringify({
    showZone: state.showZone,
    editZone: state.editZone,
    showTarget: state.showTarget,
  }));
}

function syncOverlayControls() {
  const selected = activeFrame();
  const game = gameKey(selected);
  const gameCalibration = game ? state.calibrations.games[game] : null;
  const hasGameCalibration = Boolean(gameCalibration);
  const hasReference = calibrationHasReference(gameCalibration);
  const scope = els.overlayScopeSelect.value === "default" ? "global" : `game ${game || "-"}`;
  const savedState = state.overlayDirty
    ? "unsaved"
    : gameCalibration?.frame_override && gameCalibration?.reference_frame_uid === selected?.frame_uid
      ? "saved for this frame"
      : automatedOverlayForFrame(selected)
        ? pretty(selected.zone_refinement_status || "estimated zone")
        : "no zone evidence";
  els.showZoneToggle.checked = state.showZone;
  els.editZoneToggle.checked = state.editZone;
  els.editZoneToggle.disabled = !state.showZone;
  els.showTargetToggle.checked = state.showTarget;
  els.overlayLeft.value = state.overlay.left;
  els.overlayTop.value = state.overlay.top;
  els.overlayWidth.value = state.overlay.width;
  els.overlayHeight.value = state.overlay.height;
  els.overlayReadout.textContent = `${scope} | ${savedState} | L ${fmt(state.overlay.left, 1)} | T ${fmt(state.overlay.top, 1)} | W ${fmt(state.overlay.width, 1)} | H ${fmt(state.overlay.height, 1)}`;
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

function buildFilters() {
  fillSelect(els.pitcherFilter, ["All Pitchers", ...unique(state.frames.map((row) => row.sample_pitcher_name))]);
  fillSelect(els.pitchTypeFilter, ["All Types", ...unique(state.frames.map((row) => row.pitch_type))]);
  fillSelect(els.familyFilter, ["All Families", ...unique(state.frames.map((row) => row.pitch_family))]);
  fillSelect(els.visibilityFilter, ["All Visibility", ...unique(state.frames.map((row) => row.setup_visible))]);
  fillSelect(els.statusFilter, ["All Statuses", ...unique(state.frames.map((row) => row.status_key))]);
  fillSelect(els.visionReviewFilter, ["All Vision Review", "Review Queue", "Not Queued"]);
  fillSelect(els.visionStatusFilter, ["All Vision Status", ...unique(state.frames.map((row) => row.vision_status_key))]);
}

function filteredFrames(options = {}) {
  const forceReviewQueue = Boolean(options.forceReviewQueue);
  const unreviewedOnly = Boolean(options.unreviewedOnly);
  const pitcher = els.pitcherFilter.value;
  const pitchType = els.pitchTypeFilter.value;
  const family = els.familyFilter.value;
  const visibility = els.visibilityFilter.value;
  const status = els.statusFilter.value;
  const visionReview = els.visionReviewFilter.value;
  const visionStatus = els.visionStatusFilter.value;
  const query = els.searchInput.value.trim().toLowerCase();

  let rows = state.frames.filter((row) => {
    if (pitcher !== "All Pitchers" && row.sample_pitcher_name !== pitcher) return false;
    if (pitchType !== "All Types" && row.pitch_type !== pitchType) return false;
    if (family !== "All Families" && row.pitch_family !== family) return false;
    if (visibility !== "All Visibility" && row.setup_visible !== visibility) return false;
    if (status !== "All Statuses" && row.status_key !== status) return false;
    if (forceReviewQueue && row.vision_in_review_queue !== "yes") return false;
    if (!forceReviewQueue && visionReview !== "All Vision Review" && row.vision_review_key !== visionReview) return false;
    if (unreviewedOnly && row.manual_review_label) return false;
    if (visionStatus !== "All Vision Status" && row.vision_status_key !== visionStatus) return false;
    if (!query) return true;
    return [
      row.frame_uid,
      row.pitch_uid,
      row.sample_pitcher_name,
      row.pitch_type,
      row.pitch_family,
      row.description,
      row.events,
      row.label_notes,
      row.extraction_status,
      row.vision_filter_status,
      row.vision_review_reason,
      row.vision_filter_flags,
      row.vision_exclusion_reason,
      row.zone_source,
      row.zone_anchor_source,
      row.glove_source_model,
      row.vision_fallback_used,
      row.vision_fallback_reason,
      row.zone_quality_status,
      row.zone_quality_flags,
      row.manual_review_label,
    ].join(" ").toLowerCase().includes(query);
  });

  rows = [...rows].sort((a, b) => {
    const sortValue = els.sortSelect.value;
    if (sortValue === "date_desc") return String(b.game_date).localeCompare(String(a.game_date)) || compareFrame(a, b);
    if (sortValue === "status") return a.status_key.localeCompare(b.status_key) || compareFrame(a, b);
    if (sortValue === "vision_priority") return visionSortValue(a).localeCompare(visionSortValue(b)) || compareFrame(a, b);
    if (sortValue === "vision_delta") return (b.vision_manual_delta_num ?? -1) - (a.vision_manual_delta_num ?? -1) || compareFrame(a, b);
    if (sortValue === "time") return (a.frame_time_num ?? 999) - (b.frame_time_num ?? 999) || compareFrame(a, b);
    return `${a.sample_pitcher_name}${a.pitch_type}${a.pitch_uid}${a.frame_index}`.localeCompare(`${b.sample_pitcher_name}${b.pitch_type}${b.pitch_uid}${b.frame_index}`);
  });

  return rows;
}

function visionSortValue(row) {
  const priority = { high: "0", medium: "1", low: "2" }[row.vision_review_priority] || "9";
  const rank = String(row.vision_review_rank || "9999").padStart(4, "0");
  return `${priority}${rank}`;
}

function compareFrame(a, b) {
  return String(a.pitch_uid).localeCompare(String(b.pitch_uid)) || ((a.frame_index_num ?? 0) - (b.frame_index_num ?? 0));
}

function framesForPitch(pitchUid) {
  return state.frames
    .filter((row) => row.pitch_uid === pitchUid)
    .sort((a, b) => (a.frame_index_num ?? 0) - (b.frame_index_num ?? 0));
}

function pitchOrder(rows) {
  const seen = new Set();
  const order = [];
  for (const row of rows) {
    if (row.pitch_uid && !seen.has(row.pitch_uid)) {
      seen.add(row.pitch_uid);
      order.push(row.pitch_uid);
    }
  }
  return order;
}

function selectedPitchIndex(rows) {
  const order = pitchOrder(rows);
  return {
    order,
    index: order.indexOf(state.selectedPitchUid),
  };
}

function navigatePitch(direction) {
  const rows = filteredFrames();
  const { order, index } = selectedPitchIndex(rows);
  if (!order.length || index < 0) return;
  const nextIndex = index + direction;
  if (nextIndex < 0 || nextIndex >= order.length) return;
  const nextPitchUid = order[nextIndex];
  const nextFrame = rows.find((row) => row.pitch_uid === nextPitchUid) || framesForPitch(nextPitchUid)[0];
  state.selectedPitchUid = nextPitchUid;
  state.selectedFrameUid = nextFrame?.frame_uid || null;
  if (nextFrame) activateOverlayForFrame(nextFrame);
  render();
  document.querySelector(".selected-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function navigateReviewPitch() {
  let rows = filteredFrames({ forceReviewQueue: true, unreviewedOnly: true });
  if (!pitchOrder(rows).length) {
    rows = filteredFrames({ forceReviewQueue: true });
  }
  const order = pitchOrder(rows);
  if (!order.length) return;

  const currentIndex = order.indexOf(state.selectedPitchUid);
  const nextPitchUid = currentIndex >= 0 && currentIndex < order.length - 1
    ? order[currentIndex + 1]
    : order[0];
  const nextFrame = rows.find((row) => row.pitch_uid === nextPitchUid) || framesForPitch(nextPitchUid)[0];
  state.selectedPitchUid = nextPitchUid;
  state.selectedFrameUid = nextFrame?.frame_uid || null;
  if (nextFrame) activateOverlayForFrame(nextFrame);
  render();
  document.querySelector(".selected-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function representativeFrameForGame(gamePk) {
  const rows = state.frames.filter((row) => gameKey(row) === gamePk && row.image_exists === "yes");
  return rows.find((row) => row.vision_anchor_frame === "yes")
    || rows.find((row) => row.frame_time_num !== null && row.frame_time_num >= 1.8)
    || rows[0]
    || null;
}

function navigateUncalibratedGame() {
  const games = unique(state.frames.map(gameKey));
  const pending = games.filter((gamePk) => !calibrationHasReference(state.calibrations.games[gamePk]));
  if (!pending.length) {
    els.statusText.textContent = "All games have batter-referenced zone calibrations";
    return;
  }

  const currentGame = gameKey(activeFrame());
  const currentIndex = pending.indexOf(currentGame);
  const nextGame = currentIndex >= 0 && currentIndex < pending.length - 1
    ? pending[currentIndex + 1]
    : pending[0];
  const nextFrame = representativeFrameForGame(nextGame);
  if (!nextFrame) return;
  state.selectedPitchUid = nextFrame.pitch_uid;
  state.selectedFrameUid = nextFrame.frame_uid;
  activateOverlayForFrame(nextFrame);
  render();
  document.querySelector(".selected-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderKpis(rows) {
  const pitchCount = unique(rows.map((row) => row.pitch_uid)).length;
  const extracted = rows.filter((row) => row.status_key === "extracted").length;
  const reviewPitches = unique(rows.filter((row) => row.vision_in_review_queue === "yes").map((row) => row.pitch_uid)).length;
  const trustedPitches = unique(rows.filter((row) => row.vision_trusted === "yes").map((row) => row.pitch_uid)).length;
  const missingVision = unique(rows.filter((row) => row.vision_filter_status === "needs_review" && row.vision_filter_flags?.includes("missing_required_detection")).map((row) => row.pitch_uid)).length;
  const excludedPitches = unique(rows.filter((row) => row.vision_excluded === "yes").map((row) => row.pitch_uid)).length;
  const zoneIssuePitches = unique(rows.filter((row) => ["bad_reviewed", "missing", "poor"].includes(row.zone_quality_status)).map((row) => row.pitch_uid)).length;
  const unreviewedReviewPitches = unique(filteredFrames({ forceReviewQueue: true, unreviewedOnly: true }).map((row) => row.pitch_uid)).length;
  const metrics = [
    { label: "Frames", value: rows.length, detail: `${pitchCount} pitches` },
    { label: "Vision Review", value: reviewPitches, detail: `${missingVision} missing detection` },
    { label: "Unreviewed", value: unreviewedReviewPitches, detail: "review queue pitches" },
    { label: "Trusted Vision", value: trustedPitches, detail: `${rows.length ? Math.round((extracted / rows.length) * 100) : 0}% frames extracted` },
    { label: "Excluded", value: excludedPitches, detail: "unusable video" },
    { label: "Zone Issues", value: zoneIssuePitches, detail: "poor/missing/reviewed" },
  ];

  els.kpiStrip.innerHTML = "";
  for (const metric of metrics) {
    const node = document.createElement("div");
    node.className = "metric";
    node.innerHTML = `<span>${escapeHtml(metric.label)}</span><strong>${escapeHtml(metric.value)}</strong><small>${escapeHtml(metric.detail)}</small>`;
    els.kpiStrip.append(node);
  }
}

function renderFrameMedia(frame, size) {
  const media = document.createElement("div");
  media.className = "frame-media";

  if (frame.image_url) {
    const image = document.createElement("img");
    image.loading = size === "small" ? "lazy" : "eager";
    image.src = frame.image_url;
    image.alt = `${text(frame.pitch_uid)} setup frame ${text(frame.frame_index)}`;
    media.append(image);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "frame-placeholder";
    placeholder.innerHTML = `<div><strong>${escapeHtml(pretty(frame.status_key))}</strong><span>${escapeHtml(frame.image_path)}</span></div>`;
    media.append(placeholder);
  }

  const badge = document.createElement("div");
  badge.className = "frame-badge";
  badge.textContent = `${fmt(frame.frame_time_num, 2)}s`;
  media.append(badge);

  if (frame.vision_excluded === "yes") {
    const visionBadge = document.createElement("div");
    visionBadge.className = "vision-badge vision-excluded";
    visionBadge.textContent = "excluded";
    media.append(visionBadge);
  } else if (frame.vision_in_review_queue === "yes") {
    const visionBadge = document.createElement("div");
    visionBadge.className = `vision-badge vision-${frame.vision_review_priority || "low"}`;
    visionBadge.textContent = frame.vision_review_rank ? `#${frame.vision_review_rank}` : pretty(frame.vision_review_priority);
    media.append(visionBadge);
  }

  const zone = document.createElement("div");
  const canEditZone = size === "large" && state.showZone && state.editZone;
  const overlay = size === "large" ? state.overlay : savedOverlayForFrame(frame);
  const calibration = state.calibrations.games[gameKey(frame)];
  const manualFrame = calibration?.reference_frame_uid === frame.frame_uid && (calibration.frame_override || !automatedOverlayForFrame(frame));
  const hasGeometry = Boolean(automatedOverlayForFrame(frame) || manualFrame || canEditZone || (size === "large" && state.overlayDirty));
  const uncertain = !manualFrame && ![
    "visible_four_edge_fit",
    "registered_temporal_fit",
    "fox_temporal_alpha_fit",
    "fox_at_bat_consensus_fit",
    "recovered_four_edge_frame",
  ].includes(frame.zone_refinement_status);
  zone.className = `zone-overlay${state.showZone && hasGeometry ? "" : " zone-hidden"}${canEditZone ? " zone-editable" : ""}${uncertain ? " zone-uncertain" : ""}`;
  zone.title = uncertain ? "Estimated zone — alignment needs review" : "Zone aligned to this frame";
  if (!hasGeometry) zone.style.display = "none";
  applyOverlayStyles(zone, overlay);
  if (state.showTarget && frame.target_x_num !== null && frame.target_y_num !== null) {
    const target = document.createElement("div");
    target.className = "target-mitt";
    target.style.left = `${frame.target_x_num * 100}%`;
    target.style.top = `${(1 - frame.target_y_num) * 100}%`;
    target.title = `Target ${fmt(frame.target_x_num)}, ${fmt(frame.target_y_num)}`;
    zone.append(target);
  }
  if (state.showTarget && frame.vision_target_x_num !== null && frame.vision_target_y_num !== null) {
    const visionTarget = document.createElement("div");
    visionTarget.className = "vision-mitt";
    visionTarget.style.left = `${frame.vision_target_x_num * 100}%`;
    visionTarget.style.top = `${(1 - frame.vision_target_y_num) * 100}%`;
    visionTarget.title = `Vision ${fmt(frame.vision_target_x_num)}, ${fmt(frame.vision_target_y_num)}`;
    zone.append(visionTarget);
  }
  if (canEditZone) {
    addCropHandles(zone);
    zone.addEventListener("pointerdown", (event) => startOverlayPointerDrag(event, media, zone, "move"));
  }
  media.append(zone);
  return media;
}

function applyOverlayStyles(zone, overlay) {
  zone.style.left = `${overlay.left}%`;
  zone.style.top = `${overlay.top}%`;
  zone.style.width = `${overlay.width}%`;
  zone.style.height = `${overlay.height}%`;
}

function addCropHandles(zone) {
  for (const handle of ["nw", "n", "ne", "e", "se", "s", "sw", "w"]) {
    const node = document.createElement("span");
    node.className = `crop-handle ${handle}`;
    node.addEventListener("pointerdown", (event) => startOverlayPointerDrag(event, zone.parentElement, zone, handle));
    zone.append(node);
  }
}

function resizedOverlay(startOverlay, handle, dx, dy) {
  let { left, top, width, height } = startOverlay;
  const right = left + width;
  const bottom = top + height;
  if (handle.includes("w")) {
    left += dx;
    width = right - left;
  }
  if (handle.includes("e")) {
    width += dx;
  }
  if (handle.includes("n")) {
    top += dy;
    height = bottom - top;
  }
  if (handle.includes("s")) {
    height += dy;
  }
  return normalizeOverlay({ left, top, width, height });
}

function movedOverlay(startOverlay, dx, dy) {
  return normalizeOverlay({
    ...startOverlay,
    left: startOverlay.left + dx,
    top: startOverlay.top + dy,
  });
}

function startOverlayPointerDrag(event, media, zone, handle) {
  if (!media || !zone) return;
  event.preventDefault();
  event.stopPropagation();
  const rect = media.getBoundingClientRect();
  const startX = event.clientX;
  const startY = event.clientY;
  const startOverlay = { ...state.overlay };

  const onMove = (moveEvent) => {
    const dx = ((moveEvent.clientX - startX) / rect.width) * 100;
    const dy = ((moveEvent.clientY - startY) / rect.height) * 100;
    state.overlay = handle === "move"
      ? movedOverlay(startOverlay, dx, dy)
      : resizedOverlay(startOverlay, handle, dx, dy);
    applyOverlayStyles(zone, state.overlay);
    markOverlayDirty();
  };

  const onUp = () => {
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
  };

  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp, { once: true });
}

function renderReviewControls(selected) {
  els.reviewButtons.innerHTML = "";
  els.reviewNotesInput.disabled = !selected;
  if (!selected) {
    els.reviewLabelStatus.textContent = "Not reviewed";
    els.reviewNotesInput.value = "";
    return;
  }

  const savedLabel = selected.manual_review_label || "";
  const savedAt = selected.manual_reviewed_at ? ` | ${selected.manual_reviewed_at}` : "";
  els.reviewLabelStatus.textContent = savedLabel ? `${pretty(savedLabel)}${savedAt}` : "Not reviewed";
  els.reviewNotesInput.value = selected.manual_review_notes || "";

  for (const option of reviewOptions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = option.value === savedLabel ? "active" : "ghost";
    button.textContent = option.label;
    button.addEventListener("click", () => saveVisionReview(option.value));
    els.reviewButtons.append(button);
  }
}

function renderSelected(rows) {
  const previousFrameUid = state.selectedFrameUid;
  if (!rows.some((row) => row.pitch_uid === state.selectedPitchUid)) {
    state.selectedPitchUid = rows[0]?.pitch_uid || null;
    state.selectedFrameUid = rows[0]?.frame_uid || null;
  }

  const pitchFrames = framesForPitch(state.selectedPitchUid);
  let selected = pitchFrames.find((row) => row.frame_uid === state.selectedFrameUid) || pitchFrames[0] || rows[0] || null;
  if (selected) {
    state.selectedPitchUid = selected.pitch_uid;
    state.selectedFrameUid = selected.frame_uid;
    if (selected.frame_uid !== previousFrameUid) {
      activateOverlayForFrame(selected);
    }
  }

  els.selectedStage.innerHTML = "";
  els.frameButtons.innerHTML = "";
  if (!selected) {
    els.selectedTitle.textContent = "No frame selected";
    els.prevPitchButton.disabled = true;
    els.nextPitchButton.disabled = true;
    els.nextReviewButton.disabled = true;
    els.selectedStage.innerHTML = `<div class="empty-state">No frames</div>`;
    els.selectedMeta.innerHTML = "";
    renderReviewControls(null);
    els.videoLink.style.visibility = "hidden";
    els.savantLink.style.visibility = "hidden";
    return;
  }

  const { order, index } = selectedPitchIndex(rows);
  els.prevPitchButton.disabled = index <= 0;
  els.nextPitchButton.disabled = index < 0 || index >= order.length - 1;
  const unreviewedReviewCount = pitchOrder(filteredFrames({ forceReviewQueue: true, unreviewedOnly: true })).length;
  els.nextReviewButton.disabled = unreviewedReviewCount === 0;
  els.nextReviewButton.textContent = unreviewedReviewCount ? `Next Review (${unreviewedReviewCount})` : "Next Review";
  const positionText = index >= 0 ? `${index + 1}/${order.length}` : "-";
  els.selectedTitle.textContent = `${positionText} | ${text(selected.pitch_uid)} | ${text(selected.sample_pitcher_name)}`;
  els.selectedStage.append(renderFrameMedia(selected, "large"));

  for (const frame of pitchFrames) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = frame.frame_uid === selected.frame_uid ? "active" : "ghost";
    button.textContent = `${text(frame.frame_label)} | ${fmt(frame.frame_time_num, 2)}s`;
    button.addEventListener("click", () => {
      state.selectedFrameUid = frame.frame_uid;
      activateOverlayForFrame(frame);
      render();
    });
    els.frameButtons.append(button);
  }

  const result = selected.events && selected.events !== "-" ? selected.events : selected.description;
  els.selectedMeta.innerHTML = [
    ["Pitch", `${text(selected.pitch_type)} | ${text(selected.pitch_name)}`],
    ["Family", text(selected.pitch_family)],
    ["Count", `${text(selected.balls)}-${text(selected.strikes)}`],
    ["Result", text(result)],
    ["Game", `${text(selected.game_date)} | ${text(selected.inning_topbot)} ${text(selected.inning)}`],
    ["Target", `${fmt(selected.target_x_num)}, ${fmt(selected.target_y_num)}`],
    ["Vision Target", `${fmt(selected.vision_target_x_num)}, ${fmt(selected.vision_target_y_num)}`],
    ["Vision Delta", fmt(selected.vision_manual_delta_num)],
    ["Vision Status", `${pretty(selected.vision_filter_status)} | ${text(selected.vision_trusted)}`],
    ["Excluded", selected.vision_excluded === "yes" ? text(selected.vision_exclusion_reason) : "no"],
    ["Zone Source", `${pretty(selected.zone_source)} | ${pretty(selected.zone_anchor_source)}`],
    ["Glove Source", pretty(selected.glove_source_model)],
    ["Selected Frame", selected.selected_frame_uid ? `${text(selected.selected_frame_uid)} | ${text(selected.selected_frame_time_sec)}s` : "-"],
    ["Glove Candidates", selected.glove_candidate_count ? `${text(selected.plausible_glove_candidate_count)}/${text(selected.glove_candidate_count)} | score ${text(selected.glove_selection_score)}` : "-"],
    ["Glove Selection", text(selected.glove_selection_reason)],
    ["Dense Glove", selected.dense_glove_used === "yes" ? text(selected.dense_glove_reason) : "no"],
    ["Fallback", selected.vision_fallback_used === "yes" ? text(selected.vision_fallback_reason) : "no"],
    ["Statcast Zone", `${fmt(selected.statcast_sz_bot, 2)}-${fmt(selected.statcast_sz_top, 2)} ft`],
    ["Vision Queue", selected.vision_in_review_queue === "yes" ? `${text(selected.vision_review_priority)} #${text(selected.vision_review_rank)}` : "no"],
    ["Vision Reason", text(selected.vision_review_reason)],
    ["Vision Conf", `zone ${fmt(selected.zone_confidence_num, 2)} | glove ${fmt(selected.glove_confidence_num, 2)}`],
    ["Zone Quality", `${pretty(selected.zone_quality_status)} | ${fmt(selected.zone_quality_score_num, 0)}`],
    ["Zone Flags", text(selected.zone_quality_flags)],
    ["Zone Game Delta", `center ${fmt(selected.zone_center_game_delta_num, 1)} | size ${fmt(selected.zone_size_game_delta_num, 1)}`],
    ["Actual", `${fmt(selected.actual_x_num)}, ${fmt(selected.actual_y_num)}`],
    ["Visible", text(selected.setup_visible)],
    ["Review Label", pretty(selected.manual_review_label)],
    ["Status", pretty(selected.status_key)],
    ["Frame Path", text(selected.image_path)],
    ["Pitcher", text(selected.sample_pitcher_name)],
    ["Notes", text(selected.label_notes)],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");

  renderReviewControls(selected);
  els.videoLink.href = selected.direct_mp4_url || "#";
  els.videoLink.style.visibility = selected.direct_mp4_url ? "visible" : "hidden";
  els.savantLink.href = selected.savant_video_url || "#";
  els.savantLink.style.visibility = selected.savant_video_url ? "visible" : "hidden";
}

function statusPillClass(frame) {
  if (frame.vision_excluded === "yes") return "vision-excluded";
  if (frame.vision_in_review_queue === "yes") return `vision-${frame.vision_review_priority || "low"}`;
  if (["bad_reviewed", "missing", "poor"].includes(frame.zone_quality_status)) return "zone-bad";
  if (frame.zone_quality_status === "warn") return "zone-warn";
  return `status-${frame.status_key}`;
}

function statusPillText(frame) {
  if (frame.vision_excluded === "yes") return "excluded";
  if (frame.vision_in_review_queue === "yes") return pretty(frame.vision_review_priority || "review");
  if (["bad_reviewed", "missing", "poor", "warn"].includes(frame.zone_quality_status)) {
    return pretty(frame.zone_quality_status);
  }
  return pretty(frame.status_key);
}

function applyReviewDisposition(frame, review) {
  frame.manual_review_label = review.review_label || "";
  frame.manual_review_notes = review.review_notes || "";
  frame.manual_reviewed_at = review.reviewed_at || "";
  frame.manual_review_frame_uid = review.frame_uid || "";
  frame.manual_review_key = frame.manual_review_label ? pretty(frame.manual_review_label) : "Not Reviewed";

  if (frame.manual_review_label) {
    frame.vision_in_review_queue = "no";
    frame.vision_review_priority = "";
    frame.vision_review_rank = "";
    frame.vision_review_key = "Not Queued";
  }

  if (frame.manual_review_label === "unusable_video") {
    frame.vision_excluded = "yes";
    frame.vision_exclusion_reason = "manual_unusable_video";
    frame.vision_filter_status = "excluded";
    frame.vision_trusted = "no";
  } else if (frame.manual_review_label === "wrong_zone") {
    frame.vision_filter_status = "bad_zone";
    frame.vision_trusted = "no";
    frame.zone_quality_status = "bad_reviewed";
    frame.zone_quality_score = "20";
    frame.zone_quality_score_num = 20;
    if (!String(frame.zone_quality_flags || "").includes("review_wrong_zone")) {
      frame.zone_quality_flags = [frame.zone_quality_flags, "review_wrong_zone"].filter(Boolean).join(";");
    }
  }

  frame.vision_status_key = frame.vision_filter_status ? pretty(frame.vision_filter_status) : "No Vision";
}

function renderGallery(rows) {
  const shown = rows.slice(0, state.galleryLimit);
  els.frameCount.textContent = `${shown.length} of ${rows.length} frames`;
  els.frameGrid.innerHTML = "";
  if (!rows.length) {
    els.frameGrid.innerHTML = `<div class="empty-state">No frames</div>`;
    els.loadMoreButton.hidden = true;
    return;
  }

  for (const frame of shown) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `frame-card${frame.frame_uid === state.selectedFrameUid ? " selected" : ""}`;
    card.append(renderFrameMedia(frame, "small"));

    const meta = document.createElement("div");
    meta.className = "frame-card-meta";
    meta.innerHTML = `
      <div>
        <strong>${escapeHtml(frame.pitch_uid)}</strong>
        <span>${escapeHtml(frame.sample_pitcher_name)} | ${escapeHtml(frame.pitch_type)} | M ${escapeHtml(fmt(frame.target_x_num))}, ${escapeHtml(fmt(frame.target_y_num))} | V ${escapeHtml(fmt(frame.vision_target_x_num))}, ${escapeHtml(fmt(frame.vision_target_y_num))}</span>
      </div>
      <span class="status-pill ${escapeHtml(statusPillClass(frame))}">${escapeHtml(statusPillText(frame))}</span>
    `;
    card.append(meta);
    card.addEventListener("click", () => {
      state.selectedPitchUid = frame.pitch_uid;
      state.selectedFrameUid = frame.frame_uid;
      activateOverlayForFrame(frame);
      render();
      document.querySelector(".selected-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    els.frameGrid.append(card);
  }
  els.loadMoreButton.hidden = shown.length >= rows.length;
  els.loadMoreButton.textContent = `Load More (${rows.length - shown.length} left)`;
}

function render() {
  syncOverlayControls();
  const rows = filteredFrames();
  renderKpis(rows);
  renderSelected(rows);
  renderGallery(rows);
}

async function loadFrames() {
  loadOverlay();
  syncOverlayControls();
  const response = await fetch("/api/frame-review");
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  state.frames = (data.frames || []).map(normalizeFrame);
  state.calibrations = normalizeCalibrations(data.calibrations);
  updateCalibrationProgress();
  buildFilters();
  state.selectedPitchUid = state.frames[0]?.pitch_uid || null;
  state.selectedFrameUid = state.frames[0]?.frame_uid || null;
  if (state.frames[0]) activateOverlayForFrame(state.frames[0]);
  render();
}

async function saveCurrentOverlay() {
  const selected = activeFrame();
  const scope = els.overlayScopeSelect.value;
  const payload = {
    scope,
    game_pk: selected?.game_pk || "",
    overlay: normalizeOverlay(state.overlay),
    reference: {
      pitch_uid: selected?.pitch_uid || "",
      frame_uid: selected?.frame_uid || "",
      at_bat_number: selected?.at_bat_number || "",
      batter: selected?.batter || "",
      sz_top: selected?.sz_top || selected?.statcast_sz_top || "",
      sz_bot: selected?.sz_bot || selected?.statcast_sz_bot || "",
    },
  };
  const response = await fetch("/api/frame-calibration", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  state.calibrations = normalizeCalibrations(data.calibrations);
  state.overlay = normalizeOverlay(payload.overlay);
  state.overlayDirty = false;
  updateCalibrationProgress();
  render();
}

async function saveVisionReview(reviewLabel) {
  const selected = activeFrame();
  if (!selected) return;

  const payload = {
    pitch_uid: selected.pitch_uid,
    frame_uid: selected.frame_uid,
    review_label: reviewLabel,
    review_notes: els.reviewNotesInput.value.trim(),
    sample_pitcher_name: selected.sample_pitcher_name || "",
    game_date: selected.game_date || "",
    pitch_type: selected.pitch_type || "",
    vision_filter_status: selected.vision_filter_status || "",
    vision_review_reason: selected.vision_review_reason || "",
  };

  els.reviewLabelStatus.textContent = "Saving";
  for (const button of els.reviewButtons.querySelectorAll("button")) {
    button.disabled = true;
  }

  try {
    const response = await fetch("/api/vision-review-label", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    const review = data.review || {};
    for (const frame of state.frames) {
      if (frame.pitch_uid === selected.pitch_uid) {
        applyReviewDisposition(frame, review);
      }
    }
    render();
  } catch (error) {
    els.reviewLabelStatus.textContent = `Save failed: ${error.message}`;
  } finally {
    for (const button of els.reviewButtons.querySelectorAll("button")) {
      button.disabled = false;
    }
  }
}

for (const element of [
  els.pitcherFilter,
  els.pitchTypeFilter,
  els.familyFilter,
  els.visibilityFilter,
  els.statusFilter,
  els.visionReviewFilter,
  els.visionStatusFilter,
  els.searchInput,
  els.sortSelect,
]) {
  element.addEventListener("input", () => {
    state.galleryLimit = galleryPageSize;
    render();
  });
  element.addEventListener("change", () => {
    state.galleryLimit = galleryPageSize;
    render();
  });
}

for (const [key, element] of [
  ["left", els.overlayLeft],
  ["top", els.overlayTop],
  ["width", els.overlayWidth],
  ["height", els.overlayHeight],
]) {
  element.addEventListener("input", () => {
    state.overlay = normalizeOverlay({ ...state.overlay, [key]: Number(element.value) });
    state.overlayDirty = true;
    saveOverlay();
    render();
  });
}

els.showZoneToggle.addEventListener("change", () => {
  state.showZone = els.showZoneToggle.checked;
  if (!state.showZone) {
    state.editZone = false;
  }
  saveView();
  render();
});

els.editZoneToggle.addEventListener("change", () => {
  state.editZone = els.editZoneToggle.checked && state.showZone;
  saveView();
  render();
});

els.showTargetToggle.addEventListener("change", () => {
  state.showTarget = els.showTargetToggle.checked;
  saveView();
  render();
});

els.resetOverlayButton.addEventListener("click", () => {
  const selected = activeFrame();
  state.overlay = selected ? savedOverlayForFrame(selected) : normalizeOverlay(defaultOverlay);
  state.overlayDirty = false;
  saveOverlay();
  render();
});

els.overlayScopeSelect.addEventListener("change", syncOverlayControls);

els.saveOverlayButton.addEventListener("click", () => {
  els.saveOverlayButton.disabled = true;
  els.saveOverlayButton.textContent = "Saving";
  saveCurrentOverlay()
    .catch((error) => {
      els.statusText.textContent = `Zone save failed: ${error.message}`;
    })
    .finally(() => {
      els.saveOverlayButton.disabled = false;
      els.saveOverlayButton.textContent = "Save Zone";
    });
});

els.nextCalibrationButton.addEventListener("click", navigateUncalibratedGame);

els.loadMoreButton.addEventListener("click", () => {
  state.galleryLimit += galleryPageSize;
  render();
});

els.prevPitchButton.addEventListener("click", () => navigatePitch(-1));
els.nextPitchButton.addEventListener("click", () => navigatePitch(1));
els.nextReviewButton.addEventListener("click", navigateReviewPitch);

window.addEventListener("keydown", (event) => {
  const tagName = event.target?.tagName || "";
  if (tagName === "INPUT" || tagName === "SELECT" || tagName === "TEXTAREA") return;
  if (event.key === "ArrowLeft") {
    navigatePitch(-1);
  }
  if (event.key === "ArrowRight") {
    navigatePitch(1);
  }
});

loadFrames().catch((error) => {
  els.statusText.textContent = "Frame review failed to load";
  els.kpiStrip.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
});

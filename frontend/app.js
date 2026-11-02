// Base API URL configuration
// If opened directly from filesystem (file://), target http://127.0.0.1:8000
const API_BASE = window.location.origin.startsWith("http")
  ? window.location.origin
  : "http://127.0.0.1:8000";

let currentTab = "violations";
let currentPage = 1;
let currentSearch = "";
let currentStatusFilter = "";
let currentComplianceFilter = "";
let autoRefreshTimer = null;
let activeViolation = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadStats();
  loadViolations();
  setupEventListeners();

  // Auto-refresh polling every 5 seconds
  autoRefreshTimer = setInterval(() => {
    checkHealth();
    loadStats();
    loadViolations(false); // background refresh without resetting pagination
  }, 5000);
});

function setupEventListeners() {
  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const tabId = btn.getAttribute("data-tab");
      document.getElementById(`${tabId}-tab`).classList.add("active");
      currentTab = tabId;
    });
  });

  // Filter controls
  const searchInput = document.getElementById("searchInput");
  const statusFilter = document.getElementById("statusFilter");
  const complianceFilter = document.getElementById("complianceFilter");

  let debounceTimer;
  searchInput.addEventListener("input", (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      currentSearch = e.target.value.trim();
      currentPage = 1;
      loadViolations();
    }, 350);
  });

  statusFilter.addEventListener("change", (e) => {
    currentStatusFilter = e.target.value;
    currentPage = 1;
    loadViolations();
  });

  complianceFilter.addEventListener("change", (e) => {
    currentComplianceFilter = e.target.value;
    currentPage = 1;
    loadViolations();
  });

  // Upload Studio
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "var(--accent-teal)";
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.style.borderColor = "rgba(0, 210, 180, 0.35)";
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "rgba(0, 210, 180, 0.35)";
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleImageAnalysis(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleImageAnalysis(e.target.files[0]);
    }
  });

  // Modal close
  document.getElementById("closeModalBtn").addEventListener("click", closeModal);
  document.getElementById("evidenceModal").addEventListener("click", (e) => {
    if (e.target.id === "evidenceModal") closeModal();
  });

  // Ensure mock mode is explicitly unchecked by default
  const mockCheckbox = document.getElementById("mockModeCheckbox");
  if (mockCheckbox) {
    mockCheckbox.checked = false;
  }
}

// -------------------------------------------------------------
// API Calls & Rendering
// -------------------------------------------------------------

async function checkHealth() {
  const dot = document.getElementById("healthDot");
  const text = document.getElementById("healthText");
  try {
    const res = await fetch(`${API_BASE}/api/health/database`);
    if (res.ok) {
      const data = await res.json();
      if (data.database === "connected") {
        dot.className = "dot";
        text.innerText = "System Online • Database Connected";
        return;
      }
    }
    throw new Error();
  } catch {
    dot.className = "dot offline";
    text.innerText = "Backend / Database Offline";
  }
}

async function loadStats() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    if (!res.ok) return;
    const stats = await res.json();

    document.getElementById("statTotal").innerText = stats.total_detections;
    document.getElementById("statViolations").innerText = stats.violations;
    document.getElementById("statCompliant").innerText = stats.helmet_compliant;
    document.getElementById("statPending").innerText = stats.pending_review;
    document.getElementById("statApproved").innerText = stats.approved_violations;
    document.getElementById("statRejected").innerText = stats.rejected_violations;
  } catch (err) {
    console.error("Failed to load statistics:", err);
  }
}

async function loadViolations(showLoading = true) {
  const tableBody = document.getElementById("violationsTableBody");
  if (showLoading && tableBody.children.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:24px; color:var(--text-secondary);">Loading violations...</td></tr>`;
  }

  try {
    let url = `${API_BASE}/api/violations?page=${currentPage}&page_size=15`;
    if (currentStatusFilter) url += `&review_status=${encodeURIComponent(currentStatusFilter)}`;
    if (currentComplianceFilter !== "") url += `&helmet_detected=${currentComplianceFilter === "true"}`;
    if (currentSearch) url += `&search=${encodeURIComponent(currentSearch)}`;

    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to fetch violations");
    const data = await res.json();

    renderViolationsTable(data.items);
    renderPagination(data.total, data.page, data.page_size);
  } catch (err) {
    tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:24px; color:var(--accent-rose);">Error connecting to backend API: ${err.message}</td></tr>`;
  }
}

function renderViolationsTable(items) {
  const tableBody = document.getElementById("violationsTableBody");
  if (!items || items.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:32px; color:var(--text-muted);">No records found matching filters.</td></tr>`;
    return;
  }

  tableBody.innerHTML = items
    .map((item) => {
      const isCompliant = item.helmet_detected;
      const helmetBadge = isCompliant
        ? `<span class="badge badge-compliant">With Helmet</span>`
        : `<span class="badge badge-violation">Without Helmet</span>`;

      let statusBadge = `<span class="badge badge-${item.review_status}">${item.review_status}</span>`;

      const plateDisplay = item.number_plate
        ? `<strong style="color:#ffffff;">${item.number_plate}</strong>`
        : `<span style="color:var(--text-muted);">(unreadable)</span>`;

      const formattedTime = new Date(item.created_at).toLocaleString();

      const evidenceImgSrc = item.evidence_path
        ? (item.evidence_path.startsWith("http") ? item.evidence_path : `${API_BASE}${item.evidence_path}`)
        : "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='40' height='30'%3E%3Crect width='40' height='30' fill='%23162447'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%2394A3B8' font-size='8'%3ENo Img%3C/text%3E%3C/svg%3E";

      return `
        <tr>
          <td>#${item.id}</td>
          <td>
            <img src="${evidenceImgSrc}" class="thumb-preview" alt="Evidence" onclick='openModal(${JSON.stringify(item)})' />
          </td>
          <td>${plateDisplay}</td>
          <td>${helmetBadge}</td>
          <td>${statusBadge}</td>
          <td><span style="font-size:0.8rem; color:var(--text-secondary);">${item.detection_source}</span></td>
          <td style="font-size:0.8rem; color:var(--text-muted);">${formattedTime}</td>
          <td>
            <div class="action-group">
              ${
                item.review_status !== "approved"
                  ? `<button class="btn-action btn-approve" onclick="updateStatus(${item.id}, 'approved')">Approve</button>`
                  : ""
              }
              ${
                item.review_status !== "rejected"
                  ? `<button class="btn-action btn-reject" onclick="updateStatus(${item.id}, 'rejected')">Reject</button>`
                  : ""
              }
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
}

function renderPagination(total, page, pageSize) {
  const container = document.getElementById("paginationControls");
  const totalPages = Math.ceil(total / pageSize) || 1;

  container.innerHTML = `
    <span style="font-size:0.85rem; color:var(--text-secondary);">Showing Page <strong>${page}</strong> of <strong>${totalPages}</strong> (${total} total records)</span>
    <div style="display:flex; gap:8px;">
      <button class="btn-secondary" ${page <= 1 ? "disabled" : ""} onclick="changePage(${page - 1})">◀ Prev</button>
      <button class="btn-secondary" ${page >= totalPages ? "disabled" : ""} onclick="changePage(${page + 1})">Next ▶</button>
    </div>
  `;
}

function changePage(newPage) {
  currentPage = newPage;
  loadViolations();
}

async function updateStatus(violationId, newStatus) {
  try {
    const res = await fetch(`${API_BASE}/api/violations/${violationId}/status`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_status: newStatus }),
    });

    if (res.ok) {
      loadStats();
      loadViolations(false);
      if (activeViolation && activeViolation.id === violationId) {
        activeViolation.review_status = newStatus;
        openModal(activeViolation);
      }
    } else {
      alert("Failed to update status. Check backend connection.");
    }
  } catch (err) {
    alert(`Error updating status: ${err.message}`);
  }
}

// -------------------------------------------------------------
// Image Upload & Detection Analysis
// -------------------------------------------------------------

async function handleImageAnalysis(file) {
  const resultPanel = document.getElementById("analysisResultPanel");
  const previewImg = document.getElementById("analyzedPreviewImg");
  const statusMsg = document.getElementById("analysisStatusMsg");
  const detailsBox = document.getElementById("analysisDetailsBox");
  const mockCheckbox = document.getElementById("mockModeCheckbox");

  resultPanel.style.display = "block";
  statusMsg.innerHTML = `<span style="color:var(--accent-teal);">Processing frame: Running detection & OCR pipeline...</span>`;
  previewImg.src = URL.createObjectURL(file);
  detailsBox.innerHTML = "";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("detection_source", "dashboard_studio");
  formData.append("conf_threshold", "0.2");
  const useMock = Boolean(mockCheckbox && mockCheckbox.checked);
  formData.append("use_mock", useMock ? "true" : "false");

  try {
    const res = await fetch(`${API_BASE}/api/detections/analyze`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Image analysis failed.");
    }

    const data = await res.json();
    const v = data.violation;

    // Update with annotated image from static server
    if (v.evidence_path) {
      previewImg.src = `${API_BASE}${v.evidence_path}?t=${Date.now()}`;
    }

    statusMsg.innerHTML = `<span style="color:var(--accent-emerald); font-weight:700;">✔ Analysis Complete — Record #${v.id} Persisted</span>`;

    // Safely extract helmet and plate detections with fallback
    const helmetDetections = data.helmet_classification || data.helmet_detections || [];
    const plateDetections = data.number_plate_detection || data.plate_detections || [];

    const helmetSummary = helmetDetections.length > 0
      ? helmetDetections.map((d) => {
          const labelText = d.readable_label || d.label;
          const confText = d.confidence !== undefined ? ` (Conf: ${d.confidence})` : "";
          const reasons = d.violation_reasons && d.violation_reasons.length > 0
            ? ` <span style="color:var(--accent-rose);">[${d.violation_reasons.join(", ")}]</span>`
            : "";
          return `• <strong>${labelText}</strong>${confText}${reasons}`;
        }).join("<br>")
      : "• No riders/helmets localized";

    const plateSummary = plateDetections.length > 0
      ? plateDetections.map((p) => `• "${p.plate_text || "Unreadable"}" (Conf: ${p.confidence || p.detector_confidence || 0})`).join("<br>")
      : "• No license plate localized";

    const isMock = Boolean(data.is_mock_inference);
    const modelStatus = data.model_status || (isMock ? "mock_inference_mode" : "faster_rcnn_16class_active");

    const modeBadge = isMock
      ? `<span class="badge badge-pending">Simulated Mock Inference</span>`
      : `<span class="badge" style="background:rgba(52,211,153,0.18); color:#34D399; border:1px solid #34D399;">Faster R-CNN 16-Class (Active)</span>`;

    const procTime = data.processing_time_ms ? `<span>(${data.processing_time_ms} ms)</span>` : "";

    detailsBox.innerHTML = `
      <div style="background:var(--bg-surface-elevated); padding:14px; border-radius:8px; border:1px solid var(--border-subtle); display:flex; flex-direction:column; gap:8px; font-size:0.86rem;">
        <div><strong>Inference Engine:</strong> ${modeBadge} ${procTime}</div>
        <div><strong>Detection Verdict:</strong> <strong>${data.summary}</strong></div>
        <div><strong>Rider / Helmet Breakdown:</strong> <br>${helmetSummary}</div>
        <div><strong>License Plate Recognition:</strong> <br>${plateSummary}</div>
        <div><strong>Review Status:</strong> <span class="badge badge-${v.review_status}">${v.review_status}</span></div>
        <div style="font-size:0.75rem; color:var(--text-muted); font-style:italic; border-top:1px solid var(--border-subtle); padding-top:6px;">
          Faster R-CNN ResNet-50 FPN • 17-Class Classifier Active
        </div>
      </div>
    `;

    // Refresh dashboard stats and violation table
    loadStats();
    loadViolations(false);
  } catch (err) {
    statusMsg.innerHTML = `<span style="color:var(--accent-rose); font-weight:700;">⚠ Error: ${err.message}</span>`;
  }
}

// -------------------------------------------------------------
// Modal View
// -------------------------------------------------------------

function openModal(item) {
  activeViolation = item;
  const modal = document.getElementById("evidenceModal");
  const modalImg = document.getElementById("modalEvidenceImg");
  const modalInfo = document.getElementById("modalInfo");
  const modalActions = document.getElementById("modalActions");

  const evidenceSrc = item.evidence_path
    ? (item.evidence_path.startsWith("http") ? item.evidence_path : `${API_BASE}${item.evidence_path}`)
    : "";

  modalImg.src = evidenceSrc;
  document.getElementById("modalTitle").innerText = `Violation Record #${item.id} Review`;

  modalInfo.innerHTML = `
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; font-size:0.88rem;">
      <div><strong>License Plate:</strong> ${item.number_plate || "(unreadable)"}</div>
      <div><strong>Helmet Status:</strong> ${item.helmet_detected ? "With Helmet (Compliant)" : "Without Helmet (Violation)"}</div>
      <div><strong>Source:</strong> ${item.detection_source}</div>
      <div><strong>Review State:</strong> <span class="badge badge-${item.review_status}">${item.review_status}</span></div>
      <div><strong>Logged Timestamp:</strong> ${new Date(item.created_at).toLocaleString()}</div>
    </div>
  `;

  modalActions.innerHTML = `
    <button class="btn-action btn-approve" style="padding:8px 16px; font-size:0.85rem;" onclick="updateStatus(${item.id}, 'approved')">Approve as Valid Violation</button>
    <button class="btn-action btn-reject" style="padding:8px 16px; font-size:0.85rem;" onclick="updateStatus(${item.id}, 'rejected')">Dismiss / Reject Record</button>
  `;

  modal.classList.add("open");
}

function closeModal() {
  document.getElementById("evidenceModal").classList.remove("open");
  activeViolation = null;
}

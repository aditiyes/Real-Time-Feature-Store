const labels = {
  transactions_7d: "Transactions · 7d",
  failed_payments_30d: "Failed payments · 30d",
  avg_order_value: "Average order value",
  account_age_days: "Account age · days",
};

const userSelect = document.getElementById("user-select");
const chart = document.getElementById("feature-chart");
let chartPoints = [];

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "Request failed");
  return body;
}

function localInputValue(timestamp) {
  const date = new Date(timestamp);
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

function formatDate(timestamp) {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(timestamp));
}

function renderFeatures(target, features) {
  target.replaceChildren();
  for (const [name, title] of Object.entries(labels)) {
    const item = document.createElement("div");
    item.className = "feature-item";
    const label = document.createElement("span");
    label.textContent = title;
    const value = document.createElement("strong");
    const raw = features[name];
    value.textContent = raw == null ? "--" : name === "avg_order_value" ? `$${Number(raw).toFixed(2)}` : Number(raw).toLocaleString();
    item.append(label, value);
    target.append(item);
  }
}

async function runPrediction() {
  const button = document.getElementById("run-prediction");
  const placeholder = document.getElementById("result-placeholder");
  const content = document.getElementById("result-content");
  button.disabled = true;
  button.firstChild.textContent = "Fetching... ";
  try {
    const result = await requestJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: Number(userSelect.value) }),
    });
    document.getElementById("risk-score").textContent = `${result.risk_percent.toFixed(1)}%`;
    document.getElementById("score-ring").style.setProperty("--risk", `${result.risk_percent}%`);
    const decision = document.getElementById("decision-label");
    decision.textContent = result.prediction;
    decision.classList.toggle("review", result.prediction === "review");
    document.getElementById("prediction-meta").textContent = `As of ${formatDate(result.online_as_of)} · ${result.latency_ms} ms`;
    renderFeatures(document.getElementById("feature-values"), result.features);
    placeholder.hidden = true;
    content.hidden = false;
  } catch (error) {
    placeholder.textContent = error.message;
    placeholder.hidden = false;
    content.hidden = true;
  } finally {
    button.disabled = false;
    button.firstChild.textContent = "Run prediction ";
  }
}

function drawChart() {
  const bounds = chart.getBoundingClientRect();
  if (!bounds.width || !chartPoints.length) return;
  const ratio = window.devicePixelRatio || 1;
  chart.width = Math.round(bounds.width * ratio);
  chart.height = Math.round(bounds.height * ratio);
  const context = chart.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = bounds.width;
  const height = bounds.height;
  const pad = { left: 9, right: 8, top: 9, bottom: 10 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#e6eae3";
  context.lineWidth = 1;
  for (let row = 0; row <= 3; row += 1) {
    const y = pad.top + (plotHeight * row) / 3;
    context.beginPath();
    context.moveTo(pad.left, y);
    context.lineTo(width - pad.right, y);
    context.stroke();
  }
  const drawSeries = (key, max, color) => {
    context.beginPath();
    chartPoints.forEach((point, index) => {
      const x = pad.left + (plotWidth * index) / Math.max(chartPoints.length - 1, 1);
      const y = pad.top + plotHeight * (1 - Math.min(Number(point[key]) / max, 1));
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.lineJoin = "round";
    context.stroke();
  };
  drawSeries("transactions_7d", 10, "#779c49");
  drawSeries("failed_payments_30d", 4, "#df8b70");
  document.getElementById("chart-start").textContent = formatDate(chartPoints[0].event_timestamp);
  document.getElementById("chart-end").textContent = formatDate(chartPoints[chartPoints.length - 1].event_timestamp);
}

async function loadTimeline() {
  try {
    const result = await requestJson(`/api/timeline/${userSelect.value}`);
    chartPoints = result.points;
    drawChart();
  } catch {
    chartPoints = [];
  }
}

function renderComparison(result) {
  const container = document.getElementById("pit-results");
  container.replaceChildren();
  const notice = document.createElement("div");
  notice.className = `pit-outcome${result.excluded_future_snapshot ? " leak" : ""}`;
  notice.textContent = result.excluded_future_snapshot
    ? `Future snapshot excluded · ${formatDate(result.next_snapshot_after_cutoff.event_timestamp)}`
    : "No later snapshot exists for this cutoff";
  container.append(notice);
  if (!result.next_snapshot_after_cutoff) {
    container.hidden = false;
    return;
  }
  const table = document.createElement("div");
  table.className = "compare-table";
  const header = document.createElement("div");
  header.className = "compare-head";
  for (const text of ["FEATURE", "AS OF EVENT", "NEXT SNAPSHOT"] ) {
    const cell = document.createElement("span");
    cell.textContent = text;
    header.append(cell);
  }
  table.append(header);
  for (const [name, label] of Object.entries(labels)) {
    const row = document.createElement("div");
    row.className = "compare-row";
    const title = document.createElement("span");
    title.textContent = label;
    const historical = document.createElement("strong");
    historical.textContent = Number(result.historical_features[name]).toLocaleString();
    const future = document.createElement("strong");
    future.textContent = Number(result.next_snapshot_after_cutoff.features[name]).toLocaleString();
    row.append(title, historical, future);
    table.append(row);
  }
  container.append(table);
  container.hidden = false;
}

async function runPointInTime() {
  const button = document.getElementById("run-pit");
  const output = document.getElementById("pit-results");
  button.disabled = true;
  button.textContent = "...";
  try {
    const result = await requestJson("/api/point-in-time", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: Number(userSelect.value),
        event_timestamp: new Date(document.getElementById("event-cutoff").value).toISOString(),
      }),
    });
    renderComparison(result);
  } catch (error) {
    output.replaceChildren();
    const message = document.createElement("div");
    message.className = "pit-error";
    message.textContent = error.message;
    output.append(message);
    output.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Check";
  }
}

async function initialize() {
  try {
    const [metadata, userData] = await Promise.all([
      requestJson("/api/metadata"),
      requestJson("/api/users"),
    ]);
    document.getElementById("store-value").textContent = metadata.online_store.toUpperCase();
    document.getElementById("hero-store-name").textContent = `${metadata.online_store.toUpperCase()} online store`;
    document.getElementById("feature-count").textContent = metadata.feature_count;
    document.getElementById("training-rows").textContent = Number(metadata.training_rows).toLocaleString();
    document.getElementById("model-auc").textContent = Number(metadata.roc_auc).toFixed(3);
    for (const userId of userData.users) {
      const option = document.createElement("option");
      option.value = userId;
      option.textContent = `User ${userId}`;
      userSelect.append(option);
    }
    document.getElementById("event-cutoff").value = localInputValue(metadata.recommended_cutoff);
    await Promise.all([runPrediction(), loadTimeline()]);
    await runPointInTime();
  } catch (error) {
    document.getElementById("store-value").textContent = "UNAVAILABLE";
    document.getElementById("result-placeholder").textContent = error.message;
  }
}

document.getElementById("run-prediction").addEventListener("click", runPrediction);
document.getElementById("run-pit").addEventListener("click", runPointInTime);
userSelect.addEventListener("change", () => {
  runPrediction();
  loadTimeline();
});
window.addEventListener("resize", drawChart);
initialize();
const form = document.getElementById("analyzeForm");
const input = document.getElementById("urlInput");
const analysisMode = document.getElementById("analysisMode");
const modeButtons = Array.from(document.querySelectorAll(".mode-btn"));
const sampleChips = Array.from(document.querySelectorAll(".sample-chip"));
const endpointLabel = document.getElementById("endpointLabel");
const modeHint = document.getElementById("modeHint");
const statusChip = document.getElementById("statusChip");
const modelStatus = document.getElementById("modelStatus");
const scoreLabel = document.getElementById("scoreLabel");
const riskScore = document.getElementById("riskScore");
const riskBar = document.getElementById("riskBar");
const verdict = document.getElementById("verdict");
const confidenceHeading = document.getElementById("confidenceHeading");
const confidence = document.getElementById("confidence");
const confidenceLabel = document.getElementById("confidenceLabel");
const lastChecked = document.getElementById("lastChecked");
const analysisTrack = document.getElementById("analysisTrack");
const checkedUrl = document.getElementById("checkedUrl");
const briefTitle = document.getElementById("briefTitle");
const briefSummary = document.getElementById("briefSummary");
const signalList = document.getElementById("signalList");
const probabilityCard = document.getElementById("probabilityCard");
const probabilityList = document.getElementById("probabilityList");
const historyList = document.getElementById("historyList");
const clearBtn = document.getElementById("clearBtn");
const mitigationBadge = document.getElementById("mitigationBadge");
const mitigationTitle = document.getElementById("mitigationTitle");
const mitigationDetail = document.getElementById("mitigationDetail");
const binaryMetricValue = document.getElementById("binaryMetricValue");
const binaryMetricMeta = document.getElementById("binaryMetricMeta");
const multiclassMetricValue = document.getElementById("multiclassMetricValue");
const multiclassMetricMeta = document.getElementById("multiclassMetricMeta");
const baselineMetricValue = document.getElementById("baselineMetricValue");
const baselineMetricMeta = document.getElementById("baselineMetricMeta");
const robustnessMetricValue = document.getElementById("robustnessMetricValue");
const robustnessMetricMeta = document.getElementById("robustnessMetricMeta");
const generalizationMetricValue = document.getElementById("generalizationMetricValue");
const generalizationMetricMeta = document.getElementById("generalizationMetricMeta");
const evaluationInsights = document.getElementById("evaluationInsights");

const history = [];
const endpointByTask = {
  binary: "/api/classify",
  multiclass: "/api/classify-multiclass",
};

const modeConfig = {
  binary: {
    title: "Binary phishing risk",
    endpoint: "/api/classify",
    hint: "Binary mode gives a phishing score, a confidence estimate, and a suggested action.",
    placeholder: "https://example.com/login",
    scoreLabel: "Risk Score",
    confidenceHeading: "Model Confidence",
    emptyConfidence: "No inference yet",
  },
  multiclass: {
    title: "Multiclass URL family",
    endpoint: "/api/classify-multiclass",
    hint: "Multiclass mode predicts whether a URL looks benign, defacement, malware, or phishing.",
    placeholder: "https://suspicious-domain.example/path",
    scoreLabel: "Top Class Probability",
    confidenceHeading: "Predicted Class",
    emptyConfidence: "No multiclass inference yet",
  },
};

const suspiciousKeywords = [
  "login",
  "verify",
  "update",
  "secure",
  "account",
  "bank",
  "signin",
  "password",
  "confirm",
  "invoice",
  "support",
];

const suspiciousTlds = [
  "zip",
  "mov",
  "top",
  "xyz",
  "work",
  "support",
  "click",
  "country",
  "gq",
  "cf",
  "tk",
  "ml",
];

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const url = input.value.trim();
  if (!url) return;

  setLoading(true);
  const result = await classifyUrl(url, currentTaskType());
  renderResult(url, result);
  setLoading(false);
});

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const nextMode = button.dataset.mode || "binary";
    applyModePresentation(nextMode);
    resetDashboard(nextMode);
  });
});

sampleChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.sample || "";
    input.focus();
  });
});

clearBtn.addEventListener("click", () => {
  input.value = "";
  resetDashboard(currentTaskType());
  input.focus();
});

applyModePresentation(currentTaskType());
resetDashboard(currentTaskType());
loadEvaluationSnapshot();

function currentTaskType() {
  return analysisMode.value === "multiclass" ? "multiclass" : "binary";
}

function applyModePresentation(taskType) {
  const config = modeConfig[taskType] || modeConfig.binary;
  analysisMode.value = taskType;
  endpointLabel.textContent = config.endpoint;
  modeHint.textContent = config.hint;
  input.placeholder = config.placeholder;
  scoreLabel.textContent = config.scoreLabel;
  confidenceHeading.textContent = config.confidenceHeading;
  confidenceLabel.textContent = config.emptyConfidence;
  analysisTrack.textContent = config.title;

  modeButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === taskType);
  });
}

function setLoading(isLoading) {
  if (isLoading) {
    statusChip.textContent = "Analyzing";
    statusChip.style.background = "rgba(20, 52, 74, 0.12)";
    statusChip.style.color = "var(--navy)";
    statusChip.style.borderColor = "rgba(20, 52, 74, 0.18)";
    return;
  }

  statusChip.textContent = "Idle";
  statusChip.style.background = "rgba(21, 35, 51, 0.08)";
  statusChip.style.color = "var(--muted)";
  statusChip.style.borderColor = "transparent";
}

async function classifyUrl(url, taskType) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3500);
    const response = await fetch(endpointByTask[taskType], {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error("API unavailable");
    }

    const payload = await response.json();
    if (!payload || !payload.label) {
      throw new Error("Invalid API response");
    }

    modelStatus.textContent = `${payload.model || "Live API"} · online`;
    return {
      label: payload.label,
      taskType: payload.task_type || taskType,
      score: clampNumber(payload.score ?? 50, 0, 100),
      confidence: clampNumber(payload.confidence ?? 60, 0, 100),
      signals: payload.signals || [],
      classProbabilities: payload.class_probabilities || [],
      mitigation:
        payload.mitigation ||
        buildMitigation(payload.label, clampNumber(payload.score ?? 50, 0, 100)),
      mode: "api",
    };
  } catch (error) {
    if (taskType === "multiclass") {
      modelStatus.textContent = "Multiclass endpoint unavailable";
      return {
        label: "Unavailable",
        taskType: "multiclass",
        score: 0,
        confidence: 0,
        signals: [
          {
            title: "Multiclass endpoint unavailable",
            detail:
              "The 4-class endpoint is not responding right now. Restart the local server to use the multiclass model.",
          },
        ],
        classProbabilities: [],
        mitigation: {
          severity: "neutral",
          action: "endpoint_unavailable",
          title: "Endpoint unavailable",
          detail: "No multiclass result came back, so the page cannot assign a URL family yet.",
        },
        mode: "offline",
      };
    }

    modelStatus.textContent = "Binary heuristic fallback";
    return heuristicClassify(url);
  }
}

function heuristicClassify(rawUrl) {
  const normalized = normalizeUrl(rawUrl);
  const features = extractFeatures(normalized);
  let score = 15;

  if (features.isIp) score += 22;
  if (features.hasAtSymbol) score += 15;
  if (features.hasPunycode) score += 18;
  if (features.isHttp) score += 8;
  if (features.longUrl) score += 10;
  if (features.manySubdomains) score += 12;
  if (features.suspiciousTld) score += 10;
  if (features.hasKeyword) score += 16;
  if (features.hasExcessDelimiters) score += 9;
  if (features.queryHeavy) score += 6;

  score = clampNumber(score, 0, 100);
  const confidenceScore = clampNumber(55 + (score - 50) * 0.6, 45, 92);
  const label = score >= 65 ? "Likely phishing" : score >= 35 ? "Suspicious" : "Likely benign";

  return {
    label,
    taskType: "binary",
    score,
    confidence: confidenceScore,
    signals: buildSignals(features),
    classProbabilities: [],
    mitigation: buildMitigation(label, score),
    mode: "heuristic",
  };
}

function buildMitigation(label, score) {
  if (label === "Likely phishing" || label === "Phishing" || label === "Malware") {
    return {
      severity: "high",
      action: "block_and_report",
      title: "Block and escalate",
      detail: "Treat this as high risk. Block the link, hide previews, and send it for review.",
    };
  }
  if (label === "Suspicious" || label === "Defacement") {
    return {
      severity: "medium",
      action: "quarantine_for_review",
      title: "Hold for review",
      detail: "Keep this link behind a warning and review it before trusting it.",
    };
  }
  if (label === "Unavailable") {
    return {
      severity: "neutral",
      action: "endpoint_unavailable",
      title: "Endpoint unavailable",
      detail: "The selected analysis path did not return a usable result.",
    };
  }
  return {
    severity: "low",
    action: "allow_and_monitor",
    title: "Allow with monitoring",
    detail: `This looks safe for now. Allow it, but keep the run in the audit trail because the score is ${score}/100.`,
  };
}

function normalizeUrl(url) {
  const trimmed = url.trim();
  if (!/^https?:\/\//i.test(trimmed)) {
    return `https://${trimmed}`;
  }
  return trimmed;
}

function extractFeatures(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch (error) {
    return {
      invalid: true,
      isIp: false,
      hasAtSymbol: url.includes("@"),
      hasPunycode: url.includes("xn--"),
      isHttp: url.startsWith("http://"),
      longUrl: url.length > 80,
      manySubdomains: false,
      suspiciousTld: false,
      hasKeyword: containsKeyword(url),
      hasExcessDelimiters: countDelimiters(url) > 6,
      queryHeavy: url.includes("?") && url.split("?")[1].length > 25,
    };
  }

  const host = parsed.hostname.toLowerCase();
  const hostParts = host.split(".").filter(Boolean);
  const tld = hostParts[hostParts.length - 1] || "";

  return {
    invalid: false,
    isIp: /^\d{1,3}(\.\d{1,3}){3}$/.test(host),
    hasAtSymbol: parsed.href.includes("@"),
    hasPunycode: host.includes("xn--"),
    isHttp: parsed.protocol === "http:",
    longUrl: parsed.href.length > 80,
    manySubdomains: hostParts.length >= 4,
    suspiciousTld: suspiciousTlds.includes(tld),
    hasKeyword: containsKeyword(parsed.pathname + parsed.search + host),
    hasExcessDelimiters: countDelimiters(parsed.href) > 6,
    queryHeavy: parsed.search.length > 30,
  };
}

function containsKeyword(text) {
  const lower = text.toLowerCase();
  return suspiciousKeywords.some((keyword) => lower.includes(keyword));
}

function countDelimiters(text) {
  return (text.match(/[\-_.]/g) || []).length;
}

function buildSignals(features) {
  const signals = [];

  if (features.invalid) {
    signals.push({
      title: "Invalid URL format",
      detail: "The input could not be parsed cleanly.",
    });
  }
  if (features.isIp) {
    signals.push({ title: "Raw IP address", detail: "The link points to an IP address instead of a normal domain." });
  }
  if (features.hasAtSymbol) {
    signals.push({ title: "@ symbol present", detail: "This can hide the real destination." });
  }
  if (features.hasPunycode) {
    signals.push({ title: "Punycode detected", detail: "This may be using a lookalike domain." });
  }
  if (features.isHttp) {
    signals.push({ title: "HTTP protocol", detail: "The link is using HTTP instead of HTTPS." });
  }
  if (features.longUrl) {
    signals.push({ title: "Long URL", detail: "The URL is unusually long, which can hide suspicious routing." });
  }
  if (features.manySubdomains) {
    signals.push({ title: "Many subdomains", detail: "Too many subdomains can be a sign of impersonation." });
  }
  if (features.suspiciousTld) {
    signals.push({ title: "High-risk TLD", detail: "This top-level domain shows up often in abuse cases." });
  }
  if (features.hasKeyword) {
    signals.push({ title: "Sensitive keywords", detail: "The URL includes words that often appear in phishing links." });
  }
  if (features.hasExcessDelimiters) {
    signals.push({ title: "Delimiter-heavy", detail: "Repeated dashes and dots can make a link harder to read." });
  }
  if (features.queryHeavy) {
    signals.push({ title: "Large query string", detail: "The link has a long query string with a lot of parameters." });
  }

  if (!signals.length) {
    signals.push({ title: "No strong heuristic flags", detail: "Nothing unusual stood out in the basic URL structure." });
  }
  return signals;
}

function renderResult(url, result) {
  const taskType = result.taskType || "binary";
  applyModePresentation(taskType);

  riskScore.textContent =
    taskType === "multiclass" ? `${Math.round(result.score)}%` : `${Math.round(result.score)}`;
  riskBar.style.width = `${clampNumber(result.score, 0, 100)}%`;
  riskBar.style.background = barGradientForLabel(result.label);

  verdict.textContent = result.label;
  verdict.style.color = verdictColor(result.label);

  if (taskType === "multiclass") {
    confidence.textContent = result.label;
    confidenceLabel.textContent =
      result.mode === "api"
        ? `${Math.round(result.confidence)}% top-class probability`
        : "No multiclass prediction available";
  } else {
    confidence.textContent = `${Math.round(result.confidence)}%`;
    confidenceLabel.textContent =
      result.mode === "api" ? "Live binary model response" : "Heuristic binary fallback";
  }

  const time = new Date();
  lastChecked.textContent = time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  analysisTrack.textContent = modeConfig[taskType].title;
  checkedUrl.textContent = url;
  checkedUrl.title = url;

  briefTitle.textContent = result.mode === "offline" ? "Model output unavailable" : `${result.label} decision`;
  briefSummary.textContent = buildBriefSummary(result, taskType);

  renderSignals(result.signals || []);
  renderProbabilities(result.classProbabilities || [], taskType);
  renderMitigation(result.mitigation || buildMitigation(result.label, result.score));
  updateHistory({
    url,
    score: result.score,
    label: result.label,
    taskType,
    time,
  });
}

function buildBriefSummary(result, taskType) {
  if (result.mode === "offline") {
    return "The selected model did not respond, so this run could not return a reliable result.";
  }

  if (taskType === "multiclass") {
    const ranked = (result.classProbabilities || [])
      .slice(0, 2)
      .map((item) => `${item.label} ${Math.round(item.probability)}%`)
      .join(" · ");
    return ranked
      ? `Top multiclass guesses: ${ranked}.`
      : "The multiclass model returned a label without a detailed breakdown.";
  }

  return result.mitigation?.detail || "The binary model returned a result.";
}

function renderSignals(signals) {
  signalList.innerHTML = "";

  if (!signals.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "Run a URL to inspect the strongest signals behind the result.";
    signalList.appendChild(empty);
    return;
  }

  signals.forEach((signal) => {
    const item = document.createElement("li");
    const body = document.createElement("div");
    const title = document.createElement("strong");
    const detail = document.createElement("span");

    title.textContent = signal.title || "Signal";
    detail.textContent = signal.detail || "";
    body.appendChild(title);
    body.appendChild(detail);
    item.appendChild(body);
    signalList.appendChild(item);
  });
}

function renderProbabilities(classProbabilities, taskType) {
  if (taskType !== "multiclass") {
    probabilityCard.hidden = true;
    probabilityList.innerHTML = '<li class="empty">Run a multiclass analysis to see the class breakdown.</li>';
    return;
  }

  probabilityCard.hidden = false;
  probabilityList.innerHTML = "";

  if (!classProbabilities.length) {
    probabilityList.innerHTML = '<li class="empty">Run a multiclass analysis to see the class breakdown.</li>';
    return;
  }

  classProbabilities.forEach((item) => {
    const probability = clampNumber(Number(item.probability) || 0, 0, 100);
    const li = document.createElement("li");
    li.className = "probability-item";

    const head = document.createElement("div");
    head.className = "probability-head";

    const label = document.createElement("strong");
    label.textContent = item.label || "Unknown";

    const value = document.createElement("span");
    value.className = "mono";
    value.textContent = `${Math.round(probability)}%`;

    head.appendChild(label);
    head.appendChild(value);

    const meter = document.createElement("div");
    meter.className = "probability-meter";

    const fill = document.createElement("div");
    fill.className = "probability-fill";
    fill.style.width = `${probability}%`;

    meter.appendChild(fill);
    li.appendChild(head);
    li.appendChild(meter);
    probabilityList.appendChild(li);
  });
}

function updateHistory(entry) {
  history.unshift(entry);
  history.splice(5);

  historyList.innerHTML = "";

  history.forEach((item) => {
    const li = document.createElement("li");
    const left = document.createElement("div");
    const right = document.createElement("div");
    const title = document.createElement("strong");
    const meta = document.createElement("span");
    const score = document.createElement("strong");
    const time = document.createElement("span");

    title.textContent = item.label;
    meta.textContent = `${modeConfig[item.taskType].title} · ${truncateUrl(item.url, 54)}`;
    left.appendChild(title);
    left.appendChild(meta);

    score.textContent = item.taskType === "multiclass" ? `${Math.round(item.score)}%` : `${Math.round(item.score)}`;
    score.style.color = verdictColor(item.label);
    time.textContent = item.time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    right.appendChild(score);
    right.appendChild(time);

    li.appendChild(left);
    li.appendChild(right);
    historyList.appendChild(li);
  });
}

function renderMitigation(mitigation) {
  const nextMitigation = mitigation || buildMitigation("Suspicious", 50);
  mitigationTitle.textContent = nextMitigation.title;
  mitigationDetail.textContent = nextMitigation.detail;
  mitigationBadge.textContent = String(nextMitigation.action || "no_action").replaceAll("_", " ");
  mitigationBadge.className = `mitigation-badge ${nextMitigation.severity || "neutral"}`;
}

async function loadEvaluationSnapshot() {
  const [binary, multiclass, stress, crossDataset, baseline] = await Promise.all([
    fetchJson("outputs/cnn_metrics.json"),
    fetchJson("outputs/cnn_multiclass_metrics.json"),
    fetchJson("outputs/stress_test_metrics.json"),
    fetchJson("outputs/cross_dataset_metrics.json"),
    fetchJson("outputs/baseline_metrics_random_forest.json"),
  ]);

  if (binary) {
    binaryMetricValue.textContent = formatDecimal(binary.test?.roc_auc);
    binaryMetricMeta.textContent = `ROC-AUC ${formatDecimal(binary.test?.roc_auc)} · Accuracy ${formatPercent(binary.test?.accuracy)} · F1 ${formatDecimal(binary.test?.f1_1)} on ${formatCount(binary.split_sizes?.test)} test URLs.`;
  } else {
    binaryMetricMeta.textContent = "Binary metrics could not be loaded.";
  }

  if (multiclass) {
    multiclassMetricValue.textContent = formatDecimal(multiclass.test?.f1_macro);
    multiclassMetricMeta.textContent = `Macro F1 ${formatDecimal(multiclass.test?.f1_macro)} · Accuracy ${formatPercent(multiclass.test?.accuracy)} across 4 classes.`;
  } else {
    multiclassMetricMeta.textContent = "Multiclass metrics could not be loaded.";
  }

  if (baseline) {
    baselineMetricValue.textContent = formatDecimal(baseline.test?.roc_auc);
    baselineMetricMeta.textContent = `Random Forest · Accuracy ${formatPercent(baseline.test?.accuracy)} · F1 ${formatDecimal(baseline.test?.f1_1)}.`;
  } else {
    baselineMetricMeta.textContent = "Baseline metrics could not be loaded.";
  }

  if (stress) {
    const cleanAcc = Number(stress.clean_acc ?? stress.clean?.accuracy ?? 0);
    const stressAcc = Number(stress.stress_acc ?? stress.stress?.accuracy ?? 0);
    const drop = cleanAcc - stressAcc;
    robustnessMetricValue.textContent = `${(drop * 100).toFixed(1)} pts`;
    robustnessMetricMeta.textContent = `Accuracy drops from ${formatPercent(cleanAcc)} to ${formatPercent(stressAcc)} under obfuscation stress.`;
  } else {
    robustnessMetricMeta.textContent = "Stress-test metrics could not be loaded.";
  }

  if (crossDataset) {
    const dataset2 = crossDataset["Dataset 2 (Binary)"];
    const dataset3 = crossDataset["Dataset 3 (Binary)"];
    generalizationMetricValue.textContent = formatDecimal(dataset3?.roc_auc);
    generalizationMetricMeta.textContent = `Dataset 2 ROC-AUC ${formatDecimal(dataset2?.roc_auc)} · Dataset 3 ROC-AUC ${formatDecimal(dataset3?.roc_auc)}.`;
  } else {
    generalizationMetricMeta.textContent = "Cross-dataset metrics could not be loaded.";
  }

  renderEvaluationInsights({ binary, multiclass, stress, crossDataset, baseline });
}

function renderEvaluationInsights(payload) {
  const insights = [];
  const { binary, multiclass, stress, crossDataset, baseline } = payload;

  if (binary) {
    insights.push(
      `The binary CNN reaches ROC-AUC ${formatDecimal(binary.test?.roc_auc)} with F1 ${formatDecimal(binary.test?.f1_1)} on dataset1_binary.`
    );
  }
  if (multiclass) {
    insights.push(
      `The multiclass CNN reaches macro F1 ${formatDecimal(multiclass.test?.f1_macro)} across benign, defacement, malware, and phishing.`
    );
  }
  if (baseline) {
    insights.push(
      `Random Forest is still the strongest lexical baseline here at ROC-AUC ${formatDecimal(baseline.test?.roc_auc)}.`
    );
  }
  if (stress) {
    insights.push(
      `Obfuscation still hurts performance: accuracy falls from ${formatPercent(stress.clean_acc)} to ${formatPercent(stress.stress_acc)}.`
    );
  }
  if (crossDataset) {
    insights.push(
      `Cross-dataset generalization is still the weakest part of the pipeline: Dataset 2 ROC-AUC ${formatDecimal(crossDataset["Dataset 2 (Binary)"]?.roc_auc)}, Dataset 3 ROC-AUC ${formatDecimal(crossDataset["Dataset 3 (Binary)"]?.roc_auc)}.`
    );
  }

  evaluationInsights.innerHTML = "";
  if (!insights.length) {
    const li = document.createElement("li");
    li.textContent = "Saved evaluation artifacts could not be loaded.";
    evaluationInsights.appendChild(li);
    return;
  }

  insights.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    evaluationInsights.appendChild(li);
  });
}

async function fetchJson(path) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load ${path}`);
    }
    return await response.json();
  } catch (error) {
    return null;
  }
}

function verdictColor(label) {
  const normalized = String(label).toLowerCase();
  if (normalized.includes("malware") || normalized.includes("phish")) {
    return "var(--danger)";
  }
  if (normalized.includes("suspicious") || normalized.includes("defacement")) {
    return "var(--warn)";
  }
  if (normalized.includes("unavailable")) {
    return "var(--muted)";
  }
  return "var(--safe)";
}

function barGradientForLabel(label) {
  const normalized = String(label).toLowerCase();
  if (normalized.includes("malware") || normalized.includes("phish")) {
    return "linear-gradient(90deg, #d27656, var(--danger))";
  }
  if (normalized.includes("suspicious") || normalized.includes("defacement")) {
    return "linear-gradient(90deg, #d6a146, var(--warn))";
  }
  if (normalized.includes("unavailable")) {
    return "linear-gradient(90deg, #c8ccd2, #9aa4b2)";
  }
  return "linear-gradient(90deg, #7eb38d, var(--safe))";
}

function formatDecimal(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return number.toFixed(3);
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return `${(number * 100).toFixed(1)}%`;
}

function formatCount(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return new Intl.NumberFormat("en-US").format(number);
}

function truncateUrl(url, maxLength) {
  if (url.length <= maxLength) return url;
  return `${url.slice(0, Math.max(18, maxLength - 20))}...${url.slice(-14)}`;
}

function resetDashboard(taskType) {
  const config = modeConfig[taskType] || modeConfig.binary;
  riskScore.textContent = "--";
  riskBar.style.width = "0%";
  riskBar.style.background = "linear-gradient(90deg, #7eb38d, var(--safe))";
  verdict.textContent = "Awaiting analysis";
  verdict.style.color = "var(--muted)";
  confidence.textContent = "--";
  confidenceLabel.textContent = config.emptyConfidence;
  lastChecked.textContent = "--:--";
  analysisTrack.textContent = config.title;
  checkedUrl.textContent = "No URL checked yet";
  checkedUrl.removeAttribute("title");
  briefTitle.textContent = "No URL checked yet";
  briefSummary.textContent = "Run a URL to see the model output and the suggested next step.";
  renderSignals([]);
  renderProbabilities([], taskType);
  renderMitigation({
    severity: "neutral",
    action: "no_action",
    title: "Awaiting analysis",
    detail: "The result will suggest whether the URL should be allowed, reviewed, or blocked.",
  });
}

function clampNumber(value, min, max) {
  return Math.min(Math.max(Number(value), min), max);
}

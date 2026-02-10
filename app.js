const form = document.getElementById("analyzeForm");
const input = document.getElementById("urlInput");
const statusChip = document.getElementById("statusChip");
const modelStatus = document.getElementById("modelStatus");
const riskScore = document.getElementById("riskScore");
const riskBar = document.getElementById("riskBar");
const verdict = document.getElementById("verdict");
const confidence = document.getElementById("confidence");
const confidenceLabel = document.getElementById("confidenceLabel");
const lastChecked = document.getElementById("lastChecked");
const checkedUrl = document.getElementById("checkedUrl");
const signalList = document.getElementById("signalList");
const historyList = document.getElementById("historyList");
const clearBtn = document.getElementById("clearBtn");

const history = [];

const verdicts = [
  { label: "Likely benign", threshold: 35, color: "var(--safe)" },
  { label: "Suspicious", threshold: 65, color: "var(--warn)" },
  { label: "Likely phishing", threshold: 101, color: "var(--danger)" },
];

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
  const result = await classifyUrl(url);
  renderResult(url, result);
  setLoading(false);
});

clearBtn.addEventListener("click", () => {
  input.value = "";
  input.focus();
});

function setLoading(isLoading) {
  statusChip.textContent = isLoading ? "Analyzing" : "Idle";
  statusChip.style.background = isLoading
    ? "rgba(45, 212, 191, 0.2)"
    : "rgba(148, 163, 184, 0.2)";
  statusChip.style.color = isLoading ? "var(--accent)" : "var(--muted)";
}

async function classifyUrl(url) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const response = await fetch("/api/classify", {
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
      score: clampNumber(payload.score ?? 50, 0, 100),
      confidence: clampNumber(payload.confidence ?? 60, 0, 100),
      signals: payload.signals || [],
      mode: "api",
    };
  } catch (error) {
    modelStatus.textContent = "Mock model · heuristic mode";
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
    score,
    confidence: confidenceScore,
    signals: buildSignals(features),
    mode: "heuristic",
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
      detail: "Input could not be parsed, which often appears in obfuscated links.",
    });
  }
  if (features.isIp) {
    signals.push({ title: "Raw IP address", detail: "Host is an IP instead of a domain." });
  }
  if (features.hasAtSymbol) {
    signals.push({ title: "@ symbol", detail: "May hide the true destination." });
  }
  if (features.hasPunycode) {
    signals.push({
      title: "Punycode detected",
      detail: "Potential homoglyph or IDN obfuscation.",
    });
  }
  if (features.isHttp) {
    signals.push({ title: "HTTP protocol", detail: "Unencrypted HTTP link." });
  }
  if (features.longUrl) {
    signals.push({
      title: "Long URL",
      detail: "Excessive length can hide suspicious components.",
    });
  }
  if (features.manySubdomains) {
    signals.push({
      title: "Many subdomains",
      detail: "Nested subdomains often mimic trusted domains.",
    });
  }
  if (features.suspiciousTld) {
    signals.push({
      title: "High-risk TLD",
      detail: "Known for abuse in phishing campaigns.",
    });
  }
  if (features.hasKeyword) {
    signals.push({
      title: "Sensitive keywords",
      detail: "Contains login or account-related language.",
    });
  }
  if (features.hasExcessDelimiters) {
    signals.push({
      title: "Delimiter-heavy",
      detail: "Multiple dashes or dots can obscure the true domain.",
    });
  }
  if (features.queryHeavy) {
    signals.push({
      title: "Large query string",
      detail: "Long parameters can disguise redirects or tokens.",
    });
  }

  if (signals.length === 0) {
    signals.push({ title: "No major signals", detail: "URL appears structurally normal." });
  }

  return signals;
}

function renderResult(url, result) {
  const verdictInfo = verdicts.find((item) => result.score < item.threshold) || verdicts[2];
  riskScore.textContent = `${result.score.toFixed(0)}`;
  riskBar.style.width = `${result.score}%`;
  verdict.textContent = verdictInfo.label;
  verdict.style.color = verdictInfo.color;

  confidence.textContent = `${result.confidence.toFixed(0)}%`;
  confidenceLabel.textContent = result.mode === "api" ? "Live model response" : "Heuristic estimate";

  const time = new Date();
  lastChecked.textContent = time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  // Shorten long URLs and style the ellipsis in-place
  if (url.length > 50) {
    const shortened = url.slice(0, 30) + "..." + url.slice(-15);
    checkedUrl.innerHTML = shortened.replace(
      "...",
      '<span style="color:gray;font-style:italic">...</span>'
    );
  } else {
    checkedUrl.textContent = url;
  }

  renderSignals(result.signals);
  updateHistory({ url, score: result.score, label: verdictInfo.label, time });
}

function renderSignals(signals) {
  signalList.innerHTML = "";
  signals.forEach((signal) => {
    const item = document.createElement("li");
    item.innerHTML = `<strong>${signal.title}</strong><span>${signal.detail}</span>`;
    signalList.appendChild(item);
  });
}

function updateHistory(entry) {
  history.unshift(entry);
  history.splice(5);

  historyList.innerHTML = "";
  history.forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${item.label}</span><span class="mono">${item.score.toFixed(
      0
    )}</span><span>${item.time.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    })}</span>`;
    historyList.appendChild(li);
  });
}

function clampNumber(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

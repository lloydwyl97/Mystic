// Mystic Operator Console — dual-engine DAY + SCALP dashboard
const DASHBOARD_VERSION = 54;
const REFRESH_MS = 90000;
const CANONICAL_REFRESH_MS = 30000;
const SECONDARY_POLL_MS = 3000;
const SCALP_STATUS_TIMEOUT_MS = 30000;
const CANONICAL_TIMEOUT_MS = 90000;
const SLOW_ENDPOINT_TIMEOUT_MS = 90000;
const CANONICAL_ENDPOINT = {
    path: "/api/portfolio-engine/dashboard-canonical",
    key: "dashboardCanonical",
    timeoutMs: CANONICAL_TIMEOUT_MS,
};

const LAZY_TAB_ENDPOINTS = {
    learning: [
        { path: "/api/ai-diagnostics/full", key: "aiDiagnosticsFull", timeoutMs: SLOW_ENDPOINT_TIMEOUT_MS },
        { path: "/api/ai-diagnostics/learning-health", key: "learningHealth" },
    ],
    // Feature health panel lives under Settings — load diagnostics when that tab opens.
    settings: [
        { path: "/api/ai-diagnostics/full", key: "aiDiagnosticsFull", timeoutMs: SLOW_ENDPOINT_TIMEOUT_MS },
    ],
    performance: [],
    day: [
        { path: "/api/portfolio-engine/day-health", key: "dayHealth" },
        { path: "/api/portfolio-engine/decisions", key: "decisions" },
        { path: "/api/portfolio-engine/rejects", key: "rejects" },
    ],
    scalp: [
        { path: "/api/scalp/status", key: "scalpStatus", timeoutMs: SCALP_STATUS_TIMEOUT_MS },
        { path: "/api/scalp/strategies", key: "scalpStrategies" },
        { path: "/api/scalp/positions", key: "scalpPositions" },
        { path: "/api/scalp/trades?limit=100", key: "scalpTrades" },
        { path: "/api/scalp/scoreboard?days=14", key: "scalpScoreboard" },
        { path: "/api/scalp/attribution", key: "scalpAttribution" },
        { path: "/api/scalp/learning-summary", key: "scalpLearning" },
    ],
    marketlens: [
        { path: "/api/public/mystic-marketlens-feed", key: "marketlensFeed" },
    ],
};
const BACKGROUND_ENDPOINTS = [
    { path: "/api/performance/display-context", key: "perfDisplayContext" },
    { path: "/api/system/health/quick", key: "systemHealth" },
    { path: "/api/system/health/comprehensive", key: "systemHealthFull" },
    { path: "/api/system/process-health", key: "processHealth" },
    { path: "/api/portfolio-engine/latency", key: "latency" },
    { path: "/api/portfolio-engine/invariants-detail", key: "invariantsDetail" },
    { path: "/api/portfolio-engine/learning-status?limit=20", key: "learningStatus" },
    { path: "/api/portfolio-engine/live-readiness", key: "liveReadiness" },
    { path: "/api/portfolio-engine/model-panel", key: "modelPanel" },
    { path: "/api/ai-diagnostics/missed-opportunities?limit=30", key: "missedOpportunities" },
    { path: "/api/portfolio-engine/trading-economics", key: "tradingEconomics" },
    { path: "/api/portfolio-engine/scoreboard?days=7", key: "scoreboard7d" },
];
// Legacy alias — refresh button iterates this list
const ENDPOINTS = BACKGROUND_ENDPOINTS.concat(
    (LAZY_TAB_ENDPOINTS.performance || []),
    (LAZY_TAB_ENDPOINTS.day || []),
    (LAZY_TAB_ENDPOINTS.scalp || []),
    (LAZY_TAB_ENDPOINTS.marketlens || [])
);

window._lastDashboardCanonical = null;
window._lastScalpPositions = null;
window._lastScalpTrades = null;

const CANONICAL_STALE_WIDGET_IDS = [
    "analytics-risk-status",
    "analytics-invariants",
    "analytics-xcheck",
    "status-health",
];

function setCardText(id, text, opts) {
    opts = opts || {};
    const el = typeof id === "string" ? document.getElementById(id) : id;
    if (!el) return;
    const s = text != null && text !== "" ? String(text) : "--";
    el.textContent = s;
    if (opts.title != null) {
        el.title = String(opts.title);
    } else if (s !== "--" && s.length > 14) {
        el.title = s;
    } else if (!opts.keepTitle) {
        el.title = "";
    }
    if (s !== "--" && s.length > 14) {
        el.classList.add("has-long-text");
        el.classList.toggle("has-very-long-text", s.length > 32);
    } else {
        el.classList.remove("has-long-text", "has-very-long-text");
    }
    if (opts.warn != null) {
        el.classList.toggle("readiness-warn", !!opts.warn);
    }
}

function formatPnlWithSells(pnl, sells) {
    if (pnl == null || Number.isNaN(Number(pnl))) return null;
    const amt = "$" + Number(pnl).toFixed(2);
    const n = sells != null ? Number(sells) : 0;
    return amt + "\n(" + n + " sell" + (n === 1 ? "" : "s") + ")";
}

function formatScalpOperationalMode(mode) {
    const map = {
        runner_dead: "Runner dead",
        max_open_positions_reached: "Max open — waiting for exit",
        exit_watch_active: "Exit watch active",
        entry_rejected_by_strategy: "Entry rejected (strategy)",
        entry_scan_active: "Entry scan active",
    };
    return map[mode] || mode || "--";
}

function formatScalpDecision(decision, blocker, operationalMode) {
    if (operationalMode) {
        const op = formatScalpOperationalMode(operationalMode);
        if (blocker && operationalMode === "entry_rejected_by_strategy") {
            return op + "\n" + String(blocker);
        }
        if (operationalMode === "max_open_positions_reached" || operationalMode === "exit_watch_active") {
            return op;
        }
        if (!decision && !blocker) return op;
    }
    if (!decision && !blocker) return "--";
    if (blocker) return String(decision || "BLOCKED") + "\n" + String(blocker);
    return String(decision);
}

function setPnlCard(id, pnl, sells) {
    const el = document.getElementById(id);
    if (!el) return;
    const formatted = formatPnlWithSells(pnl, sells);
    setCardText(el, formatted != null ? formatted : "--");
    el.classList.remove("pnl-pos", "pnl-neg");
    if (pnl != null && !Number.isNaN(Number(pnl))) {
        el.classList.add(Number(pnl) >= 0 ? "pnl-pos" : "pnl-neg");
    }
}

let chartPortfolio = null;
let chartDailyReturns = null;
let chartCumulativeReturns = null;
let chartPnlHistogram = null;
let chartTradeDuration = null;
let chartStrategyPerformance = null;

function init() {
    markDashboardLoading(true);
    try {
        initCharts();
    } catch (e) {
        console.error("Chart.js init failed:", e);
    }
    initTabs();
    startPolling();
    startMarketReadinessPolling();
    updateHeaderMeta();
    setInterval(updateHeaderMeta, 1000);
    initModeSwitch();
    initOperatorControls();
    initRefreshButton();
    initTradeFilters();
    setupHistoricalDiagnosticsToggle();
}

const ENDPOINT_FRESHNESS = {};

function recordEndpointFreshness(key, ok, payloadTs) {
    ENDPOINT_FRESHNESS[key] = {
        ok: !!ok,
        at: Date.now(),
        payloadTs: payloadTs || null,
    };
}

function getDisplayContextFreshness() {
    const st = ENDPOINT_FRESHNESS.perfDisplayContext;
    if (!st) return "Loading";
    if (!st.ok) return "Failed";
    const age = Date.now() - st.at;
    if (age > REFRESH_MS * 2) return "Stale";
    return "Fresh";
}

function todayUtcDateStr() {
    return new Date().toISOString().slice(0, 10);
}

function responseErrorMessage(data, fallback) {
    if (!data || typeof data !== "object") return fallback;
    for (const key of ["message", "detail", "error"]) {
        if (typeof data[key] === "string" && data[key].trim()) return data[key];
    }
    return fallback;
}

function markDashboardLoading(loading) {
    if (loading) {
        CANONICAL_STALE_WIDGET_IDS.forEach(function (id) {
            const el = document.getElementById(id);
            if (!el || el.textContent !== "--") return;
            el.textContent = "Loading…";
            el.title = "Fetching live snapshot (first load can take ~20s)";
        });
        ["status-cash", "status-equity", "status-positions"].forEach(function (id) {
            const el = document.getElementById(id);
            if (el && el.textContent === "--") {
                el.textContent = "Loading…";
                el.title = "Fetching live snapshot (first load can take ~20s)";
            }
        });
        return;
    }
    CANONICAL_STALE_WIDGET_IDS.concat(["status-cash", "status-equity", "status-positions"]).forEach(function (id) {
        const el = document.getElementById(id);
        if (el && el.textContent === "Loading…") {
            el.textContent = "--";
            el.removeAttribute("title");
        }
    });
}

function initTabs() {
    const buttons = document.querySelectorAll(".console-tabs__btn");
    buttons.forEach(function (btn) {
        btn.addEventListener("click", function () {
            const tab = btn.getAttribute("data-tab");
            if (!tab) return;
            buttons.forEach(function (b) {
                b.classList.toggle("console-tabs__btn--active", b === btn);
            });
            document.querySelectorAll(".tab-panel").forEach(function (panel) {
                const active = panel.id === "tab-" + tab;
                panel.classList.toggle("tab-panel--active", active);
                if (active) panel.removeAttribute("hidden");
                else panel.setAttribute("hidden", "hidden");
            });
            const lazy = LAZY_TAB_ENDPOINTS[tab];
            if (lazy && lazy.length) {
                lazy.forEach(function (ep, i) {
                    setTimeout(function () { pollOne(ep); }, i * 400);
                });
            }
        });
    });
}

function initTradeFilters() {
    const bar = document.getElementById("trade-filters");
    if (!bar) return;
    bar.addEventListener("change", function () {
        applyTradeFilters();
    });
}

function applyTradeFilters() {
    const engine = (document.querySelector('input[name="engine-filter"]:checked') || {}).value || "all";
    const time = getSelectedTradeTimeFilter();
    document.querySelectorAll("#tab-positions [data-engine]").forEach(function (el) {
        const eng = el.getAttribute("data-engine");
        el.hidden = engine !== "all" && eng !== engine;
    });
    if (window._lastScalpTrades) updateScalpTradesTable(window._lastScalpTrades, time);
    const canonicalPerformance = window._lastDashboardCanonical && window._lastDashboardCanonical.performance;
    if (window._lastDayTrades || canonicalPerformance) {
        updateTrades({ trades: window._lastDayTrades || canonicalPerformance.trades || [] }, time);
    }
}

const MARKET_READINESS_REFRESH_MS = 90000;

function startMarketReadinessPolling() {
    const ep = {
        path: "/api/portfolio-engine/market-data-readiness",
        key: "marketDataReadiness",
    };
    setTimeout(() => pollOne(ep), 6000);
    setInterval(() => pollOne(ep), MARKET_READINESS_REFRESH_MS);
}

function initRefreshButton() {
    const btn = document.getElementById("refresh-btn");
    if (!btn) return;
    btn.addEventListener("click", function () {
        btn.disabled = true;
        btn.textContent = "…";
        const promises = [pollOne(CANONICAL_ENDPOINT)].concat(
            ENDPOINTS.map(function (ep, i) {
                return new Promise(function (resolve) {
                    setTimeout(function () { resolve(pollOne(ep)); }, i * 400);
                });
            })
        ).concat([
            pollOne({
                path: "/api/portfolio-engine/market-data-readiness",
                key: "marketDataReadiness",
            }),
        ]);
        Promise.all(promises).then(function () {
            btn.textContent = "↻";
            btn.disabled = false;
        });
    });
}

function initOperatorControls() {
    const form = document.getElementById("operator-config-form");
    if (!form) return;
    loadOperatorConfig();
    form.addEventListener("submit", async (ev) => {
        ev.preventDefault();
        const statusEl = document.getElementById("op-config-status");
        const token = prompt("Enter ADMIN_TOKEN to save operator settings:");
        if (!token) return;
        const payload = {
            max_open_positions: Number(document.getElementById("op-max-positions").value),
            live_test_max_open_positions: Number(document.getElementById("op-live-max-positions").value),
            live_test_max_notional: Number(document.getElementById("op-live-max-notional").value),
            risk_per_trade_pct: Number(document.getElementById("op-risk-pct").value),
            max_cash_per_coin_pct: Number(document.getElementById("op-max-cash-coin").value),
            live_test_symbol_allowlist: document.getElementById("op-live-allowlist").value,
            live_test_manual_arm: document.getElementById("op-live-manual-arm").checked,
            kill_switch: document.getElementById("op-kill-switch").value,
            kill_switch_reason: "dashboard_operator_config",
        };
        if (statusEl) {
            statusEl.textContent = "Saving…";
            statusEl.className = "operator-form__status";
        }
        try {
            const res = await fetch("/api/portfolio-engine/operator-config", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: "Bearer " + token,
                },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (data.success && data.data) {
                fillOperatorConfigForm(data.data);
                if (statusEl) {
                    statusEl.textContent = "Saved — limits active without restart.";
                    statusEl.className = "operator-form__status operator-form__status--ok";
                }
                pollOne(CANONICAL_ENDPOINT);
            } else {
                const msg = responseErrorMessage(data, "Save failed");
                if (statusEl) {
                    statusEl.textContent = msg;
                    statusEl.className = "operator-form__status operator-form__status--err";
                }
            }
        } catch (e) {
            if (statusEl) {
                statusEl.textContent = e && e.message ? e.message : "Network error";
                statusEl.className = "operator-form__status operator-form__status--err";
            }
        }
    });
}

function fillOperatorConfigForm(d) {
    if (!d) return;
    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el && val != null) el.value = val;
    };
    setVal("op-max-positions", d.max_open_positions);
    setVal("op-live-max-positions", d.live_test_max_open_positions);
    setVal("op-live-max-notional", d.live_test_max_notional);
    setVal("op-risk-pct", d.risk_per_trade_pct);
    setVal("op-max-cash-coin", d.max_cash_per_coin_pct);
    if (Array.isArray(d.live_test_symbol_allowlist)) {
        setVal("op-live-allowlist", d.live_test_symbol_allowlist.join(","));
    } else if (d.live_test_symbol_allowlist) {
        setVal("op-live-allowlist", d.live_test_symbol_allowlist);
    }
    const arm = document.getElementById("op-live-manual-arm");
    if (arm) arm.checked = !!d.live_test_manual_arm;
    const kill = document.getElementById("op-kill-switch");
    if (kill && d.kill_switch) kill.value = d.kill_switch;
}

async function loadOperatorConfig() {
    try {
        const res = await fetch("/api/portfolio-engine/operator-config", { cache: "no-cache" });
        const data = await res.json();
        if (data.success && data.data) fillOperatorConfigForm(data.data);
    } catch (e) {
        console.warn("operator-config load failed:", e);
    }
}

function initModeSwitch() {
    const badge = document.getElementById("mode-badge");
    if (!badge) return;
    badge.addEventListener("click", async () => {
        const current = (badge.dataset.effectiveMode || badge.textContent || "").toLowerCase();
        const isLive = current === "live";
        const target = isLive ? "PAPER" : "LIVE";
        const goingLive = !isLive;
        const msg = isLive
            ? "Switch to PAPER mode? Real orders will stop."
            : "Enable LIVE trading? Real money orders will be placed.";
        if (!confirm(msg)) return;
        const headers = { "Content-Type": "application/json" };
        if (goingLive) {
            const token = prompt("Enter ADMIN_TOKEN to enable live trading:");
            if (!token) return;
            headers["Authorization"] = "Bearer " + token;
        }
        try {
            const res = await fetch("/api/portfolio-engine/execution-mode", {
                method: "POST",
                headers: headers,
                body: JSON.stringify({
                    mode: target.toLowerCase(),
                    live_trades_allowed: goingLive,
                }),
            });
            const data = await res.json();
            if (data.success && data.data) {
                updateExecutionMode(data);
                pollOne(CANONICAL_ENDPOINT);
            } else {
                const reason = responseErrorMessage(data, "unknown");
                const failures = (data.readiness && data.readiness.failures) ? data.readiness.failures.join(", ") : "";
                alert("Mode switch failed: " + reason + (failures ? "\n\n" + failures : ""));
            }
        } catch (e) {
            alert("Error: " + (e && e.message ? e.message : "network error"));
        }
    });
}

function initCharts() {
    if (typeof Chart === "undefined") {
        console.warn("Chart.js not loaded - charts will not render but data polling will continue");
        return;
    }

    const gridColor = "rgba(43, 49, 57, 0.3)";
    const textColor = "rgba(139, 148, 158, 0.8)";

    const commonChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 750,
            easing: 'easeInOutQuart'
        },
        interaction: {
            intersect: false,
            mode: 'index'
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(26, 31, 38, 0.95)',
                titleColor: '#f0f3f6',
                bodyColor: '#f0f3f6',
                borderColor: 'rgba(43, 49, 57, 0.8)',
                borderWidth: 1,
                padding: 12,
                cornerRadius: 8,
                titleFont: { size: 13, weight: '600', family: 'Space Grotesk' },
                bodyFont: { size: 12, family: 'Space Grotesk' },
                displayColors: true,
                boxPadding: 6
            }
        },
        scales: {
            x: {
                grid: {
                    color: gridColor,
                    drawBorder: false
                },
                ticks: {
                    color: textColor,
                    font: { size: 11 },
                    maxRotation: 0,
                    autoSkipPadding: 10
                }
            },
            y: {
                grid: {
                    color: gridColor,
                    drawBorder: false
                },
                ticks: {
                    color: textColor,
                    font: { size: 11 },
                    padding: 8
                }
            }
        }
    };

    const pvCtx = document.getElementById("chart-portfolio-value");
    if (pvCtx) {
        const ctx = pvCtx.getContext('2d');
        const gradientFill = ctx.createLinearGradient(0, 0, 0, 220);
        gradientFill.addColorStop(0, 'rgba(14, 203, 129, 0.2)');
        gradientFill.addColorStop(1, 'rgba(14, 203, 129, 0)');

        chartPortfolio = new Chart(pvCtx, {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "Portfolio Value",
                    data: [],
                    borderColor: "#0ecb81",
                    backgroundColor: gradientFill,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#0ecb81',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2
                }],
            },
            options: commonChartOptions
        });
    }

    const drCtx = document.getElementById("chart-daily-returns");
    if (drCtx) {
        chartDailyReturns = new Chart(drCtx, {
            type: "bar",
            data: {
                labels: [],
                datasets: [{
                    label: "Return %",
                    data: [],
                    backgroundColor: [],
                    borderColor: [],
                    borderWidth: 0,
                    borderRadius: 4,
                }],
            },
            options: commonChartOptions
        });
    }

    const crCtx = document.getElementById("chart-cumulative-returns");
    if (crCtx) {
        const ctx = crCtx.getContext('2d');
        const gradientFill = ctx.createLinearGradient(0, 0, 0, 220);
        gradientFill.addColorStop(0, 'rgba(14, 203, 129, 0.2)');
        gradientFill.addColorStop(1, 'rgba(14, 203, 129, 0)');

        chartCumulativeReturns = new Chart(crCtx, {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "Cumulative %",
                    data: [],
                    borderColor: "#0ecb81",
                    backgroundColor: gradientFill,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#0ecb81',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2
                }],
            },
            options: commonChartOptions
        });
    }

    const phCtx = document.getElementById("chart-pnl-histogram");
    if (phCtx) {
        chartPnlHistogram = new Chart(phCtx, {
            type: "bar",
            data: {
                labels: [],
                datasets: [{
                    label: "Trades",
                    data: [],
                    backgroundColor: [],
                    borderColor: [],
                    borderWidth: 0,
                    borderRadius: 4,
                }],
            },
            options: commonChartOptions
        });
    }

    const tdCtx = document.getElementById("chart-trade-duration");
    if (tdCtx) {
        chartTradeDuration = new Chart(tdCtx, {
            type: "bar",
            data: {
                labels: [],
                datasets: [{
                    label: "Trades",
                    data: [],
                    backgroundColor: "#0ecb81",
                    borderColor: "#0ecb81",
                    borderWidth: 0,
                    borderRadius: 4,
                }],
            },
            options: commonChartOptions
        });
    }

    const spCtx = document.getElementById("chart-strategy-performance");
    if (spCtx) {
        chartStrategyPerformance = new Chart(spCtx, {
            type: "bar",
            data: {
                labels: [],
                datasets: [{
                    label: "PnL ($)",
                    data: [],
                    backgroundColor: [],
                    borderColor: [],
                    borderWidth: 0,
                    borderRadius: 4,
                }],
            },
            options: {
                ...commonChartOptions,
                indexAxis: "y",
                scales: {
                    ...commonChartOptions.scales,
                    x: {
                        ...commonChartOptions.scales.x,
                        beginAtZero: false,
                    },
                },
            },
        });
    }
}

async function fetchEndpoint(path, timeoutOverride) {
    try {
        const controller = new AbortController();
        const isScalpStatus = path.indexOf("/api/scalp/status") === 0;
        const isCanonical = path.indexOf("/api/portfolio-engine/dashboard-canonical") === 0;
        const isSlow = path.indexOf("/api/ai-diagnostics/full") === 0
            || path.indexOf("/api/portfolio-engine/market-data-readiness") === 0;
        const timeoutMs = timeoutOverride
            || (isScalpStatus ? SCALP_STATUS_TIMEOUT_MS
                : isCanonical ? CANONICAL_TIMEOUT_MS
                : isSlow ? SLOW_ENDPOINT_TIMEOUT_MS
                : 15000);
        const t = setTimeout(() => controller.abort(), timeoutMs);
        const res = await fetch(path, { method: "GET", signal: controller.signal, cache: "no-cache" });
        clearTimeout(t);
        if (res.status === 204) {
            return { ok: true, data: {} };
        }
        const data = await res.json();
        if (!res.ok && isScalpStatus && window._lastScalpStatus) {
            return { ok: true, data: window._lastScalpStatus, stale: true };
        }
        return { ok: res.ok, data };
    } catch (e) {
        if (path.indexOf("/api/scalp/status") === 0 && window._lastScalpStatus) {
            return { ok: true, data: window._lastScalpStatus, stale: true };
        }
        return { ok: false, data: null };
    }
}

function startPolling() {
    let bgIndex = 0;
    let bootstrapDone = false;

    async function bootstrap() {
        markDashboardLoading(true);
        await pollOne({ path: "/api/performance/display-context", key: "perfDisplayContext" });
        await pollOne(CANONICAL_ENDPOINT);
        bootstrapDone = true;
        pollOne({ path: "/api/system/health/quick", key: "systemHealth" });
        pollOne({ path: "/api/system/process-health", key: "processHealth" });
        pollOne(LAZY_TAB_ENDPOINTS.scalp[0]);
    }
    bootstrap();

    setInterval(function () {
        pollOne(CANONICAL_ENDPOINT);
    }, CANONICAL_REFRESH_MS);

    setInterval(function () {
        if (!bootstrapDone) return;
        const ep = BACKGROUND_ENDPOINTS[bgIndex % BACKGROUND_ENDPOINTS.length];
        bgIndex += 1;
        pollOne(ep);
    }, SECONDARY_POLL_MS);
}

let lastUpdateTime = null;

async function pollOne(ep) {
    const result = await fetchEndpoint(ep.path, ep.timeoutMs);
    if (!result.ok) {
        if (ep.key === "perfDisplayContext") {
            recordEndpointFreshness("perfDisplayContext", false);
            updateCurrentAccountSourceLabel(window._perfDisplayContext);
        }
        if (ep.key === "dashboardCanonical") {
            if (window._lastDashboardCanonical) {
                return;
            }
            updateDashboardCanonical({ success: false });
        }
        return;
    }
    const data = result.data;
    if (data === null || data === undefined) return;
    if (ep.key === "dashboardCanonical") {
        window._lastDashboardCanonical = data;
    }
    if (ep.key === "perfDisplayContext") {
        const ctx = (data.data && data.data.ledger_principal != null) ? data.data : data;
        recordEndpointFreshness("perfDisplayContext", true, ctx && ctx.last_updated);
    }
    lastUpdateTime = new Date();
    try {
        updateUI(ep.key, typeof data === "object" ? data : {}, !!result.stale);
    } catch (err) {
        console.warn("Dashboard updateUI error for", ep.key, err);
    }
}

function updateHeaderMeta() {
    const meta = document.getElementById("header-meta");
    if (meta) meta.textContent = "Last update: " + (lastUpdateTime ? lastUpdateTime.toLocaleTimeString() : "--");
    updateCurrentAccountSourceLabel(window._perfDisplayContext);
}

function updateUI(key, data, stale) {
    switch (key) {
        case "dashboardCanonical":
            updateDashboardCanonical(data);
            break;
        case "executionMode":
            updateExecutionMode(data);
            break;
        case "operator":
            updateOperator(data);
            break;
        case "regime":
            updateRegime(data);
            break;
        case "systemHealth":
            updateSystemHealth(data);
            break;
        case "latency":
            updateLatency(data);
            break;
        case "invariantsDetail":
            updateInvariantsDetail(data);
            break;
        case "portfolioValue":
            updatePortfolioChart(data);
            break;
        case "dailyReturns":
            updateDailyReturnsChart(data);
            break;
        case "cumulativeReturns":
            updateCumulativeReturnsChart(data);
            break;
        case "tradePnl":
            updatePnlHistogramChart(data);
            break;
        case "tradeDuration":
            updateTradeDurationChart(data);
            break;
        case "strategyPerformance":
            updateStrategyPerformanceChart(data);
            break;
        case "perfDisplayContext":
            capturePerfDisplayContext(data);
            break;
        case "rejects":
            updateRejects(data);
            break;
        case "decisions":
            updateDecisions(data);
            break;
        case "systemHealthFull":
            updateSystemHealthFull(data);
            break;
        case "scoreboard7d":
            updateScoreboard7d(data);
            break;
        case "learningStatus":
            updateLearningStatus(data);
            break;
        case "liveReadiness":
            updateLiveReadiness(data);
            break;
        case "modelPanel":
            updateModelPanel(data);
            break;
        case "aiDiagnosticsFull":
            updateFeatureHealthPanel(data);
            break;
        case "missedOpportunities":
            updateMissedOpportunitiesPanel(data);
            break;
        case "marketDataReadiness":
            updateMarketDataReadiness(data);
            break;
        case "dayHealth":
            updateDayHealth(data);
            break;
        case "learningHealth":
            updateLearningHealth(data);
            break;
        case "tradingEconomics":
            updateTradingEconomics(data);
            break;
        case "scalpStatus":
            updateScalpEngineStatus(data, stale);
            break;
        case "scalpStrategies":
            updateScalpStrategies(data);
            break;
        case "scalpPositions":
            updateScalpPositions(data);
            break;
        case "scalpTrades":
            updateScalpTrades(data);
            break;
        case "scalpScoreboard":
            updateScalpScoreboard(data);
            break;
        case "scalpLearning":
            updateScalpLearning(data);
            break;
        case "scalpAttribution":
            updateScalpAttribution(data);
            break;
        case "processHealth":
            updateProcessHealth(data);
            break;
        case "marketlensFeed":
            updateMarketLensFeed(data);
            break;
    }
}

/** Scalp engine panel — never mixed into DAY scoreboard or DAY PnL. */
function updateScalpEngineStatus(res, stale) {
    const d = res && typeof res === "object" ? res : {};
    const set = setCardText;
    const active = d.runner_active === true;
    const statusLabel = stale ? (active ? "RUNNING (stale)" : "STOPPED (stale)") : (active ? "RUNNING" : "STOPPED");
    set("eng-scalp-status", statusLabel);
    set("scalp-runner", statusLabel);
    ["eng-scalp-status", "scalp-runner"].forEach(function (id) {
        const el = document.getElementById(id);
        if (!el) return;
        el.title = stale ? "Showing last good SCALP status — live poll failed/timed out" : "";
        el.classList.toggle("pnl-neg", !!stale);
    });
    const pnl = d.pnl_summary || {};
    const today = pnl.today || {};
    const todayPnl = today.realized_pnl_usd != null ? Number(today.realized_pnl_usd) : null;
    setPnlCard("eng-scalp-pnl", todayPnl, today.sells);
    const scalpPnlEl = document.getElementById("scalp-today-pnl");
    if (scalpPnlEl) {
        scalpPnlEl.textContent = todayPnl != null ? "$" + todayPnl.toFixed(2) : "--";
        scalpPnlEl.classList.remove("pnl-pos", "pnl-neg", "has-long-text", "has-very-long-text");
        if (todayPnl != null) scalpPnlEl.classList.add(todayPnl >= 0 ? "pnl-pos" : "pnl-neg");
    }
    set("scalp-today-sells", today.sells != null ? String(today.sells) : "--");
    const op = (d.operational_summary || {});
    const opMode = op.operational_mode || (active ? null : "runner_dead");
    if (active) {
        const diag = formatScalpDecision(d.overall_decision, d.top_blocker, opMode);
        set("eng-scalp-blocker", diag);
        set("scalp-overall", d.overall_decision || "--");
        set("scalp-top-blocker", d.top_blocker || op.entry_blocked_reason || "--");
        set("cc-scalp-decision", diag);
    } else {
        const diag = formatScalpDecision(null, d.note, opMode || "runner_dead");
        set("eng-scalp-blocker", diag);
        set("scalp-overall", "STOPPED");
        set("scalp-top-blocker", d.note || "--");
        set("cc-scalp-decision", diag);
    }
    set("scalp-operational-mode", formatScalpOperationalMode(opMode));
    set("scalp-entry-blocked", op.entry_blocked_reason || "--");
    set("scalp-entry-armed", d.entry_armed === true ? "ARMED" : d.entry_armed === false ? "DISARMED" : "--");
    set("scalp-open-count", d.open_scalp_positions != null ? String(d.open_scalp_positions)
        : (pnl.open_positions != null ? String(pnl.open_positions) : "--"));
    set("cc-scalp-positions", d.open_scalp_positions != null ? String(d.open_scalp_positions)
        : (pnl.open_positions != null ? String(pnl.open_positions) : "--"));
    set("scalp-calibration", d.calibration_profile || (d.calibration_mode ? "calibration" : "strict"));

    // All-time summary
    const allTime = pnl.all_time || {};
    const allTimePnl = allTime.realized_pnl_usd != null ? Number(allTime.realized_pnl_usd) : null;
    const allTimePnlEl = document.getElementById("scalp-alltime-pnl");
    if (allTimePnlEl) {
        allTimePnlEl.textContent = allTimePnl != null ? "$" + allTimePnl.toFixed(2) : "--";
        allTimePnlEl.classList.remove("pnl-pos", "pnl-neg");
        if (allTimePnl != null) allTimePnlEl.classList.add(allTimePnl >= 0 ? "pnl-pos" : "pnl-neg");
    }
    set("scalp-alltime-sells", allTime.sells != null ? String(allTime.sells) : "--");

    // Mode flags
    const modeParts = [];
    if (d.scalp_live === true) modeParts.push("LIVE");
    else if (d.scalp_paper_enabled === true) modeParts.push("paper");
    else modeParts.push("--");
    set("scalp-mode", modeParts.join(", "));

    // Fee model and warm rounds
    set("scalp-fee-ok", d.fee_model_verified === true ? "yes" : d.fee_model_verified === false ? "no" : "--");
    const warmUsed = d.warm_rounds_used != null ? d.warm_rounds_used : null;
    const warmRec = d.warm_rounds_recommended != null ? d.warm_rounds_recommended : null;
    set("scalp-warm-rounds", warmUsed != null || warmRec != null
        ? (warmUsed != null ? warmUsed : "?") + " / " + (warmRec != null ? warmRec : "?")
        : "--");
    set("scalp-warm-note", d.warm_rounds_note || "--");

    window._lastScalpStatus = d;
    updateScalpSymbolDiagnostics(d);
    updateScalpRouter(d);
    refreshEnginesPanelFromCache();
    refreshCommandCenter();
}

function updateScalpSymbolDiagnostics(d) {
    const tbody = document.getElementById("scalp-symbols-tbody");
    if (!tbody) return;
    const symbols = d.symbols || {};
    const micro = d.micro_regimes || {};
    const keys = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"].filter(function (k) {
        return symbols[k] || micro[k];
    });
    if (!keys.length) {
        keys.push.apply(keys, Object.keys(symbols).sort());
    }
    if (!keys.length) {
        tbody.innerHTML = "<tr><td colspan=\"7\">--</td></tr>";
        return;
    }
    tbody.innerHTML = keys.map(function (sym) {
        const row = symbols[sym] || {};
        const mr = micro[sym] || {};
        const dist = row.distance_to_pass || {};
        const distPct = dist.distance_to_pass_pct != null ? Number(dist.distance_to_pass_pct).toFixed(3) + "%" : "--";
        const rejectRaw = String(row.reject_reason || "--");
        const rejectShort = rejectRaw.length > 28 ? rejectRaw.slice(0, 26) + "…" : rejectRaw;
        return "<tr><td>" + sym + "</td><td>" + (row.decision || "--") +
            "</td><td class=\"td-wrap\" title=\"" + rejectRaw.replace(/"/g, "&quot;") + "\">" + rejectShort +
            "</td><td>" + (mr.micro_regime || mr.regime || "--") + "</td><td>" + distPct + "</td><td>" +
            (row.spread_pct != null ? (Number(row.spread_pct) * 100).toFixed(3) + "%" : "--") + "</td><td>" +
            (row.momentum_confirmed ? "yes" : "no") + "</td></tr>";
    }).join("");
}

function updateScalpRouter(d) {
    const pre = document.getElementById("scalp-router-pre");
    if (!pre) return;
    try {
        pre.textContent = JSON.stringify(d.strategy_router || {}, null, 2);
    } catch (_e) {
        pre.textContent = "--";
    }
}

function updateScalpStrategies(res) {
    const pre = document.getElementById("scalp-strategies-pre");
    if (!pre) return;
    try {
        pre.textContent = JSON.stringify(res || {}, null, 2);
    } catch (_e) {
        pre.textContent = "--";
    }
}

function updateScalpPositions(res) {
    window._lastScalpPositions = res;
    const positions = (res && res.positions) || [];
    const ledger = res && res.ledger;
    const renderFullRow = function (pos) {
        const setupRaw = String(pos.setup || "--");
        const setupShort = setupRaw.length > 18 ? setupRaw.slice(0, 16) + "…" : setupRaw;
        const netUsd = pos.executable_net_pnl_usd;
        const netPct = pos.executable_net_pct;
        const gap = pos.target_gap_pct;
        const reasonRaw = String(pos.lifecycle_reason || pos.last_state_reason || "--");
        const reasonShort = reasonRaw.length > 24 ? reasonRaw.slice(0, 22) + "…" : reasonRaw;
        const nextExit = String(pos.next_exit_trigger || "--");
        return "<tr><td>" + (pos.symbol || "") + "</td><td class=\"td-wrap\" title=\"" + setupRaw.replace(/"/g, "&quot;") + "\">" + setupShort + "</td><td>" +
            (pos.entry_price != null ? Number(pos.entry_price).toFixed(4) : "--") + "</td><td>" +
            (netUsd != null ? "$" + Number(netUsd).toFixed(4) : "--") + "</td><td>" +
            (netPct != null ? (Number(netPct) * 100).toFixed(3) + "%" : "--") + "</td><td>" +
            (gap != null ? (Number(gap) * 100).toFixed(3) + "%" : "--") + "</td><td>" +
            (pos.hold_seconds != null ? Number(pos.hold_seconds).toFixed(0) : "--") + "</td><td>" +
            (pos.lifecycle_state || pos.state || pos.status || "--") + "</td><td class=\"td-wrap\" title=\"" + reasonRaw.replace(/"/g, "&quot;") + "\">" + reasonShort + "</td><td class=\"td-wrap\" title=\"" + nextExit.replace(/"/g, "&quot;") + "\">" + nextExit + "</td></tr>";
    };
    const renderCompactRow = function (pos) {
        const setupRaw = String(pos.setup || "--");
        const setupShort = setupRaw.length > 16 ? setupRaw.slice(0, 14) + "…" : setupRaw;
        const netUsd = pos.executable_net_pnl_usd;
        const nextExit = String(pos.next_exit_trigger || "--");
        return "<tr><td>" + (pos.symbol || "") + "</td><td class=\"td-wrap\" title=\"" + setupRaw.replace(/"/g, "&quot;") + "\">" + setupShort + "</td><td>" +
            (pos.entry_price != null ? Number(pos.entry_price).toFixed(4) : "--") + "</td><td>" +
            (netUsd != null ? "$" + Number(netUsd).toFixed(4) : "--") + "</td><td>" +
            (pos.hold_seconds != null ? Number(pos.hold_seconds).toFixed(0) : "--") + "</td><td>" +
            (pos.lifecycle_state || pos.state || "--") + "</td><td class=\"td-wrap\" title=\"" + nextExit.replace(/"/g, "&quot;") + "\">" + nextExit + "</td></tr>";
    };
    const fullTbody = document.getElementById("scalp-positions-tbody");
    if (fullTbody) {
        fullTbody.innerHTML = !positions.length
            ? "<tr><td colspan=\"10\">No open SCALP positions</td></tr>"
            : positions.map(renderFullRow).join("");
    }
    const compactTbody = document.getElementById("pt-scalp-open-tbody");
    if (compactTbody) {
        compactTbody.innerHTML = !positions.length
            ? "<tr><td colspan=\"7\">No open SCALP positions</td></tr>"
            : positions.map(renderCompactRow).join("");
    }
    if (ledger && ledger.unrealized_pnl != null) {
        const el = document.getElementById("cc-scalp-unrealized");
        if (el) {
            const u = Number(ledger.unrealized_pnl);
            el.textContent = "$" + u.toFixed(2);
            el.classList.remove("pnl-pos", "pnl-neg");
            el.classList.add(u >= 0 ? "pnl-pos" : "pnl-neg");
        }
    }
    // Prefer open count from /api/scalp/status when available (authoritative).
    if (!(window._lastScalpStatus && window._lastScalpStatus.open_scalp_positions != null)) {
        setCcScalpPositions(positions.length);
    }
}

function setCcScalpPositions(n) {
    const el = document.getElementById("cc-scalp-positions");
    if (el) el.textContent = String(n);
}

function updateScalpTrades(res) {
    window._lastScalpTrades = res;
    const time = (document.querySelector('input[name="time-filter"]:checked') || {}).value || "all";
    updateScalpTradesTable(res, time);
}

function updateScalpTradesTable(res, timeFilter) {
    const trades = (res && res.trades) || [];
    const now = new Date();
    const filtered = trades.filter(function (t) {
        if (timeFilter === "all") return true;
        if (!t.created_at) return false;
        const dt = new Date(t.created_at.replace(" ", "T") + "Z");
        if (timeFilter === "today") {
            // Match DAY scoreboard: UTC calendar day, not local browser day.
            const utcToday = todayUtcDateStr();
            const y = dt.getUTCFullYear();
            const m = String(dt.getUTCMonth() + 1).padStart(2, "0");
            const day = String(dt.getUTCDate()).padStart(2, "0");
            return (y + "-" + m + "-" + day) === utcToday;
        }
        if (timeFilter === "7d") {
            return now - dt < 7 * 86400000;
        }
        return true;
    });
    const html = filtered.length ? filtered.map(function (t) {
        const pnl = t.pnl_usd != null ? Number(t.pnl_usd) : null;
        const pnlCls = pnl != null ? (pnl >= 0 ? "pnl-pos" : "pnl-neg") : "";
        const ts = t.created_at || "";
        const timeStr = typeof ts === "string" ? ts.slice(11, 19) : ts;
        const exitRaw = String(t.exit_reason || "");
        const exitShort = exitRaw.length > 20 ? exitRaw.slice(0, 18) + "…" : exitRaw;
        return "<tr><td class=\"td-compact\">" + timeStr + "</td><td>" + (t.symbol || "") + "</td><td>" + (t.side || "") +
            "</td><td>" + (t.price != null ? Number(t.price).toFixed(4) : "") + "</td><td class=\"" + pnlCls + "\">" +
            (pnl != null ? pnl.toFixed(2) : "--") + "</td><td class=\"td-wrap\" title=\"" + exitRaw.replace(/"/g, "&quot;") + "\">" + exitShort + "</td></tr>";
    }).join("") : "<tr><td colspan=\"6\">No SCALP trades</td></tr>";
    ["scalp-trades-tbody", "pt-scalp-closed-tbody"].forEach(function (id) {
        const tbody = document.getElementById(id);
        if (tbody) tbody.innerHTML = html;
    });
}

function updateScalpScoreboard(res) {
    const tbody = document.getElementById("scalp-scoreboard-tbody");
    if (!tbody) return;
    const rows = (res && res.rows) || [];
    if (!rows.length) {
        tbody.innerHTML = "<tr><td colspan=\"5\">No SCALP scoreboard rows yet</td></tr>";
        return;
    }
    tbody.innerHTML = rows.map(function (r) {
        return "<tr><td>" + (r.day || "") + "</td><td>" + (r.trades || 0) + "</td><td>" + (r.wins || 0) +
            "</td><td>" + (r.losses || 0) + "</td><td>" + (r.net_pnl != null ? Number(r.net_pnl).toFixed(2) : "--") + "</td></tr>";
    }).join("");
}

function updateScalpAttribution(res) {
    const tbody = document.getElementById("scalp-attribution-tbody");
    if (!tbody) return;
    const sections = [
        ["Symbol", (res && res.by_symbol) || []],
        ["Setup", (res && res.by_setup) || []],
        ["Regime", (res && res.by_regime) || []],
        ["Exit", (res && res.by_exit_reason) || []],
        ["Hold", (res && res.by_hold_bucket) || []],
        ["Cost", (res && res.by_fee_burden) || []],
    ];
    const rows = [];
    sections.forEach(function (pair) {
        const label = pair[0];
        (pair[1] || []).forEach(function (item) {
            rows.push(
                "<tr><td>" + label + "</td><td>" + (item.key || "--") + "</td><td>" +
                (item.trades != null ? item.trades : "--") + "</td><td>" +
                (item.wins != null ? item.wins : "--") + "/" + (item.losses != null ? item.losses : "--") + "</td><td>" +
                (item.net_pnl_usd != null ? "$" + Number(item.net_pnl_usd).toFixed(4) : "--") + "</td><td>" +
                (item.avg_pnl_usd != null ? "$" + Number(item.avg_pnl_usd).toFixed(4) : "--") + "</td></tr>"
            );
        });
    });
    if (!rows.length) {
        tbody.innerHTML = "<tr><td colspan=\"6\">No closed scalp trades</td></tr>";
    } else {
        tbody.innerHTML = rows.join("");
    }
    const totalEl = document.getElementById("scalp-attribution-total");
    if (totalEl && res) {
        totalEl.textContent = res.total_net_pnl_usd != null
            ? "$" + Number(res.total_net_pnl_usd).toFixed(2) + " (" + (res.closed_sells || 0) + " sells)"
            : "--";
    }
}

function updateScalpLearning(res) {
    const d = res || {};
    const set = setCardText;
    set("sl-closed-sells", d.closed_sells);
    set("sl-first-close", d.first_close_ready ? "YES" : "waiting for first close");
    const fmt = function (arr) {
        try {
            return JSON.stringify(arr || [], null, 2);
        } catch (_e) {
            return "--";
        }
    };
    const att = document.getElementById("sl-attribution-pre");
    if (att) att.textContent = fmt(d.outcome_attribution);
    const rev = document.getElementById("sl-reviews-pre");
    if (rev) rev.textContent = fmt(d.post_trade_reviews);
    const w = document.getElementById("sl-weights-pre");
    if (w) w.textContent = fmt(d.strategy_score_weights);
}

function updateProcessHealth(res) {
    window._lastProcessHealth = res;
    const procs = (res && res.processes) || {};
    const optional = (res && res.optional_processes) || {};
    const setStatus = function (id, ok) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = ok ? "UP" : "DOWN";
        el.classList.remove("status-up", "status-down");
        el.classList.add(ok ? "status-up" : "status-down");
    };
    setStatus("ph-uvicorn", procs.uvicorn);
    setStatus("ph-live-md", procs.live_market_data);
    setStatus("ph-signal", procs.ai_signal_generator);
    setStatus("ph-portfolio", procs.portfolio_engine);
    setStatus("ph-context", procs.ai_market_context);
    setStatus("ph-learning", procs.ai_learning);
    setStatus("ph-scalp", procs.scalp_runner);
    const ldcEl = document.getElementById("ph-live-data-collector");
    if (ldcEl) {
        const ldc = optional.live_data_collector || {};
        const running = ldc.running === true || procs.live_data_collector === true;
        ldcEl.textContent = running ? "UP (optional)" : "RETIRED";
        ldcEl.title = ldc.note || "Retired/optional — not part of start_mystic.sh core; OHLCV via live_market_data.";
        ldcEl.classList.remove("status-up", "status-down");
        // Retired is expected — do not paint as DOWN/red.
        if (running) ldcEl.classList.add("status-up");
    }
    const redisEl = document.getElementById("ph-redis");
    if (redisEl) {
        const ok = res && res.redis === "ok";
        redisEl.textContent = ok ? "OK" : "DOWN";
        redisEl.classList.remove("status-up", "status-down");
        redisEl.classList.add(ok ? "status-up" : "status-down");
    }
    refreshEnginesPanelFromCache();
}

function updateMarketLensFeed(res) {
    const pre = document.getElementById("ml-feed-pre");
    if (!pre) return;
    try {
        pre.textContent = JSON.stringify(res || {}, null, 2);
    } catch (_e) {
        pre.textContent = "--";
    }
}

function refreshCommandCenter() {
    const set = setCardText;
    const dh = window._lastDayHealth || {};
    set("cc-day-decision", dh.capital_idle_reason || "--");
    set("cc-day-positions", dh.open_positions_count != null ? String(dh.open_positions_count) : "--");
    const canon = window._lastDashboardCanonical || {};
    const perfBlock = (canon.performance && canon.performance.performance) || {};
    const unrealized = perfBlock.unrealized_pnl != null ? perfBlock.unrealized_pnl : (canon.risk || {}).unrealized_pnl;
    if (unrealized != null) {
        const u = Number(unrealized);
        const el = document.getElementById("cc-day-unrealized");
        if (el) {
            el.textContent = "$" + u.toFixed(2);
            el.classList.remove("pnl-pos", "pnl-neg");
            el.classList.add(u >= 0 ? "pnl-pos" : "pnl-neg");
        }
    }
    const op = canon.operator || {};
    set("cc-kill-switch", op.kill_switch || "--");
    const lr = window._lastLiveReadiness || {};
    set("cc-live-ready", lr.ready_for_tiny_live_test === true ? "READY" : (lr.live_orders_block_reason || (lr.live_readiness_blockers || [])[0] || "--"));
}

/** DAY + account slice of engines panel (called from canonical + scoreboard). */
function refreshEnginesPanelFromCache() {
    const set = setCardText;
    const ph = window._lastProcessHealth || {};
    const procs = ph.processes || {};
    const dayUp = procs.portfolio_engine === true;
    set("eng-day-status", dayUp ? "RUNNING" : (ph.status ? ph.status.toUpperCase() : "UNKNOWN"));
    const canon = window._lastDashboardCanonical || {};
    const perf = (canon.performance && canon.performance.performance) || {};
    const risk = canon.risk || {};
    const eq = perf.total_equity != null ? Number(perf.total_equity) : (risk.total_equity != null ? Number(risk.total_equity) : null);
    const pr = perf.principal != null ? Number(perf.principal) : null;
    const eqEl = document.getElementById("eng-account-equity");
    if (eqEl && eq != null) eqEl.textContent = "$" + eq.toFixed(2);
    const pnlEl = document.getElementById("eng-account-pnl");
    if (pnlEl && eq != null && pr != null) {
        const delta = eq - pr;
        pnlEl.textContent = (delta >= 0 ? "+" : "") + "$" + delta.toFixed(2);
        pnlEl.classList.remove("pnl-pos", "pnl-neg");
        pnlEl.classList.add(delta >= 0 ? "pnl-pos" : "pnl-neg");
    }
    const sb = window._lastScoreboardToday || {};
    const dayPnl = sb.realized_pnl != null ? Number(sb.realized_pnl) : null;
    setPnlCard("eng-day-pnl", dayPnl, sb.trades);
    set("eng-day-scoreboard", sb.pass_fail || sb.status || "--");
    refreshCommandCenter();
}

function updateTradingEconomics(res) {
    const wrap = res && typeof res === "object" ? res : {};
    const d = wrap.data || wrap;
    if (!d || typeof d !== "object") return;

    const set = setCardText;
    const pct = (v) => (v != null && !Number.isNaN(Number(v)) ? (Number(v) * 100).toFixed(4) + "%" : "--");
    const bps = (v) => (v != null && !Number.isNaN(Number(v)) ? Number(v).toFixed(2) + " bps" : "--");

    set("te-exchange", d.exchange || "Binance.US");
    set("te-maker", pct(d.maker_fee_pct) + "\n(" + bps(d.maker_fee_bps) + ")");
    set("te-taker", pct(d.taker_fee_pct) + "\n(" + bps(d.taker_fee_bps) + ")");
    set("te-slippage", pct(d.slippage_buffer_pct));
    set("te-half-spread", pct(d.orderbook_half_spread_estimate_pct));
    set("te-roundtrip", pct(d.roundtrip_estimated_cost_pct) + "\n(" + bps(d.roundtrip_estimated_cost_bps) + ")");
    set("te-fee-date", d.fee_schedule_source_date || "--");
    set("te-notional-mult", d.day_notional_mult != null ? String(d.day_notional_mult) + "×" : "--");
    set("te-per-slot", d.day_target_notional_per_slot_usd != null ? "$" + Number(d.day_target_notional_per_slot_usd).toFixed(0) : "--");
    set("te-max-deployed", d.day_max_deployed_usd != null ? "$" + Number(d.day_max_deployed_usd).toFixed(0) : "--");
    set("te-baseline-lock", d.baseline_lock_id || "--");

    const ver = d.binance_us_verification || {};
    const conclusion = ver.conclusion || d.fee_schedule_note || "--";
    set("te-tier-note", "BTC, ETH, SOL, XRP · Adv. Spot (0.02% taker)", { title: conclusion });

    const pre = document.getElementById("panel-trading-economics-content");
    if (pre) {
        try {
            pre.textContent = JSON.stringify(d, null, 2);
        } catch (_e) {
            pre.textContent = String(d);
        }
    }
}

function updateLearningHealth(res) {
    const wrap = res && typeof res === "object" ? res : {};
    const d = wrap.data || wrap;
    if (!d || typeof d !== "object") return;

    const set = setCardText;
    const t = d.totals || {};
    set("lh-closed", t.closed_outcome_rows != null ? String(t.closed_outcome_rows) : "--");
    set("lh-snapshots", t.candidate_snapshots != null ? String(t.candidate_snapshots) : "--");
    set("lh-labeled", t.candidate_snapshots_labeled != null ? String(t.candidate_snapshots_labeled) : "--");
    set("lh-pending", t.candidate_snapshots_pending != null ? String(t.candidate_snapshots_pending) : "--");
    set("lh-heartbeats", t.position_heartbeats != null ? String(t.position_heartbeats) : "--");
    set("lh-missed", t.missed_opportunities != null ? String(t.missed_opportunities) : "--");

    const warnings = Array.isArray(d.warnings) ? d.warnings : [];
    const starving = warnings.some((w) => String(w).indexOf("DATA_STARVATION") === 0);
    set("lh-starvation", starving ? "STARVED" : "OK");
    set("lh-warnings", warnings.length ? warnings.length + " — " + warnings[0] : "none");

    const tbody = document.getElementById("lh-symbol-rows");
    if (tbody) {
        const perSym = d.per_symbol || {};
        const syms = Object.keys(perSym).sort();
        if (!syms.length) {
            tbody.innerHTML = '<tr><td colspan="8">--</td></tr>';
        } else {
            tbody.innerHTML = syms
                .map((sym) => {
                    const s = perSym[sym] || {};
                    const promo = s.promotion_ready ? "ready" : s.tiered_fallback_eligible ? "tiered-fallback" : "starved";
                    return (
                        "<tr><td>" + sym + "</td><td>" + (s.closed_outcomes ?? "--") +
                        "</td><td>" + (s.snapshots ?? "--") +
                        "</td><td>" + (s.labeled_snapshots ?? "--") +
                        "</td><td>" + (s.heartbeats ?? "--") +
                        "</td><td>" + (s.model_active_date ?? "--") +
                        "</td><td>" + promo +
                        "</td><td>" + (s.last_promotion_event ?? "--") + "</td></tr>"
                    );
                })
                .join("");
        }
    }

    const pre = document.getElementById("panel-learning-health-content");
    if (pre) pre.textContent = JSON.stringify(d, null, 2);
}

function updateDayHealth(res) {
    const wrap = res && typeof res === "object" ? res : {};
    const d = wrap.data || wrap;
    if (!d || typeof d !== "object") return;
    window._lastDayHealth = d;

    const set = setCardText;
    set("dh-slots", d.open_positions_count != null || d.max_open_positions != null ? `${d.open_positions_count || 0}/${d.max_open_positions || 4}` : "--");
    set("dh-idle-cash", d.cash_balance != null ? "$" + Number(d.cash_balance).toFixed(2) : "--");
    set("dh-idle-reason", d.capital_idle_reason || "--");
    set("cc-day-decision", d.capital_idle_reason || "--");
    set("cc-day-positions", d.open_positions_count != null ? String(d.open_positions_count) : "--");
    const diag = d.capital_idle_diagnosis || {};
    set("dh-spread-passing", Array.isArray(diag.spread_passing_symbols) && diag.spread_passing_symbols.length
        ? diag.spread_passing_symbols.join(", ")
        : (diag.spread_passing_symbols ? String(diag.spread_passing_symbols) : "--"));
    set("dh-trapped-days", d.trapped_position_days_max != null ? Number(d.trapped_position_days_max).toFixed(1) + "d" : "--");

    const pos = Array.isArray(d.positions) && d.positions.length ? d.positions[0] : null;
    if (pos) {
        set("dh-held-symbol", pos.symbol || "--");
        set("dh-held-state", pos.state || "--");
        set("dh-held-rank", pos.rs_rank != null ? String(pos.rs_rank) : "--");
        set("dh-held-net", pos.net_pct != null ? (Number(pos.net_pct) * 100).toFixed(2) + "%" : "--");
        set("dh-best-alt", pos.best_alternate_symbol || "--");
    }

    const sf = d.signal_freshness || {};
    set("dh-signal-blocked", sf.blocked_count != null ? String(sf.blocked_count) : "0");

    const eq = d.entry_quality || {};
    const gates = eq.gates || {};
    set("dh-gates-enforced", gates.enforced ? "ON" : "OFF");
    set("dh-rs-order", Array.isArray(eq.basket_rs_order) ? eq.basket_rs_order.join(" > ") : "--");

    const last = eq.last_bar_evaluation || {};
    const det = last.detail || {};
    set("dh-last-bar", last.symbol ? `${last.symbol} ${last.gate_ok ? "ok" : last.reject_code || "blocked"}` : "--");
    set("dh-last-setup", det.setup_credit != null ? Number(det.setup_credit).toFixed(3) : "--");
    set("dh-bar-skip", d.last_bar_skip_reason || "--");

    const rejects = Array.isArray(d.entry_reject_summary_7d) ? d.entry_reject_summary_7d : [];
    set("dh-top-reject", rejects.length ? rejects[0].reject_reason + " (" + rejects[0].count + ")" : "--");

    const sp = d.spread_preflight || {};
    set("dh-spread-blocked", sp.blocked_by_exec_spread_count != null ? String(sp.blocked_by_exec_spread_count) : "--");
    set("dh-spread-cap", sp.effective_paper_spread_pct != null ? (Number(sp.effective_paper_spread_pct) * 100).toFixed(3) + "%" : "--");
    set("dh-spread-align", sp.paper_align_with_bar ? "ON" : "OFF");

    const lastBlock = d.last_execution_block || {};
    set("dh-last-exec-block", lastBlock.reject_reason ? String(lastBlock.reject_reason) + " @ " + (lastBlock.symbol || "?") : "--");

    updateDayBasketSignals(d.basket_signals);
    updateDayPositionsTable(d.positions);

    const pre = document.getElementById("panel-day-health-content");
    if (pre) {
        pre.textContent = JSON.stringify(
            {
                capital_idle_reason: d.capital_idle_reason,
                positions: d.positions,
                signal_freshness: { blocked_count: sf.blocked_count, blocked_codes: sf.blocked_codes },
                entry_quality: { gates: eq.gates, last_bar: last, rs_order: eq.basket_rs_order },
                spread_preflight: sp,
                last_execution_block: lastBlock,
            },
            null,
            2
        );
    }
    refreshCommandCenter();
}

function updateDayBasketSignals(signals) {
    const tbody = document.getElementById("day-basket-tbody");
    if (!tbody) return;
    const rows = Array.isArray(signals) ? signals : [];
    if (!rows.length) {
        tbody.innerHTML = "<tr><td colspan=\"6\">No basket signals in Redis</td></tr>";
        return;
    }
    tbody.innerHTML = rows.map(function (s) {
        const conf = s.winner_probability != null || s.confidence != null
            ? (Number(s.winner_probability != null ? s.winner_probability : s.confidence) * 100).toFixed(1) + "%"
            : "--";
        const bm = s.buy_margin != null ? Number(s.buy_margin) : null;
        const margin = bm != null ? bm.toFixed(3) : "--";
        const action = s.side || s.action || s.signal || s.recommendation || "--";
        const regime = String(s.regime || s.ctx_market_regime || "--");
        let fresh = "--";
        if (s.stale === true || s.fresh === false) fresh = "stale";
        else if (s.signal_age_sec != null) fresh = Number(s.signal_age_sec).toFixed(0) + "s";
        else if (s.fresh === true) fresh = "ok";
        const tip = "P(win)=" + conf + " buy_margin=" + margin +
            " (buys need margin≥threshold; HOLD can have high P(win))";
        return "<tr title=\"" + tip.replace(/"/g, "&quot;") + "\"><td>" + (s.symbol || s.pair || "--") +
            "</td><td>" + action + "</td><td>" + conf + "</td><td>" + margin +
            "</td><td>" + regime + "</td><td>" + fresh + "</td></tr>";
    }).join("");
}

function updateDayPositionsTable(positions) {
    const tbody = document.getElementById("day-positions-tbody");
    if (!tbody) return;
    const rows = Array.isArray(positions) ? positions : [];
    if (!rows.length) {
        tbody.innerHTML = "<tr><td colspan=\"5\">No open DAY positions</td></tr>";
        return;
    }
    tbody.innerHTML = rows.map(function (p) {
        return "<tr><td>" + (p.symbol || "") + "</td><td>" + (p.state || "--") + "</td><td>" +
            (p.rs_rank != null ? String(p.rs_rank) : "--") + "</td><td>" +
            (p.net_pct != null ? (Number(p.net_pct) * 100).toFixed(2) + "%" : "--") + "</td><td>" +
            (p.best_alternate_symbol || "--") + "</td></tr>";
    }).join("");
}

/** Full live Binance probe (slow path). Payload: `{ success, data }` where `data.rows` lists per-symbol flags. */
function updateMarketDataReadiness(res) {
    const wrap = res && typeof res === "object" ? res : {};
    const d = wrap.data || wrap;
    const loadingEl = document.getElementById("market-readiness-loading");
    const sumEl = document.getElementById("market-readiness-summary");
    const tbody = document.getElementById("market-readiness-tbody");
    if (!tbody) return;

    if (wrap.success === false || !d || d.success === false) {
        tbody.innerHTML = "";
        if (loadingEl) {
            loadingEl.textContent = "Readiness probe failed or timed out.";
            loadingEl.style.display = "block";
        }
        if (sumEl) sumEl.textContent = "";
        return;
    }

    if (loadingEl) loadingEl.style.display = "none";

    const rows = Array.isArray(d.rows) ? d.rows : [];
    tbody.innerHTML = rows
        .map(function (r) {
            const missing = Array.isArray(r.missing_fields) ? r.missing_fields.join("; ") : "";
            function yn(x) {
                return x ? "✓" : "✗";
            }
            return (
                "<tr><td>" +
                (r.symbol || "") +
                "</td><td>" +
                yn(r.price_ok) +
                "</td><td>" +
                yn(r.volume_ok) +
                "</td><td>" +
                yn(r.spread_ok) +
                "</td><td>" +
                yn(r.day_gate_ok) +
                "</td><td>" +
                yn(r.day_active_bundle_ok) +
                "</td><td>" +
                yn(r.month_context_ready) +
                "</td><td>" +
                yn(r.indicator_ok) +
                "</td><td>" +
                yn(r.ai_buy_context_ready) +
                "</td><td>" +
                yn(r.ai_hold_context_ready) +
                "</td><td>" +
                yn(r.ai_sell_context_ready) +
                "</td><td class=\"mono\">" +
                (missing || "").replace(/</g, "") +
                "</td></tr>"
            );
        })
        .join("");

    if (sumEl) {
        sumEl.textContent =
            "AI can act (all symbols): " +
            (d.ai_can_act_all_symbols ? "yes" : "no") +
            (d.ai_cannot_act_reason ? " (" + d.ai_cannot_act_reason + ")" : "") +
            " · ISO " +
            (d.timestamp_iso || "");
    }
}

// portfolio-engine/dashboard-canonical: one backend build for positions, sleeves, PnL, risk, trades, scoreboard, operator
function markCanonicalSnapshotStale(reason) {
    CANONICAL_STALE_WIDGET_IDS.forEach(function (id) {
        const el = document.getElementById(id);
        if (!el) return;
        el.title = (reason || "Canonical snapshot stale") + " — supplemental widget only";
        if (!el.dataset.lastGood) {
            el.textContent = "Unavailable";
        }
        el.classList.remove("pnl-pos", "pnl-neg");
    });
}

function updateDashboardCanonical(res) {
    if (!res || res.success === false) {
        if (window._lastDashboardCanonical) {
            return;
        }
        markCanonicalSnapshotStale(res && res.error ? String(res.error) : "Canonical snapshot failed");
        return;
    }
    const d = res.data || res;
    if (!d || typeof d !== "object") return;

    if (d.consistency_ok === false && Array.isArray(d.consistency_violations) && d.consistency_violations.length) {
        console.warn("Dashboard canonical consistency:", d.consistency_violations);
    }

    if (d.positions && Array.isArray(d.positions.positions)) {
        updatePositions({ positions: d.positions.positions });
    }
    if (d.performance) {
        updatePortfolioPerformance({
            success: true,
            performance: d.performance.performance,
            trades: d.performance.trades || [],
        });
    }
    if (d.sleeves) {
        updateSleeveStatus({
            data: {
                sleeve_enabled: d.sleeves.sleeve_enabled,
                sleeves: d.sleeves.sleeves || {},
            },
        });
    }
    if (d.risk) {
        updateRiskMetrics({ success: true, data: d.risk });
    }
    if (d.operator || d.risk) {
        const op = Object.assign({}, d.operator || {}, d.risk ? {
            cash_balance: d.risk.cash_balance,
            total_equity: d.risk.total_equity,
            account_status: d.risk.account_status,
        } : {});
        if (d.positions && d.positions.count != null) {
            op.open_positions_count = d.positions.count;
        }
        updateOperator({ success: true, data: op });
    }
    if (d.scoreboard_today) {
        updateScoreboardToday({ data: d.scoreboard_today });
        updateTodayActivity(d.scoreboard_today);
    }
    if (d.daily_performance) {
        updateDailyPerformanceSnapshot({ data: d.daily_performance });
    }
    if (d.invariants) {
        updateInvariants({ success: true, data: d.invariants });
    }
    updateCrossCheck(d);
    if (d.regime && typeof d.regime === "object") {
        updateRegime({ success: true, data: d.regime });
    }
    if (d.day_position_health && typeof d.day_position_health === "object") {
        updateDayHealth({ success: true, data: d.day_position_health });
    }

    const lite = d.market_data_readiness_dashboard;
    if (lite && typeof lite === "object") {
        const hb = document.getElementById("status-market-heartbeat");
        if (hb) {
            if (lite.redis_connected && lite.last_market_data_iso) {
                hb.textContent = "OK";
                hb.title = "Redis market_data:last_update " + lite.last_market_data_iso;
            } else if (lite.redis_connected) {
                hb.textContent = "no ts";
                hb.title = "Redis connected; market_data:last_update not set yet";
            } else {
                hb.textContent = "--";
                hb.title = "Redis market heartbeat unavailable";
            }
        }
        const box = document.getElementById("market-readiness-lite");
        if (box) {
            const syms = Array.isArray(lite.universe_api_symbols) ? lite.universe_api_symbols.join(", ") : "--";
            box.innerHTML =
                "<p><strong>Universe</strong> " +
                syms +
                " · <strong>DAY active TFs</strong> " +
                (Array.isArray(lite.day_active_required_timeframes) ? lite.day_active_required_timeframes.join(",") : "--") +
                " · <strong>Named indicators</strong> " +
                (lite.indicator_registry_named_count != null ? lite.indicator_registry_named_count : lite.feature_mapping_slots != null ? lite.feature_mapping_slots : "--") +
                " · <strong>Vector dim</strong> " +
                (lite.day_model_feature_dim != null ? lite.day_model_feature_dim : "--") +
                " · <strong>Feature version (day)</strong> " +
                (lite.feature_version_day_live != null ? lite.feature_version_day_live : "--") +
                "</p>";
        }
    }
    window._lastDashboardCanonical = d;
    refreshEnginesPanelFromCache();
    markDashboardLoading(false);
}

// system/health/quick: { status, memory_mb, cpu_percent }
function updateSystemHealth(data) {
    const status = data.status || "?";
    const badge = document.getElementById("system-badge");
    if (badge) {
        const mem = data.memory_mb != null ? Math.round(data.memory_mb) + " MB" : "";
        badge.textContent = status === "ok" ? "OK" : status === "warning" ? "Warning" : status === "critical" ? "Critical" : status;
        badge.title = mem ? "Memory: " + mem : "System health";
        badge.className = "header__system header__system--" + (status === "ok" ? "ok" : status === "warning" ? "warning" : status === "critical" ? "critical" : "unknown");
    }
    const memEl = document.getElementById("status-memory");
    if (memEl && data.memory_mb != null) memEl.textContent = Math.round(data.memory_mb) + " MB";
}

// portfolio-engine/risk: { success, data: { total_open_risk_pct, account_status, position_risks, ... } }
function updateRiskMetrics(data) {
    const d = data.data || data;
    if (!d) return;
    const el = document.getElementById("status-risk");
    if (el && d.total_open_risk_pct != null) {
        const p = Number(d.total_open_risk_pct);
        if (Number.isFinite(p)) {
            el.textContent = p > 0 && p < 0.1 ? "<0.1%" : p.toFixed(1) + "%";
        } else {
            el.textContent = "--";
        }
    }
    const statusEl = document.getElementById("analytics-risk-status");
    if (statusEl) {
        const s = d.account_status || "";
        statusEl.textContent = s || "--";
        if (s) statusEl.dataset.lastGood = s;
        statusEl.classList.remove("pnl-pos", "pnl-neg");
        if (s && s.toUpperCase() === "HEALTHY") statusEl.classList.add("pnl-pos");
        else if (s && (s.toUpperCase().includes("OVERALLOCATED") || s.toUpperCase().includes("DELEVERAGING"))) statusEl.classList.add("pnl-neg");
    }
    const detailEl = document.getElementById("panel-risk-content");
    if (detailEl) {
        const lines = [
            "account_status: " + (d.account_status || "--"),
            "total_open_risk_pct: " + (d.total_open_risk_pct != null ? d.total_open_risk_pct.toFixed(2) + "%" : "--"),
            "risk_cap_remaining_pct: " + (d.risk_cap_remaining_pct != null ? d.risk_cap_remaining_pct.toFixed(2) + "%" : "--"),
            "equity_invariant_ok: " + (d.equity_invariant_ok === true ? "yes" : d.equity_invariant_ok === false ? "no" : "--"),
        ];
        const pr = d.position_risks || [];
        if (pr.length > 0) {
            lines.push("position_risks:");
            pr.slice(0, 10).forEach(function (p) {
                lines.push("  " + (p.symbol || "?") + ": $" + (p.risk_usd != null ? p.risk_usd.toFixed(2) : "?") + " (" + (p.risk_pct != null ? p.risk_pct.toFixed(1) : "?") + "%)");
            });
        }
        detailEl.textContent = lines.join("\n");
    }
}

function formatScoreboardPassFail(passFail, failReasons, expectancyR) {
    const status = passFail != null ? String(passFail).toUpperCase() : "--";
    const reasons = failReasons != null ? String(failReasons) : "";
    const expStr = expectancyR != null && !Number.isNaN(Number(expectancyR))
        ? " · expectancy_r=" + Number(expectancyR).toFixed(2)
        : "";
    if (status === "FAIL") {
        if (reasons.indexOf("NEGATIVE_EXPECTANCY") >= 0) {
            return "FAIL — negative expectancy" + expStr;
        }
        return "FAIL" + (reasons ? " — " + reasons : "") + expStr;
    }
    if (status === "PASS" || status === "OK" || status === "PENDING") {
        return status + expStr;
    }
    return status;
}

function applyTodayScoreboardMetrics(d) {
    if (!d || typeof d !== "object") return;
    function setPnl(id, val) {
        const el = document.getElementById(id);
        if (!el || val == null) return;
        const n = Number(val);
        el.textContent = (n >= 0 ? "+" : "") + "$" + n.toFixed(2);
        el.classList.remove("pnl-pos", "pnl-neg");
        el.classList.add(n >= 0 ? "pnl-pos" : "pnl-neg");
    }
    setPnl("analytics-today-realized", d.realized_pnl);
    setPnl("analytics-today-unrealized", d.unrealized_pnl != null ? d.unrealized_pnl : d.open_unrealized_pnl);
    setPnl("analytics-today-total", d.total_pnl);
}

// portfolio-engine/scoreboard/today: { success, data: { status, pass, ... } }
function updateScoreboardToday(data) {
    const d = data.data || data;
    const el = document.getElementById("analytics-scoreboard");
    if (!el) return;
    if (!d) { el.textContent = "--"; return; }
    applyTodayScoreboardMetrics(d);
    const status = d.status || d.pass_fail || (d.pass ? "PASS" : d.fail ? "FAIL" : null);
    const failReasons = d.fail_reasons || "";
    const closedAi = d.closed_ai_trades_today != null ? d.closed_ai_trades_today : d.ai_closed_trades;
    const openBuys = d.open_buys_today != null ? d.open_buys_today : 0;
    const pfLabel = formatScoreboardPassFail(status, failReasons, d.expectancy_r);
    el.title =
        "DAY engine scoreboard only (not scalp). Closed AI SELLs today: " + (closedAi != null ? closedAi : "?") +
        ". Open buys today: " + openBuys +
        ". FAIL is strategy-quality (expectancy), not dollar total today. Stack can still be HEALTHY.";
    const sells = closedAi != null ? closedAi : (d.trades != null ? d.trades : "?");
    const wr = d.win_rate != null ? (Number(d.win_rate) <= 1 ? (Number(d.win_rate) * 100).toFixed(1) : Number(d.win_rate).toFixed(1)) + "%" : "--";
    const total = d.total_pnl != null ? Number(d.total_pnl) : null;
    const totalStr = total != null ? ((total >= 0 ? "+" : "") + "$" + total.toFixed(2)) : "--";
    setCardText(el, sells + " sells · " + wr + " · total " + totalStr + " · " + pfLabel, { title: el.title, keepTitle: true });

    el.classList.remove("pnl-pos", "pnl-neg");
    if (status && (String(status).toUpperCase() === "PASS" || String(status).toUpperCase() === "OK")) el.classList.add("pnl-pos");
    else if (status && String(status).toUpperCase() === "FAIL") el.classList.add("pnl-neg");
    else if (status && String(status).toUpperCase() === "PENDING") el.classList.add("pnl-pos");
    el.dataset.fromCanonical = "1";
    el.dataset.lastGood = el.textContent;
    window._lastScoreboardToday = d;
    refreshEnginesPanelFromCache();
}

function updateTodayActivity(d) {
    if (!d || typeof d !== "object") return;
    function set(id, val, title) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = val != null ? String(val) : "--";
        if (title) el.title = title;
    }
    set("today-closed-ai", d.closed_ai_trades_today != null ? d.closed_ai_trades_today : d.ai_closed_trades, "AI SELL closes today");
    set("today-open-buys", d.open_buys_today, "BUY rows today — not counted as closed trades");
    const aiPnl = d.ai_realized_pnl_today != null ? d.ai_realized_pnl_today : d.ai_closed_pnl;
    const aiEl = document.getElementById("today-ai-pnl");
    if (aiEl && aiPnl != null) {
        aiEl.textContent = "$" + Number(aiPnl).toFixed(2);
        aiEl.classList.remove("pnl-pos", "pnl-neg");
        aiEl.classList.add(Number(aiPnl) >= 0 ? "pnl-pos" : "pnl-neg");
    }
    set("today-admin-closes", d.admin_stale_closes_today != null ? d.admin_stale_closes_today : d.admin_synthetic_closes, "Admin/stale synthetic closes");
    const uEl = document.getElementById("today-unrealized");
    if (uEl && d.open_unrealized_pnl != null) {
        uEl.textContent = "$" + Number(d.open_unrealized_pnl).toFixed(2);
        uEl.classList.remove("pnl-pos", "pnl-neg");
        uEl.classList.add(Number(d.open_unrealized_pnl) >= 0 ? "pnl-pos" : "pnl-neg");
    }
    set("today-equity", d.total_equity != null ? "$" + Number(d.total_equity).toFixed(2) : "--", "Ledger total equity");
    set("today-paper-trades", d.paper_trades_today, "All paper_trades rows today");
    set("today-live-trades", d.live_trades_today, "paper_trades with mode=live today");
}

function updateLiveReadiness(res) {
    const d = (res && res.data) ? res.data : res;
    const pre = document.getElementById("panel-live-readiness-content");
    if (!d) {
        if (pre) pre.textContent = "No live readiness data.";
        return;
    }
    function yn(v) { return v ? "yes" : "no"; }
    function set(id, text, warn) {
        setCardText(id, text, { warn: !!warn });
    }
    set("lr-mode", d.current_local_mode || d.execution_mode);
    set("lr-live-orders", d.live_orders_permitted ? "permitted" : "blocked");
    set("lr-block-reason", d.live_orders_block_reason || (d.live_readiness_blockers || []).join("; ") || "none");
    set("lr-tiny-ready", yn(d.ready_for_tiny_live_test), !d.ready_for_tiny_live_test);
    set("lr-full-ready", yn(d.ready_for_full_live), !d.ready_for_full_live);
    set("lr-usdt", d.usdt_free_balance != null ? Number(d.usdt_free_balance).toFixed(2) : "?", d.usdt_free_balance != null && Number(d.usdt_free_balance) <= 0);
    set("lr-can-trade", d.can_trade != null ? yn(d.can_trade) : "?");
    set("lr-can-withdraw", d.can_withdraw != null ? yn(d.can_withdraw) : "?", d.can_withdraw === true);
    set("lr-open-orders", d.open_binance_orders_count != null ? d.open_binance_orders_count : "?");
    set("lr-drift", d.exchange_time_drift_ms != null ? d.exchange_time_drift_ms : "?");
    set("lr-protected", yn(d.protected_execution_enabled));
    set("lr-sleeve", d.sleeve_telemetry_only ? "telemetry only" : (d.sleeve_blocking_enabled ? "blocking ON" : "off"));
    if (pre) {
        const lines = [];
        lines.push("ready_for_tiny_live_test: " + yn(d.ready_for_tiny_live_test));
        lines.push("ready_for_full_live: " + yn(d.ready_for_full_live));
        if (Array.isArray(d.live_readiness_warnings) && d.live_readiness_warnings.length) {
            lines.push("warnings:");
            d.live_readiness_warnings.forEach(function (w) { lines.push("  - " + w); });
        }
        if (Array.isArray(d.live_readiness_blockers) && d.live_readiness_blockers.length) {
            lines.push("blockers:");
            d.live_readiness_blockers.forEach(function (b) { lines.push("  - " + b); });
        }
        if (Array.isArray(d.tiny_live_checklist)) {
            lines.push("tiny_live_checklist:");
            d.tiny_live_checklist.forEach(function (c) {
                lines.push("  - " + c.item + (c.env != null ? " env=" + c.env : c.value != null ? " val=" + JSON.stringify(c.value) : ""));
            });
        }
        pre.textContent = lines.join("\n");
    }
    window._lastLiveReadiness = d;
    refreshCommandCenter();
}

function updateModelPanel(res) {
    const el = document.getElementById("panel-model-content");
    if (!el) return;
    const d = (res && res.data) ? res.data : res;
    if (!d || !Array.isArray(d.per_symbol)) {
        el.textContent = "No model panel data.";
        return;
    }
    const lines = [];
    d.per_symbol.forEach(function (m) {
        lines.push(m.symbol + ": fv=" + (m.feature_version != null ? m.feature_version : "?") +
            " dim=" + (m.feature_dim != null ? m.feature_dim : "?") +
            " active_acc=" + (m.active_accuracy != null ? m.active_accuracy : "?") +
            " holdout_n=" + (m.holdout_sample_count != null ? m.holdout_sample_count : "?") +
            " low_conf=" + (m.holdout_low_confidence ? "yes" : "no"));
        if (m.candidate_always_buy) lines.push("  candidate ALWAYS_BUY");
        if (m.candidate_always_hold) lines.push("  candidate ALWAYS_HOLD");
        if (m.promotion_rejection_reason) lines.push("  reason: " + m.promotion_rejection_reason);
    });
    el.textContent = lines.join("\n") || "No symbols.";
}

function updateFeatureHealthPanel(res) {
    const el = document.getElementById("panel-feature-health-content");
    if (!el) return;
    const d = (res && res.data) ? res.data : res;
    if (!d) { el.textContent = "No diagnostics."; return; }
    try {
        el.textContent = JSON.stringify({
            feature_completeness: d.feature_completeness,
            feature_freshness: d.feature_freshness,
            model_freshness: d.model_freshness,
        }, null, 2).slice(0, 12000);
    } catch (e) {
        el.textContent = String(e);
    }
}

function updateMissedOpportunitiesPanel(res) {
    const el = document.getElementById("panel-missed-op-content");
    if (!el) return;
    const d = (res && res.data) ? res.data : res;
    if (!d) { el.textContent = "No missed opportunity data."; return; }
    try {
        el.textContent = JSON.stringify(d, null, 2).slice(0, 8000);
    } catch (e) {
        el.textContent = String(e);
    }
}

async function fetchTradeDrilldown(tradeId, targetEl) {
    if (!tradeId || !targetEl) return;
    targetEl.textContent = "Loading…";
    try {
        const r = await fetch("/api/portfolio-engine/trade/" + encodeURIComponent(tradeId));
        const j = await r.json();
        if (!j.success) {
            targetEl.textContent = responseErrorMessage(j, "Failed");
            return;
        }
        targetEl.textContent = JSON.stringify(j.data, null, 2).slice(0, 12000);
    } catch (e) {
        targetEl.textContent = String(e);
    }
}

// portfolio-engine/invariants: { success, data: { all_invariants_pass, snapshot_failed_keys, ... } }
function updateInvariants(data) {
    const d = data.data || data;
    const el = document.getElementById("analytics-invariants");
    if (el) {
        if (!d || d.all_invariants_pass === undefined) {
            el.textContent = "--";
            el.title = "";
        } else {
            el.textContent = d.all_invariants_pass ? "OK" : "FAIL";
            el.classList.remove("pnl-pos", "pnl-neg");
            el.classList.add(d.all_invariants_pass ? "pnl-pos" : "pnl-neg");
            const failed = Array.isArray(d.snapshot_failed_keys) ? d.snapshot_failed_keys : [];
            el.title = d.all_invariants_pass
                ? "All invariant categories OK (see Invariants Detail)."
                : failed.length
                  ? "Failed checks: " + failed.join(", ")
                  : "See Invariants Detail for guard status.";
        }
    }
}

function updateCrossCheck(d) {
    const el = document.getElementById("analytics-xcheck");
    if (!el) return;
    if (!d || d.consistency_ok === undefined) {
        el.textContent = "--";
        el.title = "";
        el.classList.remove("pnl-pos", "pnl-neg");
        return;
    }
    if (d.consistency_ok === true) {
        el.textContent = "OK";
        el.title = "Snapshot internal cross-checks passed.";
        el.classList.remove("pnl-neg");
        el.classList.add("pnl-pos");
        return;
    }
    const v = Array.isArray(d.consistency_violations) ? d.consistency_violations : [];
    el.textContent = v.length ? "WARN" : "FAIL";
    el.title = v.length ? v.join("; ") : "Snapshot cross-check failed";
    el.classList.remove("pnl-pos");
    el.classList.add("pnl-neg");
}

// portfolio-engine/invariants-detail: { success, data: { all_ok, equity_invariant, position_limit, risk_cap, ... } }
function updateInvariantsDetail(data) {
    const d = data.data || data;
    const el = document.getElementById("panel-invariants-content");
    if (!el) return;
    if (!d || typeof d !== "object") {
        el.textContent = "No invariants detail.";
        return;
    }
    const lines = [];
    lines.push("all_ok: " + (d.all_ok === true ? "yes" : d.all_ok === false ? "no" : "--"));
    lines.push(
        "snapshot_failed_count: " + (d.snapshot_failed_count != null ? d.snapshot_failed_count : "--") +
            (Array.isArray(d.snapshot_failed_keys) && d.snapshot_failed_keys.length
                ? "  [" + d.snapshot_failed_keys.join(", ") + "]"
                : ""),
    );
    lines.push(
        "invariant_events_total (cumulative): " + (d.invariant_events_total != null ? d.invariant_events_total : "--"),
    );
    lines.push("(invariant_events_total = cumulative event counter, not # currently failing)");

    const eq = d.equity_invariant || {};
    if (Object.keys(eq).length) {
        lines.push("equity: ok=" + (eq.ok ? "yes" : "no") + " expected=" + (eq.expected != null ? eq.expected : "?") + " actual=" + (eq.actual != null ? eq.actual : "?") + " diff=" + (eq.diff != null ? eq.diff : "?"));
    }
    const pl = d.position_limit || {};
    if (Object.keys(pl).length) {
        lines.push("position_limit: ok=" + (pl.ok ? "yes" : "no") + " current=" + (pl.current != null ? pl.current : "?") + " max=" + (pl.max != null ? pl.max : "?"));
    }
    const rc = d.risk_cap || {};
    if (Object.keys(rc).length) {
        lines.push("risk_cap: ok=" + (rc.ok ? "yes" : "no") + " current=$" + (rc.current_risk != null ? rc.current_risk : "?") + " max=$" + (rc.max_risk != null ? rc.max_risk : "?") + " pct=" + (rc.pct_used != null ? rc.pct_used + "%" : "?"));
    }
    const ns = d.no_stacking || {};
    if (ns.symbols && ns.symbols.length > 0) {
        lines.push("symbols: " + ns.symbols.join(", "));
    }
    const vg = d.vol_spike_guard || {};
    const dg = d.drawdown_guard || {};
    const cg = d.churn_guard || {};
    const sb = d.spread_blocks || {};
    lines.push("vol_spike_guard: ok=" + (vg.ok ? "yes" : "no") + " active=" + (vg.active ? "yes" : "no"));
    lines.push("drawdown_guard: ok=" + (dg.ok ? "yes" : "no") + " active=" + (dg.active ? "yes" : "no"));
    lines.push("churn_guard: ok=" + (cg.ok ? "yes" : "no") + " active=" + (cg.active ? "yes" : "no"));
    lines.push("spread_blocks: ok=" + (sb.ok ? "yes" : "no") + " blocked_n=" + (sb.blocked_symbols ? sb.blocked_symbols.length : 0));
    el.textContent = lines.join("\n");
}

// portfolio-engine/latency: { success, data: { health_flag, symbol_freshness, decision_to_execution_ms, ... } }
function updateLatency(data) {
    const d = data.data || data;
    const flag = (d && d.health_flag) ? d.health_flag : (d && d.health) ? d.health : null;
    const el = document.getElementById("status-latency");
    if (el) el.textContent = flag ? String(flag) : "--";
    const detailEl = document.getElementById("panel-latency-content");
    if (detailEl && d) {
        const dtem = d.decision_to_execution_ms || {};
        const lines = [
            "health_flag: " + (d.health_flag || "--"),
            "stale_symbol_count: " + (d.stale_symbol_count != null ? d.stale_symbol_count : "--") + " / " + (d.total_symbols != null ? d.total_symbols : "?"),
            "decision_to_exec avg_ms: " + (dtem.avg != null ? dtem.avg : dtem.rolling_avg != null ? dtem.rolling_avg : "--"),
            "decision_to_exec p95_ms: " + (dtem.p95 != null ? dtem.p95 : "--"),
        ];
        detailEl.textContent = lines.join("\n");
    }
}

// portfolio-engine/rejects: { success, rejects: [{ ts, symbol, side, reason, filter_name }] }
function updateRejects(data) {
    if (!data || typeof data !== "object") return;
    const list = Array.isArray(data.rejects) ? data.rejects : [];
    const el = document.getElementById("panel-rejects-content");
    if (!el) return;
    if (list.length === 0) {
        el.textContent = "No recent rejects.";
        return;
    }
    const lines = list.slice(0, 15).map(function (r) {
        return (r.ts || "") + " | " + (r.symbol || "?") + " " + (r.side || "") + " | " + (r.reason || r.filter_name || "?");
    });
    el.textContent = lines.join("\n");
}

// portfolio-engine/decisions: { success, decisions: [...] }
function updateDecisions(data) {
    if (!data || typeof data !== "object") return;
    const list = Array.isArray(data.decisions) ? data.decisions : [];
    const el = document.getElementById("panel-decisions-content");
    if (!el) return;
    if (list.length === 0) {
        el.textContent = "No recent decisions.";
        return;
    }
    const lines = list.slice(0, 10).map(function (d, i) {
        const sym = d.symbol || d.trade_id || "?";
        const entry = d.regime || d.entry_reason || "";
        const exit = d.exit_type || d.exit_trigger || d.exit_reason || "";
        return (i + 1) + ". " + sym + " " + (d.side || "") + " regime=" + entry + " exit=" + exit;
    });
    el.textContent = lines.join("\n");
}

// api/system/health/comprehensive: { overall_status, system, backend_api, redis, ... }
function updateSystemHealthFull(data) {
    const el = document.getElementById("panel-system-health-content");
    if (!el) return;
    if (!data || typeof data !== "object") {
        el.textContent = "No data.";
        return;
    }
    if (data.error || data.status === "error") {
        el.textContent = "Error: " + (data.error || data.status || "unknown");
        return;
    }
    const lines = [];
    if (data.overall_status != null) lines.push("overall_status: " + data.overall_status);
    const sys = data.system || {};
    if (Object.keys(sys).length) {
        lines.push("system: CPU " + (sys.cpu_percent != null ? sys.cpu_percent + "%" : "?") + " | RAM " + (sys.ram_used_percent != null ? sys.ram_used_percent + "%" : "?") + " | disk " + (sys.disk_used_percent != null ? sys.disk_used_percent + "%" : "?"));
    }
    const api = data.backend_api || {};
    if (api.memory_mb != null) lines.push("backend memory: " + api.memory_mb + " MB");
    const redis = data.redis || {};
    if (redis.status) lines.push("redis: " + redis.status);
    el.textContent = lines.length ? lines.join("\n") : "No data.";
}

// portfolio-engine/learning-status: { success, data: { storage_connected, total_rows, good/bad counts, rows } }
function updateLearningStatus(data) {
    const d = (data && data.data) || data || {};
    const el = document.getElementById("panel-learning-status-content");
    if (!el) return;
    if (!d || typeof d !== "object") {
        el.textContent = "No learning data.";
        return;
    }
    const lines = [];
    lines.push("storage_connected: " + (d.storage_connected ? "yes" : "no"));
    lines.push("table: " + (d.table || "--"));
    lines.push("total_rows: " + (d.total_rows != null ? d.total_rows : "--"));
    lines.push("last_written_at_utc: " + (d.last_written_at_utc || "--"));
    lines.push("good_trade_count_recent: " + (d.good_trade_count_recent != null ? d.good_trade_count_recent : 0));
    lines.push("bad_trade_count_recent: " + (d.bad_trade_count_recent != null ? d.bad_trade_count_recent : 0));
    lines.push("ai_has_enough_data: " + (d.ai_has_enough_data ? "yes" : "no"));
    if (d.last_good_recent) {
        lines.push("last_good: " + (d.last_good_recent.symbol || "?") + " net=" + (d.last_good_recent.net_profit_usd != null ? d.last_good_recent.net_profit_usd : "?"));
    }
    if (d.last_bad_recent) {
        lines.push("last_bad: " + (d.last_bad_recent.symbol || "?") + " reason=" + (d.last_bad_recent.close_reason || "?"));
    }
    const rows = Array.isArray(d.rows) ? d.rows : [];
    if (rows.length === 0) {
        lines.push("");
        lines.push("recent_rows: none yet (AI still gathering data)");
    } else {
        lines.push("");
        lines.push("recent_rows (newest first):");
        rows.slice(0, 10).forEach(function (r) {
            const tag = r.good_trade ? "GOOD" : r.bad_trade ? "BAD " : "----";
            const net = r.net_profit_usd != null ? r.net_profit_usd.toFixed ? r.net_profit_usd.toFixed(4) : r.net_profit_usd : "?";
            lines.push("  " + tag + "  " + (r.symbol || "?") + "  " + (r.close_reason || "?") + "  net=" + net + (r.lesson ? "  lesson=" + r.lesson : ""));
        });
    }
    el.textContent = lines.join("\n");
}

// portfolio-engine/scoreboard?days=7: { success, data: { aggregate, daily_records } }
function updateScoreboard7d(data) {
    const d = data.data || data;
    const el = document.getElementById("panel-scoreboard7d-content");
    if (el) {
        if (!d || typeof d !== "object") {
            el.textContent = "No scoreboard data.";
        } else {
            const agg = d.aggregate || d;
            const lines = [
                "period_days: " + (d.period_days != null ? d.period_days : "7"),
                "total_trades: " + (agg.total_trades != null ? agg.total_trades : "--"),
                "overall_win_rate: " + (agg.overall_win_rate != null ? (agg.overall_win_rate * 100).toFixed(1) + "%" : "--"),
                "avg_expectancy_r: " + (agg.avg_expectancy_r != null ? agg.avg_expectancy_r.toFixed(3) : "--"),
                "max_drawdown_pct: " + (agg.max_drawdown_pct != null ? agg.max_drawdown_pct.toFixed(2) + "%" : "--"),
                "overall_status: " + (agg.overall_status || d.overall_status || "--"),
            ];
            el.textContent = lines.join("\n");
        }
    }
    const scoreEl = document.getElementById("analytics-scoreboard");
    // Prefer canonical scoreboard/today — do not let the 7d aggregate race overwrite it.
    if (scoreEl && scoreEl.dataset.fromCanonical === "1") {
        return;
    }
    if (scoreEl) {
        const records = (d && d.daily_records) || [];
        const todayRow = records.find(function (r) {
            return r.day === todayUtcDateStr() || r.date === todayUtcDateStr();
        });
        if (todayRow) {
            applyTodayScoreboardMetrics(todayRow);
            const wr = todayRow.win_rate != null ? (Number(todayRow.win_rate) * 100).toFixed(1) + "%" : "--";
            const pnl = todayRow.total_pnl != null ? Number(todayRow.total_pnl) : (todayRow.realized_pnl != null ? Number(todayRow.realized_pnl) : null);
            const pnlStr = pnl != null ? ((pnl >= 0 ? "+" : "") + "$" + pnl.toFixed(2)) : "--";
            const pfLabel = formatScoreboardPassFail(todayRow.pass_fail, todayRow.fail_reasons, todayRow.expectancy_r);
            scoreEl.textContent = (todayRow.trades || 0) + " sells · " + wr + " · total " + pnlStr + " · " + pfLabel;
            scoreEl.dataset.lastGood = scoreEl.textContent;
            scoreEl.title = "DAY scoreboard today (UTC). FAIL is expectancy quality, not dollar total.";
            scoreEl.classList.remove("pnl-pos", "pnl-neg");
            if (String(todayRow.pass_fail || "").toUpperCase() === "FAIL") scoreEl.classList.add("pnl-neg");
            else if (String(todayRow.pass_fail || "").toUpperCase() === "PASS") scoreEl.classList.add("pnl-pos");
        } else if (!scoreEl.dataset.lastGood) {
            scoreEl.textContent = "0 sells today";
        }
    }
}

// regime: canonical or /regime — { regime, confidence, effective_risk_multiplier, guards_active }
function updateRegime(data) {
    const d = data.data || data;
    if (!d) return;
    let regime = String(d.regime || "unknown").toLowerCase().trim() || "unknown";
    const badge = document.getElementById("regime-badge");
    if (!badge) return;
    const g = d.guards_active || {};
    const mult = d.effective_risk_multiplier != null ? Number(d.effective_risk_multiplier) : 1;
    const parts = [];
    if (g.vol_spike) parts.push("VOL");
    if (g.drawdown_guard) parts.push("DD");
    if (g.churn_guard) parts.push("CHURN");
    const spreadN = Number(g.spread_blocked_count || 0);
    if (spreadN > 0) parts.push("SPREAD×" + spreadN);

    let display;
    if (regime !== "unknown" && regime !== "") {
        display = "Fear & Greed: " + regime.toUpperCase();
    } else if (parts.length > 0) {
        display = "Guards: " + parts.join(" · ");
    } else if (mult < 0.999) {
        display = "Risk scale: ×" + mult.toFixed(2);
    } else {
        display = "Market: unclassified";
    }
    badge.textContent = display;
    const conf = d.confidence != null ? Number(d.confidence) : null;
    const up = d.upstream || {};
    let explain = "";
    if (up.source === "fear_greed_api" && up.raw_value != null) {
        explain =
            "Crypto Fear & Greed " +
            Number(up.raw_value).toFixed(0) +
            "/100 (alternative.me). BULL|BEAR|SIDEWAYS = score thresholds on that index, not price-trend ML.";
    } else if (up.source === "mystic_startup_default") {
        explain = "Default label until first successful regime API write.";
    } else if (up.source === "api_error_fallback") {
        explain = "Regime API failed; safe default stored in Redis.";
    } else if (d.label_source === "redis:market_regime:global" && up.source) {
        explain = "Redis market_regime:global · source=" + up.source + ".";
    } else if (d.label_source === "engine_memory") {
        explain = "Engine memory (Redis miss or unavailable).";
    }
    badge.title =
        (explain ? explain + " " : "") +
        "Label=" +
        regime +
        (conf != null && !Number.isNaN(conf) ? " · |score|=" + conf.toFixed(2) : "") +
        " · risk×" +
        (Number.isFinite(mult) ? mult.toFixed(3) : "?") +
        (parts.length ? " · active: " + parts.join(", ") : "");
    badge.className = "header__regime header__regime--" + (regime === "bull" ? "bull" : regime === "bear" ? "bear" : "neutral");
}

// execution-mode: { success, data: { execution_mode, live_trades_allowed, effective_mode, real_orders_enabled } }
function updateExecutionMode(data) {
    const d = data.data || data;
    if (!d) return;
    const effective = (d.effective_mode || d.execution_mode || "PAPER").toLowerCase();
    const badge = document.getElementById("mode-badge");
    if (badge) {
        badge.textContent = effective === "live" ? "LIVE" : "PAPER";
        badge.className = "header__mode header__mode--clickable " + effective;
        badge.dataset.effectiveMode = effective;
        badge.dataset.realOrdersEnabled = d.real_orders_enabled ? "1" : "0";
    }
}

// operator-status: { success, data: { mode, cash_balance, total_equity, open_positions_count, account_status } }
// NOTE: Open Positions table is driven by dashboard-canonical; operator count fills in when canonical is still loading.
function updateOperator(data) {
    const d = data.data || data;
    const mode = (d.mode || "PAPER").toLowerCase();
    const badge = document.getElementById("mode-badge");
    if (badge) {
        badge.textContent = mode === "live" ? "LIVE" : "PAPER";
        badge.className = "header__mode header__mode--clickable " + mode;
        badge.dataset.effectiveMode = mode;
        if (d.real_orders_enabled !== undefined) {
            badge.dataset.realOrdersEnabled = d.real_orders_enabled ? "1" : "0";
        }
    }
    const cash = document.getElementById("status-cash");
    if (cash && d.cash_balance != null) cash.textContent = "$" + Number(d.cash_balance).toFixed(2);
    const eq = document.getElementById("status-equity");
    if (eq && d.total_equity != null) eq.textContent = "$" + Number(d.total_equity).toFixed(2);
    // Positions count: only when header still "--" (canonical updatePositions normally owns this)
    const pos = document.getElementById("status-positions");
    if (pos && pos.textContent === "--" && d.open_positions_count != null) pos.textContent = String(d.open_positions_count);
    const health = document.getElementById("status-health");
    if (health && d.account_status) health.textContent = d.account_status;
}

// positions: { positions: [...] }
function updatePositions(data) {
    if (!data || typeof data !== "object") return;
    const positions = Array.isArray(data.positions) ? data.positions : [];
    const tbody = document.getElementById("positions-tbody");
    const noData = document.getElementById("positions-no-data");
    const posCount = document.getElementById("status-positions");
    // Always sync status bar count with positions table (source of truth for positions widget)
    if (posCount) posCount.textContent = String(positions.length);
    if (!tbody) return;
    tbody.innerHTML = "";
    if (positions.length === 0) {
        if (noData) noData.classList.add("is-visible");
        return;
    }
    if (noData) noData.classList.remove("is-visible");
    positions.forEach((pos) => {
        const pnl = pos.unrealized_pnl != null ? Number(pos.unrealized_pnl) : 0;
        const sleeve = pos.sleeve || "ACTIVE";
        const badgeCls = "sleeve-badge sleeve-badge--" + sleeve.toLowerCase();
        const tr = document.createElement("tr");
        tr.innerHTML =
            "<td>" + (pos.symbol || "") + "</td>" +
            "<td><span class='" + badgeCls + "'>" + sleeve + "</span></td>" +
            "<td>" + (pos.quantity != null ? Number(pos.quantity).toFixed(6) : "") + "</td>" +
            "<td>" + (pos.average_price != null ? Number(pos.average_price).toFixed(4) : pos.entry_price != null ? Number(pos.entry_price).toFixed(4) : "") + "</td>" +
            "<td>" + (pos.current_price != null ? Number(pos.current_price).toFixed(4) : pos.average_price != null ? Number(pos.average_price).toFixed(4) : "") + "</td>" +
            "<td class='" + (pnl >= 0 ? "pnl-pos" : "pnl-neg") + "'>" + (pnl !== 0 ? pnl.toFixed(2) : "0.00") + "</td>";
        tbody.appendChild(tr);
    });
}

function capturePerfDisplayContext(data) {
    const ctx = (data && data.ledger_principal != null) ? data
        : (data && data.data && data.data.ledger_principal != null ? data.data : null);
    if (ctx && typeof ctx === "object" && ctx.ledger_principal != null) {
        window._perfDisplayContext = ctx;
        applyPerfDisplayContext(ctx);
    }
}

function applyPerfDisplayContext(ctx) {
    if (!ctx || typeof ctx !== "object") return;
    const banner = document.getElementById("perf-context-banner");
    if (banner) {
        banner.textContent =
            "DAY · Current · Principal $" + Number(ctx.ledger_principal || 0).toFixed(0) +
            " · Equity $" + Number(ctx.total_equity || 0).toFixed(2) +
            " · Today UTC " + (ctx.today_start ? String(ctx.today_start).slice(0, 10) : "") +
            " · SCALP excluded";
    }
    function setMetric(id, text, title) {
        const el = document.getElementById(id);
        if (!el || text == null) return;
        el.textContent = text;
        el.dataset.lastGood = text;
        if (title) el.title = title;
    }
    setMetric("analytics-principal", "$" + Number(ctx.ledger_principal).toFixed(0), "Ledger principal via display-context");
    const cumTitle = document.getElementById("chart-cumulative-title");
    if (cumTitle) {
        cumTitle.textContent =
            "Cumulative Returns — All-Time vs $" +
            Number(ctx.ledger_principal).toLocaleString("en-US", { maximumFractionDigits: 0 }) +
            " principal (DAY)";
    }
    setMetric("analytics-equity", "$" + Number(ctx.total_equity).toFixed(2), "Ledger total equity via display-context");
    setMetric("analytics-cash", "$" + Number(ctx.cash_balance).toFixed(2), "Ledger cash via display-context");
    setMetric("status-cash", "$" + Number(ctx.cash_balance).toFixed(2), "DAY cash via display-context");
    setMetric("status-equity", "$" + Number(ctx.total_equity).toFixed(2), "DAY equity via display-context");
    setMetric("analytics-trades", String(ctx.today_closed_sells), "Closed SELLs today (UTC) via display-context");
    setMetric("analytics-winrate", Number(ctx.today_win_rate_pct).toFixed(1) + "%", "Win rate today (UTC) via display-context");
    const ret = Number(ctx.lifetime_account_pnl_usd != null ? ctx.lifetime_account_pnl_usd : ctx.account_return_usd);
    const pnlEl = document.getElementById("analytics-pnl");
    if (pnlEl && (ctx.lifetime_account_pnl_usd != null || ctx.account_return_usd != null)) {
        pnlEl.textContent = (ret >= 0 ? "+" : "") + "$" + ret.toFixed(2);
        pnlEl.dataset.lastGood = pnlEl.textContent;
        pnlEl.classList.remove("pnl-pos", "pnl-neg");
        pnlEl.classList.add(ret >= 0 ? "pnl-pos" : "pnl-neg");
        pnlEl.title = "Lifetime account PnL = Current Equity − Principal (authoritative economic result)";
    }
    const realizedEl = document.getElementById("analytics-realized");
    if (realizedEl && (ctx.visible_history_realized_pnl != null || ctx.realized_pnl != null)) {
        const r = Number(ctx.visible_history_realized_pnl != null ? ctx.visible_history_realized_pnl : ctx.realized_pnl);
        realizedEl.textContent = "$" + r.toFixed(2);
        realizedEl.dataset.lastGood = realizedEl.textContent;
        realizedEl.classList.remove("pnl-pos", "pnl-neg");
        realizedEl.classList.add(r >= 0 ? "pnl-pos" : "pnl-neg");
        realizedEl.title = "Sum of retained closed trades only — may not explain lifetime account PnL when history is incomplete";
    }
    const unrealizedEl = document.getElementById("analytics-unrealized");
    if (unrealizedEl && ctx.unrealized_pnl != null) {
        const u = Number(ctx.unrealized_pnl);
        unrealizedEl.textContent = "$" + u.toFixed(2);
        unrealizedEl.dataset.lastGood = unrealizedEl.textContent;
        unrealizedEl.classList.remove("pnl-pos", "pnl-neg");
        unrealizedEl.classList.add(u >= 0 ? "pnl-pos" : "pnl-neg");
        unrealizedEl.title = "Current open-position mark-to-market PnL";
    }
    const histStatus = document.getElementById("analytics-history-status");
    const histNote = document.getElementById("accounting-history-note");
    if (histStatus) {
        if (ctx.history_incomplete === true) {
            histStatus.textContent = "INCOMPLETE";
            histStatus.classList.remove("pnl-pos");
            histStatus.classList.add("pnl-neg");
            histStatus.title = ctx.note || "Retained realized history does not reconcile to lifetime account PnL";
        } else {
            histStatus.textContent = "OK";
            histStatus.classList.remove("pnl-neg");
            histStatus.classList.add("pnl-pos");
            histStatus.title = "Visible realized + unrealized reconcile to lifetime account PnL within tolerance";
        }
    }
    if (histNote) {
        if (ctx.history_incomplete === true) {
            histNote.hidden = false;
            histNote.textContent =
                "History incomplete — Lifetime account PnL uses Equity − Principal. " +
                "Visible-history Realized is the sum of retained closed trades only and does not reconcile to lifetime PnL because older trade history is unavailable.";
        } else {
            histNote.hidden = true;
            histNote.textContent = "";
        }
    }
    const runEl = document.getElementById("current-run-metrics-note");
    if (runEl) {
        runEl.textContent = ctx.current_run_available
            ? "Current-run start: " + ctx.current_run_start
            : (ctx.current_run_note || "Current-run trade metrics unavailable — no explicit run start marker recorded.");
    }
    updateCurrentAccountSourceLabel(ctx);
}

function updateCurrentAccountSourceLabel(ctx) {
    const el = document.getElementById("current-account-source");
    if (!el) return;
    const state = getDisplayContextFreshness();
    el.classList.remove("source-fresh", "source-stale", "source-failed");
    if (state === "Fresh") el.classList.add("source-fresh");
    else if (state === "Stale") el.classList.add("source-stale");
    else if (state === "Failed") el.classList.add("source-failed");
    const ts = (ctx && ctx.last_updated) || (ENDPOINT_FRESHNESS.perfDisplayContext && ENDPOINT_FRESHNESS.perfDisplayContext.payloadTs) || null;
    el.textContent = "Source: current ledger (display-context) · Freshness: " + state + (ts ? " · " + ts : "");
}

function applyHistoricalDiagnosticContext(ctx) {
    if (!ctx || typeof ctx !== "object") return;
    const allTimeTradesEl = document.getElementById("analytics-trades-alltime");
    if (allTimeTradesEl && ctx.all_time_closed_sells != null) {
        allTimeTradesEl.textContent = String(ctx.all_time_closed_sells);
    }
    const allTimeWinrateEl = document.getElementById("analytics-winrate-alltime");
    if (allTimeWinrateEl && ctx.all_time_win_rate_pct != null) {
        allTimeWinrateEl.textContent = Number(ctx.all_time_win_rate_pct).toFixed(1) + "%";
    }
}

let _historicalDiagnosticsLoaded = false;
const HIST_SCOPE = "?scope=historical_diagnostic";

function loadHistoricalDiagnostics() {
    if (_historicalDiagnosticsLoaded) return;
    _historicalDiagnosticsLoaded = true;
    const statusEl = document.getElementById("hist-diag-status");
    if (statusEl) statusEl.textContent = "Loading historical diagnostics…";
    const endpoints = [
        { path: "/api/performance/portfolio-value" + HIST_SCOPE, key: "portfolioValue" },
        { path: "/api/performance/daily-returns" + HIST_SCOPE, key: "dailyReturns" },
        { path: "/api/performance/cumulative-returns" + HIST_SCOPE, key: "cumulativeReturns" },
        { path: "/api/performance/trade-pnl" + HIST_SCOPE, key: "tradePnl" },
        { path: "/api/performance/trade-duration" + HIST_SCOPE, key: "tradeDuration" },
        { path: "/api/performance/strategy-performance" + HIST_SCOPE, key: "strategyPerformance" },
    ];
    endpoints.forEach(function (ep, i) {
        setTimeout(function () { pollOne(ep); }, i * 300);
    });
    fetchEndpoint("/api/performance/display-context?scope=historical_diagnostic").then(function (result) {
        if (result.ok && result.data) {
            const ctx = (result.data.data && result.data.data.all_time_closed_sells != null) ? result.data.data : null;
            if (ctx) applyHistoricalDiagnosticContext(ctx);
        }
        if (statusEl) statusEl.textContent = "Loaded " + new Date().toLocaleTimeString() + " — all-time data, not current performance.";
        [chartPortfolio, chartDailyReturns, chartCumulativeReturns, chartPnlHistogram, chartTradeDuration, chartStrategyPerformance].forEach(function (c) {
            if (c && typeof c.resize === "function") c.resize();
        });
    });
}

function setupHistoricalDiagnosticsToggle() {
    const panel = document.getElementById("hist-diag-panel");
    if (!panel) return;
    panel.addEventListener("toggle", function () {
        if (panel.open) loadHistoricalDiagnostics();
    });
}

// portfolio-engine/performance: canonical P&L — supplemental only when display-context not loaded
function updatePortfolioPerformance(data) {
    if (!data || !data.success) return;
    const timeFilter = getSelectedTradeTimeFilter();
    if (window._perfDisplayContext && window._perfDisplayContext.ledger_principal != null) {
        updateTrades({ trades: data.trades || [] }, timeFilter);
        window._lastDayTrades = data.trades || [];
        return;
    }
    const p = data.performance || {};
    const trades = data.trades || [];
    const tradesEl = document.getElementById("analytics-trades");
    const winrateEl = document.getElementById("analytics-winrate");
    const pnlEl = document.getElementById("analytics-pnl");
    if (tradesEl && p.total_trades != null) {
        tradesEl.textContent = String(p.total_trades);
        tradesEl.title = (p.scope === "forward_epoch" ? "Forward-epoch closed SELLs (since " + (p.forward_epoch_started_at || "?") + "). Excludes synthetic smoke and admin clears." : "All-time closed SELL count (excludes admin clears).");
    }
    if (winrateEl && p.win_rate != null) {
        winrateEl.textContent = Number(p.win_rate).toFixed(1) + "%";
        winrateEl.title = (p.scope === "forward_epoch" ? "Win rate on forward-epoch strategy closes only." : "All-time win rate from closed SELL rows in paper_trades.");
    }
    const totalPnl = p.total_pnl != null ? Number(p.total_pnl) : null;
    const principal = p.principal != null ? Number(p.principal) : null;
    if (pnlEl) {
        pnlEl.textContent = totalPnl != null ? "$" + totalPnl.toFixed(2) : "--";
        pnlEl.classList.remove("pnl-pos", "pnl-neg");
        if (totalPnl != null) pnlEl.classList.add(totalPnl >= 0 ? "pnl-pos" : "pnl-neg");
        pnlEl.title =
            "Account return vs forward principal (total equity − principal). " +
            (principal != null ? "Principal $" + principal.toFixed(2) + ". " : "") +
            (p.scope === "forward_epoch" ? "Forward epoch only; synthetic smoke excluded. " : "") +
            "Matches realized + unrealized on open book.";
    }
    const realizedEl = document.getElementById("analytics-realized");
    const unrealizedEl = document.getElementById("analytics-unrealized");
    if (realizedEl && p.realized_pnl != null) {
        realizedEl.textContent = "$" + Number(p.realized_pnl).toFixed(2);
        realizedEl.classList.remove("pnl-pos", "pnl-neg");
        realizedEl.classList.add(Number(p.realized_pnl) >= 0 ? "pnl-pos" : "pnl-neg");
        realizedEl.title = "Closed-trade net PnL (forward epoch when active; excludes synthetic smoke and admin clears).";
    }
    if (unrealizedEl && p.unrealized_pnl != null) {
        unrealizedEl.textContent = "$" + Number(p.unrealized_pnl).toFixed(2);
        unrealizedEl.classList.remove("pnl-pos", "pnl-neg");
        unrealizedEl.classList.add(Number(p.unrealized_pnl) >= 0 ? "pnl-pos" : "pnl-neg");
        unrealizedEl.title = "Open positions mark-to-market vs entry (not total account return).";
    }
    updateTrades({ trades }, timeFilter);
    window._lastDayTrades = trades;
}

// performance/strategy-performance: { strategies: [{ name, data: [{ timestamp, value }] }] } or 204
function updateStrategyPerformanceChart(data) {
    const strategies = (data && data.strategies) ? data.strategies : [];
    const noData = document.getElementById("strategy-performance-no-data");
    const canvas = document.getElementById("chart-strategy-performance");
    if (noData) noData.classList.toggle("is-visible", strategies.length === 0);
    if (canvas) canvas.style.display = strategies.length > 0 ? "block" : "none";
    if (chartStrategyPerformance && strategies.length > 0) {
        const labels = [];
        const values = [];
        strategies.forEach(function (s) {
            const name = s.name || s.strategy || "?";
            const pts = s.data || [];
            const val = pts.length > 0 ? (pts[pts.length - 1].value != null ? Number(pts[pts.length - 1].value) : 0) : 0;
            labels.push(name);
            values.push(val);
        });
        chartStrategyPerformance.data.labels = labels;
        chartStrategyPerformance.data.datasets[0].data = values;
        chartStrategyPerformance.data.datasets[0].backgroundColor = values.map(function (v) {
            return v >= 0 ? "#0ecb81" : "#f6465d";
        });
        chartStrategyPerformance.data.datasets[0].borderColor = chartStrategyPerformance.data.datasets[0].backgroundColor;
        const minData = Math.min.apply(null, values);
        const maxData = Math.max.apply(null, values);
        const minV = Math.min(0, minData);
        const maxV = Math.max(0, maxData);
        const span = maxV - minV || 1;
        const pad = Math.max(span * 0.12, 1);
        if (!chartStrategyPerformance.options.scales) chartStrategyPerformance.options.scales = {};
        chartStrategyPerformance.options.scales.x = Object.assign({}, chartStrategyPerformance.options.scales.x || {}, {
            min: minV - pad,
            max: maxV + pad,
            beginAtZero: false,
        });
        chartStrategyPerformance.update("none");
    }
}

// performance/trade-duration: { duration: [values in minutes] } or 204
function updateTradeDurationChart(data) {
    const values = (data && data.duration) ? data.duration : [];
    const noData = document.getElementById("trade-duration-no-data");
    const canvas = document.getElementById("chart-trade-duration");
    if (noData) noData.classList.toggle("is-visible", values.length === 0);
    if (canvas) canvas.style.display = values.length > 0 ? "block" : "none";
    if (chartTradeDuration && values.length > 0) {
        const buckets = [
            { label: "<5m", test: function (n) { return n < 5; } },
            { label: "5-15m", test: function (n) { return n >= 5 && n < 15; } },
            { label: "15-60m", test: function (n) { return n >= 15 && n < 60; } },
            { label: "1-4h", test: function (n) { return n >= 60 && n < 240; } },
            { label: "4-24h", test: function (n) { return n >= 240 && n < 1440; } },
            { label: ">24h", test: function (n) { return n >= 1440; } },
        ];
        const counts = buckets.map(function (b) {
            return values.filter(function (v) {
                const n = Number(v);
                return !isNaN(n) && b.test(n);
            }).length;
        });
        chartTradeDuration.data.labels = buckets.map(function (b) { return b.label; });
        chartTradeDuration.data.datasets[0].data = counts;
        chartTradeDuration.update("none");
    }
}

// performance/trade-pnl: { tradePnl: [pnl values in USD] } or 204
function updatePnlHistogramChart(data) {
    const values = (data && data.tradePnl) ? data.tradePnl : [];
    const noData = document.getElementById("pnl-histogram-no-data");
    const canvas = document.getElementById("chart-pnl-histogram");
    if (noData) noData.classList.toggle("is-visible", values.length === 0);
    if (canvas) canvas.style.display = values.length > 0 ? "block" : "none";
    if (chartPnlHistogram && values.length > 0) {
        const buckets = [
            { label: "<$-10", test: function (n) { return n < -10; } },
            { label: "$-10 to -2", test: function (n) { return n >= -10 && n < -2; } },
            { label: "$-2 to 0", test: function (n) { return n >= -2 && n < 0; } },
            { label: "$0 to 2", test: function (n) { return n >= 0 && n < 2; } },
            { label: "$2 to 10", test: function (n) { return n >= 2 && n < 10; } },
            { label: ">$10", test: function (n) { return n >= 10; } },
        ];
        const colors = ["#f6465d", "#f6465d", "#f0b90b", "#0ecb81", "#0ecb81", "#0ecb81"];
        const counts = buckets.map(function (b) {
            return values.filter(function (v) {
                const n = Number(v);
                return !isNaN(n) && b.test(n);
            }).length;
        });
        chartPnlHistogram.data.labels = buckets.map(function (b) { return b.label; });
        chartPnlHistogram.data.datasets[0].data = counts;
        chartPnlHistogram.data.datasets[0].backgroundColor = colors;
        chartPnlHistogram.data.datasets[0].borderColor = colors;
        chartPnlHistogram.update("none");
    }
}

// performance/cumulative-returns: { cumulative: [{ timestamp, value }] } or 204
function updateCumulativeReturnsChart(data) {
    const series = data && data.cumulative ? data.cumulative : [];
    const noData = document.getElementById("cumulative-no-data");
    const canvas = document.getElementById("chart-cumulative-returns");
    if (noData) noData.classList.toggle("is-visible", series.length === 0);
    if (canvas) canvas.style.display = series.length > 0 ? "block" : "none";
    if (chartCumulativeReturns && series.length > 0) {
        const seen = new Map();
        chartCumulativeReturns.data.labels = series.map(function (p) {
            const ts = (p.timestamp || p.date || "");
            const base = ts.length >= 16 ? ts.slice(0, 16).replace("T", " ") : (ts.length >= 10 ? ts.slice(0, 10) : ts);
            const n = (seen.get(base) || 0) + 1;
            seen.set(base, n);
            return n > 1 ? base + " +" + n : base;
        });
        chartCumulativeReturns.data.datasets[0].data = series.map(function (p) {
            return p.value != null ? Number(p.value) : NaN;
        });
        chartCumulativeReturns.update("none");
    }
}

// portfolio-value: { portfolio: [{ timestamp, value }] }
function updatePortfolioChart(data) {
    if (!data || typeof data !== "object") return;
    const portfolio = Array.isArray(data.portfolio) ? data.portfolio : [];
    const noData = document.getElementById("portfolio-no-data");
    const canvas = document.getElementById("chart-portfolio-value");

    if (portfolio.length === 0) {
        // No data - show message
        if (noData) noData.classList.add("is-visible");
        if (canvas) canvas.style.display = "none";
        return;
    }

    // Has data
    if (canvas) canvas.style.display = "block";

    if (chartPortfolio) {
        // Chart available - render it
        if (noData) noData.classList.remove("is-visible");
        chartPortfolio.data.labels = portfolio.map((p) => {
            const ts = (p.timestamp || p.date || "");
            return ts.length >= 16 ? ts.slice(0, 16).replace("T", " ") : ts.slice(0, 10);
        });
        chartPortfolio.data.datasets[0].data = portfolio.map((p) => (p.value != null ? Number(p.value) : NaN));
        chartPortfolio.update("none");
    } else {
        // No chart - show latest value as text.
        const latestValue = portfolio[portfolio.length - 1]?.value;
        const displayText = "Portfolio Value: $" + (latestValue != null ? Number(latestValue).toFixed(2) : "N/A");
        if (noData) {
            noData.textContent = displayText;
            noData.classList.add("is-visible");
        }
    }
}

function parseTradePnl(val) {
    if (val == null || val === "" || val === "None" || val === "null") return null;
    const n = Number(val);
    return Number.isFinite(n) ? n : null;
}

// daily-returns: { returns: [{ timestamp, value }] } — collapse to one bar per calendar day (avoids duplicate x labels)
function updateDailyReturnsChart(data) {
    if (!data || typeof data !== "object") return;
    const src = data.data_source || "";
    const returns = Array.isArray(data.returns) ? data.returns : [];
    const isBaseline =
        src === "default_baseline" ||
        src === "no_data" ||
        (returns.length === 1 &&
            (returns[0].timestamp || "").indexOf("2024-01-01") === 0 &&
            Number(returns[0].value) === 0);
    const byDay = new Map();
    if (!isBaseline) {
        for (let i = 0; i < returns.length; i++) {
            const r = returns[i];
            const ts = r.timestamp || r.date || "";
            const day = ts.length >= 10 ? ts.slice(0, 10) : "";
            if (!day) continue;
            byDay.set(day, r);
        }
    }
    const sliceReturns = Array.from(byDay.entries())
        .sort((a, b) => a[0].localeCompare(b[0]))
        .map((entry) => entry[1])
        .slice(-30);
    const noData = document.getElementById("daily-returns-no-data");
    const canvas = document.getElementById("chart-daily-returns");
    if (noData) noData.classList.toggle("is-visible", sliceReturns.length === 0);
    if (canvas) canvas.style.display = sliceReturns.length > 0 ? "block" : "none";
    if (chartDailyReturns && sliceReturns.length > 0) {
        chartDailyReturns.data.labels = sliceReturns.map((r) => {
            const ts = r.timestamp || r.date || "";
            return ts.length >= 10 ? ts.slice(0, 10) : ts;
        });
        chartDailyReturns.data.datasets[0].data = sliceReturns.map((r) => (r.value != null ? Number(r.value) : 0));
        chartDailyReturns.data.datasets[0].backgroundColor = sliceReturns.map((r) =>
            (r.value != null && Number(r.value) >= 0) ? "#0ecb81" : "#f6465d"
        );
        chartDailyReturns.data.datasets[0].borderColor = chartDailyReturns.data.datasets[0].backgroundColor;
        chartDailyReturns.update("none");
    }
}

function normalizeTs(ts) {
    if (ts == null) return 0;
    if (typeof ts === "number") return ts > 1e12 ? ts : ts * 1000;
    const value = typeof ts === "string" && /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(ts) && !/[zZ]|[+-]\d{2}:?\d{2}$/.test(ts)
        ? ts.replace(" ", "T") + "Z"
        : ts;
    const d = new Date(value);
    return isNaN(d.getTime()) ? 0 : d.getTime();
}

function getSelectedTradeTimeFilter() {
    return (document.querySelector('input[name="time-filter"]:checked') || {}).value || "all";
}

function filterTradesByTime(trades, timeFilter, nowMs) {
    const filter = timeFilter || "all";
    if (filter === "all") return trades;
    const now = nowMs == null ? Date.now() : nowMs;
    const today = new Date(now).toISOString().slice(0, 10);
    return trades.filter(function (trade) {
        const ts = normalizeTs(trade.timestamp || trade.created_at || trade.time);
        if (!ts) return false;
        if (filter === "today") return new Date(ts).toISOString().slice(0, 10) === today;
        if (filter === "7d") return ts >= now - 7 * 86400000 && ts <= now;
        return true;
    });
}

// portfolio-engine/performance: { trades: [...] } — DAY ledger only
function updateTrades(data, timeFilter) {
    if (!data || typeof data !== "object") return;
    const trades = Array.isArray(data.trades) ? data.trades : [];
    const tbody = document.getElementById("trades-tbody");
    const noData = document.getElementById("trades-no-data");
    if (!tbody) return;
    tbody.innerHTML = "";
    const filtered = filterTradesByTime(trades, timeFilter);
    if (filtered.length === 0) {
        if (noData) noData.classList.add("is-visible");
        return;
    }
    if (noData) noData.classList.remove("is-visible");
    // Newest first (chronological), then SELL before BUY at same second
    const sorted = [...filtered].sort((a, b) => {
        const ta = normalizeTs(a.timestamp || a.created_at || a.time);
        const tb = normalizeTs(b.timestamp || b.created_at || b.time);
        if (tb !== ta) return tb - ta;
        const aSell = (a.side || "").toLowerCase() === "sell" ? 1 : 0;
        const bSell = (b.side || "").toLowerCase() === "sell" ? 1 : 0;
        return bSell - aSell;
    });
    sorted.slice(0, 50).forEach((t) => {
        const pnl = parseTradePnl(t.pnl);
        const ts = t.timestamp || t.created_at || t.time || "";
        const timeStr = typeof ts === "string" ? ts.slice(11, 19) : "";
        const pnlStr = pnl != null ? "$" + pnl.toFixed(2) : "—";
        const pnlClass = pnl != null ? (pnl >= 0 ? "pnl-pos" : "pnl-neg") : "";
        const sleeve = t.sleeve || "ACTIVE";
        const badgeCls = "sleeve-badge sleeve-badge--" + sleeve.toLowerCase();
        const tradeId = t.trade_id || t.id || "";
        const tr = document.createElement("tr");
        tr.innerHTML =
            "<td>" + timeStr + "</td>" +
            "<td>" + (t.symbol || "") + "</td>" +
            "<td><span class='" + badgeCls + "'>" + sleeve + "</span></td>" +
            "<td>" + (t.side || "").toLowerCase() + "</td>" +
            "<td>" + (t.quantity != null ? Number(t.quantity).toFixed(6) : "") + "</td>" +
            "<td>" + (t.price != null ? Number(t.price).toFixed(4) : t.fill_price != null ? Number(t.fill_price).toFixed(4) : "") + "</td>" +
            "<td class='" + pnlClass + "'>" + pnlStr + "</td>" +
            "<td><button type='button' class='trade-drill-btn' data-trade-id='" + String(tradeId).replace(/'/g, "") + "'>View</button></td>";
        tbody.appendChild(tr);
        if (tradeId) {
            const detailTr = document.createElement("tr");
            detailTr.className = "trade-drill-row";
            detailTr.style.display = "none";
            const detailTd = document.createElement("td");
            detailTd.colSpan = 8;
            const pre = document.createElement("pre");
            pre.className = "panel-pre trade-drill-pre";
            detailTd.appendChild(pre);
            detailTr.appendChild(detailTd);
            tbody.appendChild(detailTr);
            const btn = tr.querySelector(".trade-drill-btn");
            if (btn) {
                btn.addEventListener("click", function () {
                    const open = detailTr.style.display !== "none";
                    detailTr.style.display = open ? "none" : "table-row";
                    if (!open && pre.textContent === "") {
                        fetchTradeDrilldown(tradeId, pre);
                    }
                });
            }
        }
    });
}

function fmt(x, digits) {
    if (x == null) return "n/a";
    if (typeof digits === "undefined") digits = 4;
    if (Number.isInteger(x) || (typeof x === "number" && Number.isInteger(x))) return String(Math.round(x));
    return Number(x).toFixed(digits);
}

function updateDailyPerformanceSnapshot(data) {
    const d = data && (data.data || data);
    const container = document.getElementById("snapshot-container");
    const noData = document.getElementById("snapshot-no-data");
    if (!container) return;
    if (!d || typeof d !== "object") {
        if (noData) noData.classList.add("is-visible");
        return;
    }
    container.style.display = "block";
    if (noData) noData.classList.remove("is-visible");

    let counts = (d.trade_counts || [])
        .map((r) => `  ${(r.side || "").padEnd(4)} mode=${(r.mode || "").padEnd(5)} count=${r.count}`)
        .join("\n") || "  (no rows)";
    const cons = d.consistency || {};
    if (cons.buy_count != null && cons.sell_count != null) {
        counts += `\n  BUY/SELL: ${cons.buy_count}/${cons.sell_count}  open=${cons.open_positions ?? 0}  ledger_gap=${cons.orphan_estimate ?? "—"}`;
        counts +=
            "\n  (ledger_gap = BUY−SELL−open in SQLite; not the dashboard position count. If dashboard uses canonical engine only, compare to Open Positions there.)";
    }

    const sell = d.sell || {};
    let sellTxt = "  No SELLs yet.";
    if (sell.count > 0) {
        sellTxt =
            `  SELL count: ${sell.count}\n` +
            `  Win/Loss/Flat: ${sell.win || 0}/${sell.loss || 0}/${sell.flat || 0}  Winrate: ${fmt(sell.winrate, 3)}\n` +
            `  PnL% avg/median: ${fmt(sell.avg_pnl_pct)} / ${fmt(sell.median_pnl_pct)}`;
        if (sell.hold_sec_p50 != null) {
            sellTxt += `\n  Hold sec p10/p50/p90: ${fmt(sell.hold_sec_p10, 0)} / ${fmt(sell.hold_sec_p50, 0)} / ${fmt(sell.hold_sec_p90, 0)}`;
        }
    }

    const exitTypes = (d.exit_types || [])
        .map((r) => `  ${String(r.exit_type || "").padEnd(24)} ${r.count}`)
        .join("\n") || "  (no SELL rows)";

    const churn = d.churn || {};
    let churnTxt =
        churn.total_with_hold_time > 0
            ? `  Fast exits (<= ${churn.fast_exit_minutes || 10} min): ${churn.fast_count}/${churn.total_with_hold_time}  (${fmt(churn.fast_pct, 3)})\n`
            : "  n/a (no hold_time_seconds)\n";
    const liveRows = (d.mode_gate_sanity || {}).mode_live_rows || 0;
    churnTxt += `  mode='live' rows: ${liveRows}`;

    const topSymbols = (d.top_symbols || [])
        .map((r) => `  ${(r.symbol || "").padEnd(12)} sells=${(r.sells || 0).toString().padStart(3)}  wins=${(r.wins || 0).toString().padStart(2)}  wr=${fmt(r.winrate, 3)}  avg_pnl%=${fmt(r.avg_pnl_pct)}`)
        .join("\n") || "  (no SELL rows)";

    const countsEl = document.getElementById("snapshot-counts");
    const sellEl = document.getElementById("snapshot-sell");
    const exitEl = document.getElementById("snapshot-exit-types");
    const churnEl = document.getElementById("snapshot-churn");
    const topSymbolsEl = document.getElementById("snapshot-top-symbols");
    if (countsEl) countsEl.textContent = counts;
    if (sellEl) sellEl.textContent = sellTxt;
    if (exitEl) exitEl.textContent = exitTypes;
    if (churnEl) churnEl.textContent = churnTxt;
    if (topSymbolsEl) topSymbolsEl.textContent = topSymbols;
}

function updateSleeveStatus(data) {
    const d = data.data || data;
    const container = document.getElementById("sleeve-summary");
    if (!container) return;
    if (!d || !d.sleeve_enabled) {
        container.innerHTML = '<span class="sleeve-disabled">Sleeves disabled</span>';
        return;
    }
    const sleeves = d.sleeves || {};
    let html = "";
    for (const [name, s] of Object.entries(sleeves)) {
        const cls = name === "CORE" ? "sleeve-card--core" : "sleeve-card--active";
        const util = s.utilization_pct != null ? Number(s.utilization_pct).toFixed(1) : "0.0";
        const budget = s.capital_budget != null ? Number(s.capital_budget).toFixed(2) : "0.00";
        const notional = s.notional_used != null ? Number(s.notional_used).toFixed(2) : "0.00";
        const upnl = s.unrealized_pnl != null ? Number(s.unrealized_pnl).toFixed(2) : "0.00";
        const upnlCls = Number(upnl) >= 0 ? "pnl-pos" : "pnl-neg";
        html += '<div class="sleeve-card ' + cls + '">' +
            '<div class="sleeve-card__header"><span class="sleeve-badge sleeve-badge--' + name.toLowerCase() + '">' + name + '</span>' +
            '<span class="sleeve-card__positions">' + (s.positions || 0) + '/' + (s.max_positions || 0) + ' pos</span></div>' +
            '<div class="sleeve-card__row"><span>Budget</span><span>$' + budget + '</span></div>' +
            '<div class="sleeve-card__row"><span>Deployed</span><span>$' + notional + '</span></div>' +
            '<div class="sleeve-card__row"><span>Utilization</span><span>' + util + '%</span></div>' +
            '<div class="sleeve-card__row"><span>Unrealized</span><span class="' + upnlCls + '">$' + upnl + '</span></div>' +
            '</div>';
    }
    container.innerHTML = html;
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}

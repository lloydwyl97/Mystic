// Mystic Trading Dashboard - Live data in widgets only
// No raw JSON. Fetches APIs and populates charts, tables, and status.

const REFRESH_MS = 15000; // 15s for snappier dashboard updates
// All Mystic dashboard data is driven by live backend endpoints below.
const ENDPOINTS = [
    // Single canonical portfolio snapshot (positions, sleeves, PnL, risk, trades, scoreboard, daily snapshot, operator, regime, invariants summary)
    { path: "/api/portfolio-engine/dashboard-canonical", key: "dashboardCanonical" },
    // Mode badge: owned by dashboard-canonical operator + POST /execution-mode after toggle (avoid fighting execution-mode GET poll)
    // Regime is included in dashboard-canonical (same tick as positions/risk) — no separate poll
    { path: "/api/system/health/quick", key: "systemHealth" },
    { path: "/api/portfolio-engine/latency", key: "latency" },
    { path: "/api/portfolio-engine/invariants-detail", key: "invariantsDetail" },
    { path: "/api/performance/portfolio-value", key: "portfolioValue" },
    { path: "/api/performance/daily-returns", key: "dailyReturns" },
    { path: "/api/performance/cumulative-returns", key: "cumulativeReturns" },
    { path: "/api/performance/trade-pnl", key: "tradePnl" },
    { path: "/api/performance/trade-duration", key: "tradeDuration" },
    { path: "/api/performance/strategy-performance", key: "strategyPerformance" },
    { path: "/api/performance/analytics", key: "analytics" },
    { path: "/api/portfolio-engine/rejects", key: "rejects" },
    { path: "/api/portfolio-engine/decisions", key: "decisions" },
    { path: "/api/system/health/comprehensive", key: "systemHealthFull" },
    { path: "/api/portfolio-engine/scoreboard?days=7", key: "scoreboard7d" },
    { path: "/api/portfolio-engine/learning-status?limit=20", key: "learningStatus" },
];

let chartPortfolio = null;
let chartDailyReturns = null;
let chartCumulativeReturns = null;
let chartPnlHistogram = null;
let chartTradeDuration = null;
let chartStrategyPerformance = null;

function init() {
    try {
        initCharts();
    } catch (e) {
        console.error("Chart.js init failed:", e);
    }
    startPolling();
    startMarketReadinessPolling();
    updateHeaderMeta();
    setInterval(updateHeaderMeta, 1000);
    initModeSwitch();
    initOperatorControls();
    initRefreshButton();
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
        const promises = ENDPOINTS.map(function (ep) {
            return pollOne(ep);
        }).concat([
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
                pollOne({ path: "/api/portfolio-engine/dashboard-canonical", key: "dashboardCanonical" });
            } else {
                const msg = data.error || data.detail || "Save failed";
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
                pollOne({ path: "/api/portfolio-engine/dashboard-canonical", key: "dashboardCanonical" });
            } else {
                const reason = data.error || data.detail || "unknown";
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

async function fetchEndpoint(path) {
    try {
        const controller = new AbortController();
        const t = setTimeout(() => controller.abort(), 15000);
        const res = await fetch(path, { method: "GET", signal: controller.signal, cache: "no-cache" });
        clearTimeout(t);
        if (res.status === 204) {
            return { ok: true, data: {} };
        }
        const data = await res.json();
        return { ok: res.ok, data };
    } catch (e) {
        return { ok: false, data: null };
    }
}

function startPolling() {
    ENDPOINTS.forEach((ep, i) => {
        setTimeout(() => pollOne(ep), i * 400);
    });
    setInterval(() => ENDPOINTS.forEach(pollOne), REFRESH_MS);
}

let lastUpdateTime = null;

async function pollOne(ep) {
    const result = await fetchEndpoint(ep.path);
    if (!result.ok) {
        if (ep.key === "dashboardCanonical") {
            updateDashboardCanonical({ success: false });
        }
        return;
    }
    const data = result.data;
    if (data === null || data === undefined) return;
    lastUpdateTime = new Date();
    try {
        updateUI(ep.key, typeof data === "object" ? data : {});
    } catch (err) {
        console.warn("Dashboard updateUI error for", ep.key, err);
    }
}

function updateHeaderMeta() {
    const meta = document.getElementById("header-meta");
    if (meta) meta.textContent = "Last update: " + (lastUpdateTime ? lastUpdateTime.toLocaleTimeString() : "--");
}

function updateUI(key, data) {
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
        case "analytics":
            updateAnalytics(data);
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
        case "marketDataReadiness":
            updateMarketDataReadiness(data);
            break;
    }
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
function updateDashboardCanonical(res) {
    if (!res || res.success === false) {
        const inv = document.getElementById("analytics-invariants");
        if (inv) {
            inv.textContent = "SNAPSHOT ERR";
            inv.classList.remove("pnl-pos");
            inv.classList.add("pnl-neg");
        }
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
    }
    if (d.daily_performance) {
        updateDailyPerformanceSnapshot({ data: d.daily_performance });
    }
    if (d.invariants) {
        updateInvariants({ success: true, data: d.invariants });
    }
    if (d.consistency_ok === false) {
        const el = document.getElementById("analytics-invariants");
        if (el) {
            const v = Array.isArray(d.consistency_violations) ? d.consistency_violations : [];
            el.textContent = v.length ? "XCHECK" : "FAIL";
            el.title = v.length ? v.join("; ") : "Snapshot cross-check failed";
            el.classList.remove("pnl-pos");
            el.classList.add("pnl-neg");
        }
    }
    if (d.regime && typeof d.regime === "object") {
        updateRegime({ success: true, data: d.regime });
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

// portfolio-engine/scoreboard/today: { success, data: { status, pass, ... } }
function updateScoreboardToday(data) {
    const d = data.data || data;
    const el = document.getElementById("analytics-scoreboard");
    if (!el) return;
    if (!d) { el.textContent = "--"; return; }
    const status = d.status || d.pass_fail || (d.pass ? "PASS" : d.fail ? "FAIL" : null);
    const failReasons = d.fail_reasons || "";
    el.title =
        "Today's closed-trade stats (SQLite). PASS/FAIL is diagnostic only — it does not block buys. " +
        (failReasons || status || "");

    // Display status with fail reasons if present
    if (status && String(status).toUpperCase() === "FAIL" && failReasons) {
        el.textContent = `FAIL: ${failReasons}`;
    } else {
        el.textContent = status ? String(status) : "--";
        el.title = "Today's closed-trade stats (SQLite). Diagnostic only; does not block buys.";
    }

    el.classList.remove("pnl-pos", "pnl-neg");
    if (status && (String(status).toUpperCase() === "PASS" || String(status).toUpperCase() === "OK")) el.classList.add("pnl-pos");
    else if (status && String(status).toUpperCase() === "FAIL") el.classList.add("pnl-neg");
    else if (status && String(status).toUpperCase() === "PENDING") el.classList.add("pnl-pos");
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
    lines.push("(legacy total_violations = same as invariant_events_total; not # of failing checks now)");

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
    if (!el) return;
    if (!d || typeof d !== "object") {
        el.textContent = "No scoreboard data.";
        return;
    }
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

// paper-trading/portfolio: { portfolio: { total_balance_usd, cash_balance, ... } }
// Dashboard fix: Do NOT touch status-cash or status-equity. operator-status is the single source
// (reads from SQLite). Avoids "wrong then correct then wrong" flicker from competing endpoints.
function updatePaperPortfolio(data) {
    const p = data.portfolio || data;
    if (!p) return;
    // Previously filled cash/equity when "--"; now operator-status owns those exclusively.
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

// portfolio-engine/performance: canonical P&L for monitoring (realized, unrealized, trades)
function updatePortfolioPerformance(data) {
    if (!data || !data.success) return;
    const p = data.performance || {};
    const trades = data.trades || [];
    const tradesEl = document.getElementById("analytics-trades");
    const winrateEl = document.getElementById("analytics-winrate");
    const pnlEl = document.getElementById("analytics-pnl");
    if (tradesEl && p.total_trades != null) {
        tradesEl.textContent = String(p.total_trades);
        tradesEl.title = "All-time: total closed trades in engine performance bundle (not “today only”).";
    }
    if (winrateEl && p.win_rate != null) {
        winrateEl.textContent = Number(p.win_rate).toFixed(1) + "%";
        winrateEl.title = "All-time win rate from same performance bundle as Total Trades.";
    }
    const totalPnl = p.total_pnl != null ? Number(p.total_pnl) : null;
    const principal = p.principal != null ? Number(p.principal) : null;
    if (pnlEl) {
        pnlEl.textContent = totalPnl != null ? "$" + totalPnl.toFixed(2) : "--";
        pnlEl.classList.remove("pnl-pos", "pnl-neg");
        if (totalPnl != null) pnlEl.classList.add(totalPnl >= 0 ? "pnl-pos" : "pnl-neg");
        pnlEl.title =
            "Account return vs principal (total equity − principal). " +
            (principal != null ? "Principal $" + principal.toFixed(2) + ". " : "") +
            "Not the same as realized + unrealized on open-book MTM.";
    }
    const realizedEl = document.getElementById("analytics-realized");
    const unrealizedEl = document.getElementById("analytics-unrealized");
    if (realizedEl && p.realized_pnl != null) {
        realizedEl.textContent = "$" + Number(p.realized_pnl).toFixed(2);
        realizedEl.classList.remove("pnl-pos", "pnl-neg");
        realizedEl.classList.add(Number(p.realized_pnl) >= 0 ? "pnl-pos" : "pnl-neg");
        realizedEl.title = "Closed-trade net PnL from ledger (can be negative even when account return is positive).";
    }
    if (unrealizedEl && p.unrealized_pnl != null) {
        unrealizedEl.textContent = "$" + Number(p.unrealized_pnl).toFixed(2);
        unrealizedEl.classList.remove("pnl-pos", "pnl-neg");
        unrealizedEl.classList.add(Number(p.unrealized_pnl) >= 0 ? "pnl-pos" : "pnl-neg");
        unrealizedEl.title = "Open positions mark-to-market vs entry (not total account return).";
    }
    updateTrades({ trades });
}

// paper-trading/performance: { performance: { total_trades, win_rate, total_pnl } }
// Dashboard fix: Do NOT update analytics-trades/winrate/pnl. portfolio-engine/performance
// is the canonical source. Prevents multiple endpoints fighting over the same DOM.
function updatePaperPerformance(data) {
    const p = data.performance || data;
    if (!p) return;
    // Trades/winrate/PnL widgets: owned by updatePortfolioPerformance only.
}

// performance/analytics: { current_metrics: { total_trades, win_rate, total_pnl } }
// Dashboard fix: Do NOT update analytics-trades/winrate/pnl. portfolio-engine/performance
// is the canonical source. Prevents multiple endpoints fighting over the same DOM.
function updateAnalytics(data) {
    const m = data.current_metrics || data.performance || data;
    if (!m) return;
    // Trades/winrate/PnL widgets: owned by updatePortfolioPerformance only.
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
    const returns = Array.isArray(data.returns) ? data.returns : [];
    const byDay = new Map();
    for (let i = 0; i < returns.length; i++) {
        const r = returns[i];
        const ts = r.timestamp || r.date || "";
        const day = ts.length >= 10 ? ts.slice(0, 10) : "";
        if (!day) continue;
        byDay.set(day, r);
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
    const d = new Date(ts);
    return isNaN(d.getTime()) ? 0 : d.getTime();
}

// paper-trading/trades or portfolio-engine/performance: { trades: [{ symbol, side, quantity, price, pnl, timestamp }] }
function updateTrades(data) {
    if (!data || typeof data !== "object") return;
    const trades = Array.isArray(data.trades) ? data.trades : [];
    const tbody = document.getElementById("trades-tbody");
    const noData = document.getElementById("trades-no-data");
    if (!tbody) return;
    tbody.innerHTML = "";
    if (trades.length === 0) {
        if (noData) noData.classList.add("is-visible");
        return;
    }
    if (noData) noData.classList.remove("is-visible");
    // Newest first (chronological), then SELL before BUY at same second
    const sorted = [...trades].sort((a, b) => {
        const ta = normalizeTs(a.timestamp || a.created_at || a.time);
        const tb = normalizeTs(b.timestamp || b.created_at || b.time);
        if (tb !== ta) return tb - ta;
        const aSell = (a.side || "").toLowerCase() === "sell" ? 1 : 0;
        const bSell = (b.side || "").toLowerCase() === "sell" ? 1 : 0;
        return bSell - aSell;
    });
    sorted.slice(0, 30).forEach((t) => {
        const pnl = parseTradePnl(t.pnl);
        const ts = t.timestamp || t.created_at || t.time || "";
        const timeStr = typeof ts === "string" ? ts.slice(11, 19) : "";
        const pnlStr = pnl != null ? "$" + pnl.toFixed(2) : "—";
        const pnlClass = pnl != null ? (pnl >= 0 ? "pnl-pos" : "pnl-neg") : "";
        const sleeve = t.sleeve || "ACTIVE";
        const badgeCls = "sleeve-badge sleeve-badge--" + sleeve.toLowerCase();
        const tr = document.createElement("tr");
        tr.innerHTML =
            "<td>" + timeStr + "</td>" +
            "<td>" + (t.symbol || "") + "</td>" +
            "<td><span class='" + badgeCls + "'>" + sleeve + "</span></td>" +
            "<td>" + (t.side || "").toLowerCase() + "</td>" +
            "<td>" + (t.quantity != null ? Number(t.quantity).toFixed(6) : "") + "</td>" +
            "<td>" + (t.price != null ? Number(t.price).toFixed(4) : t.fill_price != null ? Number(t.fill_price).toFixed(4) : "") + "</td>" +
            "<td class='" + pnlClass + "'>" + pnlStr + "</td>";
        tbody.appendChild(tr);
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

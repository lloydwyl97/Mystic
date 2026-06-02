"""
Chart Generator - Live Configuration Only

Generates trading charts and performance visualizations.
All configuration values come from live config - no hardcoded values.
"""

import logging
import os
import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# Configure logging
logger = logging.getLogger(__name__)

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_default_symbol() -> str:
    """Get default symbol from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "trading_universe") and hasattr(value.trading_universe, "top10_symbols"):
                symbols = value.trading_universe.top10_symbols
                if isinstance(symbols, list) and symbols:
                    return str(symbols[0])
        except (AttributeError, ValueError, TypeError, IndexError):
            pass

    symbol = os.getenv("CHART_GENERATOR_DEFAULT_SYMBOL", "").strip()
    if symbol:
        return symbol

    return "ETHUSDT"


def _get_default_db_path() -> str:
    """Get default database path from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "default_db_path"):
                db_path = value.chart_generator.default_db_path
                if isinstance(db_path, str) and db_path:
                    return db_path.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    db_path = os.getenv("CHART_GENERATOR_DEFAULT_DB_PATH", "").strip()
    if db_path:
        return db_path

    return os.getenv("SIMULATION_TRADES_DB", "simulation_trades.db")


def _get_figure_width() -> float:
    """Get figure width from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "figure_width"):
                width = value.chart_generator.figure_width
                if isinstance(width, (int, float)) and width > 0:
                    return float(width)
        except (AttributeError, ValueError, TypeError):
            pass

    width = os.getenv("CHART_GENERATOR_FIGURE_WIDTH", "").strip()
    if width:
        try:
            return float(width)
        except (ValueError, TypeError):
            pass

    return 12.0


def _get_figure_height() -> float:
    """Get figure height from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "figure_height"):
                height = value.chart_generator.figure_height
                if isinstance(height, (int, float)) and height > 0:
                    return float(height)
        except (AttributeError, ValueError, TypeError):
            pass

    height = os.getenv("CHART_GENERATOR_FIGURE_HEIGHT", "").strip()
    if height:
        try:
            return float(height)
        except (ValueError, TypeError):
            pass

    return 6.0


def _get_scatter_alpha() -> float:
    """Get scatter plot alpha from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "scatter_alpha"):
                alpha = value.chart_generator.scatter_alpha
                if isinstance(alpha, (int, float)) and 0.0 <= alpha <= 1.0:
                    return float(alpha)
        except (AttributeError, ValueError, TypeError):
            pass

    alpha = os.getenv("CHART_GENERATOR_SCATTER_ALPHA", "").strip()
    if alpha:
        try:
            alpha_val = float(alpha)
            if 0.0 <= alpha_val <= 1.0:
                return alpha_val
        except (ValueError, TypeError):
            pass

    return 0.7


def _get_line_alpha() -> float:
    """Get line plot alpha from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "line_alpha"):
                alpha = value.chart_generator.line_alpha
                if isinstance(alpha, (int, float)) and 0.0 <= alpha <= 1.0:
                    return float(alpha)
        except (AttributeError, ValueError, TypeError):
            pass

    alpha = os.getenv("CHART_GENERATOR_LINE_ALPHA", "").strip()
    if alpha:
        try:
            alpha_val = float(alpha)
            if 0.0 <= alpha_val <= 1.0:
                return alpha_val
        except (ValueError, TypeError):
            pass

    return 0.2


def _get_profit_color_positive() -> str:
    """Get positive profit color from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "profit_color_positive"):
                color = value.chart_generator.profit_color_positive
                if isinstance(color, str) and color:
                    return color.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    color = os.getenv("CHART_GENERATOR_PROFIT_COLOR_POSITIVE", "").strip()
    if color:
        return color

    return "green"


def _get_profit_color_negative() -> str:
    """Get negative profit color from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "profit_color_negative"):
                color = value.chart_generator.profit_color_negative
                if isinstance(color, str) and color:
                    return color.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    color = os.getenv("CHART_GENERATOR_PROFIT_COLOR_NEGATIVE", "").strip()
    if color:
        return color

    return "red"


def _get_trade_chart_output_path() -> str:
    """Get trade chart output path from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "trade_chart_output_path"):
                path = value.chart_generator.trade_chart_output_path
                if isinstance(path, str) and path:
                    return path.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    path = os.getenv("CHART_GENERATOR_TRADE_CHART_OUTPUT_PATH", "").strip()
    if path:
        return path

    return "trade_chart.png"


def _get_performance_chart_output_path() -> str:
    """Get performance chart output path from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "performance_chart_output_path"):
                path = value.chart_generator.performance_chart_output_path
                if isinstance(path, str) and path:
                    return path.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    path = os.getenv("CHART_GENERATOR_PERFORMANCE_CHART_OUTPUT_PATH", "").strip()
    if path:
        return path

    return "performance_chart.png"


def _get_chart_title_trade() -> str:
    """Get trade chart title from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "chart_title_trade"):
                title = value.chart_generator.chart_title_trade
                if isinstance(title, str) and title:
                    return title.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    title = os.getenv("CHART_GENERATOR_CHART_TITLE_TRADE", "").strip()
    if title:
        return title

    return "Trade Chart for {symbol}"


def _get_chart_title_performance() -> str:
    """Get performance chart title from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "chart_title_performance"):
                title = value.chart_generator.chart_title_performance
                if isinstance(title, str) and title:
                    return title.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    title = os.getenv("CHART_GENERATOR_CHART_TITLE_PERFORMANCE", "").strip()
    if title:
        return title

    return "AI Trading Performance Over Time"


def _get_xlabel_default() -> str:
    """Get default x-axis label from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "xlabel_default"):
                label = value.chart_generator.xlabel_default
                if isinstance(label, str) and label:
                    return label.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    label = os.getenv("CHART_GENERATOR_XLABEL_DEFAULT", "").strip()
    if label:
        return label

    return "Time"


def _get_ylabel_price() -> str:
    """Get price y-axis label from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "ylabel_price"):
                label = value.chart_generator.ylabel_price
                if isinstance(label, str) and label:
                    return label.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    label = os.getenv("CHART_GENERATOR_YLABEL_PRICE", "").strip()
    if label:
        return label

    return "Price"


def _get_ylabel_profit() -> str:
    """Get profit y-axis label from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "ylabel_profit"):
                label = value.chart_generator.ylabel_profit
                if isinstance(label, str) and label:
                    return label.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    label = os.getenv("CHART_GENERATOR_YLABEL_PROFIT", "").strip()
    if label:
        return label

    return "Cumulative Profit ($)"


def _get_legend_label_profit() -> str:
    """Get profit legend label from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "legend_label_profit"):
                label = value.chart_generator.legend_label_profit
                if isinstance(label, str) and label:
                    return label.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    label = os.getenv("CHART_GENERATOR_LEGEND_LABEL_PROFIT", "").strip()
    if label:
        return label

    return "Cumulative Profit"


def _get_show_grid() -> bool:
    """Get show grid setting from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "show_grid"):
                show = value.chart_generator.show_grid
                if isinstance(show, bool):
                    return show
        except (AttributeError, ValueError, TypeError):
            pass

    show = os.getenv("CHART_GENERATOR_SHOW_GRID", "").strip().lower()
    return show not in ("false", "0", "no")


def _get_show_legend() -> bool:
    """Get show legend setting from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "chart_generator") and hasattr(value.chart_generator, "show_legend"):
                show = value.chart_generator.show_legend
                if isinstance(show, bool):
                    return show
        except (AttributeError, ValueError, TypeError):
            pass

    show = os.getenv("CHART_GENERATOR_SHOW_LEGEND", "").strip().lower()
    return show not in ("false", "0", "no")


def plot_trades(symbol: str | None = None, db_path: str | None = None) -> None:
    """Plot trades chart with live configuration."""
    chart_symbol = symbol if symbol is not None else _get_default_symbol()
    chart_db_path = db_path if db_path is not None else _get_default_db_path()

    conn = sqlite3.connect(chart_db_path)
    try:
        cursor = conn.execute(
            """
            SELECT timestamp, price, simulated_profit FROM simulated_trades
            WHERE symbol = ? ORDER BY timestamp
        """,
            (chart_symbol,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("[Chart] No trades to plot.")
        return

    times = [datetime.fromisoformat(row[0]) for row in rows]
    prices = [row[1] for row in rows]
    profits = [row[2] for row in rows]

    positive_color = _get_profit_color_positive()
    negative_color = _get_profit_color_negative()
    colors = [positive_color if p > 0 else negative_color for p in profits]

    figure_width = _get_figure_width()
    figure_height = _get_figure_height()
    scatter_alpha = _get_scatter_alpha()
    line_alpha = _get_line_alpha()

    plt.figure(figsize=(figure_width, figure_height))
    plt.scatter(times, prices, c=colors, label=chart_symbol, alpha=scatter_alpha)
    plt.plot(times, prices, alpha=line_alpha)

    chart_title_template = _get_chart_title_trade()
    chart_title = chart_title_template.format(symbol=chart_symbol)
    plt.title(chart_title)

    xlabel = _get_xlabel_default()
    ylabel_price = _get_ylabel_price()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel_price)

    show_grid = _get_show_grid()
    plt.grid(show_grid)

    plt.tight_layout()

    output_path = _get_trade_chart_output_path()
    plt.savefig(output_path)
    logger.info(f"[Chart] Chart saved to {output_path}")


def plot_performance_over_time(db_path: str | None = None) -> None:
    """Plot performance over time chart with live configuration."""
    chart_db_path = db_path if db_path is not None else _get_default_db_path()

    conn = sqlite3.connect(chart_db_path)
    try:
        cursor = conn.execute(
            """
            SELECT timestamp, simulated_profit FROM simulated_trades
            ORDER BY timestamp
        """,
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("[Chart] No performance data to plot.")
        return

    times = [datetime.fromisoformat(row[0]) for row in rows]
    profits = [row[1] for row in rows]
    cumulative = []
    total = 0
    for profit in profits:
        total += profit
        cumulative.append(total)

    figure_width = _get_figure_width()
    figure_height = _get_figure_height()

    plt.figure(figsize=(figure_width, figure_height))
    legend_label = _get_legend_label_profit()
    plt.plot(times, cumulative, label=legend_label)

    chart_title = _get_chart_title_performance()
    plt.title(chart_title)

    xlabel = _get_xlabel_default()
    ylabel_profit = _get_ylabel_profit()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel_profit)

    show_grid = _get_show_grid()
    plt.grid(show_grid)

    show_legend = _get_show_legend()
    if show_legend:
        plt.legend()

    plt.tight_layout()

    output_path = _get_performance_chart_output_path()
    plt.savefig(output_path)
    logger.info(f"[Chart] Performance chart saved to {output_path}")

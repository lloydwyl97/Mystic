import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

mpl.use("Agg")

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e


def _to_ccxt_symbol(base: str, quote: str) -> str:
    return f"{base}/{quote}"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPORTS_DIR = "./exports"
DEFAULT_OUTPUT = str(Path(EXPORTS_DIR) / "strategy_mutation_graph.png")


def plot_strategy_graph(mutation_history: Iterable[tuple[str, str, float]], out_path: str | None = None) -> str:
    """
    Plot a directed strategy mutation graph and save to file.

    mutation_history: iterable of (parent_id, child_id, profit_weight)
    out_path: destination image path; defaults to ./exports/strategy_mutation_graph.png
    """
    out_path = out_path or DEFAULT_OUTPUT
    dirpath = Path(out_path).parent
    dirpath.mkdir(parents=True, exist_ok=True)

    G = nx.DiGraph()
    added = False
    for parent, child, profit in mutation_history:
        try:
            p_val = float(profit)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            p_val = 0.0
        G.add_edge(str(parent), str(child), weight=p_val)
        added = True

    if not added:
        logger.warning("No edges provided; skipping graph render")
        return out_path

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, "weight")

    plt.figure(figsize=(10, 7), dpi=120)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="#cfe8ff",
        node_size=900,
        font_size=9,
        font_weight="bold",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Strategy Mutation Graph")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Strategy mutation graph saved: %s", out_path)
    return out_path


def load_history_from_json(path: str) -> list[tuple[str, str, float]]:
    """
    Load mutation history from a JSON file.
    Expected format: list of [parent, child, profit] or list of objects with keys parent, child, profit.
    """
    path_obj = Path(path)
    with path_obj.open(encoding="utf-8") as f:
        data = json.load(f)
    history: list[tuple[str, str, float]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    profit = float(item[2])
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    profit = 0.0
                history.append((str(item[0]), str(item[1]), profit))
            elif isinstance(item, dict):
                parent = item.get("parent", "")
                child = item.get("child", "")
                try:
                    profit = float(item.get("profit", 0.0))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    profit = 0.0
                history.append((str(parent), str(child), profit))
    return history


def visualize_strategy_evolution(mutation_history: Iterable[tuple[str, str, float]], out_path: str | None = None) -> str:
    """
    Build and save the strategy mutation graph from a provided history iterable.
    """
    return plot_strategy_graph(mutation_history, out_path)


if __name__ == "__main__":
    import sys

    history_path = os.environ.get("MUTATION_HISTORY_JSON")
    output_path = os.environ.get("MUTATION_GRAPH_PATH")
    if len(sys.argv) >= 2:
        history_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    if not history_path or not Path(history_path).exists():
        logger.error("No valid mutation history JSON provided. Set MUTATION_HISTORY_JSON or pass a file path argument.")
        sys.exit(2)

    try:
        history = load_history_from_json(history_path)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Failed to load history: %s", e)
        sys.exit(3)

    try:
        out = visualize_strategy_evolution(history, output_path)
        logger.info("Graph ready at: %s", out)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Visualization failed: %s", e)
        sys.exit(4)

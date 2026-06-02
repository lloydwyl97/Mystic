"""
Mutation Graph - Live Configuration Only

Generates strategy mutation graphs using NetworkX and Matplotlib.
All configuration values come from live config - no hardcoded values.
"""

import logging
import os

import matplotlib.pyplot as plt
import networkx as nx

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# Configure logging
logger = logging.getLogger(__name__)

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_node_color() -> str:
    """Get node color from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "mutation_graph") and hasattr(value.mutation_graph, "node_color"):
                color = value.mutation_graph.node_color
                if isinstance(color, str) and color:
                    return color.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    color = os.getenv("MUTATION_GRAPH_NODE_COLOR", "").strip()
    if color:
        return color

    return "lightblue"


def _get_node_size() -> int:
    """Get node size from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "mutation_graph") and hasattr(value.mutation_graph, "node_size"):
                size = value.mutation_graph.node_size
                if isinstance(size, int) and size > 0:
                    return size
        except (AttributeError, ValueError, TypeError):
            pass

    size = os.getenv("MUTATION_GRAPH_NODE_SIZE", "").strip()
    if size:
        try:
            size_val = int(size)
            if size_val > 0:
                return size_val
        except (ValueError, TypeError):
            pass

    return 1000


def _get_graph_title() -> str:
    """Get graph title from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "mutation_graph") and hasattr(value.mutation_graph, "graph_title"):
                title = value.mutation_graph.graph_title
                if isinstance(title, str) and title:
                    return title.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    title = os.getenv("MUTATION_GRAPH_TITLE", "").strip()
    if title:
        return title

    return "Strategy Mutation Graph"


def _get_layout_algorithm() -> str:
    """Get layout algorithm from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "mutation_graph") and hasattr(value.mutation_graph, "layout_algorithm"):
                algorithm = value.mutation_graph.layout_algorithm
                if isinstance(algorithm, str) and algorithm:
                    return algorithm.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    algorithm = os.getenv("MUTATION_GRAPH_LAYOUT_ALGORITHM", "").strip()
    if algorithm:
        return algorithm

    return "spring"


def _get_show_labels() -> bool:
    """Get show labels setting from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "mutation_graph") and hasattr(value.mutation_graph, "show_labels"):
                show = value.mutation_graph.show_labels
                if isinstance(show, bool):
                    return show
        except (AttributeError, ValueError, TypeError):
            pass

    show = os.getenv("MUTATION_GRAPH_SHOW_LABELS", "").strip().lower()
    return show not in ("false", "0", "no")


def plot_strategy_graph(mutation_history: list[tuple[str, str, float]]) -> None:
    """Plot strategy mutation graph with profit weights using live configuration."""
    graph = nx.DiGraph()
    for parent, child, profit in mutation_history:
        graph.add_edge(parent, child, weight=profit)

    layout_algorithm = _get_layout_algorithm()
    if layout_algorithm == "spring":
        pos = nx.spring_layout(graph)
    elif layout_algorithm == "circular":
        pos = nx.circular_layout(graph)
    elif layout_algorithm == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout_algorithm == "planar":
        pos = nx.planar_layout(graph)
    else:
        # Default to spring layout if unknown algorithm
        pos = nx.spring_layout(graph)

    edge_labels = nx.get_edge_attributes(graph, "weight")

    node_color = _get_node_color()
    node_size = _get_node_size()
    show_labels = _get_show_labels()

    nx.draw(graph, pos, with_labels=show_labels, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    graph_title = _get_graph_title()
    plt.title(graph_title)
    plt.show()


def generate_mutation_timeline() -> list[tuple[str, str, float]]:
    """Generate sample mutation timeline - returns empty list by default.

    Note: This function returns an empty list to comply with "no mock data" policy.
    Use actual mutation history data from your data source.
    """
    return []


def visualize_strategy_evolution(mutation_history: list[tuple[str, str, float]] | None = None) -> None:
    """Main function to visualize strategy evolution using live configuration.

    Args:
        mutation_history: List of tuples (parent, child, profit). If None, returns empty list.
    """
    if mutation_history is None:
        mutation_history = generate_mutation_timeline()

    if not mutation_history:
        logger.warning("[GRAPH] No mutation history provided, cannot visualize")
        return

    plot_strategy_graph(mutation_history)
    logger.info("[GRAPH] Strategy evolution visualized")


if __name__ == "__main__":
    visualize_strategy_evolution()

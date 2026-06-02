import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from db_logger import get_session

from models import Strategy, StrategyPerformance, Trade

logger = logging.getLogger(__name__)


def evaluate_strategies(min_trades: int = 5, days: int = 7) -> dict[str, Any]:
    session = get_session()
    try:
        strategies = session.query(Strategy).filter_by(is_active=True).all()
        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=days)

        evaluation_results = {
            "total_strategies": len(strategies),
            "evaluated_strategies": 0,
            "updated_strategies": 0,
            "strategy_details": [],
        }

        for strat in strategies:
            trades = session.query(Trade).filter(Trade.strategy_id == strat.id).filter(Trade.timestamp >= cutoff_date).filter(Trade.exit_price.isnot(None)).all()

            if len(trades) >= min_trades:
                evaluation_results["evaluated_strategies"] += 1

                total_profit = sum(t.profit for t in trades if t.profit is not None)
                win_count = sum(1 for t in trades if getattr(t, "success", False))
                avg_profit = total_profit / len(trades) if trades else 0.0
                win_rate = win_count / len(trades) if trades else 0.0

                profits = [t.profit for t in trades if t.profit is not None]
                max_profit = max(profits) if profits else 0.0
                max_loss = min(profits) if profits else 0.0

                strat.win_rate = round(win_rate, 4)
                strat.avg_profit = round(avg_profit, 4)
                strat.trades_executed = len(trades)
                strat.total_profit = round(total_profit, 4)
                strat.updated_at = now

                performance = StrategyPerformance(
                    strategy_id=strat.id,
                    strategy_name=strat.name,
                    date=now,
                    win_rate=round(win_rate, 4),
                    avg_profit=round(avg_profit, 4),
                    total_trades=len(trades),
                    total_profit=round(total_profit, 4),
                    max_drawdown=abs(max_loss) if max_loss < 0 else 0.0,
                    sharpe_ratio=(calculate_sharpe_ratio(profits) if profits else 0.0),
                )
                session.add(performance)

                evaluation_results["updated_strategies"] += 1

                evaluation_results["strategy_details"].append(
                    {
                        "id": strat.id,
                        "name": strat.name,
                        "win_rate": round(win_rate, 4),
                        "avg_profit": round(avg_profit, 4),
                        "total_profit": round(total_profit, 4),
                        "trades_count": len(trades),
                        "max_profit": max_profit,
                        "max_loss": max_loss,
                    }
                )

                logger.info(f"Updated: {strat.name} | Win: {win_rate:.2%} | Profit: {avg_profit:.2f} | Total: {total_profit:.2f}")
            else:
                logger.debug(f"Strategy {strat.name} has only {len(trades)} trades, skipping evaluation")

        session.commit()
        logger.info(f"Strategy evaluation completed: {evaluation_results['updated_strategies']}/{evaluation_results['total_strategies']} strategies updated")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        session.rollback()
        logger.exception(f"Strategy evaluation failed: {e}")
        return {"error": str(e)}
    else:
        return evaluation_results
    finally:
        session.close()


def calculate_sharpe_ratio(profits: list[float], risk_free_rate: float = 0.02) -> float:
    if not profits or len(profits) < 2:
        return 0.0

    returns = np.array(profits)
    avg_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # risk_free_rate is annual; convert to per-period (assuming daily) for comparison with profits
    sharpe = (avg_return - risk_free_rate / 365) / std_return * np.sqrt(365)
    return round(float(sharpe), 4)


def get_top_performers(top_n: int = 5, min_trades: int = 5) -> list[dict[str, Any]]:
    session = get_session()
    try:
        strategies = session.query(Strategy).filter(Strategy.trades_executed >= min_trades).filter(Strategy.is_active).order_by(Strategy.win_rate.desc(), Strategy.avg_profit.desc()).limit(top_n).all()

        return [
            {
                "id": strat.id,
                "name": strat.name,
                "win_rate": strat.win_rate,
                "avg_profit": strat.avg_profit,
                "total_profit": strat.total_profit,
                "trades_executed": strat.trades_executed,
            }
            for strat in strategies
        ]
    finally:
        session.close()


def get_poor_performers(min_trades: int = 5, max_win_rate: float = 0.4) -> list[dict[str, Any]]:
    session = get_session()
    try:
        strategies = session.query(Strategy).filter(Strategy.trades_executed >= min_trades).filter(Strategy.win_rate <= max_win_rate).filter(Strategy.is_active).order_by(Strategy.win_rate.asc()).all()

        return [
            {
                "id": strat.id,
                "name": strat.name,
                "win_rate": strat.win_rate,
                "avg_profit": strat.avg_profit,
                "total_profit": strat.total_profit,
                "trades_executed": strat.trades_executed,
            }
            for strat in strategies
        ]
    finally:
        session.close()


def deactivate_strategy(strategy_id: int) -> bool:
    session = get_session()
    try:
        strategy = session.query(Strategy).filter_by(id=strategy_id).first()
        if strategy:
            strategy.is_active = False
            strategy.updated_at = datetime.now(timezone.utc)
            session.commit()
            logger.info(f"Deactivated strategy: {strategy.name}")
            result = True
        else:
            result = False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        session.rollback()
        logger.exception(f"Failed to deactivate strategy {strategy_id}: {e}")
        return False
    else:
        return result
    finally:
        session.close()


def get_strategy_performance_history(strategy_id: int, days: int = 30) -> list[dict[str, Any]]:
    session = get_session()
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        performances = (
            session.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == strategy_id).filter(StrategyPerformance.date >= cutoff_date).order_by(StrategyPerformance.date.desc()).all()
        )

        return [
            {
                "date": perf.date.isoformat() if perf.date else None,
                "win_rate": perf.win_rate,
                "avg_profit": perf.avg_profit,
                "total_trades": perf.total_trades,
                "total_profit": perf.total_profit,
                "max_drawdown": perf.max_drawdown,
                "sharpe_ratio": perf.sharpe_ratio,
            }
            for perf in performances
        ]
    finally:
        session.close()


def run_daily_evaluation() -> dict[str, Any]:
    logger.info("Starting daily strategy evaluation...")
    results = evaluate_strategies(min_trades=3, days=1)
    top_performers = get_top_performers(top_n=3, min_trades=5)
    poor_performers = get_poor_performers(min_trades=5, max_win_rate=0.4)
    results["top_performers"] = top_performers
    results["poor_performers"] = poor_performers
    logger.info(f"Daily evaluation completed. Top performers: {len(top_performers)}, Poor performers: {len(poor_performers)}")
    return results

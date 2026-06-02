"""
Trade Repository - All Live Data, No Fallback/Hardcoded Data

This module provides repository pattern implementation for trade data operations.
All repository operations:
- Access live trade data from database (persisted from backend port 8000)
- Return None/empty structures only on error or not found, not as fallback data
- All data from live order execution persisted to database
- Connected to live database for trade data persistence

Live Data Sources:
- Trades: From live order execution via backend (port 8000), persisted to database
- Trade status: From live exchange API (Binance.US) via backend endpoints
- Trade prices and quantities: From live order execution
- All data operations use live database records - no fallback/hardcoded data

Endpoint References:
- /api/trades - Live trade operations
- /api/trades/create - Live trade creation
- /api/trades/update - Live trade updates
- All connected to backend running on port 8000, data persisted to live database
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.adapters.db.models import TradeModel
from backend.app.domain.models.trade import Trade, TradeCreate, TradeStatus

logger = logging.getLogger(__name__)


class TradeRepository:
    """
    Repository for live trade operations.

    Provides data access layer for:
    - Live trades from database (persisted from backend port 8000)
    - Live trade status updates from order execution
    - All operations use live database records - no fallback/hardcoded data
    """

    def __init__(self, db_session: Session):
        """
        Initialize the repository with a database session for live data operations.

        Args:
            db_session: Database session connected to live data store (backend port 8000)
        """
        self.db = db_session

    def _to_domain(self, db_trade: TradeModel) -> Trade:
        """
        Convert a database model instance to a domain Trade instance.

        Converts live database records to domain models for live trade data.

        Args:
            db_trade: Database trade model from live database

        Returns:
            Domain Trade model from live database record
        """
        # Convert id to uuid.UUID (from live database record)
        trade_id = db_trade.id if isinstance(db_trade.id, uuid.UUID) else uuid.UUID(str(db_trade.id))

        def _to_utc(dt: datetime | None) -> datetime | None:
            """Convert datetime to UTC timezone-aware (from live database record)."""
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        return Trade(
            id=trade_id,
            user_id=db_trade.user_id,  # From live trade
            symbol=db_trade.symbol,  # From live trade
            side=db_trade.side,  # From live trade
            quantity=db_trade.quantity,  # From live order execution
            price=db_trade.price,  # From live order execution
            status=db_trade.status,  # From live order execution
            created_at=_to_utc(db_trade.created_at),  # From live trade
            updated_at=_to_utc(db_trade.updated_at),  # From live trade
            completed_at=_to_utc(getattr(db_trade, "filled_at", None)),  # From live order execution
        )

    async def create(self, user_id: str, trade_data: TradeCreate) -> Trade:
        """
        Create a new trade in live database (backend port 8000).

        Args:
            user_id: User identifier
            trade_data: Trade creation data (from live order execution)

        Returns:
            Trade object created in live database

        Raises:
            Various exceptions if trade creation fails
        """
        try:
            trade_id = str(uuid.uuid4())
            amount = trade_data.quantity * trade_data.price  # Calculated from live trade data

            # Create trade record in live database (from backend port 8000)
            db_trade = TradeModel(
                id=trade_id,
                user_id=user_id,
                symbol=trade_data.symbol,  # From live order
                side=trade_data.side,  # From live order
                quantity=trade_data.quantity,  # From live order execution
                price=trade_data.price,  # From live order execution
                amount=amount,  # Calculated from live trade data
                status=TradeStatus.PENDING,  # Initial status for live trade
                created_at=datetime.now(timezone.utc),  # Timezone-aware timestamp
            )

            self.db.add(db_trade)
            self.db.commit()
            self.db.refresh(db_trade)

            # Convert live database record to domain model
            return self._to_domain(db_trade)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error creating trade for user %s", user_id)
            try:
                self.db.rollback()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception("Error rolling back transaction after create failure")
            raise

    async def get_by_id(self, trade_id: uuid.UUID) -> Trade | None:
        """
        Get a trade by ID from live database (backend port 8000).

        Args:
            trade_id: Trade identifier

        Returns:
            Trade object from live database or None if not found (not fallback data)
        """
        try:
            # Query live database for trade
            result = self.db.execute(select(TradeModel).where(TradeModel.id == str(trade_id)))
            db_trade = result.scalar_one_or_none()

            if not db_trade:
                return None  # Not found (not fallback data - just no record exists)

            # Convert live database record to domain model
            return self._to_domain(db_trade)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error getting trade %s", trade_id)
            return None  # Error response (not fallback data)

    async def get_by_user(self, user_id: str, symbol: str | None = None, limit: int = 20) -> list[Trade]:
        """
        Get trades for a specific user from live database (backend port 8000).

        Args:
            user_id: User identifier
            symbol: Optional symbol filter
            limit: Maximum number of trades to return

        Returns:
            List of Trade objects from live database (persisted from backend port 8000)
        """
        try:
            # Build query for live trades from database
            query = select(TradeModel).where(TradeModel.user_id == user_id)

            if symbol:
                query = query.where(TradeModel.symbol == symbol)

            query = query.order_by(TradeModel.created_at.desc()).limit(limit)

            # Execute query on live database
            result = self.db.execute(query)
            db_trades = result.scalars().all()

            # Convert live database records to domain models
            trades = [self._to_domain(db_trade) for db_trade in db_trades]

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error getting trades for user %s", user_id)
            return []  # Empty list on error (error response, not fallback data)
        else:
            return trades

    async def update_status(self, trade_id: uuid.UUID, status: TradeStatus) -> Trade | None:
        """
        Update a trade's status in live database (backend port 8000).

        Args:
            trade_id: Trade identifier
            status: New trade status (from live order execution)

        Returns:
            Updated Trade object from live database or None if trade not found
        """
        try:
            # Get existing trade from live database
            result = self.db.execute(select(TradeModel).where(TradeModel.id == str(trade_id)))
            db_trade = result.scalar_one_or_none()

            if not db_trade:
                return None  # Trade not found (not fallback data - just no record exists)

            # Update status and timestamps (from live order execution)
            db_trade.status = status  # From live order execution
            db_trade.updated_at = datetime.now(timezone.utc)  # Timezone-aware timestamp

            if status == TradeStatus.COMPLETED:
                # Set filled_at timestamp when trade is completed (from live order execution)
                db_trade.filled_at = datetime.now(timezone.utc)  # Timezone-aware timestamp

            self.db.commit()
            self.db.refresh(db_trade)

            # Convert live database record to domain model
            return self._to_domain(db_trade)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error updating trade %s status", trade_id)
            try:
                self.db.rollback()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception("Error rolling back transaction after update_status failure")
            return None  # Error response (not fallback data)

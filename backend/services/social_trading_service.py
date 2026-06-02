"""Social Trading Service
Handles user profiles, strategies, followers, and social features
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database_schema import DATABASE_PATH
from backend.models.social_trading_models import (
    Base,
    CopyTrade,
    SocialPost,
    TradingStrategy,
    User,
    strategy_followers,
    strategy_likes,
    user_followers,
)

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SocialTradingService:
    """Service for managing social trading features."""

    def __init__(self, database_pool_service, cache_service=None):
        self.db_pool = database_pool_service
        self.cache = cache_service
        self._initialized = False
        self._engine: AsyncEngine | None = None
        self._sessionmaker: sessionmaker[AsyncSession] | None = None
        self._db_path = Path(DATABASE_PATH).resolve()

    async def initialize(self) -> None:
        """Initialize the social trading service."""
        if self._initialized and self._sessionmaker is not None:
            return

        try:
            db_url = f"sqlite+aiosqlite:///{self._db_path.as_posix()}"
            self._engine = create_async_engine(db_url, echo=False, future=True)
            self._sessionmaker = sessionmaker(self._engine, expire_on_commit=False, class_=AsyncSession)
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._initialized = True
            logger.info("Social trading service initialized")
        except Exception as exc:
            self._initialized = False
            logger.exception("Failed to initialize social trading service: %s", exc)
            raise

    def _require_session_factory(self) -> sessionmaker[AsyncSession]:
        if self._sessionmaker is None:
            msg = "Social trading service not initialized"
            raise RuntimeError(msg)
        return self._sessionmaker

    async def _cache_invalidate(self, *keys: str) -> None:
        if not self.cache:
            return
        deleter = getattr(self.cache, "delete", None)
        if deleter is None:
            return
        for key in keys:
            try:
                result = deleter(key)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug("Cache delete failed for key %s", key, exc_info=True)

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    async def create_user(
        self,
        username: str,
        email: str,
        display_name: str | None = None,
        bio: str = "",
        avatar_url: str | None = None,
    ) -> dict[str, Any] | None:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                existing = await session.execute(select(User.id).where(or_(User.username == username, User.email == email)))
                if existing.scalar_one_or_none():
                    logger.warning("User %s or email %s already exists", username, email)
                    return None

                user = User(
                    username=username,
                    email=email,
                    display_name=display_name or username,
                    bio=bio,
                    avatar_url=avatar_url,
                )
                session.add(user)
                await session.flush()
                await session.refresh(user)
                await session.commit()
                logger.info("[OK] Created user: %s", username)
                return self._user_to_dict(user)
            except Exception as exc:
                await session.rollback()
                logger.exception("[ERROR] Failed to create user %s: %s", username, exc)
                return None

    async def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                user = await session.get(User, user_id)
                if not user:
                    return None
                return self._user_to_dict(user)
            except Exception as exc:
                logger.exception("[ERROR] Failed to get user %s: %s", user_id, exc)
                return None

    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                result = await session.execute(select(User).where(User.username == username).limit(1))
                user = result.scalar_one_or_none()
                if not user:
                    return None
                return self._user_to_dict(user)
            except Exception as exc:
                logger.exception("[ERROR] Failed to get user %s: %s", username, exc)
                return None

    async def update_user_stats(self, user_id: int, stats: dict[str, Any]) -> bool:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                user = await session.get(User, user_id)
                if not user:
                    return False

                for key, value in stats.items():
                    if value is not None and hasattr(user, key):
                        setattr(user, key, value)

                user.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await self._cache_invalidate(f"user:{user_id}", f"user:{user.username}")
                logger.info("[OK] Updated stats for user %s", user_id)
            except Exception as exc:
                await session.rollback()
                logger.exception("[ERROR] Failed to update user stats %s: %s", user_id, exc)
                return False
            else:
                return True

    # ------------------------------------------------------------------
    # Strategy management
    # ------------------------------------------------------------------

    async def create_strategy(
        self,
        author_id: int,
        name: str,
        description: str,
        strategy_type: str,
        symbol: str,
        timeframe: str,
        config: dict[str, Any],
        indicators: list[str] | None = None,
        entry_conditions: dict[str, Any] | None = None,
        exit_conditions: dict[str, Any] | None = None,
        is_public: bool = True,
    ) -> dict[str, Any] | None:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                author = await session.get(User, author_id)
                if not author:
                    logger.error("Author %s not found", author_id)
                    return None

                strategy = TradingStrategy(
                    author_id=author_id,
                    name=name,
                    description=description,
                    strategy_type=strategy_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    config=str(config),
                    indicators=str(indicators or []),
                    entry_conditions=str(entry_conditions or {}),
                    exit_conditions=str(exit_conditions or {}),
                    is_public=is_public,
                )
                session.add(strategy)
                author.strategies_count += 1
                await session.flush()
                await session.refresh(strategy)
                await session.commit()
                logger.info("[OK] Created strategy: %s by user %s", name, author_id)
                return self._strategy_to_dict(strategy)
            except Exception as exc:
                await session.rollback()
                logger.exception("[ERROR] Failed to create strategy %s: %s", name, exc)
                return None

    async def get_strategy_by_id(self, strategy_id: int) -> dict[str, Any] | None:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                strategy = await session.get(TradingStrategy, strategy_id)
                if not strategy:
                    return None
                return self._strategy_to_dict(strategy)
            except Exception as exc:
                logger.exception("[ERROR] Failed to get strategy %s: %s", strategy_id, exc)
                return None

    async def get_public_strategies(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "created_at",
    ) -> list[dict[str, Any]]:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                order_mapping = {
                    "created_at": TradingStrategy.created_at,
                    "likes_count": TradingStrategy.likes_count,
                    "win_rate": TradingStrategy.win_rate,
                    "total_pnl_percentage": TradingStrategy.total_pnl_percentage,
                }
                order_column = order_mapping.get(sort_by, TradingStrategy.created_at)
                stmt = (
                    select(TradingStrategy, User).join(User, TradingStrategy.author_id == User.id).where(TradingStrategy.is_public.is_(True)).order_by(order_column.desc()).limit(limit).offset(offset)
                )
                result = await session.execute(stmt)
                rows = result.all()
                strategies: list[dict[str, Any]] = []
                for strategy, author in rows:
                    strategy_dict = self._strategy_to_dict(strategy)
                    strategy_dict["author"] = {
                        "id": author.id,
                        "username": author.username,
                        "display_name": author.display_name,
                        "avatar_url": author.avatar_url,
                    }
                    strategies.append(strategy_dict)
            except Exception as exc:
                logger.exception("[ERROR] Failed to get public strategies: %s", exc)
                return []
            else:
                return strategies

    async def update_strategy_performance(self, strategy_id: int, performance: dict[str, Any]) -> bool:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                strategy = await session.get(TradingStrategy, strategy_id)
                if not strategy:
                    return False

                for key, value in performance.items():
                    if value is not None and hasattr(strategy, key):
                        setattr(strategy, key, value)

                strategy.updated_at = datetime.now(timezone.utc)
                await session.commit()
                logger.info("[OK] Updated performance for strategy %s", strategy_id)
            except Exception as exc:
                await session.rollback()
                logger.exception(
                    "[ERROR] Failed to update strategy performance %s: %s",
                    strategy_id,
                    exc,
                )
                return False
            else:
                return True

    # ------------------------------------------------------------------
    # Social interactions
    # ------------------------------------------------------------------

    async def follow_user(self, follower_id: int, following_id: int) -> bool:
        if follower_id == following_id:
            return False

        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                existing = await session.execute(
                    select(user_followers.c.follower_id).where(
                        user_followers.c.follower_id == follower_id,
                        user_followers.c.following_id == following_id,
                    )
                )
                if existing.scalar_one_or_none():
                    logger.info("User %s already follows %s", follower_id, following_id)
                    return True

                await session.execute(user_followers.insert().values(follower_id=follower_id, following_id=following_id))
                follower = await session.get(User, follower_id)
                following = await session.get(User, following_id)
                if follower:
                    follower.following_count += 1
                if following:
                    following.followers_count += 1
                await session.commit()
                await self._cache_invalidate(f"user:{follower_id}", f"user:{following_id}")
                logger.info("[OK] User %s now follows %s", follower_id, following_id)
            except Exception as exc:
                await session.rollback()
                logger.exception(
                    "[ERROR] Failed to follow user %s -> %s: %s",
                    follower_id,
                    following_id,
                    exc,
                )
                return False
            else:
                return True

    async def unfollow_user(self, follower_id: int, following_id: int) -> bool:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                result = await session.execute(
                    delete(user_followers).where(
                        user_followers.c.follower_id == follower_id,
                        user_followers.c.following_id == following_id,
                    )
                )
                if result.rowcount == 0:
                    logger.warning("User %s was not following %s", follower_id, following_id)
                    await session.commit()
                    return True

                follower = await session.get(User, follower_id)
                following = await session.get(User, following_id)
                if follower:
                    follower.following_count = max(follower.following_count - 1, 0)
                if following:
                    following.followers_count = max(following.followers_count - 1, 0)
                await session.commit()
                await self._cache_invalidate(f"user:{follower_id}", f"user:{following_id}")
                logger.info("[OK] User %s unfollowed %s", follower_id, following_id)
            except Exception as exc:
                await session.rollback()
                logger.exception(
                    "[ERROR] Failed to unfollow user %s -> %s: %s",
                    follower_id,
                    following_id,
                    exc,
                )
                return False
            else:
                return True

    async def like_strategy(self, user_id: int, strategy_id: int) -> bool:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                existing = await session.execute(
                    select(strategy_likes.c.user_id).where(
                        strategy_likes.c.user_id == user_id,
                        strategy_likes.c.strategy_id == strategy_id,
                    )
                )
                if existing.scalar_one_or_none():
                    logger.info("User %s already liked strategy %s", user_id, strategy_id)
                    return True

                await session.execute(strategy_likes.insert().values(user_id=user_id, strategy_id=strategy_id))
                strategy = await session.get(TradingStrategy, strategy_id)
                if strategy:
                    strategy.likes_count += 1
                await session.commit()
                logger.info("[OK] User %s liked strategy %s", user_id, strategy_id)
            except Exception as exc:
                await session.rollback()
                logger.exception(
                    "[ERROR] Failed to like strategy %s by user %s: %s",
                    strategy_id,
                    user_id,
                    exc,
                )
                return False
            else:
                return True

    async def follow_strategy(self, user_id: int, strategy_id: int) -> bool:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                existing = await session.execute(
                    select(strategy_followers.c.user_id).where(
                        strategy_followers.c.user_id == user_id,
                        strategy_followers.c.strategy_id == strategy_id,
                    )
                )
                if existing.scalar_one_or_none():
                    logger.info("User %s already follows strategy %s", user_id, strategy_id)
                    return True

                await session.execute(strategy_followers.insert().values(user_id=user_id, strategy_id=strategy_id))
                strategy = await session.get(TradingStrategy, strategy_id)
                if strategy:
                    strategy.followers_count += 1
                await session.commit()
                logger.info("[OK] User %s followed strategy %s", user_id, strategy_id)
            except Exception as exc:
                await session.rollback()
                logger.exception(
                    "[ERROR] Failed to follow strategy %s by user %s: %s",
                    strategy_id,
                    user_id,
                    exc,
                )
                return False
            else:
                return True

    # ------------------------------------------------------------------
    # Leaderboards and traders
    # ------------------------------------------------------------------

    async def get_leaderboard(self, _period: str = "monthly", category: str = "pnl", limit: int = 100) -> list[dict[str, Any]]:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                category_map = {
                    "pnl": User.total_pnl_percentage,
                    "win_rate": User.win_rate,
                    "sharpe_ratio": User.sharpe_ratio,
                    "followers": User.followers_count,
                }
                column = category_map.get(category, User.total_pnl_percentage)
                stmt = select(User).order_by(column.desc(), User.id.asc()).limit(max(limit, 1))
                result = await session.execute(stmt)
                users = result.scalars().all()
                column_name = column.key if hasattr(column, "key") else category
                leaderboard: list[dict[str, Any]] = []
                for position, user in enumerate(users, start=1):
                    score_value = getattr(user, column_name, 0)
                    leaderboard.append(
                        {
                            "rank": position,
                            "score": score_value,
                            "followers": user.followers_count,
                            "user": {
                                "id": user.id,
                                "username": user.username,
                                "display_name": user.display_name,
                                "avatar_url": user.avatar_url,
                            },
                        }
                    )
            except Exception as exc:
                logger.exception("[ERROR] Failed to get leaderboard: %s", exc)
                return []
            else:
                return leaderboard

    async def get_traders(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self.get_leaderboard(_period="all_time", category="pnl", limit=limit)

    # ------------------------------------------------------------------
    # Copy trading
    # ------------------------------------------------------------------

    async def start_copy_trading(
        self,
        follower_id: int,
        leader_id: int,
        strategy_id: int,
        allocation_percentage: float = 100.0,
        max_position_size: float | None = None,
    ) -> dict[str, Any] | None:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                existing = await session.execute(
                    select(CopyTrade.id).where(
                        CopyTrade.follower_id == follower_id,
                        CopyTrade.leader_id == leader_id,
                        CopyTrade.strategy_id == strategy_id,
                        CopyTrade.is_active.is_(True),
                    )
                )
                if existing.scalar_one_or_none():
                    logger.warning(
                        "Copy trading relationship already exists follower=%s leader=%s strategy=%s",
                        follower_id,
                        leader_id,
                        strategy_id,
                    )
                    return None

                copy_trade = CopyTrade(
                    follower_id=follower_id,
                    leader_id=leader_id,
                    strategy_id=strategy_id,
                    allocation_percentage=allocation_percentage,
                    max_position_size=max_position_size,
                )
                session.add(copy_trade)
                await session.flush()
                await session.refresh(copy_trade)
                await session.commit()
                logger.info(
                    "[OK] Started copy trading: follower=%s leader=%s strategy=%s",
                    follower_id,
                    leader_id,
                    strategy_id,
                )
                return self._copy_trade_to_dict(copy_trade)
            except Exception as exc:
                await session.rollback()
                logger.exception("[ERROR] Failed to start copy trading: %s", exc)
                return None

    async def stop_copy_trading(self, copy_trade_id: int) -> bool:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                copy_trade = await session.get(CopyTrade, copy_trade_id)
                if not copy_trade:
                    return False
                copy_trade.is_active = False
                copy_trade.updated_at = datetime.now(timezone.utc)
                await session.commit()
                logger.info("[OK] Stopped copy trading %s", copy_trade_id)
            except Exception as exc:
                await session.rollback()
                logger.exception("[ERROR] Failed to stop copy trading %s: %s", copy_trade_id, exc)
                return False
            else:
                return True

    async def get_copy_trades(self, active_only: bool = True) -> list[dict[str, Any]]:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                stmt = select(CopyTrade)
                if active_only:
                    stmt = stmt.where(CopyTrade.is_active.is_(True))
                result = await session.execute(stmt)
                trades = result.scalars().all()
                return [self._copy_trade_to_dict(trade) for trade in trades]
            except Exception as exc:
                logger.exception("[ERROR] Failed to fetch copy trades: %s", exc)
                return []

    # ------------------------------------------------------------------
    # Social feed / posts
    # ------------------------------------------------------------------

    async def create_post(
        self,
        author_id: int,
        content: str,
        post_type: str = "insight",
        symbol: str | None = None,
        strategy_id: int | None = None,
    ) -> dict[str, Any] | None:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                post = SocialPost(
                    author_id=author_id,
                    content=content,
                    post_type=post_type,
                    symbol=symbol,
                    strategy_id=strategy_id,
                )
                session.add(post)
                await session.flush()
                await session.refresh(post)
                await session.commit()
                logger.info("[OK] Created post by user %s", author_id)
                return self._post_to_dict(post)
            except Exception as exc:
                await session.rollback()
                logger.exception("[ERROR] Failed to create post: %s", exc)
                return None

    async def get_feed(self, user_id: int | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                stmt = select(SocialPost, User, TradingStrategy.name).join(User, SocialPost.author_id == User.id).outerjoin(TradingStrategy, SocialPost.strategy_id == TradingStrategy.id)
                if user_id is not None:
                    following_subq = select(user_followers.c.following_id).where(user_followers.c.follower_id == user_id)
                    stmt = stmt.where(
                        or_(
                            SocialPost.author_id == user_id,
                            SocialPost.author_id.in_(following_subq),
                        )
                    )
                stmt = stmt.order_by(SocialPost.created_at.desc()).limit(limit).offset(offset)
                result = await session.execute(stmt)
                rows = result.all()
                posts: list[dict[str, Any]] = []
                for post, author, strategy_name in rows:
                    post_dict = self._post_to_dict(post)
                    post_dict["author"] = {
                        "id": author.id,
                        "username": author.username,
                        "display_name": author.display_name,
                        "avatar_url": author.avatar_url,
                    }
                    if strategy_name:
                        post_dict["strategy"] = {"name": strategy_name}
                    posts.append(post_dict)
            except Exception as exc:
                logger.exception("[ERROR] Failed to get social feed: %s", exc)
                return []
            else:
                return posts

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def get_social_performance(self) -> dict[str, Any]:
        session_factory = self._require_session_factory()
        async with session_factory() as session:
            try:
                total_traders = await session.scalar(select(func.count(User.id))) or 0
                total_strategies = await session.scalar(select(func.count(TradingStrategy.id))) or 0
                avg_performance = await session.scalar(select(func.avg(User.total_pnl_percentage))) or 0.0
            except Exception as exc:
                logger.exception("[ERROR] Failed to compute social performance: %s", exc)
                return {
                    "total_traders": 0,
                    "total_strategies": 0,
                    "top_performers": [],
                    "average_performance": 0.0,
                    "timestamp": _utc_iso(),
                }

        top_performers = await self.get_leaderboard(_period="monthly", category="pnl", limit=10)
        return {
            "total_traders": int(total_traders),
            "total_strategies": int(total_strategies),
            "top_performers": top_performers,
            "average_performance": float(avg_performance),
            "timestamp": _utc_iso(),
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _user_to_dict(self, user: User) -> dict[str, Any]:
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "display_name": user.display_name,
            "avatar_url": user.avatar_url,
            "bio": user.bio,
            "location": user.location,
            "website": user.website,
            "verified": user.verified,
            "premium": user.premium,
            "trading_stats": {
                "total_trades": user.total_trades,
                "win_rate": user.win_rate,
                "total_pnl": user.total_pnl,
                "total_pnl_percentage": user.total_pnl_percentage,
                "best_trade": user.best_trade,
                "worst_trade": user.worst_trade,
                "avg_trade_duration": user.avg_trade_duration,
                "sharpe_ratio": user.sharpe_ratio,
                "max_drawdown": user.max_drawdown,
            },
            "social_stats": {
                "followers_count": user.followers_count,
                "following_count": user.following_count,
                "strategies_count": user.strategies_count,
                "reputation_score": user.reputation_score,
            },
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        }

    def _strategy_to_dict(self, strategy: TradingStrategy) -> dict[str, Any]:
        return {
            "id": strategy.id,
            "author_id": strategy.author_id,
            "name": strategy.name,
            "description": strategy.description,
            "strategy_type": strategy.strategy_type,
            "symbol": strategy.symbol,
            "timeframe": strategy.timeframe,
            "config": strategy.config,
            "indicators": strategy.indicators,
            "entry_conditions": strategy.entry_conditions,
            "exit_conditions": strategy.exit_conditions,
            "performance": {
                "total_trades": strategy.total_trades,
                "win_rate": strategy.win_rate,
                "total_pnl": strategy.total_pnl,
                "total_pnl_percentage": strategy.total_pnl_percentage,
                "sharpe_ratio": strategy.sharpe_ratio,
                "max_drawdown": strategy.max_drawdown,
                "calmar_ratio": strategy.calmar_ratio,
            },
            "social": {
                "is_public": strategy.is_public,
                "is_featured": strategy.is_featured,
                "likes_count": strategy.likes_count,
                "followers_count": strategy.followers_count,
                "copies_count": strategy.copies_count,
            },
            "pricing": {
                "is_premium": strategy.is_premium,
                "price": strategy.price,
                "currency": strategy.currency,
            },
            "tags": strategy.tags,
            "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
            "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None,
        }

    def _copy_trade_to_dict(self, copy_trade: CopyTrade) -> dict[str, Any]:
        return {
            "id": copy_trade.id,
            "follower_id": copy_trade.follower_id,
            "leader_id": copy_trade.leader_id,
            "strategy_id": copy_trade.strategy_id,
            "allocation_percentage": copy_trade.allocation_percentage,
            "max_position_size": copy_trade.max_position_size,
            "risk_multiplier": copy_trade.risk_multiplier,
            "is_active": copy_trade.is_active,
            "performance": {
                "total_trades_copied": copy_trade.total_trades_copied,
                "successful_trades": copy_trade.successful_trades,
                "total_pnl": copy_trade.total_pnl,
            },
            "created_at": copy_trade.created_at.isoformat() if copy_trade.created_at else None,
            "updated_at": copy_trade.updated_at.isoformat() if copy_trade.updated_at else None,
            "last_trade_at": copy_trade.last_trade_at.isoformat() if copy_trade.last_trade_at else None,
        }

    def _post_to_dict(self, post: SocialPost) -> dict[str, Any]:
        return {
            "id": post.id,
            "author_id": post.author_id,
            "content": post.content,
            "post_type": post.post_type,
            "symbol": post.symbol,
            "strategy_id": post.strategy_id,
            "engagement": {
                "likes_count": post.likes_count,
                "comments_count": post.comments_count,
                "shares_count": post.shares_count,
            },
            "media": {"images": post.images, "charts": post.charts},
            "is_pinned": post.is_pinned,
            "created_at": post.created_at.isoformat() if post.created_at else None,
            "updated_at": post.updated_at.isoformat() if post.updated_at else None,
        }


# Social trading service state - using dict to avoid global keyword
_social_trading_service_state: dict[str, Any | None] = {"instance": None}


def get_social_trading_service():
    """Get the global social trading service instance."""
    return _social_trading_service_state["instance"]


def set_social_trading_service(service):
    """Set the global social trading service instance."""
    _social_trading_service_state["instance"] = service

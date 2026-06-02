"""
Data Collector
Collects and prepares data for AI training.

Adds lightweight types `DatasetMetadata` and `DataBatch` for test compatibility.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

try:
    import numpy as np  # type: ignore[import-untyped]

    _HAS_NUMPY = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    _HAS_NUMPY = False

try:
    import pandas as pd  # type: ignore[import-untyped]

    _HAS_PANDAS = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    _HAS_PANDAS = False

try:
    from backend.services.redis_service import get_redis_service

    _HAS_REDIS = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    _HAS_PANDAS = False

from pydantic import BaseModel, Field


class MissingRequiredFieldsError(ValueError):
    """Missing required fields in training sample"""

    def __init__(self, missing_list: str) -> None:
        super().__init__(f"Missing required fields: {missing_list}")
        self.missing_list = missing_list


# Optional imports - try at top level
try:
    from backend.services.market_data import MarketDataService  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    MarketDataService = None

# CoinCache deleted - using dict instead

logger = logging.getLogger(__name__)


class DatasetMetadata(BaseModel):
    """Batch metadata used by tests (attribute access + model_dump)."""

    name: str = Field(...)
    timeframe: str = Field(default="1h")
    feature_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    num_samples: int = 0
    symbols: list[str] = Field(default_factory=list)
    model_config = {"protected_namespaces": ()}


class DataBatch(TypedDict):
    # Use Any for broad compatibility with numpy arrays or lists
    features: Any
    targets: Any
    timestamps: Any
    symbols: list[str]
    metadata: DatasetMetadata


class DataCollector:
    """Collects and prepares data for AI training."""

    def __init__(
        self,
        market_data_service: Any | None = None,
        *,
        batch_size: int | None = None,
        cache_size: int | None = None,
    ) -> None:
        """Initialize the collector - config from backend.config.settings"""
        self.market_data_service = market_data_service if market_data_service is not None else self._load_market_service()
        self.batch_size = int(batch_size) if batch_size is not None else 1024
        self.cache_size = int(cache_size) if cache_size is not None else 10_000
        self._last_batch: DataBatch | None = None
        logger.info("Data Collector initialized")

    @staticmethod
    def _load_market_service() -> Any:
        """Best-effort import of a market data service used in tests."""
        try:
            # Preferred path in tests
            if MarketDataService is None:
                return None

            return MarketDataService.shared()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Fallback removed - deprecated service no longer available
            return None

    # -------------------------------
    # Raw data access helpers (kept)
    # -------------------------------

    async def collect_market_data(
        self,
        symbols: list[str] | None = None,
        timeframe: str = "1h",
    ) -> DataBatch:
        """Collect live market data and return a model-ready DataBatch.

        - Uses provided symbols or discovers via market_data_service.get_available_symbols()
        - Falls back to ["BTC/USD", "ETH/USD"] if discovery fails
        - Returns numpy arrays for features/targets/timestamps
        """
        # Resolve symbols
        resolved: list[str] = []
        if symbols and len(symbols) > 0:
            resolved = list(symbols)
        else:
            try:
                avail = await self.market_data_service.get_available_symbols()  # type: ignore[attr-defined]
                if isinstance(avail, list) and avail:
                    resolved = avail[:2]
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                resolved = []
        if not resolved:
            resolved = ["BTC/USD", "ETH/USD"]

        # Fetch frames
        if not _HAS_PANDAS:
            msg = "pandas not available"
            raise ImportError(msg)

        frames: dict[str, Any] = {}
        for sym in resolved:
            try:
                df = await self.market_data_service.get_market_data(sym, timeframe=timeframe)  # type: ignore[attr-defined]
                if df is not None and not getattr(df, "empty", True):
                    frames[sym] = df
                    continue
            except TypeError:
                try:
                    df = await self.market_data_service.get_market_data(sym)  # type: ignore[attr-defined]
                    if df is not None and not getattr(df, "empty", True):
                        frames[sym] = df
                        continue
                except (ValueError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass
            except (ValueError, AttributeError, KeyError, IndexError, RuntimeError):
                pass
        # If direct getter failed, try a cached bulk source if provided (used by tests)
        if not frames:
            try:
                cached = await self.market_data_service.get_all_cached_data()  # type: ignore[attr-defined]
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                cached = getattr(self.market_data_service, "get_all_cached_data", dict)()
            if isinstance(cached, dict):
                for sym, df in cached.items():
                    if df is not None and not getattr(df, "empty", True):
                        frames[sym] = df

        if not frames:
            msg = "No valid data collected"
            raise ValueError(msg)

        # Build combined features/targets
        feature_frames: list[pd.DataFrame] = []
        target_frames: list[pd.DataFrame] = []
        ts_series: list[Any] = []
        sym_list: list[str] = []

        for sym, df in frames.items():
            try:
                feats = self._extract_features(df)
                targs = self._extract_targets(df)
                # align on index
                common_index = feats.index.intersection(targs.index)
                feats = feats.loc[common_index]
                targs = targs.loc[common_index]
                if len(feats) == 0:
                    continue
                feature_frames.append(feats)
                target_frames.append(targs)
                ts_series.append(common_index.values)
                sym_list.append(sym)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                continue

        if not feature_frames:
            msg = "No valid data collected"
            raise ValueError(msg)

        if not _HAS_NUMPY:
            msg = "numpy not available"
            raise ImportError(msg)

        features = np.vstack([f.to_numpy(dtype=np.float64) for f in feature_frames])
        targets = np.vstack([t.to_numpy(dtype=np.float64) for t in target_frames])
        timestamps = np.concatenate([np.asarray(s, dtype="datetime64[ns]") for s in ts_series])

        metadata = DatasetMetadata(
            name=f"market_data_{timeframe}",
            timeframe=timeframe,
            feature_columns=list(feature_frames[0].columns),
            target_columns=list(target_frames[0].columns),
            num_samples=int(features.shape[0]),
            symbols=sym_list,
        )

        batch: DataBatch = {
            "features": features,
            "targets": targets,
            "timestamps": timestamps,
            "symbols": sym_list,
            "metadata": metadata,
        }

        self._last_batch = batch
        return batch

    async def get_active_symbols(self) -> list[str]:
        """Get list of active trading symbols"""
        return await self.market_data_service.get_active_symbols()

    async def get_coin_data(self, symbol: str) -> dict | None:
        """Get cached data for a specific coin"""
        return await self.market_data_service.get_coin_data(symbol)

    async def get_all_cached_data(self) -> list[dict]:
        """Get all cached coin data"""
        return await self.market_data_service.get_all_cached_data()

    # -------------------------------
    # Dataset building
    # -------------------------------

    async def build_dataset(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        feature_columns: list[str] | None = None,
        target_columns: list[str] | None = None,
        dataset_name: str = "market_dataset",
        align_by_timestamp: bool = False,
        limit_per_symbol: int | None = None,
        use_numpy: bool | None = None,
    ) -> DataBatch:
        """Build a model-ready dataset (features/targets/timestamps/symbols + metadata).

        Args:
            symbols: Symbols to include (e.g., ["BTC-USD", "ETH-USD"])
            timeframe: Candle timeframe (e.g., "1h", "5m")
            feature_columns: Which fields to extract as features. Defaults to ["close","volume"] if available.
            target_columns: Which fields to extract as targets. Defaults to ["close"] if available.
            dataset_name: Name for metadata.
            align_by_timestamp: If True, only keep rows with timestamps that appear for ALL symbols.
            limit_per_symbol: Optional cap on samples per symbol (most recent N).
            use_numpy: Force numpy arrays (True) or lists (False). Default: auto (numpy if available).

        Returns:
            DataBatch: features/targets/timestamps/symbols and metadata.
        """
        if use_numpy is None:
            use_numpy = _HAS_NUMPY

        raw = await self.collect_market_data(symbols, timeframe=timeframe)
        if not raw:
            # empty dataset shell
            meta = DatasetMetadata(
                name=dataset_name,
                timeframe=timeframe,
                feature_columns=feature_columns or [],
                target_columns=target_columns or [],
                num_samples=0,
                symbols=symbols,
            )
            return {
                "features": [] if not use_numpy else (np.empty((0, 0)) if _HAS_NUMPY else []),
                "targets": [] if not use_numpy else (np.empty((0, 0)) if _HAS_NUMPY else []),
                "timestamps": [],
                "symbols": [],
                "metadata": meta,
            }

        # Defaults if not provided
        if feature_columns is None:
            feature_columns = ["close", "volume"]
        if target_columns is None:
            target_columns = ["close"]

        # Prepare symbol -> rows with (ts, data)
        per_symbol: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for sym, rows in raw.items():
            normalized = []
            # enforce limit per symbol (most recent)
            it = rows[-limit_per_symbol:] if (limit_per_symbol and limit_per_symbol > 0) else rows
            for r in it:
                ts = self._extract_ts(r)
                if ts is None:
                    continue
                normalized.append((ts, r))
            # Keep in chronological order
            normalized.sort(key=lambda x: x[0])
            if normalized:
                per_symbol[sym] = normalized

        if not per_symbol:
            meta = DatasetMetadata(
                name=dataset_name,
                timeframe=timeframe,
                feature_columns=feature_columns,
                target_columns=target_columns,
                num_samples=0,
                symbols=symbols,
            )
            return {
                "features": [] if not use_numpy else (np.empty((0, 0)) if _HAS_NUMPY else []),
                "targets": [] if not use_numpy else (np.empty((0, 0)) if _HAS_NUMPY else []),
                "timestamps": [],
                "symbols": [],
                "metadata": meta,
            }

        # Optional inner join by timestamp across symbols
        if align_by_timestamp:
            common_ts = self._common_timestamps(per_symbol.values())
            per_symbol = {s: [(ts, row) for ts, row in seq if ts in common_ts] for s, seq in per_symbol.items()}

        # Build rows (concatenate across symbols; 1 sample per (symbol, timestamp))
        feat_list: list[list[float]] = []
        targ_list: list[list[float]] = []
        ts_list: list[int] = []
        sym_list: list[str] = []

        for sym in symbols:
            seq = per_symbol.get(sym)
            if not seq:
                continue
            for ts, row in seq:
                feats = self._extract_columns(row, feature_columns)
                targs = self._extract_columns(row, target_columns)
                if feats is None or targs is None:
                    # skip rows missing required columns
                    continue
                feat_list.append(feats)
                targ_list.append(targs)
                ts_list.append(ts)
                sym_list.append(sym)

        num_samples = len(feat_list)
        if use_numpy and _HAS_NUMPY:
            features = np.asarray(feat_list, dtype=float) if num_samples > 0 else np.empty((0, len(feature_columns)))
            targets = np.asarray(targ_list, dtype=float) if num_samples > 0 else np.empty((0, len(target_columns)))
            timestamps = np.asarray(ts_list, dtype=np.int64) if num_samples > 0 else np.empty((0,), dtype=np.int64)
        else:
            features = feat_list
            targets = targ_list
            timestamps = ts_list

        meta = DatasetMetadata(
            name=dataset_name,
            timeframe=timeframe,
            feature_columns=feature_columns,
            target_columns=target_columns,
            num_samples=num_samples,
            symbols=symbols,
        )

        return {
            "features": features,
            "targets": targets,
            "timestamps": timestamps,
            "symbols": sym_list,
            "metadata": meta,
        }

    # -------------------------------
    # Batching utilities
    # -------------------------------

    def iter_batches(
        self,
        batch: DataBatch,
        batch_size: int = 1024,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterable[DataBatch]:
        """Yield smaller DataBatch chunks (same metadata, sliced rows).

        Works with numpy arrays or python lists.
        """
        n = int(getattr(batch["metadata"], "num_samples", batch["metadata"]["num_samples"]))
        if n <= 0:
            return

        idx = list(range(n))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(idx)

        # index helper for arrays/lists
        def _take(arr: Any, ix: list[int]):
            if _HAS_NUMPY and "numpy" in str(type(arr)):
                return arr[ix]
            # Assume list-like
            return [arr[i] for i in ix]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sel = idx[start:end]
            yield {
                "features": _take(batch["features"], sel),
                "targets": _take(batch["targets"], sel),
                "timestamps": _take(batch["timestamps"], sel),
                "symbols": _take(batch["symbols"], sel),
                "metadata": DatasetMetadata(
                    **(batch["metadata"].model_dump() if hasattr(batch["metadata"], "model_dump") else batch["metadata"]),
                    num_samples=len(sel),
                ),
            }

    # -------------------------------
    # Helpers
    # -------------------------------

    @staticmethod
    def _extract_ts(row: dict[str, Any]) -> int | None:
        """Extract a POSIX ms timestamp from a candle row."""
        # try common keys: 'ts' or 'timestamp'
        ts = row.get("ts")
        if ts is None:
            ts = row.get("timestamp")
        if ts is None:
            return None
        try:
            # if ISO string
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    return int(dt.timestamp() * 1000)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    # maybe numeric string
                    return int(float(ts))
            # if seconds
            if isinstance(ts, (int, float)) and ts < 10_000_000_000:
                return int(ts * 1000)
            # already ms
            return int(ts)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return None

    @staticmethod
    def _extract_columns(row: dict[str, Any], cols: list[str]) -> list[float] | None:
        out: list[float] = []
        try:
            for c in cols:
                v = row.get(c)
                if v is None:
                    return None
                out.append(float(v))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return None
        else:
            return out

    @staticmethod
    def _common_timestamps(
        sequences: Iterable[list[tuple[int, dict[str, Any]]]],
    ) -> set[int]:
        """Return timestamps common to all sequences (inner join)."""
        itr = iter(sequences)
        try:
            first = next(itr)
        except StopIteration:
            return set()
        common = {ts for ts, _ in first}
        for seq in itr:
            common &= {ts for ts, _ in seq}
            if not common:
                break
        return common

    # -------------------------------
    # Feature/Target engineering used in tests
    # -------------------------------

    def _extract_features(self, df: Any) -> Any:
        if not _HAS_PANDAS:
            msg = "pandas not available"
            raise ImportError(msg)

        data = df.copy()
        # Ensure float dtypes
        for c in ["open", "high", "low", "close", "volume"]:
            if c in data.columns:
                data[c] = data[c].astype("float64")

        features = pd.DataFrame(index=data.index)
        features["open"] = data["open"]
        features["high"] = data["high"]
        features["low"] = data["low"]
        features["close"] = data["close"]
        features["volume"] = data["volume"]
        # Derived
        features["price_change"] = data["close"].pct_change().fillna(0.0)
        features["high_low_range"] = (data["high"] - data["low"]).fillna(0.0)
        features["volume_change"] = data["volume"].pct_change().fillna(0.0)

        for win in (5, 10, 20, 50):
            features[f"ma_{win}"] = data["close"].rolling(window=win, min_periods=1).mean().bfill()
        for win in (5, 10, 20, 50):
            features[f"volume_ma_{win}"] = data["volume"].rolling(window=win, min_periods=1).mean().bfill()
        # Rolling volatility (std of returns over 20)
        returns = data["close"].pct_change()
        features["volatility"] = returns.rolling(window=20, min_periods=1).std().fillna(0.0)
        return features.fillna(0.0)

    def _extract_targets(self, df: Any) -> Any:
        if not _HAS_PANDAS:
            msg = "pandas not available"
            raise ImportError(msg)

        data = df.copy()
        for c in ["open", "high", "low", "close", "volume"]:
            if c in data.columns:
                data[c] = data[c].astype("float64")

        t = pd.DataFrame(index=data.index)
        close = data["close"]
        for h in (1, 5, 10):
            t[f"return_{h}"] = close.shift(-h) / close - 1.0
        # Future volatility over 20 bars
        ret = close.pct_change()
        t["future_volatility"] = ret.rolling(window=20, min_periods=1).std().shift(-20)
        return t.dropna()

    # -------------------------------
    # Persistence helpers used in tests
    # -------------------------------

    def save_batch(self, batch: DataBatch | None = None) -> Any:
        if not _HAS_NUMPY:
            msg = "numpy not available"
            raise ImportError(msg)

        b = batch or self._last_batch
        if b is None:
            msg = "No batch available to save"
            raise ValueError(msg)

        data_cfg = getattr(self.config, "data", {})
        base_dir = None
        base_dir = data_cfg.get("data_dir") if isinstance(data_cfg, dict) else getattr(data_cfg, "data_dir", None)

        base_path = Path(str(base_dir or "data"))
        batches_dir = base_path / "batches"
        batches_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = batches_dir / f"batch_{ts}.npz"

        meta = b["metadata"].model_dump() if hasattr(b["metadata"], "model_dump") else b["metadata"]
        np.savez_compressed(
            path,
            features=b["features"],
            targets=b["targets"],
            timestamps=b["timestamps"],
            symbols=np.array(b["symbols"], dtype=object),
            metadata=json.dumps(meta).encode("utf-8"),
        )
        return path

    def load_batch(self, path: Any) -> DataBatch:
        if not _HAS_NUMPY:
            msg = "numpy not available"
            raise ImportError(msg)

        with np.load(path, allow_pickle=True) as npz:
            features = npz["features"]
            targets = npz["targets"]
            timestamps = npz["timestamps"]
            symbols = list(npz["symbols"].tolist())
            meta_raw = json.loads(bytes(npz["metadata"]).decode("utf-8"))
            metadata = DatasetMetadata(**meta_raw)

        batch: DataBatch = {
            "features": features,
            "targets": targets,
            "timestamps": timestamps,
            "symbols": symbols,
            "metadata": metadata,
        }
        return batch

    async def collect_sample(self, training_sample: dict[str, Any]) -> None:
        """
        Collect a single training sample for AI learning.

        Stores individual training samples (from paper trading feedback, etc.)
        for later batch processing during model retraining.

        Args:
            training_sample: Dictionary containing training data with keys:
                - features: Feature vector for the sample
                - label: Target label (0/1 for classification)
                - symbol: Trading symbol
                - timestamp: When the sample was created
                - metadata: Additional information about the sample
        """

        def _raise_missing_fields(missing_list: str) -> None:
            raise MissingRequiredFieldsError(missing_list)

        try:
            # Validate required fields
            required_fields = ["features", "label", "symbol", "timestamp"]
            missing_fields = [field for field in required_fields if field not in training_sample]

            if missing_fields:
                missing_list = ", ".join(missing_fields)
                _raise_missing_fields(missing_list)

            # For now, we'll use Redis to store training samples
            # This allows the realtime model trainer to collect them later
            redis_service = get_redis_service()
            redis_client = redis_service.redis_client

            # Create a unique key for this sample
            sample_key = f"training_sample:{training_sample['symbol']}:{training_sample['timestamp']}"

            # Store the sample as JSON
            sample_data = {
                "features": training_sample["features"],
                "label": training_sample["label"],
                "symbol": training_sample["symbol"],
                "timestamp": training_sample["timestamp"],
                "metadata": training_sample.get("metadata", {}),
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }

            # Store with expiration (24 hours)
            redis_client.setex(sample_key, 86400, json.dumps(sample_data))

            # Add to a set of training samples for this symbol
            symbol_samples_key = f"training_samples:{training_sample['symbol']}"
            redis_client.sadd(symbol_samples_key, sample_key)
            redis_client.expire(symbol_samples_key, 86400)  # Expire the set too

            logger.debug(f"Collected training sample for {training_sample['symbol']}")

        except Exception as e:
            logger.exception(f"Failed to collect training sample: {e}")
            # Don't raise - we don't want sample collection to break trading

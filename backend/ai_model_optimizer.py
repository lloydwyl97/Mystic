"""
AI Model Optimizer for Mystic Trading Platform (Repaired & Hardened)

Key fixes & improvements:
- Robust concurrency: single-flight loading per model using an asyncio.Lock
- Deterministic LRU eviction with memory *budget* (uses psutil total RAM and MAX_MODEL_MEMORY)
- Timeout protection for load operations via asyncio.wait_for
- Safer memory estimation across different model objects (HF pipeline, dict(model/tokenizer), torch.nn.Module)
- Consistent cache stats + proactive GC on evictions/unloads
- Graceful handling when optional ML libs are missing
- Background preloading respects existing loads and reports per-model failures
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Direct imports for production
import joblib
import psutil
import torch
from transformers import AutoModel, AutoTokenizer, pipeline

from backend.services.task_manager import task_manager

# SentenceTransformer feature temporarily disabled due to dependency conflicts
# sentence-transformers conflicts with torch 2.5.1 in requirements.txt
# If needed in future, resolve torch/transformers version conflicts first
# For now, sentence-transformer model types are not supported
SentenceTransformer = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Model configuration
MODEL_CACHE_DIR = "models/cache"
MODEL_DOWNLOAD_TIMEOUT = 300  # seconds (upper bound we wait for any blocking load to finish)
MODEL_LOAD_TIMEOUT = 120  # seconds (upper bound for our async wait per model)
MAX_MODEL_MEMORY = 0.8  # fraction of total system RAM we allow the cache to occupy
BACKGROUND_LOADING = True


@dataclass
class ModelInfo:
    """Model information and metadata"""

    name: str
    model_type: str  # 'transformer', 'sentiment', 'embedding', 'custom'
    path: str
    size_mb: float
    loaded: bool = False
    load_time: float = 0.0
    memory_usage: float = 0.0
    last_used: float = 0.0
    access_count: int = 0


class ModelCache:
    """In-memory model cache with LRU eviction"""

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self.models: dict[str, Any] = {}
        self.model_info: dict[str, ModelInfo] = {}
        self.access_order: list[str] = []
        self.lock = threading.Lock()

    def get(self, model_name: str) -> Any | None:
        """Get model from cache and mark as MRU"""
        with self.lock:
            if model_name in self.models:
                if model_name in self.access_order:
                    with contextlib.suppress(ValueError):
                        self.access_order.remove(model_name)
                self.access_order.append(model_name)
                if model_name in self.model_info:
                    info = self.model_info[model_name]
                    info.access_count += 1
                    info.last_used = time.time()
                return self.models[model_name]
            return None

    def set(self, model_name: str, model: Any, model_info: ModelInfo):
        """Add/update model in cache (LRU). Evicts if capacity exceeded."""
        with self.lock:
            if model_name not in self.models and len(self.models) >= self.max_size:
                self._evict_lru()

            self.models[model_name] = model
            self.model_info[model_name] = model_info

            if model_name in self.access_order:
                with contextlib.suppress(ValueError):
                    self.access_order.remove(model_name)
            self.access_order.append(model_name)

    def _evict_lru(self):
        """Evict least-recently used model"""
        if not self.access_order:
            return
        lru_model = self.access_order.pop(0)
        # drop data
        self.models.pop(lru_model, None)
        self.model_info.pop(lru_model, None)
        gc.collect()
        logger.info(f"Evicted model from cache (LRU): {lru_model}")

    def evict_until(self, condition_fn) -> int:
        """
        Evict LRU models until condition_fn() returns True or cache is empty.
        Returns number of evicted models.
        """
        evicted = 0
        with self.lock:
            while self.access_order and not condition_fn():
                lru_model = self.access_order.pop(0)
                self.models.pop(lru_model, None)
                self.model_info.pop(lru_model, None)
                evicted += 1
                logger.info(f"Evicted model from cache (budget): {lru_model}")
        if evicted:
            gc.collect()
        return evicted

    def clear(self):
        """Clear all models from cache"""
        with self.lock:
            self.models.clear()
            self.model_info.clear()
            self.access_order.clear()
        gc.collect()

    def total_memory_mb(self) -> float:
        with self.lock:
            return float(sum(info.memory_usage for info in self.model_info.values()))

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "cache_size": len(self.models),
                "max_size": self.max_size,
                "models": list(self.models.keys()),
                "total_memory_mb": float(sum(info.memory_usage for info in self.model_info.values())),
                "access_order": self.access_order.copy(),
            }


class ModelLoader:
    """Optimized model loader with background loading and memory budgeting"""

    def __init__(self) -> None:
        self.cache = ModelCache()
        self.loading_models: dict[str, asyncio.Future] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        # asyncio.Lock is safe to instantiate here; it will bind to the running loop when used.
        self._async_lock = asyncio.Lock()  # serializes single-flight decisions

        Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Public API
    # ----------------------------
    async def load_model(self, model_name: str, model_type: str = "transformer") -> Any:
        """Load model with caching, concurrency control, and timeouts."""
        # Cache fast path
        cached = self.cache.get(model_name)
        if cached is not None:
            logger.debug(f"Model {model_name} served from cache")
            return cached

        async with self._async_lock:
            # Re-check cache after acquiring lock
            cached = self.cache.get(model_name)
            if cached is not None:
                return cached

            # If already loading, await the same future
            if model_name in self.loading_models:
                logger.debug(f"Model {model_name} is already loading; awaiting existing task")
                fut = self.loading_models[model_name]
            else:
                logger.info(f"Loading model: {model_name} ({model_type})")
                fut = await task_manager.create_task(self._load_model_async(model_name, model_type), name="ai_model_optimizer:load_model_async")
                self.loading_models[model_name] = fut

        # Await outside the lock to allow other models to proceed
        try:
            # Timeout guard (prevents hanging forever on network downloads)
            return await asyncio.wait_for(fut, timeout=MODEL_LOAD_TIMEOUT)
        except asyncio.TimeoutError:
            logger.exception(f"Timed out loading model {model_name} after {MODEL_LOAD_TIMEOUT}s")
            # Best effort cancel; underlying thread may continue but we won't await it.
            with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                fut.cancel()
            raise
        finally:
            # Clean up registry if *this* wait was for the canonical future
            async with self._async_lock:
                if self.loading_models.get(model_name) is fut:
                    self.loading_models.pop(model_name, None)

    async def preload_models(self, model_configs: list[dict[str, str]]):
        """Preload models in background."""
        if not BACKGROUND_LOADING:
            return

        logger.info(f"Preloading {len(model_configs)} models")
        tasks = []
        for cfg in model_configs:
            name = cfg["name"]
            mtype = cfg.get("type", "transformer")
            # Skip if already loaded or loading
            if self.cache.get(name) is not None or name in self.loading_models:
                continue
            tasks.append(await task_manager.create_task(self._preload_one(name, mtype), name="ai_model_optimizer:preload_one"))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = sum(1 for r in results if isinstance(r, Exception))
            if failures:
                logger.warning(f"Model preloading completed with {failures} failure(s)")
            else:
                logger.info("Model preloading completed successfully")

    def unload_model(self, model_name: str):
        """Unload model from cache and run GC."""
        with self.cache.lock:
            if model_name in self.cache.models:
                self.cache.models.pop(model_name, None)
                self.cache.model_info.pop(model_name, None)
                with contextlib.suppress(ValueError):
                    self.cache.access_order.remove(model_name)
                logger.info(f"Unloaded model: {model_name}")
        gc.collect()

    def optimize_memory(self):
        """Evict LRU models until total cache memory fits within budget."""
        budget_mb = self._cache_budget_mb()
        evicted = self.cache.evict_until(lambda: self.cache.total_memory_mb() <= budget_mb)
        if evicted:
            logger.info(f"Memory optimization evicted {evicted} model(s) to meet budget {budget_mb:.1f} MB")

    def get_model_stats(self) -> dict[str, Any]:
        """Get model loading statistics"""
        stats = self.cache.get_stats()
        total_mb, avail_mb = (
            self._get_total_memory_mb(),
            self._get_available_memory_mb(),
        )
        budget_mb = self._cache_budget_mb()
        stats.update(
            {
                "loading_models": list(self.loading_models.keys()),
                "total_models": len(self.cache.models) + len(self.loading_models),
                "system_total_memory_mb": total_mb,
                "system_available_memory_mb": avail_mb,
                "cache_budget_mb": budget_mb,
                "cache_budget_used_pct": round((stats["total_memory_mb"] / budget_mb) * 100, 2) if budget_mb > 0 else 0.0,
            },
        )
        return stats

    def cleanup(self):
        """Cleanup resources"""
        self.cache.clear()
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.executor.shutdown(wait=True)
        logger.info("Model loader cleaned up")

    # ----------------------------
    # Internals
    # ----------------------------
    async def _preload_one(self, model_name: str, model_type: str):
        try:
            # Longer timeout allowance during preloading
            await asyncio.wait_for(self.load_model(model_name, model_type), timeout=MODEL_DOWNLOAD_TIMEOUT)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Preload failed for {model_name}: {e}")

    async def _load_model_async(self, model_name: str, model_type: str) -> Any:
        """Load model asynchronously in a thread pool, then enforce memory budget."""
        try:
            loop = asyncio.get_running_loop()
            model, info = await loop.run_in_executor(self.executor, self._load_model_sync, model_name, model_type)

            # Memory budget enforcement: evict until within budget BEFORE admitting
            budget_mb = self._cache_budget_mb()
            if budget_mb > 0:
                # Prepare to add this model; ensure total + this <= budget
                def within_budget() -> bool:
                    return (self.cache.total_memory_mb() + info.memory_usage) <= budget_mb

                evicted = self.cache.evict_until(within_budget)
                if evicted:
                    logger.info(f"Evicted {evicted} model(s) to admit '{model_name}' within budget ({budget_mb:.1f} MB)")

            # Admit to cache
            self.cache.set(model_name, model, info)
            logger.info(f"Model {model_name} loaded in {info.load_time:.2f}s; est. mem {info.memory_usage:.1f} MB")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to load model {model_name}: {e}")
            raise
        else:
            return model

    def _load_model_sync(self, model_name: str, model_type: str) -> tuple[Any, ModelInfo]:
        """Load model synchronously inside a worker thread."""
        start = time.time()
        model: Any

        if model_type == "sentiment":
            if pipeline is None:
                msg = "transformers library not available (pipeline)"
                raise ImportError(msg)
            model = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if (torch and getattr(torch, "cuda", None) and torch.cuda.is_available()) else -1,
            )

        elif model_type == "embedding":
            if SentenceTransformer is None:
                msg = "sentence-transformers library not available"
                raise ImportError(msg)
            model = SentenceTransformer(model_name)

        elif model_type == "transformer":
            if AutoTokenizer is None or AutoModel is None:
                msg = "transformers library not available (AutoTokenizer/AutoModel)"
                raise ImportError(msg)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            nn_model = AutoModel.from_pretrained(model_name)
            model = {"tokenizer": tokenizer, "model": nn_model}

        else:
            # Custom model
            if joblib is None:
                msg = "joblib library not available"
                raise ImportError(msg)
            path = str(Path(MODEL_CACHE_DIR) / f"{model_name}.joblib")
            if not Path(path).exists():
                msg = f"Model file not found: {path}"
                raise FileNotFoundError(msg)
            model = joblib.load(path)

        load_time = time.time() - start
        memory_usage = self._estimate_memory_usage(model)

        info = ModelInfo(
            name=model_name,
            model_type=model_type,
            path=model_name,
            size_mb=memory_usage,
            loaded=True,
            load_time=load_time,
            memory_usage=memory_usage,
            last_used=time.time(),
            access_count=1,
        )
        return model, info

    def _estimate_memory_usage(self, model: Any) -> float:
        """Estimate model memory usage in MB."""
        try:
            # HF pipeline -> try .model
            if hasattr(model, "model") and hasattr(model.model, "parameters"):
                torch_model = model.model
                return self._estimate_torch_params_mb(torch_model)

            # Dict {"model": nn.Module, ...}
            if isinstance(model, dict) and "model" in model and hasattr(model["model"], "parameters"):
                return self._estimate_torch_params_mb(model["model"])

            # SentenceTransformer often wraps a torch model
            if SentenceTransformer is not None and isinstance(model, SentenceTransformer):
                inner = getattr(model, "auto_model", None)
                if inner is None:
                    attr = getattr(model, "_first_module", None)
                    if callable(attr):
                        try:
                            inner = attr()
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            inner = None
                    else:
                        inner = attr
                if inner is not None and hasattr(inner, "parameters"):
                    return self._estimate_torch_params_mb(inner)

            # torch.nn.Module directly
            if torch is not None and hasattr(model, "parameters"):
                return self._estimate_torch_params_mb(model)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass

        # Fallback conservative estimate
        return 100.0

    @staticmethod
    def _estimate_torch_params_mb(nn_module: Any) -> float:
        try:
            params_iter = getattr(nn_module, "parameters", None)
            if params_iter is None:
                return 150.0
            total_params = sum(int(p.numel()) for p in params_iter())
            # Assume float32 unless known otherwise
            memory_bytes = total_params * 4
            return memory_bytes / (1024 * 1024)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 150.0

    def _get_available_memory_mb(self) -> float:
        try:
            return float(psutil.virtual_memory().available) / (1024 * 1024)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 1024.0

    def _get_total_memory_mb(self) -> float:
        try:
            return float(psutil.virtual_memory().total) / (1024 * 1024)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 4096.0

    def _cache_budget_mb(self) -> float:
        """Maximum cache memory in MB based on system RAM and MAX_MODEL_MEMORY fraction."""
        total_mb = self._get_total_memory_mb()
        return float(total_mb * MAX_MODEL_MEMORY)


# Global model loader instance
model_loader = ModelLoader()

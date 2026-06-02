from backend.config.redis_config import SharedRedisState


def get_redis_service():
    return SharedRedisState.get_async_client()


def get_sync_redis_service():
    return SharedRedisState.get_sync_client()

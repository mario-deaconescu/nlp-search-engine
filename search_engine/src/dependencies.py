import redis.asyncio as redis

from src.utils.cache import FileCache, BaseFileCache

# redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
file_cache = BaseFileCache("nlp-search-engine")

async def get_file_cache() -> FileCache:
    return file_cache

# async def get_redis():
#     return redis_client
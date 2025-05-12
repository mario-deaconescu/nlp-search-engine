import asyncio
import codecs
import hashlib
import os
import pickle
from asyncio import Lock
from functools import cache
import fcntl
from typing import Optional, TypeVar, Generic, Awaitable, Union, Literal

from fastapi import UploadFile
from platformdirs import user_cache_dir
from redis.typing import ResponseT

T = TypeVar("T")
MaybeAwaitable = T | T

class FileLock:
    def __init__(self, file_path, mode: Literal['w', 'wb', 'w+', 'wb+'], lock_type=fcntl.LOCK_EX):
        self.file_path = file_path
        self.mode = mode
        self.lock_type = lock_type
        self.file = None

    def __enter__(self):
        self.file = open(self.file_path, self.mode)
        fcntl.flock(self.file, self.lock_type)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.file:
            return
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


class ReadFileLock:
    def __init__(self, file_path, mode: Literal['r', 'rb', 'r+', 'rb+'], lock_type=fcntl.LOCK_EX):
        self.file_path = file_path
        self.mode = mode
        self.lock_type = lock_type
        self.file = None

    def __enter__(self):
        try:
            self.file = open(self.file_path, self.mode)
        except FileNotFoundError:
            return None
        fcntl.flock(self.file, self.lock_type)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.file:
            return
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


class BaseFileCache:
    class SubCache:
        def __init__(self, parent: "BaseFileCache | BaseFileCache.SubCache", key: str):
            self.parent = parent
            self.key = key

        def subcache(self, key: str) -> "BaseFileCache.SubCache":
            return BaseFileCache.SubCache(self, key)

        def _key(self, item: Optional[str] = None) -> str:
            if item is None:
                return self.key
            return f"{self.key}:{item}"

        def get(self, item: str) -> Optional[str]:
            return self.parent.get(self._key(item))

        def set(self, item: str, value: str) -> None:
            return self.parent.set(self._key(item), value)

        def delete(self, item: str) -> None:
            return self.parent.delete(self._key(item))

        def exists(self, item: str) -> bool:
            return self.parent.exists(self._key(item))

        def get_pickled(self, item: str) -> Optional[T]:
            return self.parent.get_pickled(self._key(item))

        def set_pickled(self, item: str, value: object) -> None:
            return self.parent.set_pickled(self._key(item), value)

        def __getitem__(self, item: str) -> Optional[str]:
            return self.parent[self._key(item)]

        def __setitem__(self, item: str, value: str) -> None:
            self.parent[self._key(item)] = value

        def subkeys(self, key: Optional[str] = None) -> list[str]:
            return self.parent.subkeys(self._key(key))

    def __init__(self, appname: str):
        self.path = user_cache_dir(appname)

    def subcache(self, key: str) -> "BaseFileCache.SubCache":
        return BaseFileCache.SubCache(self, key)

    def get(self, item: str) -> Optional[str]:
        segments = item.split(":")
        path = os.path.join(self.path, *segments)
        with ReadFileLock(path, "r") as f:
            return f.read() if f else None

    def set(self, item: str, value: str) -> None:
        segments = item.split(":")
        path = os.path.join(self.path, *segments)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with FileLock(path, "w") as f:
            f.write(value)

    def delete(self, item: str) -> None:
        segments = item.split(":")
        path = os.path.join(self.path, *segments)
        with ReadFileLock(path, "rb") as f:
            if f:
                os.remove(path)

    def exists(self, item: str) -> bool:
        segments = item.split(":")
        path = os.path.join(self.path, *segments)
        return os.path.exists(path)

    def subkeys(self, key: Optional[str] = None) -> list[str]:
        if key is None:
            path = self.path
        else:
            segments = key.split(":")
            path = os.path.join(self.path, *segments)
        if not os.path.exists(path) or not os.path.isdir(path):
            return []
        return os.listdir(path)

    def get_pickled(self, item: str) -> Optional[T]:
        segments = item.split(":")
        path = os.path.join(self.path, *segments)
        with ReadFileLock(path, "rb") as f:
            return pickle.load(f) if f else None

    def set_pickled(self, item: str, value: object) -> None:
        segments = item.split(":")
        path = os.path.join(self.path, *segments)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with FileLock(path, "wb") as f:
            pickle.dump(value, f)

    def __getitem__(self, item: str) -> Optional[str]:
        return self.get(item)

    def __setitem__(self, item: str, value: str) -> None:
        self.set(item, value)


FileCache = BaseFileCache | BaseFileCache.SubCache


async def make_hash(file: UploadFile, reuse: bool = True, chunk_size: int = 8192) -> str:
    hasher = hashlib.sha256()
    file_size = 0

    while True:
        data = await file.read(chunk_size)
        if not data:
            break
        hasher.update(data)
        file_size += len(data)

    hasher.update(str(file_size).encode())

    if reuse:
        await file.seek(0)

    return hasher.hexdigest()


@cache
def get_cache_dir(child: Optional[str] = None) -> str:
    path = user_cache_dir("nlp-search-engine")
    if child is not None:
        path = os.path.join(path, child)
    os.makedirs(path, exist_ok=True)
    return path

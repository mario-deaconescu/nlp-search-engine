import os
import pickle
from typing import TypeVar, Generic, Callable, Optional

from search_engine.src.constants import CACHE_PATH

T = TypeVar("T")

def no_collate(batch: list[T]) -> T:
    assert len(batch) == 1
    return batch[0]

class CachedList(Generic[T]):
    def __init__(self, cache_path: str, length: int):
        self.cache_path = cache_path
        self.length = length

    def __len__(self) -> int:
        return self.length

    def _build_path(self, idx: int) -> str:
        return os.path.join(CACHE_PATH, self.cache_path, f"{idx}.pkl")

    def __getitem__(self, idx: int) -> Optional[T]:
        path = self._build_path(idx)
        if os.path.exists(path):
            with open(path, "rb") as f:
                x = pickle.load(f)
            return x
        return None

    def __setitem__(self, idx: int, value: T) -> None:
        path = self._build_path(idx)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(value, f)

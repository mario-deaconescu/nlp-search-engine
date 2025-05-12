from typing import TypeVar

import faiss

T = TypeVar("T")

def no_collate(batch: list[T]) -> T:
    assert len(batch) == 1
    return batch[0]


def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

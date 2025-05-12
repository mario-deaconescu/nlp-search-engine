from abc import ABC, abstractmethod
from typing import TypedDict, Optional, Iterable

import torch.utils.data
from rank_bm25 import BM25Okapi

from src.preprocessing.preprocess import preprocess_document
from src.utils.cache import FileCache


class Bm25Input(TypedDict):
    document: str

class Bm25Output(TypedDict):
    document: str
    preprocessed: str
    bm25: BM25Okapi

class Bm25BatchedOutput(TypedDict):
    documents: list[str]
    bm25: BM25Okapi


class BaseBm25DocumentDataset(torch.utils.data.Dataset, ABC):

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def _get_document(self, idx: int) -> str:
        pass

    @abstractmethod
    def _get_preprocessed_document(self, idx: int) -> tuple[str, str]:
        pass

    @staticmethod
    def __tokenize(x: str) -> list[str]:
        return x.split()

    def __getitem__(self, idx: int) -> Bm25Output:
        preprocessed, document = self._get_preprocessed_document(idx)
        tokenized_doc = [self.__tokenize(preprocessed)]
        bm25 = BM25Okapi(tokenized_doc)
        return {
            "document": document,
            "preprocessed": preprocessed,
            "bm25": bm25,
        }

    def get_chunked(self, idxs: Iterable[int], cached: Optional[BM25Okapi] = None) -> Bm25BatchedOutput:
        documents = [self._get_preprocessed_document(idx) for idx in idxs]
        preprocessed_docs, documents = zip(*documents)
        tokenized_corpus = [self.__tokenize(doc) for doc in preprocessed_docs]

        bm25 = BM25Okapi(tokenized_corpus) if cached is None else cached

        return {
            "documents": documents,
            "preprocessed": preprocessed_docs,
            "bm25": bm25,
        }

    def get_all(self) -> Bm25BatchedOutput:
        return self.get_chunked(range(len(self)))

class Bm25DocumentDataset(BaseBm25DocumentDataset):

    def __init__(self, documents: list[Bm25Input]):
        self.documents = documents

    def __len__(self) -> int:
        return len(self.documents)

    def _get_document(self, idx: int) -> str:
        return self.documents[idx]["document"]

    def _get_preprocessed_document(self, idx: int) -> tuple[str, str]:
        document = self._get_document(idx)
        preprocessed = preprocess_document(document)
        return preprocessed, document

class Bm25FileDocumentDataset(BaseBm25DocumentDataset):

    def __init__(self, cache: FileCache, length: int):
        self.document_cache = cache.subcache("documents")
        self.preprocessed_cache = cache.subcache("preprocessed")
        self.length = length

    def __len__(self) -> int:
        return self.length

    def _get_document(self, idx: int) -> str:
        document = self.document_cache[str(idx)]
        if document is None:
            raise ValueError(f"Document {idx} not found")
        return document

    def _get_preprocessed_document(self, idx: int) -> tuple[str, str]:
        document = self.document_cache[str(idx)]
        preprocessed = self.preprocessed_cache[str(idx)]
        if document is None or preprocessed is None:
            raise ValueError(f"Document {idx} not found")
        return preprocessed, document


class Bm25ChunkedDocumentDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: BaseBm25DocumentDataset, chunk_size: int, cache: Optional[FileCache] = None):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.cache = cache

    def __len__(self):
        return len(self.dataset) // self.chunk_size + (1 if len(self.dataset) % self.chunk_size > 0 else 0)

    def _get(self, idx: int, cached: Optional[BM25Okapi]) -> Bm25BatchedOutput:
        start_idx = idx * self.chunk_size
        end_idx = min((idx + 1) * self.chunk_size, len(self.dataset))
        return self.dataset.get_chunked(range(start_idx, end_idx), cached)

    def __getitem__(self, idx: int) -> Bm25BatchedOutput:
        cached = self.cache.get_pickled(str(idx)) if self.cache is not None else None
        result = self._get(idx, cached)
        if self.cache is not None and cached is None:
            self.cache.set_pickled(str(idx), result["bm25"])
        return result

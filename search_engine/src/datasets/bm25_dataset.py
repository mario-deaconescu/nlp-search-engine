from rank_bm25 import BM25Okapi
from typing import TypedDict, Optional, Iterable
import torch.utils.data

from src.datasets.utils import CachedList
from src.preprocessing.preprocess import preprocess_document


class Bm25Input(TypedDict):
    document: str

class Bm25Output(TypedDict):
    document: str
    bm25: BM25Okapi

class Bm25BatchedOutput(TypedDict):
    documents: list[str]
    bm25: BM25Okapi


class Bm25DocumentDataset(torch.utils.data.Dataset):

    def __init__(self, documents: list[Bm25Input]):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    @staticmethod
    def __tokenize(x: str) -> list[str]:
        return x.split()

    def __getitem__(self, idx: int) -> Bm25Output:
        document = preprocess_document(self.documents[idx]["document"])
        tokenized_doc = [self.__tokenize(document)]
        bm25 = BM25Okapi(tokenized_doc)
        return {
            "document": document,
            "bm25": bm25,
        }

    def get_document(self, idx: int) -> str:
        return self.documents[idx]["document"]

    def get_chunked(self, idxs: Iterable[int], cached: Optional[BM25Okapi] = None) -> Bm25BatchedOutput:
        documents = [self.documents[idx]["document"] for idx in idxs]
        documents = list(map(preprocess_document, documents))
        tokenized_corpus = [self.__tokenize(doc) for doc in documents]

        bm25 = BM25Okapi(tokenized_corpus) if cached is None else cached

        return {
            "documents": documents,
            "bm25": bm25,
        }

    def get_all(self) -> Bm25BatchedOutput:
        return self.get_chunked(range(len(self.documents)))



class Bm25ChunkedDocumentDataset(torch.utils.data.Dataset):

    def __init__(self, documents: list[Bm25Input], chunk_size: int, cache_path: Optional[str] = None):
        self.dataset = Bm25DocumentDataset(documents)
        self.chunk_size = chunk_size
        if cache_path is not None:
            self.cache: CachedList[BM25Okapi] | None = CachedList(
                cache_path=cache_path,
                length=len(self)
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.dataset) // self.chunk_size + (1 if len(self.dataset) % self.chunk_size > 0 else 0)

    def _get(self, idx: int, cached: Optional[BM25Okapi]) -> Bm25BatchedOutput:
        start_idx = idx * self.chunk_size
        end_idx = min((idx + 1) * self.chunk_size, len(self.dataset))
        return self.dataset.get_chunked(range(start_idx, end_idx), cached)

    def __getitem__(self, idx: int) -> Bm25BatchedOutput:
        cached = self.cache[idx] if self.cache is not None else None
        result = self._get(idx, cached)
        if self.cache is not None and cached is None:
            self.cache[idx] = result["bm25"]
        return result

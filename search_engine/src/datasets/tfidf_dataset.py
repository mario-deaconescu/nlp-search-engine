import time
from abc import ABC, abstractmethod
from typing import TypedDict, Optional, Iterable

import torch.utils.data
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.preprocess import preprocess_document
from src.utils.cache import FileCache


class TfIdfInput(TypedDict):
    document: str

class TfIdfOutput(TypedDict):
    document: str
    preprocessed: str
    tfidf: csr_matrix
    vectorizer: TfidfVectorizer

class TfIdfBatchedOutput(TypedDict):
    documents: list[str]
    preprocessed: list[str]
    tfidf: csr_matrix
    vectorizer: TfidfVectorizer

class TfIdfBaseDocumentDataset(torch.utils.data.Dataset, ABC):

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
    def __preprocess(x):
        return x

    @staticmethod
    def __tokenize(x):
        return x.split()

    def __getitem__(self, idx: int, cached: Optional[tuple[TfidfVectorizer, csr_matrix]] = None) -> TfIdfOutput:
        preprocessed, document = self._get_preprocessed_document(idx)
        if cached is None:
            vectorizer = TfidfVectorizer(
                preprocessor=TfIdfBaseDocumentDataset.__preprocess,
                tokenizer=TfIdfBaseDocumentDataset.__tokenize,
                token_pattern=None,
                lowercase=False)
            tfidf = vectorizer.fit_transform(preprocessed)
        else:
            vectorizer, tfidf = cached
        return {
            "document": document,
            "preprocessed": preprocessed,
            "tfidf": tfidf, # type: ignore
            "vectorizer": vectorizer,
        }

    def get_chunked(self, idxs: Iterable[int], cached: Optional[tuple[TfidfVectorizer, csr_matrix]] = None) -> TfIdfBatchedOutput:
        # start_time = time.time()
        documents = [self._get_preprocessed_document(idx) for idx in idxs]
        preprocessed, documents = zip(*documents)
        # print(f"Preprocessing took {time.time() - start_time:.2f} seconds")

        if cached is None:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(preprocessed) # type: ignore
        else:
            vectorizer, tfidf = cached

        return {
            "documents": documents,
            "preprocessed": preprocessed,
            "tfidf": tfidf, # type: ignore
            "vectorizer": vectorizer,
        }

    def get_all(self) -> TfIdfBatchedOutput:
        return self.get_chunked(range(len(self)))

class TfIdfDocumentDataset(TfIdfBaseDocumentDataset):
    def __init__(self, documents: list[TfIdfInput]):
        self.documents = documents

    def __len__(self) -> int:
        return len(self.documents)

    def _get_document(self, idx: int) -> str:
        return self.documents[idx]["document"]

    def _get_preprocessed_document(self, idx: int) -> tuple[str, str]:
        document = self.documents[idx]["document"]
        preprocessed = preprocess_document(document)
        return preprocessed, document

class TfIdfFileDocumentDataset(TfIdfBaseDocumentDataset):
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


class TfIdfChunkedDocumentDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: TfIdfBaseDocumentDataset, chunk_size: int, cache: Optional[FileCache] = None):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.cache = cache

    def __len__(self):
        return len(self.dataset) // self.chunk_size + (1 if len(self.dataset) % self.chunk_size > 0 else 0)

    def _get(self, idx: int, cached: Optional[tuple[TfidfVectorizer, csr_matrix]]) -> TfIdfBatchedOutput:
        start_idx = idx * self.chunk_size
        end_idx = min((idx + 1) * self.chunk_size, len(self.dataset))
        return self.dataset.get_chunked(range(start_idx, end_idx), cached)

    def __getitem__(self, idx: int) -> TfIdfBatchedOutput:
        cached = None
        if self.cache is not None:
            cached = self.cache.get_pickled(str(idx))

        result = self._get(idx, cached)
        if self.cache is not None and cached is None:
            self.cache.set_pickled(str(idx), (result["vectorizer"], result["tfidf"]))

        return result

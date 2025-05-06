from typing import TypedDict, Optional, Iterable

import torch.utils.data
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer

from search_engine.src.datasets.utils import CachedList
from search_engine.src.preprocessing.preprocess import nlp, preprocess_document
from scipy.sparse import csr_matrix

from search_engine.src.preprocessing.vectorizer import StemmedTfidfVectorizer


class TfIdfInput(TypedDict):
    document: str

class TfIdfOutput(TypedDict):
    document: str
    tfidf: csr_matrix
    vectorizer: TfidfVectorizer

class TfIdfBatchedOutput(TypedDict):
    documents: list[str]
    tfidf: csr_matrix
    vectorizer: TfidfVectorizer

class TfIdfDocumentDataset(torch.utils.data.Dataset):

    def __init__(self, documents: list[TfIdfInput]):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    @staticmethod
    def __preprocess(x):
        return x

    @staticmethod
    def __tokenize(x):
        return x.split()

    def __getitem__(self, idx: int) -> TfIdfOutput:
        document = self.documents[idx]["document"]
        document = preprocess_document(document)
        vectorizer = TfidfVectorizer(
            preprocessor=TfIdfDocumentDataset.__preprocess,
            tokenizer=TfIdfDocumentDataset.__tokenize,
            token_pattern=None,
            lowercase=False)
        tfidf = vectorizer.fit_transform(document)
        return {
            "document": document,
            "tfidf": tfidf, # type: ignore
            "vectorizer": vectorizer,
        }

    def get_document(self, idx: int) -> str:
        return self.documents[idx]["document"]

    def get_chunked(self, idxs: Iterable[int], cached: Optional[tuple[TfidfVectorizer, csr_matrix]] = None) -> TfIdfBatchedOutput:
        documents = [self.documents[idx]["document"] for idx in idxs]
        documents = list(map(preprocess_document, documents))

        if cached is None:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(documents) # type: ignore
        else:
            vectorizer, tfidf = cached

        return {
            "documents": documents,
            "tfidf": tfidf, # type: ignore
            "vectorizer": vectorizer,
        }

    def get_all(self) -> TfIdfBatchedOutput:
        return self.get_chunked(range(len(self.documents)))

class TfIdfChunkedDocumentDataset(torch.utils.data.Dataset):

    def __init__(self, documents: list[TfIdfInput], chunk_size: int, cache_path: Optional[str] = None):
        self.dataset = TfIdfDocumentDataset(documents)
        self.chunk_size = chunk_size
        if cache_path is not None:
            self.cache: CachedList[tuple[TfidfVectorizer, csr_matrix]] | None = CachedList(
                cache_path=cache_path,
                length=len(self)
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.dataset) // self.chunk_size + (1 if len(self.dataset) % self.chunk_size > 0 else 0)

    def _get(self, idx: int, cached: Optional[tuple[TfidfVectorizer, csr_matrix]]) -> TfIdfBatchedOutput:
        start_idx = idx * self.chunk_size
        end_idx = min((idx + 1) * self.chunk_size, len(self.dataset))
        return self.dataset.get_chunked(range(start_idx, end_idx), cached)

    def __getitem__(self, idx: int) -> TfIdfBatchedOutput:
        cached = None
        if self.cache is not None:
            cached = self.cache[idx]

        result = self._get(idx, cached)
        if self.cache is not None and cached is None:
            self.cache[idx] = result["vectorizer"], result["tfidf"]

        return result

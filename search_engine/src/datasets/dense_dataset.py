from abc import ABC, abstractmethod
from typing import TypedDict, Optional, Iterable
import numpy as np
from faiss import Index, IndexFlatIP,  write_index, read_index

import torch.utils.data

from src.preprocessing.preprocess import nlp, preprocess_document
from sentence_transformers import SentenceTransformer

from src.constants import EMBEDDINGS_DIMENSION
from src.utils.cache import FileCache


class DenseInput(TypedDict):
    document: str

class DenseOutput(TypedDict):
    document: str
    preprocessed: str
    embedding: np.ndarray
    model: Optional[SentenceTransformer]

class DenseBatchedOutput(TypedDict):
    documents: list[str]
    embeddings: np.ndarray  # shape: (batch_size, embedding_dim)
    model: SentenceTransformer

class BaseDenseDocumentDataset(torch.utils.data.Dataset, ABC):

    def __len__(self):
        return len(self.documents)

    @staticmethod
    def __preprocess(x):
        return x

    @staticmethod
    def __tokenize(x):
        return x.split()

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def _get_document(self, idx: int) -> str:
        pass

    @abstractmethod
    def _get_preprocessed_document(self, idx: int) -> tuple[str, str]:
        pass

    def __getitem__(self, idx: int) -> DenseOutput:
        preprocessed, document = self._get_preprocessed_document(idx)
        # Don't encode here; batching will happen in get_chunked
        return {
            "document": document,
            "preprocessed": preprocessed,
            "embedding": np.array([]),  # placeholder, not used
            "model": None,  # placeholder, not used
        }

    def get_chunked(self, idxs: Iterable[int], model: SentenceTransformer, cached: Optional[np.ndarray] = None) -> DenseBatchedOutput:
        documents = [self._get_preprocessed_document(idx) for idx in idxs]
        preprocessed, documents = zip(*documents)

        if cached is None:
            embeddings = model.encode(
                preprocessed,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=True
            )
        else:
            embeddings = cached

        return {
            "documents": documents,
            "preprocessed": preprocessed,
            "embeddings": embeddings,
            "model": model,
        }

    def get_all(self, model: SentenceTransformer) -> DenseBatchedOutput:
        return self.get_chunked(range(len(self)), model)

class DenseDocumentDataset(BaseDenseDocumentDataset):

    def __init__(self, documents: list[DenseInput]):
        self.documents = documents

    def _get_document(self, idx: int) -> str:
        return self.documents[idx]["document"]

    def _get_preprocessed_document(self, idx: int) -> tuple[str, str]:
        document = self._get_document(idx)
        preprocessed = preprocess_document(document)
        return preprocessed, document

    def __len__(self) -> int:
        return len(self.documents)

class DenseFileDocumentDataset(BaseDenseDocumentDataset):

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


class DenseChunkedDocumentDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: BaseDenseDocumentDataset, chunk_size: int, cache: Optional[FileCache] = None, index_path: Optional[str] = None):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.index_path = index_path

        self.cache = cache

        self.index: Optional[IndexFlatIP] = None

        if index_path is not None:
            try:
                self.index : Optional[IndexFlatIP] | None = read_index(index_path)  # Try loading the existing index
            except Exception as e:
                print(f"Could not load index from {index_path}. Error: {e}")
                self.index = None
       
        if self.index is None:  # If no index found, create a new one
            self.index = IndexFlatIP(EMBEDDINGS_DIMENSION)


    def __len__(self):
        return len(self.dataset) // self.chunk_size + (1 if len(self.dataset) % self.chunk_size > 0 else 0)

    def _get(self, idx: int, model: SentenceTransformer, cached: Optional[np.ndarray]) -> DenseBatchedOutput:
        start_idx = idx * self.chunk_size
        end_idx = min((idx + 1) * self.chunk_size, len(self.dataset))
        return self.dataset.get_chunked(range(start_idx, end_idx), model, cached)


    def get_chunk(self, idx: int, model: SentenceTransformer) -> DenseBatchedOutput:
        cached = None
        if self.cache is not None:
            cached = self.cache.get_pickled(str(idx))

        result = self._get(idx, model, cached)


        # Insert into FAISS index if provided
        if self.index is not None:
            self.index.add(result["embeddings"].astype(np.float32))  # FAISS requires float32
            self.save_index()

        if self.cache is not None and cached is None:
            self.cache.set_pickled(str(idx), result["embeddings"])

        return result

    def save_index(self):
        """ Save the FAISS index to disk """
        if self.index and self.index_path:
            write_index(self.index, self.index_path)
            print(f"FAISS index saved to {self.index_path}")
        else:
            print("No index to save.")

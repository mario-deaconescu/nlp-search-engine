from typing import TypedDict, Optional, Iterable
import numpy as np
from faiss import Index, IndexFlatIP,  write_index, read_index

import torch.utils.data

from search_engine.src.datasets.utils import CachedList
from search_engine.src.preprocessing.preprocess import nlp, preprocess_document
from sentence_transformers import SentenceTransformer

from constants import EMBEDDINGS_DIMENSION


class DenseInput(TypedDict):
    document: str

class DenseOutput(TypedDict):
    document: str
    embedding: np.ndarray

class DenseBatchedOutput(TypedDict):
    documents: list[str]
    embeddings: np.ndarray  # shape: (batch_size, embedding_dim)

class DenseDocumentDataset(torch.utils.data.Dataset):

    def __init__(self, documents: list[DenseInput]):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    @staticmethod
    def __preprocess(x):
        return x

    @staticmethod
    def __tokenize(x):
        return x.split()

    def __getitem__(self, idx: int) -> DenseOutput:
        document = self.documents[idx]["document"]
        document = preprocess_document(document)
        # Don't encode here; batching will happen in get_chunked
        return {
            "document": document,
            "embedding": np.array([]),  # placeholder, not used
        }

    def get_document(self, idx: int) -> str:
        return self.documents[idx]["document"]

    def get_chunked(self, idxs: Iterable[int], model: SentenceTransformer, cached: Optional[np.ndarray] = None) -> DenseBatchedOutput:
        documents = [self.documents[idx]["document"] for idx in idxs]
        documents = list(map(preprocess_document, documents))

        if cached is None:
            embeddings = model.encode(
                documents,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=True
            )
        else:
            embeddings = cached

        return {
            "documents": documents,
            "embeddings": embeddings,
        }

    def get_all(self) -> DenseBatchedOutput:
        return self.get_chunked(range(len(self.documents)))

class DenseChunkedDocumentDataset(torch.utils.data.Dataset):

    def __init__(self, documents: list[DenseInput], chunk_size: int, cache_path: Optional[str] = None, index_path: Optional[str] = None):
        self.dataset = DenseDocumentDataset(documents)
        self.chunk_size = chunk_size
        self.index_path = index_path

        if cache_path is not None:
            self.cache: CachedList[np.ndarray] | None = CachedList(
                cache_path=cache_path,
                length=len(self)
            )
        else:
            self.cache = None

        if index_path is not None:
            try:
                self.index : Optional[IndexFlatIP] | None = read_index(index_path)  # Try loading the existing index
            except Exception as e:
                print(f"Could not load index from {index_path}. Error: {e}")
                self.index = None
        else:
            self.index = None

        if self.index is None:  # If no index found, create a new one
            self.index = IndexFlatIP(EMBEDDINGS_DIMENSION)


    def __len__(self):
        return len(self.dataset) // self.chunk_size + (1 if len(self.dataset) % self.chunk_size > 0 else 0)

    def _get(self, idx: int, model: SentenceTransformer, cached: Optional[np.ndarray]) -> DenseBatchedOutput:
        start_idx = idx * self.chunk_size
        end_idx = min((idx + 1) * self.chunk_size, len(self.dataset))
        return self.dataset.get_chunked(range(start_idx, end_idx), model, cached)


    from faiss import Index

    def get_chunk(self, idx: int, model: SentenceTransformer) -> DenseBatchedOutput:
        cached = None
        if self.cache is not None:
            cached = self.cache[idx]

        result = self._get(idx, model, cached)


        # Insert into FAISS index if provided
        if self.index is not None:
            self.index.add(result["embeddings"].astype(np.float32))  # FAISS requires float32
            self.save_index()

        if self.cache is not None and cached is None:
            self.cache[idx] = result["embeddings"]

        return result

    def save_index(self):
        """ Save the FAISS index to disk """
        if self.index and self.index_path:
            write_index(self.index, self.index_path)
            print(f"FAISS index saved to {self.index_path}")
        else:
            print("No index to save.")

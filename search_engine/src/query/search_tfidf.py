import pickle
import time
from multiprocessing.managers import ListProxy
from multiprocessing.synchronize import Lock
from typing import Iterable, Optional, TypedDict, Generator
from multiprocessing import Pool, Manager

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from src.datasets.tfidf_dataset import TfIdfDocumentDataset, TfIdfBatchedOutput, TfIdfChunkedDocumentDataset
from src.datasets.utils import no_collate
from src.preprocessing.preprocess import preprocess_document


class SearchResult(TypedDict):
    document: str
    score: float
    id: int


def search_tfidf_chunked(args: tuple[str, TfIdfChunkedDocumentDataset, int, ListProxy, Lock]) -> list[SearchResult]:
    query = args[0]
    dataset = args[1]
    idx = args[2]
    global_results = args[3]
    lock = args[4]

    chunk = dataset[idx]
    tfidf = chunk['tfidf']
    vectorizer = chunk['vectorizer']
    documents = chunk['documents']
    chunk_size = len(documents)
    query = preprocess_document(query)

    query_tfidf = vectorizer.transform([query])

    # 6. Compute cosine similarity
    cosine_similarities: list[float] = cosine_similarity(query_tfidf, tfidf).flatten()

    # 7. Rank sentences
    ranked_indices = np.argsort(cosine_similarities)[::-1]

    ranked_documents: list[SearchResult] = [{
        'document': documents[i],
        'score': cosine_similarities[i],
        'id': idx * chunk_size + i,
    } for i in ranked_indices]
    # 8. Lock
    ranked_documents.sort(key=lambda x: x['score'], reverse=True)
    with lock:
        global_results.extend(ranked_documents[:chunk_size])

        # 8. Sort by score
        tmp = sorted(global_results, key=lambda x: x['score'], reverse=True)[:chunk_size]
        global_results[:] = tmp
    return tmp

def search_tfidf(query: str, dataset: TfIdfChunkedDocumentDataset) -> Generator[list[SearchResult], None, None]:
    if dataset.cache is not None and len(dataset.cache.subkeys()) > 0:
        multiprocessing = False
    else:
        multiprocessing = True
    with Manager() as manager:
        results = manager.list()
        lock = manager.Lock()
        pickled_dataset = pickle.dumps(dataset)
        if multiprocessing:
            iterable = [
                (query,
                 dataset,
                 i,
                 results,
                 lock) for i in range(len(dataset))
            ]
            with Pool() as pool:
                for result in pool.imap_unordered(search_tfidf_chunked, iterable):
                    yield result
        else:
            for i in range(len(dataset)):
                yield search_tfidf_chunked((query, dataset, i, results, lock))

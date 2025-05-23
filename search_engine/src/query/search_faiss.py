import pickle
from multiprocessing.managers import ListProxy
from multiprocessing.synchronize import Lock
from typing import Iterable, Optional, TypedDict, Generator
from multiprocessing import Pool, Manager

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from src.datasets.dense_dataset import DenseDocumentDataset, DenseBatchedOutput, DenseChunkedDocumentDataset
from src.datasets.utils import no_collate
from src.preprocessing.preprocess import preprocess_document

from sentence_transformers import SentenceTransformer

from src.utils.l2_normalizer import l2_normalize

from faiss import read_index
from src.constants import INDEX_PATH, TOP_K


class SearchResult(TypedDict):
    document: str
    score: float
    id: int


def search_faiss_chunked(args: tuple[str, DenseChunkedDocumentDataset, int, list[SearchResult], Lock, SentenceTransformer, int]) -> list[SearchResult]:
    query = args[0]
    dataset = args[1]
    idx = args[2]
    global_results = args[3]
    lock = args[4]
    model = args[5]
    top_k = args[6]

    chunk = dataset.get_chunk(idx, model)

    documents = chunk['documents']
    #model = chunk['model']
    # faiss_index = read_index(INDEX_PATH)
    faiss_index = dataset.index

    chunk_size = len(documents)
    query = preprocess_document(query)


    query_encoded = model.encode(
        [query],
        convert_to_numpy=True,
    )
    normalized_query = l2_normalize(query_encoded)

    distances, ranked_indices = faiss_index.search(normalized_query, k=top_k)

    ranked_documents: list[SearchResult] = [{
        'id': idx*chunk_size + i,
        'document': documents[i],
        'score': distances[0][iteration]
    } for iteration, i in enumerate(ranked_indices[0])]

    # 8. Lock
    with lock:
        global_results.extend(ranked_documents[:chunk_size])

        # 8. Sort by score
        tmp = sorted(global_results, key=lambda x: x['score'], reverse=True)[:chunk_size]
        global_results[:] = tmp

    return tmp

def search_faiss(query: str, dataset: DenseChunkedDocumentDataset, model: SentenceTransformer, top_k : int) -> Generator[list[SearchResult], None, None]:
    if dataset.cache is not None and len(dataset.cache.subkeys()) > 0:
        multiprocessing = False
    else:
        multiprocessing = True

    multiprocessing = False

    multiprocessing = False
    with Manager() as manager:
        results = manager.list()
        lock = manager.Lock()
        if multiprocessing:
            iterable = [
                (query,
                 dataset,
                 i,
                 results,
                 lock,
                 model,
                 top_k) for i in range(len(dataset))
            ]
            with Pool() as pool:
                for result in pool.imap_unordered(search_faiss_chunked, iterable):
                    yield result
        else:
            for i in range(len(dataset)):
                result = search_faiss_chunked((query, dataset, i, results, lock, model, top_k))
                yield result

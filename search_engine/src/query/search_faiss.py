import pickle
from multiprocessing.managers import ListProxy
from multiprocessing.synchronize import Lock
from typing import Iterable, Optional, TypedDict, Generator
from multiprocessing import Pool, Manager

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from search_engine.src.datasets.dense_dataset import DenseDocumentDataset, DenseBatchedOutput, DenseChunkedDocumentDataset
from search_engine.src.datasets.utils import no_collate
from search_engine.src.preprocessing.preprocess import preprocess_document

from sentence_transformers import SentenceTransformer

from utils.l2_normalizer import l2_normalize

from faiss import read_index
from constants import INDEX_PATH


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

    # with open(f'.cache/articles/{idx}.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # print(data)


    chunk = dataset.get_chunk(idx)

    documents = chunk['documents']
    model = chunk['model']
    faiss_index = read_index(INDEX_PATH)

    chunk_size = len(documents)
    query = preprocess_document(query)


    query_encoded = model.model.encode(
        [query],
        convert_to_numpy=True,
    )
    normalized_query = l2_normalize(query_encoded)

    distances, ranked_indices = faiss_index.search(normalized_query, k=top_k)

    ranked_documents: list[SearchResult] = [{
        'id': idx*chunk_size + i,
        'document': documents[i],
        'score': distances[i]
    } for i in ranked_indices]

    # 8. Lock
    with lock:
        global_results.extend(ranked_documents[:chunk_size])

        # 8. Sort by score
        tmp = sorted(global_results, key=lambda x: x['score'], reverse=True)[:chunk_size]
        global_results[:] = tmp
    return tmp

def search_faiss(query: str, dataset: DenseChunkedDocumentDataset, model: SentenceTransformer, top_k : int) -> Generator[list[SearchResult], None, None]:
    with Manager() as manager:
        results = manager.list()
        lock = manager.Lock()
        print("Making iterable...")
        iterable = [
            (query,
             dataset,
             i,
             results,
             lock,
             model,
             top_k) for i in range(len(dataset))
        ]
        print("Creating Pool...")
        with Pool() as pool:
            print("Searching...")
            for result in pool.imap_unordered(search_faiss_chunked, iterable):
                yield result

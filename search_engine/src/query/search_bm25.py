from multiprocessing import Pool, Manager
from typing import Generator, TypedDict
from multiprocessing.synchronize import Lock
from src.datasets.bm25_dataset import Bm25ChunkedDocumentDataset
from src.preprocessing.preprocess import preprocess_document
from src.query.search_tfidf import SearchResult
import numpy as np



def search_bm25_chunked(args: tuple[str, Bm25ChunkedDocumentDataset, int, list[SearchResult], Lock]) -> list[SearchResult]:
    query = args[0]
    dataset = args[1]
    idx = args[2]
    global_results = args[3]
    lock = args[4]


    chunk = dataset[idx]
    bm25 = chunk['bm25']
    documents = chunk['documents']
    chunk_size = len(documents)
    query = preprocess_document(query)


    query_bm25 = bm25.transform([query])

    # 6. Compute document scores
    scores = bm25.get_scores(query_bm25)

    # 7. Rank sentences
    ranked_indices = np.argsort(scores)[::-1]
    
    ranked_documents: list[SearchResult] = [{
        'id': idx*chunk_size + i,
        'document': documents[i],
        'score': scores[i]
    } for i in ranked_indices]
    # 8. Lock
    with lock:
        global_results.extend(ranked_documents[:chunk_size])

        # 8. Sort by score
        tmp = sorted(global_results, key=lambda x: x['score'], reverse=True)[:chunk_size]
        global_results[:] = tmp
    return tmp


def search_bm25(query: str, dataset: Bm25ChunkedDocumentDataset) -> Generator[list[SearchResult], None, None]:
    with Manager() as manager:
        results = manager.list()
        lock = manager.Lock()
        print("Making iterable...")
        iterable = [
            (query,
             dataset,
             i,
             results,
             lock) for i in range(len(dataset))
        ]
        print("Creating Pool...")
        with Pool() as pool:
            print("Searching...")
            for result in pool.imap_unordered(search_bm25_chunked, iterable):
                yield result

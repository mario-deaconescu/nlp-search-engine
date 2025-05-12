import json
import os
import shutil
from typing import Generator

import fitz
import torch
from sentence_transformers import SentenceTransformer

from src.constants import TOP_K
from src.datasets.bm25_dataset import Bm25ChunkedDocumentDataset
from src.datasets.dense_dataset import DenseChunkedDocumentDataset
from src.datasets.tfidf_dataset import TfIdfChunkedDocumentDataset
from src.query.search_bm25 import search_bm25
from src.query.search_faiss import search_faiss
from src.query.search_tfidf import search_tfidf


def clear_cache_dir(cache_path: str = "articles"):
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)  # Delete the directory and all contents
        os.makedirs(cache_path)


def extract_text_from_pdf(contents: bytes) -> Generator[str, None, None]:
    """
    Extract text from a PDF file.
    :param contents: The PDF file contents to extract text from.
    :return: List with extracted text per page.
    """

    doc = fitz.open(stream=contents, filetype="pdf")
    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text()
        yield text


def search_in_dataset(dataset: TfIdfChunkedDocumentDataset | DenseChunkedDocumentDataset | Bm25ChunkedDocumentDataset, search: str) -> Generator[str, None, None]:
    if isinstance(dataset, TfIdfChunkedDocumentDataset):
        generator = search_tfidf(search, dataset)
    elif isinstance(dataset, DenseChunkedDocumentDataset):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        generator = search_faiss(search, dataset, model, TOP_K)
    elif isinstance(dataset, Bm25ChunkedDocumentDataset):
        generator = search_bm25(search, dataset)

    length = len(dataset)

    for page, results in enumerate(generator):
        data = {
            "results": [{
                "page": int(result["id"]) + 1,
                "score": float(result["score"])
            } for result in results],
            "current": page,
            "total": length
        }
        yield f"data: {json.dumps(data)}\n\n"

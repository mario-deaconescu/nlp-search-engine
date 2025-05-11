import os
import shutil
import fitz
import json

from src.datasets.tfidf_dataset import TfIdfChunkedDocumentDataset
from src.datasets.dense_dataset import DenseChunkedDocumentDataset
from src.query.search_tfidf import search_tfidf
from src.query.search_faiss import search_faiss

def clear_cache_dir(cache_path: str = "articles"):
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)  # Delete the directory and all contents
        os.makedirs(cache_path)

def extract_text_from_pdf(contents: bytes) -> list[str]:
    """
    Extract text from a PDF file.
    :param contents: The PDF file contents to extract text from.
    :return: List with extracted text per page.
    """

    text_by_page = []

    doc = fitz.open(stream=contents, filetype="pdf")
    for page_number in range(len(doc)):

        page = doc[page_number]
        text = page.get_text()
        text_by_page.append(text)

    return text_by_page


def search_in_dataset(dataset: TfIdfChunkedDocumentDataset | DenseChunkedDocumentDataset, search: str):
    if isinstance(dataset, TfIdfChunkedDocumentDataset):
        search_results = search_tfidf(search, dataset)
    else:
        search_results = search_faiss(search, dataset)

    for result in search_results:
        data = {
            "page": int(result[0]["id"]),
            "score": float(result[0]["score"]),
        }
        yield f"data: {json.dumps(data)}\n\n"

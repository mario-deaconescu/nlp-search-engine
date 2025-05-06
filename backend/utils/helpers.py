import os
import shutil
import fitz
import json


from search_engine.src.datasets.tfidf_dataset import TfIdfChunkedDocumentDataset
from search_engine.src.query.search_tfidf import search_tfidf

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
    

def search_in_dataset(dataset: TfIdfChunkedDocumentDataset, search: str):
    
    search_results = search_tfidf(search, dataset)
    
    for result in search_results:
        # print(f"Found result: {result[0]['id']} with score: {result[0]['score']} document: {result[0]['document']}")
        data = {
            "page": int(result[0]["id"]),
            "score": float(result[0]["score"]),
        }   
        yield f"data: {json.dumps(data)}\n\n"
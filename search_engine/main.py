import json
import os
import pickle
import time
import uuid
from multiprocessing import Pool

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.constants import CHUNK_SIZE
from src.datasets.dense_dataset import DenseChunkedDocumentDataset
from src.datasets.tfidf_dataset import TfIdfChunkedDocumentDataset, TfIdfFileDocumentDataset
from src.dependencies import get_file_cache
from src.preprocessing.preprocess import preprocess_document
from src.preprocessing.string_list_utils import preprocess_string_list
from src.utils.cache import make_hash, FileCache
from src.utils.helpers import extract_text_from_pdf, search_in_dataset

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_and_cache(file_cache: FileCache, key: str) -> None:
    if file_cache.exists(f"preprocessed:{key}"):
        return

    document = file_cache.get(f"documents:{key}")
    if document is None:
        raise ValueError(f"Document {key} not found in cache.")
    preprocessed = preprocess_document(document)
    file_cache.set(f"preprocessed:{key}", preprocessed)


@app.post("/upload")
async def upload(file: UploadFile = File(...), file_cache: FileCache = Depends(get_file_cache)):
    # clear_cache_dir(BASE_CACHE_DIR)
    if file.filename is None or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_hash = await make_hash(file)

    contents = await file.read()
    content_pages = extract_text_from_pdf(contents)
    pdf_cache = file_cache.subcache(f"pdf:{file_hash}")

    for i, page in enumerate(content_pages):
        pdf_cache.set(f"documents:{i}", page)

    pdf_cache.set("length", str(len(content_pages)))
    # preprocessed_pages = preprocess_string_list(content_pages)

    session_id = str(uuid.uuid4())

    file_cache.set(f"session:{session_id}", file_hash)

    return {"session_id": session_id}


@app.get("/search-tf-idf")
async def search(session_id: str, search: str, file_cache: FileCache = Depends(get_file_cache)):
    """
    Search for a string in a PDF file.
    :param file: The PDF file to search in.
    :param search: The string to search for.
    :return: A JSON response with the number of pages in the PDF.
    """

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    print(f"Received search request for session {session_id} with search string: {search}")

    file_hash = file_cache.get(f"session:{session_id}")

    if not file_hash:
        raise HTTPException(status_code=404, detail="Session not found.")

    pdf_cache = file_cache.subcache(f"pdf:{file_hash}")
    length = pdf_cache.get("length")
    if length is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    length = int(length)

    existing_preprocessed = set(pdf_cache.subkeys("preprocessed"))
    print(existing_preprocessed)
    missing_preprocessed = [str(i) for i in range(length) if str(i) not in existing_preprocessed]
    print(missing_preprocessed)
    start = time.time()
    with Pool() as pool:
        print(len(pickle.dumps(pdf_cache)))
        pool.starmap(preprocess_and_cache, [
            (pdf_cache, str(i)) for i in missing_preprocessed
        ], chunksize=50)
    end = time.time()
    print(f"Preprocessing took {end - start:.2f} seconds")

    base_dataset = TfIdfFileDocumentDataset(pdf_cache, length=length)

    tfidf_cache = file_cache.subcache(f"tfidf:{file_hash}")

    dataset = TfIdfChunkedDocumentDataset(base_dataset, chunk_size=CHUNK_SIZE, cache=tfidf_cache)

    tfidf_results = search_in_dataset(dataset, search)

    # Modify this
    return StreamingResponse(tfidf_results, media_type="text/event-stream")


@app.get("/search-faiss")
async def search_faiss(session_id: str, search: str):
    """
    Search for a string in a PDF file using FAISS.
    :param session_id: The session ID for the uploaded PDF.
    :param search: The string to search for.
    :return: A JSON response with the search results.
    """

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    print(f"Received FAISS search request for session {session_id} with search string: {search}")

    session_path = os.path.join(PREPROCESSING_CACHE_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found.")

    # Load preprocessed_text
    with open(session_path, "r", encoding="utf-8") as f:
        preprocessed_text = json.load(f)
    dataset = DenseChunkedDocumentDataset(preprocessed_text, chunk_size=CHUNK_SIZE, cache_path='dense_articles')
    dense_results = search_in_dataset(dataset, search)
    # Modify this
    return StreamingResponse(dense_results, media_type="text/event-stream")

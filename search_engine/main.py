import asyncio
import json
import multiprocessing
import os
import pickle
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from multiprocessing import Pool
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.constants import CHUNK_SIZE
from src.datasets.dense_dataset import DenseChunkedDocumentDataset, DenseFileDocumentDataset
from src.datasets.tfidf_dataset import TfIdfChunkedDocumentDataset, TfIdfFileDocumentDataset
from src.dependencies import get_file_cache
from src.preprocessing.preprocess import preprocess_document
from src.preprocessing.string_list_utils import preprocess_string_list
from src.utils import pool_executor
from src.utils.cache import make_hash, FileCache
from src.utils.helpers import extract_text_from_pdf, search_in_dataset


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool_executor.executor = ProcessPoolExecutor()
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(pool_executor.executor, pool_executor.__init__worker) for _ in range(10)]
    yield
    pool_executor.executor.shutdown(wait=True)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_and_cache(args: tuple[FileCache, str]) -> None:
    file_cache, key = args
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
    content_pages = list(extract_text_from_pdf(contents))
    pdf_cache = file_cache.subcache(f"pdf:{file_hash}")

    for i, page in enumerate(content_pages):
        pdf_cache.set(f"documents:{i}", page)

    pdf_cache.set("length", str(len(content_pages)))
    # preprocessed_pages = preprocess_string_list(content_pages)

    session_id = str(uuid.uuid4())

    file_cache.set(f"session:{session_id}", file_hash)

    return {"session_id": session_id}


async def _search(session_id: str, search: str, file_cache: FileCache, mode: Literal['tfidf', 'faiss']):
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
    list(pool_executor.executor.map(preprocess_and_cache, [
        (pdf_cache, str(i)) for i in missing_preprocessed
    ], chunksize=50))
    end = time.time()
    print(f"Preprocessing took {end - start:.2f} seconds")

    if mode == 'tfidf':
        base_dataset = TfIdfFileDocumentDataset(pdf_cache, length=length)
        cache = file_cache.subcache(f"tfidf:{file_hash}")
        dataset = TfIdfChunkedDocumentDataset(base_dataset, chunk_size=CHUNK_SIZE, cache=cache)
    elif mode == 'faiss':
        base_dataset = DenseFileDocumentDataset(pdf_cache, length=length)
        cache = file_cache.subcache(f"faiss:{file_hash}")
        dataset = DenseChunkedDocumentDataset(base_dataset, chunk_size=CHUNK_SIZE, cache=cache)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'tfidf' or 'faiss'.")

    results = search_in_dataset(dataset, search)

    # Modify this
    return StreamingResponse(results, media_type="text/event-stream")


@app.get("/search-tf-idf")
async def search_tfidf(session_id: str, search: str, file_cache: FileCache = Depends(get_file_cache)):
    """
    Search for a string in a PDF file.
    :param file: The PDF file to search in.
    :param search: The string to search for.
    :param file_cache: The file cache to use for storing and retrieving the PDF.
    :return: A JSON response with the number of pages in the PDF.
    """

    return await _search(session_id, search, file_cache, mode='tfidf')


@app.get("/search-faiss")
async def search_faiss(session_id: str, search: str, file_cache: FileCache = Depends(get_file_cache)):
    """
    Search for a string in a PDF file using FAISS.
    :param session_id: The session ID for the uploaded PDF.
    :param search: The string to search for.
    :param file_cache: The file cache to use for storing and retrieving the PDF.
    :return: A JSON response with the search results.
    """

    return await _search(session_id, search, file_cache, mode='faiss')

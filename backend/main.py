import uuid
import json
import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from search_engine.src.preprocessing.string_list_utils import preprocess_string_list
from search_engine.src.datasets.tfidf_dataset import TfIdfChunkedDocumentDataset

from backend.utils.helpers import extract_text_from_pdf, search_in_dataset, clear_cache_dir
from backend.utils.constants import CHUNK_SIZE, BASE_CACHE_DIR, PREPROCESSING_CACHE_DIR, FRONTEND_URL

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(BASE_CACHE_DIR, exist_ok=True)

@app.post("/upload") 
async def upload(file: UploadFile = File(...)):

    clear_cache_dir(BASE_CACHE_DIR)
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    contents = await file.read()
    extracted_text = extract_text_from_pdf(contents)
    preprocessed_text = preprocess_string_list(extracted_text)

    session_id = str(uuid.uuid4())

    os.makedirs(PREPROCESSING_CACHE_DIR, exist_ok=True)
    session_path = os.path.join(PREPROCESSING_CACHE_DIR, f"{session_id}.json")
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(preprocessed_text, f)

    return {"session_id": session_id}

@app.get("/search-stream")
async def search(session_id: str, search: str):
    """
    Search for a string in a PDF file.
    :param file: The PDF file to search in.
    :param search: The string to search for.
    :return: A JSON response with the number of pages in the PDF.
    """

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")
    
    print(f"Received search request for session {session_id} with search string: {search}")
    
    session_path = os.path.join(PREPROCESSING_CACHE_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found.")
    
    # Load preprocessed_text
    with open(session_path, "r", encoding="utf-8") as f:
        preprocessed_text = json.load(f)
    
    dataset = TfIdfChunkedDocumentDataset(preprocessed_text, chunk_size=CHUNK_SIZE, cache_path='articles')
    
    return StreamingResponse(search_in_dataset(dataset, search), media_type="text/event-stream")



    
   

    
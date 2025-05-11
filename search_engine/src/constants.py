import os

CHUNK_SIZE = 100
BASE_CACHE_DIR = ".cache"
PREPROCESSING_CACHE_DIR = os.path.join(BASE_CACHE_DIR, "preprocessed_text")

FRONTEND_URL = "http://localhost:5173"  # Vite default

CACHE_PATH = '.cache'

DENSE_CACHE_PATH = '.dense_cache'
EMBEDDINGS_DIMENSION = 384
INDEX_PATH = '.faiss_index'

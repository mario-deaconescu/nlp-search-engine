# NLP Search Engine

## Members:
- Sabin-Mario Deaconescu
- Bianca-Daniela Popa
- Anastasia Ștefănescu
- David-Constantin Berbece

# How to run:

## Frontend
`cd frontend`,
`npm run dev`

## Backend
`cd search_engine`,
`uvicorn main:app --reload`

# How to use
1. Upload a pdf document and wait for it to be loaded
2. Select a search model (Tf-Idf, Bm25, or Faiss)
3. Enter a search query
4. Press "Search" and wait for the whole document to be analyzed
5. Press any of the search results to jump to the respective page
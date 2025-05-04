from datasets import load_dataset, Split
from tqdm import tqdm

from src.datasets.tfidf_dataset import TfIdfChunkedDocumentDataset
from src.preprocessing.article_utils import preprocess_articles_dataset
from src.query.search_tfidf import search_tfidf

ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=Split.ALL)
ds.set_transform(preprocess_articles_dataset)
truncated_ds = ds.select(range(100))
dataset = TfIdfChunkedDocumentDataset(ds, chunk_size=100, cache_path='articles')
search = search_tfidf("politics", dataset)
for result in tqdm(search):
    pass
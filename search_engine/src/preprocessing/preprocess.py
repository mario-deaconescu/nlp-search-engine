import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.utils.spacy_cache import load_spacy_model

nlp = load_spacy_model("en_core_web_sm")

def lemmatize(text: str, nlp_preprocess = None) -> str:
    if nlp_preprocess is None:
        nlp_preprocess = nlp(text)
    return " ".join([token.lemma_ for token in nlp_preprocess if not token.is_punct])

def remove_stopwords(text) -> str:
    return " ".join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])

def preprocess_document(document: str, nlp_preprocess = None) -> str:
    # Lowercase
    document = document.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r"[^\w\s]", "", document)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Lemmatization
    text = lemmatize(text, nlp_preprocess)
    text = text.lower()
    # Remove stopwords
    text = remove_stopwords(text)

    return text
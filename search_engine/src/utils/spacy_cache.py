from spacy.cli import download # type: ignore

import spacy


def load_spacy_model(name="en_core_web_sm"):
    try:
        return spacy.load(name)
    except OSError:
        print(f"Model '{name}' not found. Downloading...")
        download(name)
        return spacy.load(name)
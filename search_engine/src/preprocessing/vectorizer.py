import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer

english_stemmer = Stemmer.Stemmer('en')

class StemmedAnalyzer:
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def __call__(self, doc):
        english_stemmer.stemWords(self.analyzer(doc))

class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return StemmedAnalyzer(analyzer)



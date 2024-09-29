from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

class FeatureExtractor:

    def __init__(self):

        self.vectorizer = None


    def vectorize(self, corpus, method="tfidf"):

        if method == "tfidf":
            self.vectorizer = TfidfVectorizer()
            vectors = self.vectorizer.fit_transform(corpus)
            return vectors
        elif method == "bow":
            self.vectorizer = CountVectorizer()
            vectors = self.vectorizer.fit_transform(corpus)
            return vectors
        else:
            raise ValueError(f"Method ’{method}’ not supported. Use ’tfidf’ or ’bow’.")


    #usually performs better with TF-IDF
    def use_lsa(self, vectors, n_topics=5):

        svd_model = TruncatedSVD(n_components=n_topics, algorithm="randomized", random_state=42)
        lsa_vectors = svd_model.fit_transform(vectors)
        return lsa_vectors

    #usually performs better with BoW
    def apply_lda(self, vectors, n_topics=5):

        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_vectors = lda_model.fit_transform(vectors)
        return lda_vectors

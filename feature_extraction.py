from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

class FeatureExtractor:

    def __init__(self):

        self.vectorizer = None


    def vectorize(self, corpus, method="tfidf", return_model=False):

        if method == "tfidf":
            self.vectorizer = TfidfVectorizer()

        elif method == "bow":
            self.vectorizer = CountVectorizer()
        else:
            raise ValueError(f"Method ’{method}’ not supported. Use ’tfidf’ or ’bow’.")

        vectors = self.vectorizer.fit_transform(corpus)

        if return_model:

            return self.vectorizer  

        return vectors

    #usually performs better with TF-IDF
    def use_lsa(self, vectors, n_topics=6, return_model=False):

        svd_model = TruncatedSVD(n_components=n_topics, algorithm="randomized", random_state=42)
        lsa_vectors = svd_model.fit_transform(vectors)
        if return_model:
            return svd_model
        return lsa_vectors

    #usually performs better with BoW
    def apply_lda(self, vectors, n_topics=6):

        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_vectors = lda_model.fit_transform(vectors)
        return lda_vectors

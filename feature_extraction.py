from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

class FeatureExtractor:
    """
    A class for extracting features from textual data using various vectorization
    and dimensionality reduction techniques, such as TF-IDF, Bag-of-Words (BoW),
    Latent Semantic Analysis (LSA), and Latent Dirichlet Allocation (LDA).
    
    Methods:
        vectorize(corpus, method='tfidf', return_model=False): Converts text data
            into numerical vectors using TF-IDF or BoW.
        use_lsa(vectors, n_topics=6, return_model=False): Applies Latent Semantic
            Analysis (LSA) for dimensionality reduction.
        apply_lda(vectors, n_topics=6): Applies Latent Dirichlet Allocation (LDA) for
            topic modeling, typically used with BoW.
    """
    def __init__(self):
        """
        Initializes the FeatureExtractor with a placeholder for the vectorizer.
        """
        self.vectorizer = None


    def vectorize(self, corpus, method="tfidf", return_model=False):
        """
        Converts a corpus into numerical vectors using either TF-IDF or BoW.

        Args:
            corpus (list of str): The input text data.
            method (str): The vectorization method to use, either 'tfidf' or 'bow'.
            return_model (bool): Whether to return the vectorizer model itself.

        Returns:
                - The vectorized data as a sparse matrix.
                - The vectorizer model if 'return_model' is True.
        
        Raises:
            ValueError: If an unsupported method is specified.
        """
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer()

        elif method == "bow":
            self.vectorizer = CountVectorizer()
        else:
            raise ValueError(f"Method ’{method}’ not supported. Use ’tfidf’ or ’bow’.")

        vectors = self.vectorizer.fit_transform(corpus)

        if return_model:

            return self.vectorizer  # Return the vectorizer model if requested

        return vectors


    def use_lsa(self, vectors, n_topics=6, return_model=False):
        """
        Applies LSA for dimensionality reduction.

        Args:
            vectors : The input vectorized data.
            n_topics (int): The number of topics (dimensions) for reduction.
            return_model (bool): Return the LSA model itself if True.

        Returns:
                - The reduced vectors.
                - The LSA model if 'return_model' is True.
        """
        svd_model = TruncatedSVD(n_components=n_topics, algorithm="randomized", random_state=42)
        lsa_vectors = svd_model.fit_transform(vectors)
        if return_model:
            return svd_model
        return lsa_vectors


    def apply_lda(self, vectors, n_topics=6, return_model=False):
        """
        Applies LDA for topic modeling.

        Args:
            vectors : The input vectorized data.
            n_topics (int): The number of topics to extract.

        Returns:
                - The topic distribution for each document.
                - The LDA model if 'return_model' is True.
        """
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_vectors = lda_model.fit_transform(vectors)
        if return_model:
            return lda_model
        return lda_vectors

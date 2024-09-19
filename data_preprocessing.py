from nltk.corpus import stopwords as nltk_stopwords

class DataPreprocessor:


    def __init__(self, stopwords=None):

        self.stopwords = stopwords if stopwords is not None else set(nltk_stopwords.words("english"))
           




import re
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize

class DataPreprocessor:


    def __init__(self, stopwords=None):

        #remove stopwords
        self.stopwords = stopwords if stopwords is not None else set(nltk_stopwords.words("english"))
           


    def clean_text(self, text):

        #delete everything except []
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower()
        
        #remove multiple white spaces
        text = re.sub(r"\s+", " ", text)

        #Remove extra spaces
        text = text.strip()
        
        return text

    def tokenize(self, text):

        #Tokenize the text
        tokens = word_tokenize(text)
        
        #Remove stopwords
        tokens = [word for word in tokens if word not in self.stopwords]
        
        return tokens

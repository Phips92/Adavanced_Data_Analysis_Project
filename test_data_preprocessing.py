import pytest
from nltk.corpus import stopwords as nltk_stopwords
from data_preprocessing import DataPreprocessor  

def test_custom_stopwords():

    custom_stopwords = {"test", "something", "for", "testing"}
    processor = DataPreprocessor(stopwords=custom_stopwords)
    assert processor.stopwords == custom_stopwords

def test_default_stopwords():

    processor = DataPreprocessor()
    assert processor.stopwords == set(nltk_stopwords.words("english"))

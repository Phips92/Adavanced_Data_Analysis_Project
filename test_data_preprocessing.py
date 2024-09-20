import pytest
from nltk.corpus import stopwords as nltk_stopwords
from data_preprocessing import DataPreprocessor  
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def test_custom_stopwords():

    custom_stopwords = {"test", "something", "for", "testing"}
    processor = DataPreprocessor(stopwords=custom_stopwords)
    assert processor.stopwords == custom_stopwords

def test_default_stopwords():

    processor = DataPreprocessor()
    assert processor.stopwords == set(nltk_stopwords.words("english"))


preprocessor = DataPreprocessor()

def test_clean_text():
    #testcase1: only english letters
    input_text = "Only Letters"
    expected_output = "only letters"
    assert preprocessor.clean_text(input_text) == expected_output

    #testcase2: + numbers and spezial caracter
    input_text = "6Letters!"
    expected_output = "letters"
    assert preprocessor.clean_text(input_text) == expected_output

    #testcase3: multiple spezial characters
    input_text = "Le3Tt3er2s & @§$%?"
    expected_output = "letters"
    assert preprocessor.clean_text(input_text) == expected_output

    #testcase4: only spezial characters
    input_text = "@!§$%&/()=?"
    expected_output = ""
    assert preprocessor.clean_text(input_text) == expected_output

    #testcase5: empty string
    input_text = ""
    expected_output = ""
    assert preprocessor.clean_text(input_text) == expected_output

    #testcase6: extra whitespaces
    input_text = "   l   e   t t   e  r   s      "
    expected_output = "l e t t e r s"
    assert preprocessor.clean_text(input_text) == expected_output


def test_tokenize():

    input_text = "testing some words to be vectorized by a function called tokenize"
    expected_tokens = ["testing", "words", "vectorized", "function", "called", "tokenize"] 
    
    assert preprocessor.tokenize(input_text) == expected_tokens

def test_vectorize_tfidf():

    corpus = ['["testing", "words", "vectorized", "function", "called", "tokenize"]','["next", "tweet", "words", "vectorized"]']
    
    vectors = preprocessor.vectorize(corpus, method="tfidf")

    # Check the type of the returned vectors and ensure the correct shape
    assert isinstance(vectors, type(TfidfVectorizer().fit_transform(corpus)))
    assert vectors.shape == (2, 8)

def test_vectorize_bow():

    corpus = ['["testing", "words", "vectorized", "function", "called", "tokenize"]','["next", "tweet", "words", "vectorized"]']
    
    vectors = preprocessor.vectorize(corpus, method="bow")

    assert isinstance(vectors, type(CountVectorizer().fit_transform(corpus)))
    assert vectors.shape == (2, 8)  














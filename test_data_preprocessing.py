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



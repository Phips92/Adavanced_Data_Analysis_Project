import pandas as pd
from data_preprocessing import DataPreprocessor 


preprocessor = DataPreprocessor()

df = pd.read_csv("test_tweets.csv")
df['text'] = df['text'].apply(preprocessor.clean_text)
df.to_csv("cleaned_test_tweets.csv", index=False)

df = pd.read_csv("cleaned_test_tweets.csv")
df['text'] = df['text'].apply(preprocessor.tokenize)
df.to_csv("tokenized_cleaned_test_tweets.csv", index=False)

df = pd.read_csv("tokenized_cleaned_test_tweets.csv")
corpus = df["text"].values.tolist()
vectors_tfidf = preprocessor.vectorize(corpus, method="tfidf")
vectors_bow = preprocessor.vectorize(corpus, method="bow")

#safe vectors
tfidf_array = vectors_tfidf.toarray()
bow_array = vectors_bow.toarray()
pd.DataFrame(tfidf_array).to_csv("TF-IDF_vectors.csv", index=False)
pd.DataFrame(bow_array).to_csv("BoW_vectors.csv", index=False)

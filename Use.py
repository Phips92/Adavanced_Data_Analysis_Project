import pandas as pd
from data_preprocessing import DataPreprocessor 
from feature_extraction import FeatureExtractor

preprocessor = DataPreprocessor()
featurizer = FeatureExtractor()

df = pd.read_csv("test_tweets.csv")
df['text'] = df['text'].apply(preprocessor.clean_text)
df.to_csv("cleaned_test_tweets.csv", index=False)

df = pd.read_csv("cleaned_test_tweets.csv")
df['text'] = df['text'].apply(preprocessor.tokenize)
df.to_csv("tokenized_cleaned_test_tweets.csv", index=False)

df = pd.read_csv("tokenized_cleaned_test_tweets.csv")
corpus = df["text"].values.tolist()
vectors_tfidf = featurizer.vectorize(corpus, method="tfidf")
vectors_bow = featurizer.vectorize(corpus, method="bow")

#safe vectors
tfidf_array = vectors_tfidf.toarray()
bow_array = vectors_bow.toarray()
pd.DataFrame(tfidf_array).to_csv("TF-IDF_vectors.csv", index=False)
pd.DataFrame(bow_array).to_csv("BoW_vectors.csv", index=False)

#lsa with tfidf and bow
lsa_tfidf_vectors = featurizer.use_lsa(vectors_tfidf, n_topics=3)
print(lsa_tfidf_vectors)
lsa_bow_vectors = featurizer.use_lsa(vectors_bow, n_topics=3)
print(lsa_bow_vectors)

#lda with bow
lda_bow_vectors = featurizer.apply_lda(vectors_bow, n_topics=3)
print(lda_bow_vectors)

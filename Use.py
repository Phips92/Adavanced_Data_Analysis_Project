import pandas as pd
from data_preprocessing import DataPreprocessor 
from feature_extraction import FeatureExtractor
from model_training import ModelTrainer
from sklearn.model_selection import train_test_split
from evaluation_and_comparison import ModelEvaluator

preprocessor = DataPreprocessor()
featurizer = FeatureExtractor()
trainer = ModelTrainer()
evaluator = ModelEvaluator()

df = pd.read_csv("synthetic_unique_tweets_dataset_chatgpt.csv")
df["Text"] = df["Text"].apply(preprocessor.clean_text)
df.to_csv("cleaned_tweets.csv", index=False)

df = pd.read_csv("cleaned_tweets.csv")
df["Text"] = df["Text"].apply(preprocessor.tokenize)
df.to_csv("tokenized_cleaned_test_tweets.csv", index=False)

df = pd.read_csv("tokenized_cleaned_test_tweets.csv")
corpus = df["Text"].values.tolist()
labels = df["Label"].values
vectors_tfidf = featurizer.vectorize(corpus, method="tfidf")
vectors_bow = featurizer.vectorize(corpus, method="bow")

#safe vectors
tfidf_array = vectors_tfidf.toarray()
bow_array = vectors_bow.toarray()
pd.DataFrame(tfidf_array).to_csv("TF-IDF_vectors.csv", index=False)
pd.DataFrame(bow_array).to_csv("BoW_vectors.csv", index=False)

#lsa with tfidf and bow
lsa_tfidf_vectors = featurizer.use_lsa(vectors_tfidf, n_topics=6)
#print(lsa_tfidf_vectors)
lsa_bow_vectors = featurizer.use_lsa(vectors_bow, n_topics=6)
#print(lsa_bow_vectors)

#lda with bow
lda_bow_vectors = featurizer.apply_lda(vectors_bow, n_topics=6)
#print(lda_bow_vectors)

#train regression model with lsa (tfidf and bow) and lda (bow)
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(lsa_tfidf_vectors, labels, test_size=0.2, random_state=20)
model_lsa_tfidf = trainer.train(X_train_tfidf, y_train, model_type="logistic_regression")

#make prediction
report, accuracy = evaluator.evaluate(model_lsa_tfidf, X_test_tfidf, y_test)

#results
print(f"Accuracy: {accuracy}")
print(report)


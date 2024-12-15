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

#synthetic data set
#lamda to convert to string
df_train = pd.read_csv("synthetic_unique_tweets_dataset_chatgpt.csv")
df_train["Text"] = df_train["Text"].apply(preprocessor.clean_text)
df_train["Text"] = df_train["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))

corpus_train = df_train["Text"].values.tolist()
labels_train = df_train["Label"].values


#Kagel data set for testing
df_test = pd.read_csv("Kagel_dataset_tweets.csv")
df_test["Text"] = df_test["Text"].apply(preprocessor.clean_text)
df_test["Text"] = df_test["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))

corpus_test = df_test["Text"].values.tolist()
labels_test = df_test["Label"].values


#testing dataset synthetic
df_test_syn = pd.read_csv("synthetic_unique_tweets_test_dataset.csv")
df_test_syn["Text"] = df_test["Text"].apply(preprocessor.clean_text)
df_test_syn["Text"] = df_test["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))

corpus_test_syn = df_test_syn["Text"].values.tolist()
labels_test_syn = df_test_syn["Label"].values


#vectorize
tfidf_vectorizer = featurizer.vectorize(corpus_test, method="tfidf",return_model=True)
bow_vectorizer =  featurizer.vectorize(corpus_test, method="bow", return_model=True)

#train TF-IDF on synthetic data set
vectors_tfidf_train = tfidf_vectorizer.fit_transform(corpus_train)
vectors_tfidf_test = tfidf_vectorizer.transform(corpus_test_syn)

#train bow on synthetic data set
vectors_bow_train = bow_vectorizer.fit_transform(corpus_train)
vectors_bow_test = bow_vectorizer.transform(corpus_test_syn)


# LSA with TF-IDF 
svd_model_tfidf = featurizer.use_lsa(vectors_tfidf_train, n_topics=100, return_model=True)
lsa_tfidf_vectors_train = svd_model_tfidf.transform(vectors_tfidf_train)
lsa_tfidf_vectors_test = svd_model_tfidf.transform(vectors_tfidf_test)


# LSA with BoW 
svd_model_bow = featurizer.use_lsa(vectors_bow_train, n_topics=100, return_model=True)
lsa_bow_vectors_train = svd_model_bow.transform(vectors_bow_train)
lsa_bow_vectors_test = svd_model_bow.transform(vectors_bow_test)


# LDA 
lda_bow_vectors_train = featurizer.apply_lda(vectors_bow_train, n_topics=100)
lda_bow_vectors_test = featurizer.apply_lda(vectors_bow_test, n_topics=100)

print("_______Models trained with synthetic data and tested on another syntethic dataset______\n")

model_tfidf = trainer.train(vectors_bow_train, labels_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_tfidf, vectors_bow_test, labels_test_syn)

print("model with TF-IDF")
print(f"Accuracy: {accuracy}")
print(report)



#train regression model with lsa_tfidf
model_lsa_tfidf = trainer.train(lsa_tfidf_vectors_train, labels_train, model_type="logistic_regression")

#make prediction
report, accuracy = evaluator.evaluate(model_lsa_tfidf, lsa_tfidf_vectors_test, labels_test_syn)

#results lsa_tfidf
print("LSA with TF-IDF")
print(f"Accuracy: {accuracy}")
print(report)


#train regression model with lsa_bow
model_lsa_bow = trainer.train(lsa_bow_vectors_train, labels_train, model_type="logistic_regression")

#make prediction
report, accuracy = evaluator.evaluate(model_lsa_bow, lsa_bow_vectors_test, labels_test_syn)

#results lsa_bow
print("LSA with BoW")
print(f"Accuracy: {accuracy}")
print(report)


#train regression model with lda_bow
model_lda = trainer.train(lda_bow_vectors_train, labels_train, model_type="logistic_regression")

#make prediction
report, accuracy = evaluator.evaluate(model_lda, lda_bow_vectors_test, labels_test_syn)

#results lda_bow
print("LDA with BoW")
print(f"Accuracy: {accuracy}")
print(report)








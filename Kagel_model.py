import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import DataPreprocessor
from feature_extraction import FeatureExtractor
from model_training import ModelTrainer
from evaluation_and_comparison import ModelEvaluator

preprocessor = DataPreprocessor()
featurizer = FeatureExtractor()
trainer = ModelTrainer()
evaluator = ModelEvaluator()

#load kagel dataset and clean it
df_kagel = pd.read_csv("Kagel_dataset_tweets.csv")
df_kagel["Text"] = df_kagel["Text"].apply(preprocessor.clean_text)
df_kagel["Text"] = df_kagel["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))

#4998 random entries for training (833 x 6) equaly representing each class
df_kagel_train = df_kagel.groupby("Label", group_keys=False).apply(lambda x: x.sample(833, random_state=42))

#remove training data from dataset
df_kagel_test = df_kagel.drop(df_kagel_train.index)

#print(df_kagel_train)
#print(df_kagel_train["Label"].value_counts())

corpus_kagel_train = df_kagel_train["Text"].values.tolist()
labels_kagel_train = df_kagel_train["Label"].values
corpus_kagel_test = df_kagel_test["Text"].values.tolist()
labels_kagel_test = df_kagel_test["Label"].values



#vectorize with tf-idf and bow
tfidf_vectorizer = featurizer.vectorize(corpus_kagel_train, method="tfidf", return_model=True)
bow_vectorizer = featurizer.vectorize(corpus_kagel_train, method="bow", return_model=True)


vectors_tfidf_kagel_train = tfidf_vectorizer.fit_transform(corpus_kagel_train)
vectors_tfidf_kagel_test = tfidf_vectorizer.transform(corpus_kagel_test)

vectors_bow_kagel_train = bow_vectorizer.fit_transform(corpus_kagel_train)
vectors_bow_kagel_test = bow_vectorizer.transform(corpus_kagel_test)


#LSA with tf-idf
svd_model = featurizer.use_lsa(vectors_tfidf_kagel_train, n_topics=6, return_model=True)
lsa_kagel_train = svd_model.transform(vectors_tfidf_kagel_train)
lsa_kagel_test = svd_model.transform(vectors_tfidf_kagel_test)

#train model with 5000 datapoints and test on rest
model_kagel = trainer.train(lsa_kagel_train, labels_kagel_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_kagel, lsa_kagel_test, labels_kagel_test)

print("___________model trained and tested with kagel dataset_________")
print("LSA with Tf-IDF")
print(f"Accuracy: {accuracy}")
print(report)


# LSA with BoW
svd_model_bow = featurizer.use_lsa(vectors_bow_kagel_train, n_topics=6, return_model=True)
lsa_bow_train = svd_model_bow.transform(vectors_bow_kagel_train)
lsa_bow_test = svd_model_bow.transform(vectors_bow_kagel_test)


model_lsa_bow = trainer.train(lsa_bow_train, labels_kagel_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lsa_bow, lsa_bow_test, labels_kagel_test)
print("LSA with BoW")
print(f"Accuracy: {accuracy}")
print(report)


#LDA
lda_bow_train = featurizer.apply_lda(vectors_bow_kagel_train, n_topics=6)
lda_bow_test = featurizer.apply_lda(vectors_bow_kagel_test, n_topics=6)

model_lda_bow = trainer.train(lda_bow_train, labels_kagel_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lda_bow, lda_bow_test, labels_kagel_test)
print("LDA with BoW")
print(f"Accuracy: {accuracy}")
print(report)





























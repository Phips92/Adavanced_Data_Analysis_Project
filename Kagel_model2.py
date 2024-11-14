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


df_kagel = pd.read_csv("Kagel_dataset_tweets.csv")
df_kagel["Text"] = df_kagel["Text"].apply(preprocessor.clean_text)
df_kagel["Text"] = df_kagel["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))


corpus = df_kagel["Text"].values.tolist()
labels = df_kagel["Label"].values
corpus_train, corpus_test, labels_train, labels_test = train_test_split(corpus, labels, test_size=0.7, stratify=labels, random_state=42)

tfidf_vectorizer = featurizer.vectorize(corpus_train, method="tfidf", return_model=True)
bow_vectorizer = featurizer.vectorize(corpus_train, method="bow", return_model=True)


vectors_tfidf_train = tfidf_vectorizer.fit_transform(corpus_train)
vectors_tfidf_test = tfidf_vectorizer.transform(corpus_test)

vectors_bow_train = bow_vectorizer.fit_transform(corpus_train)
vectors_bow_test = bow_vectorizer.transform(corpus_test)



model_bow = trainer.train(vectors_bow_train, labels_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_bow, vectors_bow_test, labels_test)
print("Bow(Kaggle train/test split, test_size=0.95)")
print(f"Accuracy: {accuracy}")
print(report)

model_tfidf = trainer.train(vectors_tfidf_train, labels_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_tfidf, vectors_tfidf_test, labels_test)
print("TF-IDF (Kaggle train/test split, test_size=0.95)")
print(f"Accuracy: {accuracy}")
print(report)




"""
# LSA with TF-IDF
svd_model_tfidf = featurizer.use_lsa(vectors_tfidf_train, n_topics=6, return_model=True)
lsa_tfidf_train = svd_model_tfidf.transform(vectors_tfidf_train)
lsa_tfidf_test = svd_model_tfidf.transform(vectors_tfidf_test)

# LSA with BoW
svd_model_bow = featurizer.use_lsa(vectors_bow_train, n_topics=6, return_model=True)
lsa_bow_train = svd_model_bow.transform(vectors_bow_train)
lsa_bow_test = svd_model_bow.transform(vectors_bow_test)

# LDA 
lda_bow_train = featurizer.apply_lda(vectors_bow_train, n_topics=6)
lda_bow_test = featurizer.apply_lda(vectors_bow_test, n_topics=6)




# LSA with TF-IDF
model_lsa_tfidf = trainer.train(lsa_tfidf_train, labels_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lsa_tfidf, lsa_tfidf_test, labels_test)
print("LSA with TF-IDF (Kaggle train/test split)")
print(f"Accuracy: {accuracy}")
print(report)

# LSA with BoW
model_lsa_bow = trainer.train(lsa_bow_train, labels_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lsa_bow, lsa_bow_test, labels_test)
print("LSA with BoW (Kaggle train/test split)")
print(f"Accuracy: {accuracy}")
print(report)

#LDA
model_lda_bow = trainer.train(lda_bow_train, labels_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lda_bow, lda_bow_test, labels_test)
print("LDA with BoW (Kaggle train/test split)")
print(f"Accuracy: {accuracy}")
print(report)

"""

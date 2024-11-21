import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import DataPreprocessor
from feature_extraction import FeatureExtractor
from model_training import ModelTrainer
from evaluation_and_comparison import ModelEvaluator

# Initialize classes
preprocessor = DataPreprocessor() # Handles text cleaning and preprocessing
featurizer = FeatureExtractor()   # Manages feature extraction methods (BoW, TF-IDF, LSA, LDA)
trainer = ModelTrainer()          # Trains ML models (logistic_regression, random_forest, svm, naive_bayes, gradient_boosting)
evaluator = ModelEvaluator()      # Evaluates and compares model performance

# Load and preprocess the Kaggle dataset
df_kaggle = pd.read_csv("Kagel_dataset_tweets.csv")
df_kaggle["Text"] = df_kaggle["Text"].apply(preprocessor.clean_text)
df_kaggle["Text"] = df_kaggle["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))

#4998 random entries for training (833 x 6) equaly representing each class
df_kaggle_train = df_kaggle.groupby("Label", group_keys=False).apply(lambda x: x.sample(833, random_state=42))


# Load and preprocess the Synthetic dataset
df_train = pd.read_csv("synthetic_unique_tweets_dataset_chatgpt.csv")
df_train["Text"] = df_train["Text"].apply(preprocessor.clean_text)
df_train["Text"] = df_train["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))

# Prepare the corpus and labels
corpus_kaggle_train = df_kaggle_train["Text"].values.tolist()
labels_kaggle_train = df_kaggle_train["Label"].values
corpus_synthetic_test = df_train["Text"].values.tolist()
labels_synthetic_test = df_train["Label"].values


#vectorize with tf-idf and bow
tfidf_vectorizer = featurizer.vectorize(corpus_kaggle_train, method="tfidf", return_model=True)
bow_vectorizer = featurizer.vectorize(corpus_kaggle_train, method="bow", return_model=True)


vectors_tfidf_kaggle_train = tfidf_vectorizer.fit_transform(corpus_kaggle_train)
vectors_tfidf_synthetic_test = tfidf_vectorizer.transform(corpus_synthetic_test)

vectors_bow_kaggle_train = bow_vectorizer.fit_transform(corpus_kaggle_train)
vectors_bow_synthetic_test = bow_vectorizer.transform(corpus_synthetic_test)


# Train and evaluate models using BoW
model_bow = trainer.train(vectors_bow_kaggle_train, labels_kaggle_train, model_type="svm")
report, accuracy = evaluator.evaluate(model_bow, vectors_bow_synthetic_test, labels_synthetic_test)
print("___________model trained with kaggle dataset and tested with synthetic_________")
print("model with BoW")
print(f"Accuracy: {accuracy}")
print(report)


# Train and evaluate models using TF-IDF
model_tfidf = trainer.train(vectors_tfidf_kaggle_train, labels_kaggle_train, model_type="svm")
report, accuracy = evaluator.evaluate(model_tfidf, vectors_tfidf_synthetic_test, labels_synthetic_test)
print("___________model trained with kaggle dataset and tested with synthetic_________")
print("model with TF-IDF")
print(f"Accuracy: {accuracy}")
print(report)


#LSA with tf-idf
svd_model = featurizer.use_lsa(vectors_tfidf_kagel_train, n_topics=1700, return_model=True)
lsa_kaggle_train = svd_model.transform(vectors_tfidf_kaggle_train)
lsa_kaggle_test = svd_model.transform(vectors_tfidf_synthetic_test)

# Train and evaluate the model using LSA with TF-IDF 
model_lsa = trainer.train(lsa_kaggle_train, labels_kaggle_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lsa, lsa_kaggle_test, labels_synthetic_test)
print("___________model trained with kaggle dataset and tested with synthetic_________")
print("LSA with Tf-IDF")
print(f"Accuracy: {accuracy}")
print(report)


# LSA with BoW
svd_model_bow = featurizer.use_lsa(vectors_bow_kaggle_train, n_topics=1650, return_model=True)
lsa_bow_train = svd_model_bow.transform(vectors_bow_kaggle_train)
lsa_bow_test = svd_model_bow.transform(vectors_bow_synthetic_test)

# Train and evaluate the model using LSA with BoW 
model_lsa_bow = trainer.train(lsa_bow_train, labels_kaggle_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lsa_bow, lsa_bow_test, labels_synthetic_test)
print("___________model trained with kaggle dataset and tested with synthetic_________")
print("LSA with BoW")
print(f"Accuracy: {accuracy}")
print(report)


# LDA
lda_bow_train = featurizer.apply_lda(vectors_bow_kaggle_train, n_topics=900)
lda_bow_test = featurizer.apply_lda(vectors_bow_synthetic_test, n_topics=900)

# Train and evaluate the model using LDA with BoW
model_lda_bow = trainer.train(lda_bow_train, labels_kaggle_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lda_bow, lda_bow_test, labels_synthetic_test)
print("___________model trained with kaggle dataset and tested with synthetic_________")
print("LDA with BoW")
print(f"Accuracy: {accuracy}")
print(report)


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

# Load and preprocess the synthetic training dataset
df_train = pd.read_csv("synthetic_unique_tweets_dataset_chatgpt.csv")
df_train["Text"] = df_train["Text"].apply(preprocessor.clean_text)
df_train["Text"] = df_train["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))

# Load and preprocess the synthetic test training dataset
df_test_syn = pd.read_csv("synthetic_unique_tweets_test_dataset.csv")
df_test_syn["Text"] = df_test_syn["Text"].apply(preprocessor.clean_text)
df_test_syn["Text"] = df_test_syn["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))

# Combine synthetic training and test datasets for training
df_combined = pd.concat([df_train, df_test_syn], ignore_index=True)
corpus_train = df_combined["Text"].values.tolist() # List of combined tweets for training
labels_train = df_combined["Label"].values         # Corresponding labels


# Load and preprocess the Kaggle dataset for testing
df_test = pd.read_csv("Kagel_dataset_tweets.csv")
df_test["Text"] = df_test["Text"].apply(preprocessor.clean_text)
df_test["Text"] = df_test["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))

corpus_test = df_test["Text"].values.tolist()
labels_test = df_test["Label"].values

# Vectorize the text data using TF-IDF and Bag-of-Words
tfidf_vectorizer = featurizer.vectorize(corpus_train, method="tfidf",return_model=True)
bow_vectorizer =  featurizer.vectorize(corpus_train, method="bow", return_model=True)

# Transform the training and testing data with TF-IDF
vectors_tfidf_train = tfidf_vectorizer.fit_transform(corpus_train)
vectors_tfidf_test = tfidf_vectorizer.transform(corpus_test)

# Transform the training and testing data with Bag-of-Words
vectors_bow_train = bow_vectorizer.fit_transform(corpus_train)
vectors_bow_test = bow_vectorizer.transform(corpus_test)

# Train and evaluate the model with Bag-of-Words
model_tfidf = trainer.train(vectors_bow_train, labels_train, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_tfidf, vectors_bow_test, labels_test)
print("model with TF-IDF")
print(f"Accuracy: {accuracy}")
print(report)

# Train and evaluate the model with TF-IDF
model_tfidf = trainer.train(vectors_tfidf_train, labels_train, model_type="logistic_regression") 
report, accuracy = evaluator.evaluate(model_tfidf, vectors_tfidf_test, labels_test) 
print("Model with TF-IDF")
print(f"Accuracy: {accuracy}")
print(report)

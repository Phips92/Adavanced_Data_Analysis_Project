import pandas as pd
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


corpus_kaggle = df_kaggle["Text"].values.tolist()
labels_kaggle = df_kaggle["Label"].values

# Load and preprocess the synthetic dataset
df_synthetic = pd.read_csv("synthetic_unique_tweets_dataset_chatgpt.csv")
df_synthetic["Text"] = df_synthetic["Text"].apply(preprocessor.clean_text)
df_synthetic["Text"] = df_synthetic["Text"].apply(lambda x: " ".join(preprocessor.tokenize(x)))

corpus_synthetic = df_synthetic["Text"].values.tolist()
labels_synthetic = df_synthetic["Label"].values

# Vectorize text data using TF-IDF and Bag-of-Words
tfidf_vectorizer = featurizer.vectorize(corpus_kagel, method="tfidf", return_model=True)
bow_vectorizer = featurizer.vectorize(corpus_kagel, method="bow", return_model=True)

# Transform Kaggle and synthetic data using TF-IDF
vectors_tfidf_kagel = tfidf_vectorizer.fit_transform(corpus_kagel)
vectors_tfidf_synthetic = tfidf_vectorizer.transform(corpus_synthetic)

# Transform Kaggle and synthetic data using Bag-of-Words
vectors_bow_kagel = bow_vectorizer.fit_transform(corpus_kagel)
vectors_bow_synthetic = bow_vectorizer.transform(corpus_synthetic)

# Train and evaluate the model using TF-IDF
model_tfidf = trainer.train(vectors_tfidf_kagel, labels_kagel, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_tfidf, vectors_tfidf_synthetic, labels_synthetic)
print("TF-IDF (Kaggle train, Synthetic test)")
print(f"Accuracy: {accuracy}")
print(report)

# LSA with TF-IDF
svd_model_tfidf = featurizer.use_lsa(vectors_tfidf_kagel, n_topics=6, return_model=True)
lsa_tfidf_kagel = svd_model_tfidf.transform(vectors_tfidf_kagel)
lsa_tfidf_synthetic = svd_model_tfidf.transform(vectors_tfidf_synthetic)

# Train and evaluate the model using LSA with TF-IDF
model_lsa_tfidf = trainer.train(lsa_tfidf_kagel, labels_kagel, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lsa_tfidf, lsa_tfidf_synthetic, labels_synthetic)
print("LSA with TF-IDF (Kaggle train, Synthetic test)")
print(f"Accuracy: {accuracy}")
print(report)


# LSA with BoW
svd_model_bow = featurizer.use_lsa(vectors_bow_kagel, n_topics=6, return_model=True)
lsa_bow_kagel = svd_model_bow.transform(vectors_bow_kagel)
lsa_bow_synthetic = svd_model_bow.transform(vectors_bow_synthetic)

# Train and evaluate the model using LSA with BoW
model_lsa_bow = trainer.train(lsa_bow_kagel, labels_kagel, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lsa_bow, lsa_bow_synthetic, labels_synthetic)
print("LSA with BoW (Kaggle train, Synthetic test)")
print(f"Accuracy: {accuracy}")
print(report)


# LDA 
lda_bow_kagel = featurizer.apply_lda(vectors_bow_kagel, n_topics=6)
lda_bow_synthetic = featurizer.apply_lda(vectors_bow_synthetic, n_topics=6)

# Train and evaluate the model using LDA with BoW
model_lda_bow = trainer.train(lda_bow_kagel, labels_kagel, model_type="logistic_regression")
report, accuracy = evaluator.evaluate(model_lda_bow, lda_bow_synthetic, labels_synthetic)
print("LDA with BoW (Kaggle train, Synthetic test)")
print(f"Accuracy: {accuracy}")
print(report)

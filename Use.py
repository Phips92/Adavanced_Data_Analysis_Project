import pandas as pd
from data_preprocessing import DataPreprocessor 


preprocessor = DataPreprocessor()

df = pd.read_csv("test_tweets.csv")
df['text'] = df['text'].apply(preprocessor.clean_text)
df.to_csv("cleaned_test_tweets.csv", index=False)


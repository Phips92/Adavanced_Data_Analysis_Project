import pandas as pd
import matplotlib.pyplot as plt

df_kagel = pd.read_csv("Kagel_dataset_tweets.csv")

class_counts = df_kagel["Label"].value_counts()
class_labels = ["Sadness (0)", "Joy (1)", "Love (2)", "Anger (3)", "Fear (4)", "Surprise (5)"]

plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_labels)
plt.title("Class Distribution Kagel Dataset")
plt.show()



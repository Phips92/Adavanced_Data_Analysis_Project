import pandas as pd
import matplotlib.pyplot as plt

# Load the Kaggle dataset
df_kaggle = pd.read_csv("Kagel_dataset_tweets.csv")

# Count the number of samples for each label
class_counts = df_kaggle["Label"].value_counts()
class_labels = ["Sadness (0)", "Joy (1)", "Love (2)", "Anger (3)", "Fear (4)", "Surprise (5)"]


# Plot a pie chart for class distribution
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)  # Pie chart with percentage labels
plt.title("Class Distribution Kaggle Dataset")
plt.show()



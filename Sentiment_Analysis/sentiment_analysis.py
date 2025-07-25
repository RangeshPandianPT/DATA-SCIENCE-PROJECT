
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
training_df = pd.read_csv('twitter_training_cleaned.csv')

# Plot sentiment distribution
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.countplot(data=training_df, x='sentiment', order=training_df['sentiment'].value_counts().index, palette='viridis')
plt.title("Overall Sentiment Distribution (Training Data)", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sentiment_distribution_overall.png")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Trending_Movies.csv")

# Summary statistics
print(df.describe())

# Numeric features
numeric_cols = ["popularity", "vote_average", "vote_count"]

# Histograms and boxplots
for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[col], kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f'{col} - Histogram')
    sns.boxplot(x=df[col], ax=axes[1], color="lightgreen")
    axes[1].set_title(f'{col} - Boxplot')
    plt.tight_layout()
    plt.savefig(f"{col}_hist_box.png")
    plt.close()

# Correlation matrix
corr = df[numeric_cols].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

# Pairplot (sampled)
sample_df = df[numeric_cols].sample(n=500, random_state=42)
sns.pairplot(sample_df)
plt.savefig("pairplot.png")
plt.close()

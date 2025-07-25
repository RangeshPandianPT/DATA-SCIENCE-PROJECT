
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from termcolor import colored

# Load Dataset
print(colored("🔍 Loading Dataset...", "cyan"))
df = pd.read_csv("Titanic-Dataset.csv")
print(colored(f"✅ Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns", "green"))

# Initial Overview
def data_summary(title, data):
    print(colored(f"\n📊 {title}", "blue", attrs=["bold"]))
    print(data.describe(include='all'))
    print("\nMissing Values:\n", data.isnull().sum())

data_summary("Initial Data Summary", df)

# Handle Missing Values
print(colored("\n🔧 Handling Missing Values...", "cyan"))
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
print(colored("✅ Missing values handled.", "green"))

# Encode Categorical Variables
print(colored("\n🔤 Encoding Categorical Variables...", "cyan"))
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df.drop(columns=['Name', 'Ticket'], inplace=True)
print(colored("✅ Encoding complete.", "green"))

# Normalize Numerical Features
print(colored("\n📏 Normalizing Numerical Features...", "cyan"))
scaler = StandardScaler()
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print(colored("✅ Standardization applied.", "green"))

# Outlier Visualization
print(colored("\n📦 Visualizing Outliers...", "cyan"))
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(f'{col} Boxplot')
plt.tight_layout()
plt.savefig("outlier_boxplots.png")
plt.close()

# Remove Outliers with IQR
print(colored("\n🧹 Removing Outliers...", "cyan"))
def remove_outliers_iqr(data, col):
    Q1, Q3 = data[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

for col in numerical_cols:
    df = remove_outliers_iqr(df, col)

print(colored(f"✅ Outliers removed. Remaining rows: {df.shape[0]}", "green"))

# Final Data Summary
data_summary("Cleaned Data Summary", df)

# Export Cleaned Dataset
df.to_csv("Cleaned_Titanic_Dataset.csv", index=False)
print(colored("\n💾 Cleaned dataset saved as 'Cleaned_Titanic_Dataset.csv'", "yellow", attrs=["bold"]))

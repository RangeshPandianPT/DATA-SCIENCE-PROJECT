
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("car data.csv")

# Feature Engineering
df['Brand'] = df['Car_Name'].apply(lambda x: x.split()[0])
df.drop('Car_Name', axis=1, inplace=True)
df['Car_Age'] = 2025 - df['Year']
df.drop('Year', axis=1, inplace=True)
df['Log_Driven_kms'] = np.log(df['Driven_kms'] + 1)
df.drop('Driven_kms', axis=1, inplace=True)
df['Price_Gap'] = df['Present_Price'] - df['Selling_Price']

# Encode categorical variables
df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission', 'Brand'], drop_first=True)

# Split data
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
importances.sort_values().plot(kind='barh', color='skyblue')
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")

# Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Determine the base directory (directory of this script, fallback to current working directory)
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_path = os.getcwd()

data_path = os.path.join(base_path, "Unemployment_Rate_upto_11_2020.csv")

# Load the dataset
df = pd.read_csv(data_path)

# Data Cleaning
df.columns = df.columns.str.strip()
df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour Participation Rate',
    'Region.1': 'Zone',
    'longitude': 'Longitude',
    'latitude': 'Latitude'
}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Set plot style
sns.set(style="whitegrid")

# Plot: National Average Unemployment Rate Over Time
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='Date', y='Unemployment Rate', estimator='mean', ci=None)
plt.axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', label='COVID-19 Start')
plt.title('National Average Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.tight_layout()

# Save the visualization as an image
viz_path = os.path.join(base_path, "unemployment_trend.png")
plt.savefig(viz_path)
plt.close()

print("Analysis complete. Visualization saved as unemployment_trend.png")

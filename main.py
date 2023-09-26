# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset (replace 'data.csv' with your dataset file)
df = pd.read_csv('data.csv')

# Data Cleaning and Preprocessing
# Handle missing values and convert data types as needed

# Exploratory Data Analysis (EDA)
# Summary statistics
summary_stats = df.describe()

# Visualization of customer demographics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['Age'].hist(bins=20)
plt.title('Age Distribution')
plt.subplot(1, 2, 2)
df['Income'].hist(bins=20)
plt.title('Income Distribution')
plt.show()

# Customer Segmentation (K-Means clustering)
X = df[['Recency', 'Frequency', 'Monetary']]
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(X)

# Market Basket Analysis (Apriori algorithm)
basket = (df.groupby(['TransactionID', 'Product'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('TransactionID'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Customer Profiling
customer_profiles = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Income': 'mean',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).reset_index()

# Visualization and Reporting
# Create visualizations and reports summarizing your findings

# Recommendations
# Provide actionable recommendations based on customer segments and market basket analysis

# Optional: Build a dashboard or implement machine learning models

# Save the updated dataset (if needed)
df.to_csv('segmented_data.csv', index=False)

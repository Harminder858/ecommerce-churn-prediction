#%%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the dataset
df = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2009-2010')

#%%
# Basic data cleaning
df['Customer ID'] = df['Customer ID'].astype('str')
df = df.dropna(subset=['Customer ID'])
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Calculate total price for each transaction
df['TotalPrice'] = df['Quantity'] * df['Price']

# Create customer-level features
customer_features = df.groupby('Customer ID').agg({
    'Invoice': 'count',
    'TotalPrice': 'sum',
    'InvoiceDate': ['min', 'max']
})

customer_features.columns = ['TotalTransactions', 'TotalSpent', 'FirstPurchase', 'LastPurchase']

# Calculate recency, frequency, and monetary value
last_date = df['InvoiceDate'].max() + timedelta(days=1)
customer_features['Recency'] = (last_date - customer_features['LastPurchase']).dt.days
customer_features['Frequency'] = customer_features['TotalTransactions']
customer_features['MonetaryValue'] = customer_features['TotalSpent']

# Calculate average order value
customer_features['AvgOrderValue'] = customer_features['MonetaryValue'] / customer_features['Frequency']

# Define churn (e.g., customers who haven't made a purchase in the last 90 days)
customer_features['Churn'] = (customer_features['Recency'] > 90).astype(int)

# Reset index to make Customer ID a column
customer_features = customer_features.reset_index()

# Save the preprocessed data
customer_features.to_csv('ecommerce_data.csv', index=False)

print(customer_features.head())
print(f"\nShape of the dataset: {customer_features.shape}")
print(f"\nChurn rate: {customer_features['Churn'].mean():.2%}")
# %%

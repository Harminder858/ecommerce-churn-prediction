#%%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

#%%
# Load the preprocessed dataset
df = pd.read_csv('ecommerce_data.csv')

# Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df[['Recency', 'Frequency', 'MonetaryValue', 'AvgOrderValue', 'TotalTransactions', 'Churn']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Prepare data for modeling
X = df[['Recency', 'Frequency', 'MonetaryValue', 'AvgOrderValue', 'TotalTransactions']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_predictions = rf_model.predict(X_test_scaled)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# XGBoost Model
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train)

xgb_predictions = xgb_model.predict(X_test_scaled)
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_predictions))

# Neural Network Model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

nn_predictions = (nn_model.predict(X_test_scaled) > 0.5).astype(int)
print("\nNeural Network Classification Report:")
print(classification_report(y_test, nn_predictions))

# Customer Segmentation
kmeans = KMeans(n_clusters=3, random_state=42)
segments = kmeans.fit_predict(X_scaled)
df['segment'] = segments

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Frequency', y='MonetaryValue', hue='segment', data=df)
plt.title('Customer Segments')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Save the models
import joblib
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(xgb_model, 'xgb_model.joblib')
nn_model.save('nn_model.h5')
# %%

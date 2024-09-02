import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Load the data and models
df = pd.read_csv('data/preprocessed_data.csv')
rf_model = joblib.load('data/rf_model.joblib')
xgb_model = joblib.load('data/xgb_model.joblib')

# Prepare the data for prediction
X = df[['Recency', 'Frequency', 'MonetaryValue', 'AvgOrderValue', 'TotalTransactions']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Make predictions
df['rf_churn_prob'] = rf_model.predict_proba(X_scaled)[:, 1]
df['xgb_churn_prob'] = xgb_model.predict_proba(X_scaled)[:, 1]

# Create the Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for Render

app.layout = html.Div([
    html.H1("E-commerce Customer Churn Dashboard"),
    
    dcc.Dropdown(
        id='model-selector',
        options=[
            {'label': 'Random Forest', 'value': 'rf'},
            {'label': 'XGBoost', 'value': 'xgb'}
        ],
        value='rf',
        style={'width': '50%'}
    ),
    
    dcc.Graph(id='churn-probability-histogram'),
    
    dcc.Graph(id='recency-frequency-scatter')
])

@app.callback(
    Output('churn-probability-histogram', 'figure'),
    Input('model-selector', 'value')
)
def update_churn_histogram(selected_model):
    prob_column = 'rf_churn_prob' if selected_model == 'rf' else 'xgb_churn_prob'
    fig = px.histogram(df, x=prob_column, nbins=30,
                       title=f'Churn Probability Distribution ({selected_model.upper()})')
    return fig

@app.callback(
    Output('recency-frequency-scatter', 'figure'),
    Input('model-selector', 'value')
)
def update_recency_frequency_scatter(selected_model):
    prob_column = 'rf_churn_prob' if selected_model == 'rf' else 'xgb_churn_prob'
    fig = px.scatter(df, x='Recency', y='Frequency', color=prob_column,
                     title='Recency vs Frequency',
                     labels={'Recency': 'Days Since Last Purchase', 'Frequency': 'Number of Purchases'},
                     color_continuous_scale='viridis')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

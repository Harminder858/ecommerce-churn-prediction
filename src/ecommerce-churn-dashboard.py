import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the data and models
df = pd.read_csv('ecommerce_data.csv')
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
scaler = StandardScaler()

# Preprocess the data
X = df[['Recency', 'Frequency', 'MonetaryValue', 'AvgOrderValue', 'TotalTransactions']]
X_scaled = scaler.fit_transform(X)

# Make predictions
df['rf_churn_prob'] = rf_model.predict_proba(X_scaled)[:, 1]
df['xgb_churn_prob'] = xgb_model.predict_proba(X_scaled)[:, 1]

# Create the Dash app
app = dash.Dash(__name__)

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
    
    dcc.Graph(id='segment-scatter-plot'),
    
    dcc.Graph(id='feature-importance-plot')
])

@app.callback(
    Output('churn-probability-histogram', 'figure'),
    Input('model-selector', 'value')
)
def update_churn_histogram(selected_model):
    prob_column = 'rf_churn_prob' if selected_model == 'rf' else 'xgb_churn_prob'
    fig = px.histogram(df, x=prob_column, nbins=30, title=f'Churn Probability Distribution ({selected_model.upper()})')
    return fig

@app.callback(
    Output('segment-scatter-plot', 'figure'),
    Input('model-selector', 'value')
)
def update_segment_scatter(selected_model):
    prob_column = 'rf_churn_prob' if selected_model == 'rf' else 'xgb_churn_prob'
    fig = px.scatter(df, x='Frequency', y='MonetaryValue', color=prob_column,
                     title='Customer Segments', color_continuous_scale='viridis')
    return fig

@app.callback(
    Output('feature-importance-plot', 'figure'),
    Input('model-selector', 'value')
)
def update_feature_importance(selected_model):
    if selected_model == 'rf':
        importance = rf_model.feature_importances_
    else:
        importance = xgb_model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                 title=f'Feature Importance ({selected_model.upper()})')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
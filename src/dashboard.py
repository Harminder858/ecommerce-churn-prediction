from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data/preprocessed_data.csv')

# Calculate Customer Lifetime Value (CLV)
df['CLV'] = df['Frequency'] * df['MonetaryValue']

# Create the Dash app
app = Dash(__name__)
server = app.server  # Expose the server variable for Render

# Define color schemes
color_scheme = px.colors.qualitative.Set1

app.layout = html.Div([
    html.H1("E-commerce Customer Churn Dashboard"),
    
    dcc.Dropdown(
        id='model-selector',
        options=[
            {'label': 'Random Forest', 'value': 'rf'},
            {'label': 'XGBoost', 'value': 'xgb'}
        ],
        value='rf',
        style={'width': '50%', 'marginBottom': '20px'}
    ),
    
    html.Div([
        dcc.Graph(id='churn-probability-histogram', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='clv-distribution', style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='customer-segments-pie', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='country-distribution', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='age-distribution', style={'width': '33%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='feature-importance', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='retention-rate', style={'width': '50%', 'display': 'inline-block'})
    ])
])

@app.callback(
    Output('churn-probability-histogram', 'figure'),
    Input('model-selector', 'value')
)
def update_churn_histogram(selected_model):
    prob_column = f'{selected_model}_churn_prob'
    fig = px.histogram(df, x=prob_column, nbins=30,
                       title=f'Churn Probability Distribution ({selected_model.upper()})',
                       color_discrete_sequence=[color_scheme[0]])
    return fig

@app.callback(
    Output('clv-distribution', 'figure'),
    Input('model-selector', 'value')
)
def update_clv_distribution(selected_model):
    prob_column = f'{selected_model}_churn_prob'
    
    # Create CLV bins
    df['CLV_bins'] = pd.qcut(df['CLV'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Calculate average churn probability for each CLV bin
    clv_churn = df.groupby('CLV_bins')[prob_column].mean().reset_index()
    
    fig = px.bar(clv_churn, x='CLV_bins', y=prob_column,
                 title='Average Churn Probability by Customer Lifetime Value',
                 labels={'CLV_bins': 'Customer Lifetime Value', prob_column: 'Avg Churn Probability'},
                 color=prob_column,
                 color_continuous_scale='Viridis_r')
    
    fig.update_layout(xaxis_title='Customer Lifetime Value', yaxis_title='Average Churn Probability')
    
    return fig

@app.callback(
    Output('customer-segments-pie', 'figure'),
    Input('model-selector', 'value')
)
def update_customer_segments_pie(selected_model):
    segment_counts = df['CustomerSegment'].value_counts()
    fig = px.pie(values=segment_counts.values, names=segment_counts.index, 
                 title='Customer Segments Distribution',
                 color_discrete_sequence=color_scheme)
    return fig

@app.callback(
    Output('country-distribution', 'figure'),
    Input('model-selector', 'value')
)
def update_country_distribution(selected_model):
    country_counts = df['CountryCode'].value_counts()
    fig = px.bar(x=country_counts.index, y=country_counts.values, 
                 title='Customer Distribution by Country',
                 labels={'x': 'Country', 'y': 'Number of Customers'},
                 color_discrete_sequence=[color_scheme[2]])
    return fig

@app.callback(
    Output('age-distribution', 'figure'),
    Input('model-selector', 'value')
)
def update_age_distribution(selected_model):
    fig = px.histogram(df, x='CustomerAge', nbins=20,
                       title='Customer Age Distribution',
                       labels={'CustomerAge': 'Age'},
                       color_discrete_sequence=[color_scheme[3]])
    fig.update_xaxes(range=[18, 80])  # Set x-axis range to match the data generation
    return fig

@app.callback(
    Output('feature-importance', 'figure'),
    Input('model-selector', 'value')
)
def update_feature_importance(selected_model):
    # This is a mock-up. In a real scenario, you'd get this from your model
    features = ['Recency', 'Frequency', 'MonetaryValue', 'CustomerAge', 'TotalProducts']
    importances = np.random.rand(len(features))
    importances = importances / importances.sum()
    
    fig = px.bar(x=features, y=importances, 
                 title=f'Feature Importance ({selected_model.upper()})',
                 labels={'x': 'Feature', 'y': 'Importance'},
                 color_discrete_sequence=[color_scheme[4]])
    return fig

@app.callback(
    Output('retention-rate', 'figure'),
    Input('model-selector', 'value')
)
def update_retention_rate(selected_model):
    retention_rate = 1 - df['Churn'].mean()
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = retention_rate * 100,
        title = {'text': "Customer Retention Rate"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color_scheme[5]},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

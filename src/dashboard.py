from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data/preprocessed_data.csv')

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
        dcc.Graph(id='rfm-heatmap', style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='customer-segments-pie', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='country-distribution', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='age-segments', style={'width': '33%', 'display': 'inline-block'})
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
    Output('rfm-heatmap', 'figure'),
    Input('model-selector', 'value')
)
def update_rfm_heatmap(selected_model):
    prob_column = f'{selected_model}_churn_prob'
    
    df['R_Segment'] = pd.qcut(df['Recency'], q=5, labels=['1', '2', '3', '4', '5'])
    df['F_Segment'] = pd.qcut(df['Frequency'], q=5, labels=['5', '4', '3', '2', '1'])
    
    rfm_seg_map = df.groupby(['R_Segment', 'F_Segment'])[prob_column].mean().reset_index()
    
    fig = px.density_heatmap(rfm_seg_map, x='R_Segment', y='F_Segment', z=prob_column,
                             title='RFM Segmentation Heatmap',
                             labels={'R_Segment': 'Recency', 'F_Segment': 'Frequency', prob_column: 'Avg Churn Probability'},
                             color_continuous_scale='Viridis_r')
    
    fig.update_layout(
        xaxis_title='Recency (1=Best, 5=Worst)',
        yaxis_title='Frequency (5=Best, 1=Worst)',
        coloraxis_colorbar=dict(title='Avg Churn Probability')
    )
    
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
    Output('age-segments', 'figure'),
    Input('model-selector', 'value')
)
def update_age_segments(selected_model):
    # Create age groups
    df['AgeGroup'] = pd.cut(df['CustomerAge'], 
                            bins=[0, 25, 35, 45, 55, 65, 100],
                            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    age_segment_counts = df['AgeGroup'].value_counts().sort_index()
    
    fig = px.bar(x=age_segment_counts.index, y=age_segment_counts.values,
                 title='Customer Age Segments',
                 labels={'x': 'Age Group', 'y': 'Number of Customers'},
                 color_discrete_sequence=[color_scheme[3]])
    
    fig.update_layout(
        xaxis_title='Age Group',
        yaxis_title='Number of Customers'
    )
    
    return fig

@app.callback(
    Output('feature-importance', 'figure'),
    Input('model-selector', 'value')
)
def update_feature_importance(selected_model):
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

import pandas as pd
import dash
from dash import html, dcc
import plotly.graph_objs as go

# Load model results from CSV
df = pd.read_csv("results/results.csv")  # Ensure this CSV exists and is updated

app = dash.Dash(__name__)
app.title = "Model Performance Dashboard"

# Create bar chart traces for each metric
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

graphs = []

for metric, color in zip(metrics, colors):
    graphs.append(go.Bar(
        x=df['Model'],
        y=df[metric],
        name=metric,
        marker=dict(color=color)
    ))

layout = go.Layout(
    title="Model Comparison",
    xaxis=dict(title="Model"),
    yaxis=dict(title="Score", range=[0, 1.1]),
    barmode="group",
    plot_bgcolor="#f9f9f9"
)

app.layout = html.Div([
    html.H1("Model Evaluation Dashboard", style={"textAlign": "center"}),
    dcc.Graph(
        id="model-metrics",
        figure=go.Figure(data=graphs, layout=layout)
    )
])

if __name__ == "__main__":
    app.run(debug=True)

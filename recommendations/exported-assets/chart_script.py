import plotly.graph_objects as go
import json

# Data for the chart
data = [
  {"category": "Agent Collaboration", "before": 1, "after": 9},
  {"category": "Backend Intelligence", "before": 3, "after": 9},
  {"category": "Context Persistence", "before": 2, "after": 9},
  {"category": "Performance Monitoring", "before": 4, "after": 8},
  {"category": "Security Hardening", "before": 5, "after": 8},
  {"category": "Code Modularity", "before": 3, "after": 9}
]

# Abbreviate categories to fit 15 character limit
categories = ["Agent Collab", "Backend Intel", "Context Persist", "Perf Monitor", "Security Hard", "Code Modular"]
before_values = [item["before"] for item in data]
after_values = [item["after"] for item in data]

# Create the grouped bar chart
fig = go.Figure()

# Add Before bars
fig.add_trace(go.Bar(
    name='Before',
    x=categories,
    y=before_values,
    marker_color='#1FB8CD',
    cliponaxis=False
))

# Add After bars
fig.add_trace(go.Bar(
    name='After',
    x=categories,
    y=after_values,
    marker_color='#DB4545',
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='MCP Server: Before vs After',
    xaxis_title='Categories',
    yaxis_title='Score',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Save the chart
fig.write_image("mcp_server_comparison.png")
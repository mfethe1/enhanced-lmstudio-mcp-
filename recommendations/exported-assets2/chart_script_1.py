import plotly.graph_objects as go

# Data from the provided JSON
labels = ["1-5", "6-10", "11-15", "16+"]
values = [26, 18, 10, 7]
colors = ["#1FB8CD", "#DB4545", "#2E8B57", "#5D878F"]

# Create pie chart
fig = go.Figure(data=[go.Pie(
    labels=labels, 
    values=values, 
    marker_colors=colors
)])

# Update layout
fig.update_layout(
    title="Cyclomatic Complexity Distribution",
    uniformtext_minsize=14, 
    uniformtext_mode='hide'
)

# Save as PNG
fig.write_image("complexity_distribution.png")
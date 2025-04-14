import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category20

# Read the data
df_combined = pd.read_csv('./data/silver/df_matched.csv')

# Create a bipartite graph
G = nx.Graph()

# Add nodes for organizations and MEPs
organizations = df_combined['Name'].unique()
meps = df_combined['member_name'].unique()

# Add nodes with different colors for organizations and MEPs
G.add_nodes_from(organizations, bipartite=0, node_type='organization')
G.add_nodes_from(meps, bipartite=1, node_type='mep')

# Add edges between organizations and MEPs
for _, row in df_combined.iterrows():
    G.add_edge(row['Name'], row['member_name'])

# Calculate node degrees for sizing
degrees = dict(G.degree())
max_degree = max(degrees.values())
min_degree = min(degrees.values())

# Create a layout
pos = nx.spring_layout(G, k=0.15, iterations=20)

# Prepare data for Bokeh
x = [pos[node][0] for node in G.nodes()]
y = [pos[node][1] for node in G.nodes()]
node_names = list(G.nodes())
node_types = [G.nodes[node]['node_type'] for node in G.nodes()]
node_degrees = [degrees[node] for node in G.nodes()]
node_sizes = [5 + 15 * (degrees[node] - min_degree) / (max_degree - min_degree) for node in G.nodes()]

# Create a Bokeh plot
plot = figure(title="Network of Organizations and MEPs", 
             tools="pan,wheel_zoom,box_zoom,reset,save",
             x_range=(-1.1, 1.1), y_range=(-1.1, 1.1))

# Create data source
source = ColumnDataSource(data=dict(
    x=x,
    y=y,
    name=node_names,
    type=node_types,
    degree=node_degrees
))

# Add nodes
plot.circle('x', 'y', size=node_sizes, 
           color=[Category20[2][0] if t == 'organization' else Category20[2][1] for t in node_types],
           alpha=0.6, source=source)

# Add edges
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    plot.line([x0, x1], [y0, y1], line_color='gray', line_alpha=0.1)

# Add hover tool
hover = HoverTool(tooltips=[
    ("Name", "@name"),
    ("Type", "@type"),
    ("Connections", "@degree")
])
plot.add_tools(hover)

# Output to HTML
output_file("network_visualization.html")
show(plot)

# Print network statistics
print(f"Number of organizations: {len(organizations)}")
print(f"Number of MEPs: {len(meps)}")
print(f"Number of connections: {G.number_of_edges()}")
print(f"Average degree: {sum(degrees.values()) / G.number_of_nodes():.2f}") 
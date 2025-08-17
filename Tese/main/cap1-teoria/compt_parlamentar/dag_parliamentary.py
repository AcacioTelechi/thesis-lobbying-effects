import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create a directed graph
G = nx.DiGraph()

# Define nodes by their type (observed vs unobserved)
observed_nodes = [
    'Lobbying',  # Observed through lobbying registries
    'Legislative Activities',  # Observed through voting records, bills
    'Oversight Activities',  # Observed through parliamentary questions
    'Publicity Activities',  # Observed through speeches, media presence
    'District Size',  # Observed through electoral data
    'Socioeconomic Diversity',  # Observed through census data
    'Party Ideology',  # Observed through party manifestos
    'Government Position',  # Observed through cabinet positions
    'Professional Background',  # Observed through biographical data
    'Gender'  # Observed through demographic data
]

unobserved_nodes = [
    'Parliamentary Behavior',  # Latent construct
    'Electoral System',  # Institutional context
    'Party Characteristics',  # Latent party features
    'Individual Characteristics',  # Latent individual features
    'Personal Vote',  # Latent electoral behavior
    'Party Vote',  # Latent electoral behavior
    'Party Discipline',  # Latent party feature
    'Veto Player Status',  # Latent institutional position
    'Expertise',  # Latent individual feature
    'Experience'  # Latent individual feature
]

# Add all nodes to the graph
for node in observed_nodes + unobserved_nodes:
    G.add_node(node)

# Add edges for main relationships
main_edges = [
    ('Lobbying', 'Parliamentary Behavior'),
    ('Electoral System', 'Parliamentary Behavior'),
    ('Party Characteristics', 'Parliamentary Behavior'),
    ('Individual Characteristics', 'Parliamentary Behavior'),
    ('Parliamentary Behavior', 'Legislative Activities'),
    ('Parliamentary Behavior', 'Oversight Activities'),
    ('Parliamentary Behavior', 'Publicity Activities')
]

# Add edges for electoral system relationships
electoral_edges = [
    ('Electoral System', 'Personal Vote'),
    ('Electoral System', 'Party Vote'),
    ('District Size', 'Electoral System'),
    ('Socioeconomic Diversity', 'Electoral System')
]

# Add edges for party relationships
party_edges = [
    ('Party Characteristics', 'Party Ideology'),
    ('Party Characteristics', 'Party Discipline'),
    ('Party Characteristics', 'Government Position'),
    ('Party Characteristics', 'Veto Player Status')
]

# Add edges for individual relationships
individual_edges = [
    ('Individual Characteristics', 'Expertise'),
    ('Individual Characteristics', 'Experience'),
    ('Individual Characteristics', 'Professional Background'),
    ('Individual Characteristics', 'Gender')
]

# Add all edges to the graph
for edge in main_edges + electoral_edges + party_edges + individual_edges:
    G.add_edge(*edge)

# Set up the plot
plt.figure(figsize=(15, 10))

# Define temporal layers (from left to right)
temporal_layers = {
    'Layer 1 (Institutional/Background)': [
        'Electoral System', 'District Size', 'Socioeconomic Diversity',
        'Party Characteristics', 'Individual Characteristics',
        'Professional Background', 'Gender'
    ],
    'Layer 2 (Pre-Behavior)': [
        'Lobbying', 'Party Ideology', 'Party Discipline',
        'Government Position', 'Veto Player Status',
        'Expertise', 'Experience', 'Personal Vote', 'Party Vote'
    ],
    'Layer 3 (Behavior)': [
        'Parliamentary Behavior'
    ],
    'Layer 4 (Outcomes)': [
        'Legislative Activities', 'Oversight Activities', 'Publicity Activities'
    ]
}

# Create custom positions based on temporal layers
pos = {}
layer_width = 1.0 / (len(temporal_layers) + 1)
for i, (layer_name, nodes) in enumerate(temporal_layers.items()):
    x = (i + 1) * layer_width
    for j, node in enumerate(nodes):
        y = 1.0 - (j + 1) / (len(nodes) + 1)
        pos[node] = np.array([x, y])

# Draw observed nodes (squares)
nx.draw_networkx_nodes(G, pos, 
                      nodelist=observed_nodes,
                      node_color='lightblue',
                      node_size=2000,
                      node_shape='s',  # square
                      alpha=0.7)

# Draw unobserved nodes (circles)
nx.draw_networkx_nodes(G, pos,
                      nodelist=unobserved_nodes,
                      node_color='lightgreen',
                      node_size=2000,
                      node_shape='o',  # circle
                      alpha=0.7)

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

# Add labels
nx.draw_networkx_labels(G, pos, font_size=8)

# Add title and legend
plt.title('DAG of Parliamentary Behavior and Lobbying Effects\nSquares = Observed Variables, Circles = Unobserved Variables\nTemporal Flow: Left → Right', pad=20)

# Remove axis
plt.axis('off')

# Save the figure
plt.savefig('Tese/main/cap1-teoria/compt_parlamentar/dag_parliamentary.png', 
            bbox_inches='tight', 
            dpi=300)
plt.close() 
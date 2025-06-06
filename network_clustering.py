import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from collections import Counter

# Load the data
df_nodes = pd.read_csv("./data/silver/df_nodes.csv")
df_edges = pd.read_csv("./data/silver/df_edges.csv")

# Create a network graph
G = nx.Graph()

# First add all nodes with their attributes
for _, row in df_nodes.iterrows():
    G.add_node(row['id'], label=row['Label'], type=row['type'])

# Then add edges with weights
for _, row in df_edges.iterrows():
    # Add nodes if they don't exist (some might be missing from df_nodes)
    if row['Source'] not in G:
        G.add_node(row['Source'])
    if row['Target'] not in G:
        G.add_node(row['Target'])
    G.add_edge(row['Source'], row['Target'], weight=row['weight'])

# Perform Louvain community detection
partition = community_louvain.best_partition(G, weight='weight')

# Add community information to nodes
for node in G.nodes():
    G.nodes[node]['community'] = partition[node]

# Count communities and their sizes
community_sizes = Counter(partition.values())
print(f"Number of communities: {len(community_sizes)}")
print("Top 10 largest communities:")
for comm_id, size in community_sizes.most_common(10):
    print(f"Community {comm_id}: {size} nodes")

# Create a subgraph for visualization (taking a sample for better visualization)
sample_size = min(1000, len(G.nodes()))
sample_nodes = list(G.nodes())[:sample_size]
G_sample = G.subgraph(sample_nodes)

# Visualize the network with communities
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G_sample, seed=42)
nx.draw(G_sample, pos, 
        node_color=[G_sample.nodes[n]['community'] for n in G_sample.nodes()],
        node_size=50,
        cmap=plt.cm.rainbow,
        with_labels=False)
plt.title("Network Clustering Visualization")
plt.savefig("network_clusters.png")
plt.close()

# Calculate some network metrics
print("\nNetwork Metrics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"Network density: {nx.density(G):.4f}")

# Analyze community composition
def analyze_community_composition(community_id):
    community_nodes = [n for n in G.nodes() if G.nodes[n]['community'] == community_id]
    types = []
    for n in community_nodes:
        node_type = G.nodes[n].get('type', 'Unknown')
        types.append(node_type)
    type_counts = Counter(types)
    return type_counts

# Print composition of top 5 communities
print("\nCommunity Composition Analysis:")
for comm_id, _ in community_sizes.most_common(5):
    composition = analyze_community_composition(comm_id)
    print(f"\nCommunity {comm_id} composition:")
    for type_name, count in composition.items():
        print(f"{type_name}: {count} nodes")

# Save the community assignments
community_assignments = pd.DataFrame({
    'node_id': list(partition.keys()),
    'community_id': list(partition.values())
})
community_assignments.to_csv("./data/silver/community_assignments.csv", index=False) 
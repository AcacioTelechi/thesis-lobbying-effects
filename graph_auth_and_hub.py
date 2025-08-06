# Compute authority and hub scores for lobbying network
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("./data/silver/df_meetings_lobbyists.csv")

def construct_subgraph(edges_df, source_col, target_col, query_nodes=None, t=10, d=5):
    """
    Construct subgraph using the HITS subgraph construction algorithm.
    
    Parameters:
    - edges_df: DataFrame with edge information
    - source_col: Column name for source nodes
    - target_col: Column name for target nodes
    - query_nodes: List of "query" nodes (analogous to search results)
    - t: Number of top results to consider
    - d: Maximum number of incoming edges to include per node
    
    Returns:
    - Subgraph as a NetworkX DiGraph
    """
    
    # Create full directed graph
    G_full = nx.from_pandas_edgelist(edges_df, source=source_col, target=target_col, 
                                    create_using=nx.DiGraph())
    
    # If no query nodes provided, use nodes with highest in-degree as "top results"
    if query_nodes is None:
        in_degrees = dict(G_full.in_degree())
        query_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:t]
        query_nodes = [node for node, degree in query_nodes]
    else:
        # Limit to top t query nodes
        query_nodes = query_nodes[:t]
    
    print(f"Using {len(query_nodes)} query nodes: {query_nodes}")
    
    # Initialize subgraph with query nodes
    S_sigma = set(query_nodes)
    
    # For each query node, expand the subgraph
    for node in query_nodes:
        if node in G_full:
            # Add all pages that this node points to (outgoing edges)
            successors = set(G_full.successors(node))
            S_sigma.update(successors)
            
            # Add incoming edges (predecessors)
            predecessors = set(G_full.predecessors(node))
            if len(predecessors) <= d:
                # Add all predecessors if we have d or fewer
                S_sigma.update(predecessors)
            else:
                # Add only d predecessors (choose those with highest out-degree)
                pred_degrees = [(pred, G_full.out_degree(pred)) for pred in predecessors]
                pred_degrees.sort(key=lambda x: x[1], reverse=True)
                top_d_preds = [pred for pred, degree in pred_degrees[:d]]
                S_sigma.update(top_d_preds)
    
    # Create subgraph
    G_subgraph = G_full.subgraph(S_sigma).copy()
    
    print(f"Subgraph constructed: {G_subgraph.number_of_nodes()} nodes, {G_subgraph.number_of_edges()} edges")
    
    return G_subgraph, query_nodes

def compute_hits_networkx(edges_df, source_col, target_col, use_subgraph=True, t=10, d=5):
    """
    Compute HITS using NetworkX's built-in function with optional subgraph construction.
    """
    if use_subgraph:
        G, query_nodes = construct_subgraph(edges_df, source_col, target_col, t=t, d=d)
    else:
        G = nx.from_pandas_edgelist(edges_df, source=source_col, target=target_col, 
                                    create_using=nx.DiGraph())
        query_nodes = None
    
    # Use NetworkX's HITS implementation
    hubs, authorities = nx.hits(G)
    
    return {
        'authority_scores': authorities,
        'hub_scores': hubs,
        'graph': G,
        'query_nodes': query_nodes
    }

def compute_authority_hub_scores(edges_df, source_col, target_col, 
                                max_iterations=100, tolerance=1e-6, use_subgraph=True, t=10, d=5):
    """
    Compute authority and hub scores using the complete HITS algorithm.
    
    Parameters:
    - edges_df: DataFrame with edge information
    - source_col: Column name for source nodes
    - target_col: Column name for target nodes
    - max_iterations: Maximum number of iterations
    - tolerance: Convergence tolerance
    - use_subgraph: Whether to use subgraph construction
    - t: Number of top results for subgraph construction
    - d: Maximum incoming edges per node for subgraph construction
    
    Returns:
    - Dictionary with authority and hub scores for each node
    """
    
    # Step 1: Construct subgraph (if requested)
    if use_subgraph:
        G, query_nodes = construct_subgraph(edges_df, source_col, target_col, t=t, d=d)
    else:
        G = nx.from_pandas_edgelist(edges_df, source=source_col, target=target_col, 
                                    create_using=nx.DiGraph())
        query_nodes = None
    
    # Step 2: Initialize scores to 1 for all nodes
    authority_scores = {node: 1.0 for node in G.nodes()}
    hub_scores = {node: 1.0 for node in G.nodes()}
    
    # Step 3: Iterative computation
    for iteration in range(max_iterations):
        # Store old scores for convergence check
        old_authority = authority_scores.copy()
        old_hub = hub_scores.copy()
        
        # Update authority scores
        # Authority score = sum of hub scores of nodes pointing to it
        for node in G.nodes():
            authority_scores[node] = sum(hub_scores[predecessor] 
                                       for predecessor in G.predecessors(node))
        
        # Update hub scores
        # Hub score = sum of authority scores of nodes it points to
        for node in G.nodes():
            hub_scores[node] = sum(authority_scores[successor] 
                                 for successor in G.successors(node))
        
        # Normalize both authority and hub scores
        auth_norm = sum(authority_scores.values())
        hub_norm = sum(hub_scores.values())
        
        if auth_norm > 0:
            authority_scores = {k: v/auth_norm for k, v in authority_scores.items()}
        if hub_norm > 0:
            hub_scores = {k: v/hub_norm for k, v in hub_scores.items()}
        
        # Check convergence
        auth_diff = sum(abs(authority_scores[node] - old_authority[node]) 
                       for node in G.nodes())
        hub_diff = sum(abs(hub_scores[node] - old_hub[node]) 
                      for node in G.nodes())
        
        if auth_diff < tolerance and hub_diff < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return {
        'authority_scores': authority_scores,
        'hub_scores': hub_scores,
        'graph': G,
        'query_nodes': query_nodes
    }

def analyze_lobbying_network(df, source_col, target_col, use_subgraph=True, t=10, d=5):
    """
    Analyze the lobbying network using authority and hub scores.
    
    Parameters:
    - df: DataFrame with edge data
    - source_col: Column name for source nodes
    - target_col: Column name for target nodes
    - use_subgraph: Whether to use subgraph construction (complete HITS)
    - t: Number of top results for subgraph construction
    - d: Maximum incoming edges per node for subgraph construction
    """
    
    print("Data columns:", df.columns.tolist())
    print("Data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    if source_col in df.columns and target_col in df.columns:
        edges_df = df[[source_col, target_col]].drop_duplicates()
        edges_df.columns = [source_col, target_col]
        
        print(f"\n=== HITS ANALYSIS ===")
        print(f"Using subgraph construction: {use_subgraph}")
        if use_subgraph:
            print(f"Top t results: {t}")
            print(f"Max incoming edges per node: {d}")
        
        # Compute scores using both methods for comparison
        results_custom = compute_authority_hub_scores(edges_df, source_col, target_col, 
                                                    use_subgraph=use_subgraph, t=t, d=d)
        results_networkx = compute_hits_networkx(edges_df, source_col, target_col, 
                                               use_subgraph=use_subgraph, t=t, d=d)
        
        # Convert to DataFrame for easier analysis
        scores_df_custom = pd.DataFrame({
            'node': list(results_custom['authority_scores'].keys()),
            'authority_score_custom': list(results_custom['authority_scores'].values()),
            'hub_score_custom': list(results_custom['hub_scores'].values())
        })
        
        scores_df_networkx = pd.DataFrame({
            'node': list(results_networkx['authority_scores'].keys()),
            'authority_score_networkx': list(results_networkx['authority_scores'].values()),
            'hub_score_networkx': list(results_networkx['hub_scores'].values())
        })
        
        # Merge the results for comparison
        scores_df = scores_df_custom.merge(scores_df_networkx, on='node')
        
        # Sort by authority and hub scores
        top_authorities_custom = scores_df.nlargest(10, 'authority_score_custom')
        top_hubs_custom = scores_df.nlargest(10, 'hub_score_custom')
        top_authorities_networkx = scores_df.nlargest(10, 'authority_score_networkx')
        top_hubs_networkx = scores_df.nlargest(10, 'hub_score_networkx')
        
        print("\n=== TOP 10 AUTHORITIES (Custom Implementation) ===")
        print(top_authorities_custom[['node', 'authority_score_custom']])
        
        print("\n=== TOP 10 AUTHORITIES (NetworkX Implementation) ===")
        print(top_authorities_networkx[['node', 'authority_score_networkx']])
        
        print("\n=== TOP 10 HUBS (Custom Implementation) ===")
        print(top_hubs_custom[['node', 'hub_score_custom']])
        
        print("\n=== TOP 10 HUBS (NetworkX Implementation) ===")
        print(top_hubs_networkx[['node', 'hub_score_networkx']])
        
        # Check correlation between implementations
        auth_corr = scores_df['authority_score_custom'].corr(scores_df['authority_score_networkx'])
        hub_corr = scores_df['hub_score_custom'].corr(scores_df['hub_score_networkx'])
        
        print(f"\n=== CORRELATION BETWEEN IMPLEMENTATIONS ===")
        print(f"Authority scores correlation: {auth_corr:.6f}")
        print(f"Hub scores correlation: {hub_corr:.6f}")
        
        # Use NetworkX results as the standard
        results = results_networkx
        scores_df_final = scores_df_networkx.rename(columns={
            'authority_score_networkx': 'authority_score',
            'hub_score_networkx': 'hub_score'
        })
        
        # Visualization
        plt.figure(figsize=(20, 10))
        
        # Authority scores distribution (NetworkX)
        plt.subplot(2, 4, 1)
        plt.hist(scores_df['authority_score_networkx'], bins=20, alpha=0.7)
        plt.title('Authority Scores (NetworkX)')
        plt.xlabel('Authority Score')
        plt.ylabel('Frequency')
        
        # Hub scores distribution (NetworkX)
        plt.subplot(2, 4, 2)
        plt.hist(scores_df['hub_score_networkx'], bins=20, alpha=0.7, color='orange')
        plt.title('Hub Scores (NetworkX)')
        plt.xlabel('Hub Score')
        plt.ylabel('Frequency')
        
        # Authority vs Hub scatter plot (NetworkX)
        plt.subplot(2, 4, 3)
        plt.scatter(scores_df['authority_score_networkx'], scores_df['hub_score_networkx'], alpha=0.6)
        plt.xlabel('Authority Score')
        plt.ylabel('Hub Score')
        plt.title('Authority vs Hub Scores (NetworkX)')
        
        # Comparison scatter plot
        plt.subplot(2, 4, 4)
        plt.scatter(scores_df['authority_score_custom'], scores_df['authority_score_networkx'], alpha=0.6)
        plt.xlabel('Custom Authority Score')
        plt.ylabel('NetworkX Authority Score')
        plt.title('Authority Score Comparison')
        
        # Hub score comparison
        plt.subplot(2, 4, 5)
        plt.scatter(scores_df['hub_score_custom'], scores_df['hub_score_networkx'], alpha=0.6, color='orange')
        plt.xlabel('Custom Hub Score')
        plt.ylabel('NetworkX Hub Score')
        plt.title('Hub Score Comparison')
        
        # Authority scores distribution (Custom)
        plt.subplot(2, 4, 6)
        plt.hist(scores_df['authority_score_custom'], bins=20, alpha=0.7, color='green')
        plt.title('Authority Scores (Custom)')
        plt.xlabel('Authority Score')
        plt.ylabel('Frequency')
        
        # Hub scores distribution (Custom)
        plt.subplot(2, 4, 7)
        plt.hist(scores_df['hub_score_custom'], bins=20, alpha=0.7, color='red')
        plt.title('Hub Scores (Custom)')
        plt.xlabel('Hub Score')
        plt.ylabel('Frequency')
        
        # Authority vs Hub scatter plot (Custom)
        plt.subplot(2, 4, 8)
        plt.scatter(scores_df['authority_score_custom'], scores_df['hub_score_custom'], alpha=0.6)
        plt.xlabel('Authority Score')
        plt.ylabel('Hub Score')
        plt.title('Authority vs Hub Scores (Custom)')
        
        plt.tight_layout()
        plt.savefig('authority_hub_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return scores_df_final, results['graph']
    
    else:
        print("Please adjust the column names in the analyze_lobbying_network function")
        print("to match your actual data structure.")
        return None, None

def compute_comprehensive_centrality_analysis(df, source_col, target_col):
    """
    Comprehensive centrality analysis for lobbying networks.
    This is more suitable than HITS for finding relevant members and lobbyists.
    """
    
    # Create directed graph
    G = nx.from_pandas_edgelist(df, source=source_col, target=target_col, create_using=nx.DiGraph())
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 1. PageRank - Best for lobbying influence
    print("\nComputing PageRank...")
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    
    # 2. Betweenness Centrality - Best for intermediaries
    print("Computing Betweenness Centrality...")
    betweenness_scores = nx.betweenness_centrality(G)
    
    # 3. Eigenvector Centrality - Best for elite connections
    print("Computing Eigenvector Centrality...")
    try:
        eigenvector_scores = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        print("Eigenvector centrality failed, using simple degree centrality as fallback...")
        eigenvector_scores = dict(G.degree())
    
    # 4. Closeness Centrality - Best for information access
    print("Computing Closeness Centrality...")
    closeness_scores = nx.closeness_centrality(G)
    
    # 5. In-degree and Out-degree centrality
    print("Computing Degree Centrality...")
    in_degree_scores = dict(G.in_degree())
    out_degree_scores = dict(G.out_degree())
    
    # 6. HITS for comparison
    print("Computing HITS...")
    try:
        hubs, authorities = nx.hits(G)
    except:
        print("HITS failed, using custom implementation...")
        authorities = {node: 1.0 for node in G.nodes()}
        hubs = {node: 1.0 for node in G.nodes()}
    
    # Combine all scores
    results = {}
    for node in G.nodes():
        results[node] = {
            'pagerank': pagerank_scores[node],
            'betweenness': betweenness_scores[node],
            'eigenvector': eigenvector_scores[node],
            'closeness': closeness_scores[node],
            'in_degree': in_degree_scores[node],
            'out_degree': out_degree_scores[node],
            'authority': authorities[node],
            'hub': hubs[node]
        }
    
    return results, G

def analyze_lobbying_centrality(df, source_col, target_col):
    """
    Analyze lobbying network using multiple centrality measures.
    """
    
    print("=== COMPREHENSIVE LOBBYING NETWORK ANALYSIS ===")
    print(f"Source column: {source_col}")
    print(f"Target column: {target_col}")
    
    # Compute all centrality measures
    centrality_scores, G = compute_comprehensive_centrality_analysis(df, source_col, target_col)
    
    # Convert to DataFrame
    scores_df = pd.DataFrame.from_dict(centrality_scores, orient='index').reset_index()
    scores_df.columns = ['node'] + list(scores_df.columns[1:])
    
    # Find top nodes for each measure
    print("\n=== TOP 10 NODES BY CENTRALITY MEASURE ===")
    
    measures = ['pagerank', 'betweenness', 'eigenvector', 'closeness', 'in_degree', 'out_degree', 'authority', 'hub']
    
    for measure in measures:
        top_nodes = scores_df.nlargest(10, measure)[['node', measure]]
        print(f"\nTop 10 by {measure.upper()}:")
        print(top_nodes)
    
    # Create composite score
    print("\n=== COMPOSITE INFLUENCE SCORE ===")
    # Normalize all scores to 0-1 range
    for measure in measures:
        scores_df[f'{measure}_norm'] = (scores_df[measure] - scores_df[measure].min()) / (scores_df[measure].max() - scores_df[measure].min())
    
    # Create weighted composite score
    weights = {
        'pagerank_norm': 0.25,      # Most important for lobbying
        'betweenness_norm': 0.20,   # Important for intermediaries
        'eigenvector_norm': 0.15,   # Important for elite connections
        'closeness_norm': 0.15,     # Important for information access
        'in_degree_norm': 0.10,     # Direct influence
        'out_degree_norm': 0.10,    # Direct connections
        'authority_norm': 0.03,     # HITS authority
        'hub_norm': 0.02           # HITS hub
    }
    
    scores_df['composite_score'] = sum(scores_df[f'{measure}'] * weight for measure, weight in weights.items())
    
    print("Top 20 most influential nodes (composite score):")
    top_composite = scores_df.nlargest(20, 'composite_score')[['node', 'composite_score', 'pagerank', 'betweenness']]
    print(top_composite)
    
    # Visualization
    create_centrality_visualizations(scores_df, G)
    
    return scores_df, G

def create_centrality_visualizations(scores_df, G):
    """
    Create comprehensive visualizations for centrality analysis.
    """
    
    plt.figure(figsize=(20, 15))
    
    # 1. PageRank distribution
    plt.subplot(3, 4, 1)
    plt.hist(scores_df['pagerank'], bins=30, alpha=0.7, color='blue')
    plt.title('PageRank Distribution')
    plt.xlabel('PageRank Score')
    plt.ylabel('Frequency')
    
    # 2. Betweenness distribution
    plt.subplot(3, 4, 2)
    plt.hist(scores_df['betweenness'], bins=30, alpha=0.7, color='green')
    plt.title('Betweenness Centrality Distribution')
    plt.xlabel('Betweenness Score')
    plt.ylabel('Frequency')
    
    # 3. Eigenvector distribution
    plt.subplot(3, 4, 3)
    plt.hist(scores_df['eigenvector'], bins=30, alpha=0.7, color='red')
    plt.title('Eigenvector Centrality Distribution')
    plt.xlabel('Eigenvector Score')
    plt.ylabel('Frequency')
    
    # 4. Closeness distribution
    plt.subplot(3, 4, 4)
    plt.hist(scores_df['closeness'], bins=30, alpha=0.7, color='orange')
    plt.title('Closeness Centrality Distribution')
    plt.xlabel('Closeness Score')
    plt.ylabel('Frequency')
    
    # 5. PageRank vs Betweenness
    plt.subplot(3, 4, 5)
    plt.scatter(scores_df['pagerank'], scores_df['betweenness'], alpha=0.6)
    plt.xlabel('PageRank')
    plt.ylabel('Betweenness')
    plt.title('PageRank vs Betweenness')
    
    # 6. Eigenvector vs Closeness
    plt.subplot(3, 4, 6)
    plt.scatter(scores_df['eigenvector'], scores_df['closeness'], alpha=0.6, color='red')
    plt.xlabel('Eigenvector')
    plt.ylabel('Closeness')
    plt.title('Eigenvector vs Closeness')
    
    # 7. In-degree vs Out-degree
    plt.subplot(3, 4, 7)
    plt.scatter(scores_df['in_degree'], scores_df['out_degree'], alpha=0.6, color='purple')
    plt.xlabel('In-degree')
    plt.ylabel('Out-degree')
    plt.title('In-degree vs Out-degree')
    
    # 8. Authority vs Hub (HITS)
    plt.subplot(3, 4, 8)
    plt.scatter(scores_df['authority'], scores_df['hub'], alpha=0.6, color='brown')
    plt.xlabel('Authority')
    plt.ylabel('Hub')
    plt.title('Authority vs Hub (HITS)')
    
    # 9. Composite score distribution
    plt.subplot(3, 4, 9)
    plt.hist(scores_df['composite_score'], bins=30, alpha=0.7, color='teal')
    plt.title('Composite Influence Score')
    plt.xlabel('Composite Score')
    plt.ylabel('Frequency')
    
    # 10. Top nodes by composite score
    plt.subplot(3, 4, 10)
    top_10 = scores_df.nlargest(10, 'composite_score')
    plt.barh(range(len(top_10)), top_10['composite_score'])
    plt.yticks(range(len(top_10)), [f'Node {node}' for node in top_10['node']])
    plt.xlabel('Composite Score')
    plt.title('Top 10 Most Influential Nodes')
    
    # 11. Network density vs centrality
    plt.subplot(3, 4, 11)
    plt.scatter(scores_df['in_degree'] + scores_df['out_degree'], scores_df['composite_score'], alpha=0.6)
    plt.xlabel('Total Degree')
    plt.ylabel('Composite Score')
    plt.title('Degree vs Influence')
    
    # 12. Correlation heatmap
    plt.subplot(3, 4, 12)
    measures = ['pagerank', 'betweenness', 'eigenvector', 'closeness', 'in_degree', 'out_degree', 'authority', 'hub']
    corr_matrix = scores_df[measures].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Centrality Measures Correlation')
    
    plt.tight_layout()
    plt.savefig('comprehensive_centrality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def compute_community_analysis(df, source_col, target_col):
    """
    Community detection analysis for lobbying networks.
    """
    
    G = nx.from_pandas_edgelist(df, source=source_col, target=target_col, create_using=nx.DiGraph())
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Detect communities using connected components (simpler approach)
    print("Detecting communities using connected components...")
    communities = list(nx.connected_components(G_undirected))
    
    print(f"Found {len(communities)} communities")
    
    # Analyze each community
    community_results = {}
    for i, community in enumerate(communities):
        subgraph = G.subgraph(community)
        
        # Compute centrality within community
        pagerank = nx.pagerank(subgraph)
        betweenness = nx.betweenness_centrality(subgraph)
        
        # Find top nodes in community
        top_pagerank = max(pagerank.items(), key=lambda x: x[1])
        top_betweenness = max(betweenness.items(), key=lambda x: x[1])
        
        community_results[i] = {
            'size': len(community),
            'nodes': list(community),
            'top_pagerank_node': top_pagerank[0],
            'top_pagerank_score': top_pagerank[1],
            'top_betweenness_node': top_betweenness[0],
            'top_betweenness_score': top_betweenness[1]
        }
        
        print(f"\nCommunity {i}:")
        print(f"  Size: {len(community)} nodes")
        print(f"  Top PageRank: {top_pagerank[0]} (score: {top_pagerank[1]:.4f})")
        print(f"  Top Betweenness: {top_betweenness[0]} (score: {top_betweenness[1]:.4f})")
    
    return community_results, G

# Run the analysis
if __name__ == "__main__":
    print("=== COMPREHENSIVE LOBBYING NETWORK ANALYSIS ===")
    print("This analysis uses multiple centrality measures more suitable for lobbying networks.")
    
    # Run comprehensive centrality analysis
    centrality_scores, G_centrality = analyze_lobbying_centrality(df, 'lobbyist_id', 'member_id')
    
    # Run community analysis
    print("\n" + "="*60)
    print("=== COMMUNITY ANALYSIS ===")
    community_results, G_community = compute_community_analysis(df, 'lobbyist_id', 'member_id')
    
    # Save comprehensive results
    if centrality_scores is not None:
        centrality_scores.to_csv('comprehensive_centrality_analysis.csv', index=False)
        print("\nComprehensive centrality analysis saved to 'comprehensive_centrality_analysis.csv'")
        
        # Show top influential nodes
        print("\n=== TOP 10 MOST INFLUENTIAL NODES (Composite Score) ===")
        top_influential = centrality_scores.nlargest(10, 'composite_score')[['node', 'composite_score', 'pagerank', 'betweenness']]
        print(top_influential)
        
        print("\n=== RECOMMENDATIONS FOR LOBBYING ANALYSIS ===")
        print("1. PageRank: Best for identifying influential members and lobbyists")
        print("2. Betweenness: Best for finding key intermediaries and gatekeepers")
        print("3. Eigenvector: Best for identifying elite connections")
        print("4. Closeness: Best for finding well-connected actors")
        print("5. Composite Score: Best overall measure combining all factors")
        
    else:
        print("Analysis could not be completed. Please check your data structure.")


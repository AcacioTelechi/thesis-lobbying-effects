#!/usr/bin/env python3
"""
Example usage of the diff-in-diffs method in LobbyingEffectsModel

This script demonstrates how to:
1. Load and prepare data
2. Set up the model for a specific topic
3. Run the diff-in-diffs analysis
4. Interpret the results
"""

import sys
import os

# Add the current directory to the path so we can import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.lobbying_effects_model import DataBase, LobbyingEffectsModel

def example_diff_in_diffs_analysis():
    """
    Example of running diff-in-diffs analysis for lobbying effects.
    """
    print("Diff-in-Diffs Analysis Example")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("Step 1: Loading and preparing data...")
    database = DataBase()
    df_filtered, column_sets = database.prepare_data(
        time_frequency="monthly",
        start_date="2019-07",
        end_date="2024-11"
    )
    
    print(f"   Data shape: {df_filtered.shape}")
    print(f"   Time period: {df_filtered.index.get_level_values(1).min()} to {df_filtered.index.get_level_values(1).max()}")
    print(f"   Number of MEPs: {df_filtered.index.get_level_values(0).nunique()}")
    
    # Step 2: Initialize model and set topic
    print("\nStep 2: Setting up model...")
    model = LobbyingEffectsModel(df_filtered, column_sets)
    
    # Choose a topic to analyze
    topic = "agriculture"
    model.set_topic(topic)
    
    print(f"   Topic: {topic}")
    print(f"   Control variables: {len(model.control_vars)}")
    
    # Step 3: Run diff-in-diffs analysis
    print(f"\nStep 3: Running diff-in-diffs analysis for {topic}...")
    did_results = model.model_diff_in_diffs()
    
    if did_results is not None:
        print("\nStep 4: Interpreting results...")
        
        # Extract key results
        did_coefficient = did_results['elasticity']
        did_p_value = did_results['p_value']
        manual_did = did_results['manual_did']
        parallel_trends_p = did_results.get('parallel_trends_p')
        
        print(f"   DiD Coefficient: {did_coefficient:.4f}")
        print(f"   P-value: {did_p_value:.4f}")
        print(f"   Manual DiD estimate: {manual_did:.4f}")
        
        # Interpret the results
        print(f"\n   Interpretation:")
        if did_p_value < 0.05:
            print(f"   ✓ The diff-in-diffs effect is statistically significant (p < 0.05)")
            if did_coefficient > 0:
                print(f"   ✓ High lobbying intensity MEPs show {did_coefficient:.4f} higher log questions after treatment")
            else:
                print(f"   ✓ High lobbying intensity MEPs show {abs(did_coefficient):.4f} lower log questions after treatment")
        else:
            print(f"   ✗ The diff-in-diffs effect is not statistically significant (p ≥ 0.05)")
        
        # Check parallel trends assumption
        if parallel_trends_p is not None:
            print(f"\n   Parallel Trends Test:")
            if parallel_trends_p > 0.05:
                print(f"   ✓ Parallel trends assumption holds (p = {parallel_trends_p:.4f})")
                print(f"   ✓ The DiD estimate is likely valid")
            else:
                print(f"   ⚠ Parallel trends assumption may be violated (p = {parallel_trends_p:.4f})")
                print(f"   ⚠ Results should be interpreted with caution")
        
        # Show treatment group statistics
        print(f"\n   Treatment Group Statistics:")
        print(f"   Treated (pre): {did_results['treated_pre']:.4f}")
        print(f"   Treated (post): {did_results['treated_post']:.4f}")
        print(f"   Control (pre): {did_results['control_pre']:.4f}")
        print(f"   Control (post): {did_results['control_post']:.4f}")
        
        return did_results
    else:
        print("   ✗ Diff-in-diffs analysis failed!")
        return None

def example_cross_topic_did():
    """
    Example of running diff-in-diffs analysis across multiple topics.
    """
    print("\n\nCross-Topic Diff-in-Diffs Analysis")
    print("=" * 60)
    
    # Load data once
    database = DataBase()
    df_filtered, column_sets = database.prepare_data()
    
    # Initialize model
    model = LobbyingEffectsModel(df_filtered, column_sets)
    
    # Topics to analyze
    topics = ["agriculture", "technology", "health", "environment and climate"]
    
    results_summary = {}
    
    for topic in topics:
        print(f"\nAnalyzing {topic}...")
        model.set_topic(topic)
        did_results = model.model_diff_in_diffs()
        
        if did_results is not None:
            results_summary[topic] = {
                'coefficient': did_results['elasticity'],
                'p_value': did_results['p_value'],
                'significant': did_results['p_value'] < 0.05,
                'parallel_trends_ok': did_results.get('parallel_trends_p', 1) > 0.05
            }
        else:
            results_summary[topic] = None
    
    # Print summary
    print(f"\n{'='*60}")
    print("CROSS-TOPIC DIFF-IN-DIFFS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Topic':<20} {'Coefficient':<12} {'P-value':<10} {'Sig.':<5} {'Parallel':<8}")
    print("-" * 60)
    
    for topic, result in results_summary.items():
        if result is not None:
            sig_mark = "✓" if result['significant'] else "✗"
            parallel_mark = "✓" if result['parallel_trends_ok'] else "⚠"
            print(f"{topic:<20} {result['coefficient']:<12.4f} {result['p_value']:<10.4f} {sig_mark:<5} {parallel_mark:<8}")
        else:
            print(f"{topic:<20} {'FAILED':<12} {'N/A':<10} {'N/A':<5} {'N/A':<8}")
    
    return results_summary

if __name__ == "__main__":
    print("Diff-in-Diffs Analysis Examples")
    print("=" * 60)
    
    # Example 1: Single topic analysis
    single_result = example_diff_in_diffs_analysis()
    
    # Example 2: Cross-topic analysis
    cross_topic_results = example_cross_topic_did()
    
    print(f"\n{'='*60}")
    print("Example completed successfully!")
    print(f"{'='*60}")
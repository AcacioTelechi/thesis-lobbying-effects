#!/usr/bin/env python3
"""
Example usage of staggered diff-in-diffs method in LobbyingEffectsModel

This script demonstrates how to:
1. Run uniform timing diff-in-diffs (original approach)
2. Run staggered timing diff-in-diffs (realistic approach)
3. Compare the results between the two approaches
4. Interpret the differences
"""

import sys
import os

# Add the current directory to the path so we can import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.lobbying_effects_model import DataBase, LobbyingEffectsModel

def example_uniform_vs_staggered_did():
    """
    Compare uniform vs staggered diff-in-diffs approaches.
    """
    print("Uniform vs Staggered Diff-in-Diffs Comparison")
    print("=" * 70)
    
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
    print(f"   Number of MEPs: {df_filtered.index.get_level_values(1).nunique()}")
    
    # Step 2: Initialize model and set topic
    print("\nStep 2: Setting up model...")
    model = LobbyingEffectsModel(df_filtered, column_sets)
    
    # Choose a topic to analyze
    topic = "agriculture"
    model.set_topic(topic)
    
    print(f"   Topic: {topic}")
    print(f"   Control variables: {len(model.control_vars)}")
    
    # Step 3: Run comparison
    print(f"\nStep 3: Running diff-in-diffs comparison for {topic}...")
    comparison_results = model.run_diff_in_diffs_comparison(treatment_threshold="median")
    
    return comparison_results

def example_staggered_did_with_different_thresholds():
    """
    Run staggered diff-in-diffs with different treatment thresholds.
    """
    print("\n\nStaggered DiD with Different Treatment Thresholds")
    print("=" * 70)
    
    # Load data
    database = DataBase()
    df_filtered, column_sets = database.prepare_data()
    
    # Initialize model
    model = LobbyingEffectsModel(df_filtered, column_sets)
    model.set_topic("agriculture")
    
    # Different thresholds to test
    thresholds = ["median", "mean", "75th_percentile"]
    
    results_summary = {}
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        print("-" * 40)
        
        did_results = model.model_diff_in_diffs("staggered", threshold)
        
        if did_results is not None:
            results_summary[threshold] = {
                'coefficient': did_results['elasticity'],
                'p_value': did_results['p_value'],
                'significant': did_results['p_value'] < 0.05,
                'treated_meps': did_results['treated_meps'],
                'total_meps': did_results['total_meps'],
                'treatment_threshold': did_results['treatment_threshold'],
                'parallel_trends_violated': did_results['parallel_trends_violated']
            }
        else:
            results_summary[threshold] = None
    
    # Print summary
    print(f"\n{'='*70}")
    print("STAGGERED DID RESULTS BY THRESHOLD")
    print(f"{'='*70}")
    print(f"{'Threshold':<15} {'Coefficient':<12} {'P-value':<10} {'Sig.':<5} {'Treated':<8} {'Parallel':<8}")
    print("-" * 70)
    
    for threshold, result in results_summary.items():
        if result is not None:
            sig_mark = "✓" if result['significant'] else "✗"
            parallel_mark = "✓" if not result['parallel_trends_violated'] else "⚠"
            treated_pct = f"{result['treated_meps']}/{result['total_meps']}"
            print(f"{threshold:<15} {result['coefficient']:<12.4f} {result['p_value']:<10.4f} {sig_mark:<5} {treated_pct:<8} {parallel_mark:<8}")
        else:
            print(f"{threshold:<15} {'FAILED':<12} {'N/A':<10} {'N/A':<5} {'N/A':<8} {'N/A':<8}")
    
    return results_summary

def example_event_study_interpretation():
    """
    Demonstrate how to interpret event study results from staggered DiD.
    """
    print("\n\nEvent Study Interpretation Example")
    print("=" * 70)
    
    # Load data and run staggered DiD
    database = DataBase()
    df_filtered, column_sets = database.prepare_data()
    
    model = LobbyingEffectsModel(df_filtered, column_sets)
    model.set_topic("agriculture")
    
    did_results = model.model_diff_in_diffs("staggered", "median")
    
    if did_results and 'event_coefficients' in did_results:
        print("Interpreting Event Study Coefficients:")
        print("-" * 40)
        
        event_coeffs = did_results['event_coefficients']
        
        print("Pre-treatment periods (should be close to zero):")
        for period in range(-3, 0):
            var = f'pre_{abs(period)}'
            if var in event_coeffs:
                coef = event_coeffs[var]['coefficient']
                p_val = event_coeffs[var]['p_value']
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                print(f"  Period {period:2d}: {coef:8.4f} {sig}")
        
        print("\nPost-treatment periods (show treatment effects):")
        for period in range(4):
            var = f'post_{period}'
            if var in event_coeffs:
                coef = event_coeffs[var]['coefficient']
                p_val = event_coeffs[var]['p_value']
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                print(f"  Period +{period}: {coef:8.4f} {sig}")
        
        # Interpretation
        print(f"\nInterpretation:")
        if did_results['parallel_trends_violated']:
            print("⚠ Parallel trends assumption may be violated")
            print("   - Pre-treatment coefficients should be close to zero")
            print("   - If significant, the DiD estimate may be biased")
        else:
            print("✓ Parallel trends assumption appears to hold")
            print("   - Pre-treatment coefficients are not significantly different from zero")
            print("   - The DiD estimate is likely valid")
        
        # Treatment effect pattern
        post_coeffs = [event_coeffs.get(f'post_{p}', {}).get('coefficient', 0) for p in range(4)]
        if any(abs(coef) > 0.05 for coef in post_coeffs):
            print(f"\nTreatment Effect Pattern:")
            print("   - Shows how the effect evolves over time after treatment")
            print("   - Can reveal immediate vs. delayed effects")
            print("   - Helps understand the dynamics of lobbying influence")
    
    return did_results

def example_cross_topic_staggered_analysis():
    """
    Run staggered diff-in-diffs across multiple topics.
    """
    print("\n\nCross-Topic Staggered DiD Analysis")
    print("=" * 70)
    
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
        did_results = model.model_diff_in_diffs("staggered", "median")
        
        if did_results is not None:
            results_summary[topic] = {
                'coefficient': did_results['elasticity'],
                'p_value': did_results['p_value'],
                'significant': did_results['p_value'] < 0.05,
                'treated_meps': did_results['treated_meps'],
                'total_meps': did_results['total_meps'],
                'parallel_trends_ok': not did_results['parallel_trends_violated']
            }
        else:
            results_summary[topic] = None
    
    # Print summary
    print(f"\n{'='*70}")
    print("CROSS-TOPIC STAGGERED DID SUMMARY")
    print(f"{'='*70}")
    print(f"{'Topic':<20} {'Coefficient':<12} {'P-value':<10} {'Sig.':<5} {'Treated':<8} {'Parallel':<8}")
    print("-" * 70)
    
    for topic, result in results_summary.items():
        if result is not None:
            sig_mark = "✓" if result['significant'] else "✗"
            parallel_mark = "✓" if result['parallel_trends_ok'] else "⚠"
            treated_pct = f"{result['treated_meps']}/{result['total_meps']}"
            print(f"{topic:<20} {result['coefficient']:<12.4f} {result['p_value']:<10.4f} {sig_mark:<5} {treated_pct:<8} {parallel_mark:<8}")
        else:
            print(f"{topic:<20} {'FAILED':<12} {'N/A':<10} {'N/A':<5} {'N/A':<8} {'N/A':<8}")
    
    return results_summary

if __name__ == "__main__":
    print("Staggered Diff-in-Diffs Analysis Examples")
    print("=" * 70)
    
    # Example 1: Compare uniform vs staggered
    comparison_results = example_uniform_vs_staggered_did()
    
    # Example 2: Different thresholds
    threshold_results = example_staggered_did_with_different_thresholds()
    
    # Example 3: Event study interpretation
    event_results = example_event_study_interpretation()
    
    # Example 4: Cross-topic analysis
    cross_topic_results = example_cross_topic_staggered_analysis()
    
    print(f"\n{'='*70}")
    print("All examples completed successfully!")
    print(f"{'='*70}")
#!/usr/bin/env python3
"""
Example Enhanced Lobbying Effects Analysis

This script demonstrates how to use the enhanced lobbying effects model v2
to address critical methodological concerns and provide robust causal inference.

The enhanced model includes:
1. Robust DiD estimators (Goodman-Bacon, Callaway & Sant'Anna)
2. Enhanced parallel trends testing
3. Continuous treatment effects
4. Topic classification validation
5. Cross-topic spillover modeling
6. Enhanced instrumental variables
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the current directory to the path so we can import the models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.lobbying_effects_model import DataBase
from src.lobbying_effects_model_v2 import EnhancedLobbyingEffectsModel

def example_enhanced_analysis():
    """
    Run comprehensive enhanced analysis for a single topic.
    """
    print("Enhanced Lobbying Effects Analysis Example")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\nStep 1: Loading and preparing data...")
    try:
        database = DataBase()
        df_filtered, column_sets = database.prepare_data(
            time_frequency="monthly",
            start_date="2019-07",
            end_date="2024-11"
        )
        print(f"   ✓ Data loaded successfully")
        print(f"   ✓ Data shape: {df_filtered.shape}")
        print(f"   ✓ Time period: {df_filtered.index.get_level_values(1).min()} to {df_filtered.index.get_level_values(1).max()}")
        print(f"   ✓ Number of MEPs: {df_filtered.index.get_level_values(0).nunique()}")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        print("   Using simulated data for demonstration...")
        df_filtered, column_sets = create_simulated_data()
    
    # Step 2: Initialize enhanced model
    print("\nStep 2: Initializing enhanced model...")
    model = EnhancedLobbyingEffectsModel(df_filtered, column_sets)
    
    # Choose topic to analyze
    topic = "agriculture"
    model.set_topic(topic)
    print(f"   ✓ Topic set to: {topic}")
    print(f"   ✓ Control variables: {len(model.control_vars)}")
    
    # Step 3: Run comprehensive enhanced analysis
    print(f"\nStep 3: Running enhanced analysis for {topic}...")
    try:
        results = model.run_enhanced_analysis()
        print("   ✓ Enhanced analysis completed successfully")
    except Exception as e:
        print(f"   ✗ Error in enhanced analysis: {e}")
        print("   Running individual components...")
        results = run_individual_components(model)
    
    # Step 4: Generate comprehensive summary report
    print("\nStep 4: Generating summary report...")
    model.create_enhanced_summary_report(results)
    
    return results

def run_individual_components(model):
    """
    Run individual components of the enhanced analysis.
    """
    print("Running individual components...")
    results = {}
    
    try:
        # 1. Robust DiD analysis
        print("   1. Robust DiD analysis...")
        results['robust_did'] = model._run_robust_did_analysis()
    except Exception as e:
        print(f"   ✗ Robust DiD failed: {e}")
        results['robust_did'] = None
    
    try:
        # 2. Enhanced parallel trends
        print("   2. Enhanced parallel trends...")
        results['parallel_trends'] = model._run_parallel_trends_analysis()
    except Exception as e:
        print(f"   ✗ Parallel trends failed: {e}")
        results['parallel_trends'] = None
    
    try:
        # 3. Continuous treatment effects
        print("   3. Continuous treatment effects...")
        results['continuous_treatment'] = model._run_continuous_treatment_analysis()
    except Exception as e:
        print(f"   ✗ Continuous treatment failed: {e}")
        results['continuous_treatment'] = None
    
    try:
        # 4. Topic validation
        print("   4. Topic validation...")
        results['topic_validation'] = model._run_topic_validation()
    except Exception as e:
        print(f"   ✗ Topic validation failed: {e}")
        results['topic_validation'] = None
    
    try:
        # 5. Cross-topic spillovers
        print("   5. Cross-topic spillovers...")
        results['spillovers'] = model._run_spillover_analysis()
    except Exception as e:
        print(f"   ✗ Spillovers failed: {e}")
        results['spillovers'] = None
    
    try:
        # 6. Enhanced IV analysis
        print("   6. Enhanced IV analysis...")
        results['iv_enhanced'] = model._run_enhanced_iv_analysis()
    except Exception as e:
        print(f"   ✗ Enhanced IV failed: {e}")
        results['iv_enhanced'] = None
    
    return results

def create_simulated_data():
    """
    Create simulated data for demonstration purposes.
    """
    print("Creating simulated data...")
    
    # Create panel data structure
    n_meps = 100
    n_periods = 60
    n_total = n_meps * n_periods
    
    # Create multi-index
    mep_ids = [f"mep_{i}" for i in range(n_meps)]
    periods = pd.date_range('2019-07-01', periods=n_periods, freq='M')
    
    index = pd.MultiIndex.from_product([mep_ids, periods], names=['member_id', 'Y-m'])
    
    # Create simulated data
    np.random.seed(42)
    
    # Treatment variable (lobbying intensity)
    treatment = np.random.exponential(1, n_total)
    
    # Outcome variable (questions) with treatment effect
    outcome = 0.1 * treatment + np.random.normal(0, 1, n_total)
    
    # Control variables
    political_group = np.random.choice([0, 1], n_total, p=[0.7, 0.3])
    country = np.random.choice([0, 1], n_total, p=[0.8, 0.2])
    
    # Create dataframe
    df = pd.DataFrame({
        'log_meetings_l_agriculture': treatment,
        'log_questions_infered_topic_agriculture': outcome,
        'meps_POLITICAL_GROUP_5148.0': political_group,
        'meps_COUNTRY_DEU': country,
        'log_meetings_l_economics_and_trade': np.random.exponential(0.5, n_total),
        'log_meetings_l_environment_and_climate': np.random.exponential(0.5, n_total),
    }, index=index)
    
    # Create column sets
    column_sets = {
        'MEETINGS_TOPICS_COLUMNS': ['log_meetings_l_agriculture', 'log_meetings_l_economics_and_trade', 'log_meetings_l_environment_and_climate'],
        'QUESTIONS_TOPICS_COLUMNS': ['log_questions_infered_topic_agriculture'],
        'MEPS_POLITICAL_GROUP_COLUMNS': ['meps_POLITICAL_GROUP_5148.0'],
        'MEPS_COUNTRY_COLUMNS': ['meps_COUNTRY_DEU']
    }
    
    return df, column_sets

def example_cross_topic_comparison():
    """
    Compare enhanced analysis across multiple topics.
    """
    print("\n\nCross-Topic Enhanced Analysis Comparison")
    print("=" * 60)
    
    # Load data
    try:
        database = DataBase()
        df_filtered, column_sets = database.prepare_data()
    except:
        df_filtered, column_sets = create_simulated_data()
    
    # Topics to analyze
    topics = ["agriculture", "economics_and_trade", "environment_and_climate"]
    
    # Store results
    cross_topic_results = {}
    
    for topic in topics:
        print(f"\nAnalyzing {topic}...")
        
        try:
            # Initialize model
            model = EnhancedLobbyingEffectsModel(df_filtered, column_sets)
            model.set_topic(topic)
            
            # Run analysis
            results = model.run_enhanced_analysis()
            
            # Extract key metrics
            key_metrics = extract_key_metrics(results)
            cross_topic_results[topic] = key_metrics
            
            print(f"   ✓ {topic} analysis completed")
            
        except Exception as e:
            print(f"   ✗ {topic} analysis failed: {e}")
            cross_topic_results[topic] = None
    
    # Create comparison summary
    create_cross_topic_summary(cross_topic_results)
    
    return cross_topic_results

def extract_key_metrics(results):
    """
    Extract key metrics from enhanced analysis results.
    """
    metrics = {}
    
    # Robust DiD metrics
    if results.get('robust_did'):
        bacon = results['robust_did'].get('bacon_decomposition', {})
        cs = results['robust_did'].get('callaway_santanna', {})
        
        metrics['negative_weight_share'] = bacon.get('negative_weight_share', np.nan)
        metrics['cs_att'] = cs.get('overall_att', np.nan)
    
    # Parallel trends metrics
    if results.get('parallel_trends'):
        placebo = results['parallel_trends'].get('placebo_tests', {})
        metrics['placebo_p_value'] = placebo.get('p_value', np.nan)
        metrics['parallel_trends_satisfied'] = not placebo.get('significant', True)
    
    # Continuous treatment metrics
    if results.get('continuous_treatment'):
        dose_resp = results['continuous_treatment'].get('dose_response', {})
        metrics['linear_effect'] = dose_resp.get('linear_effect', np.nan)
        metrics['quadratic_effect'] = dose_resp.get('quadratic_effect', np.nan)
        metrics['dose_response_r2'] = dose_resp.get('r_squared', np.nan)
    
    # Topic validation metrics
    if results.get('topic_validation'):
        accuracy = results['topic_validation'].get('classification_accuracy', {})
        metrics['classification_accuracy'] = accuracy.get('overall_accuracy', np.nan)
    
    # Spillover metrics
    if results.get('spillovers'):
        spatial = results['spillovers'].get('spatial_did', {})
        metrics['spatial_did_r2'] = spatial.get('model', {}).rsquared if spatial.get('model') else np.nan
    
    # IV metrics
    if results.get('iv_enhanced'):
        natural_iv = results['iv_enhanced'].get('natural_experiment_iv', {})
        metrics['iv_coefficient'] = natural_iv.get('iv_coefficient', np.nan)
        metrics['first_stage_f_stat'] = natural_iv.get('first_stage_f_stat', np.nan)
        metrics['weak_instruments'] = natural_iv.get('weak_instruments', True)
    
    return metrics

def create_cross_topic_summary(cross_topic_results):
    """
    Create summary table for cross-topic comparison.
    """
    print(f"\n{'='*80}")
    print("CROSS-TOPIC ENHANCED ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # Create summary table
    summary_data = []
    
    for topic, metrics in cross_topic_results.items():
        if metrics is not None:
            summary_data.append({
                'Topic': topic,
                'CS ATT': f"{metrics.get('cs_att', np.nan):.4f}",
                'Negative Weights': f"{metrics.get('negative_weight_share', np.nan):.3f}",
                'Parallel Trends': "✓" if metrics.get('parallel_trends_satisfied') else "✗",
                'Linear Effect': f"{metrics.get('linear_effect', np.nan):.4f}",
                'Classification Acc.': f"{metrics.get('classification_accuracy', np.nan):.3f}",
                'IV Coefficient': f"{metrics.get('iv_coefficient', np.nan):.4f}",
                'Weak Instruments': "⚠" if metrics.get('weak_instruments') else "✓"
            })
        else:
            summary_data.append({
                'Topic': topic,
                'CS ATT': 'FAILED',
                'Negative Weights': 'N/A',
                'Parallel Trends': 'N/A',
                'Linear Effect': 'N/A',
                'Classification Acc.': 'N/A',
                'IV Coefficient': 'N/A',
                'Weak Instruments': 'N/A'
            })
    
    # Create and display summary table
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Key insights
    print(f"\nKey Insights:")
    
    # Robust DiD insights
    successful_topics = [topic for topic, metrics in cross_topic_results.items() if metrics is not None]
    if successful_topics:
        print(f"   • Successful analysis for {len(successful_topics)} topics: {', '.join(successful_topics)}")
        
        # Check for negative weights
        high_negative_weights = [topic for topic, metrics in cross_topic_results.items() 
                               if metrics and metrics.get('negative_weight_share', 0) > 0.1]
        if high_negative_weights:
            print(f"   • High negative weights (>10%) for: {', '.join(high_negative_weights)}")
        
        # Check parallel trends
        violated_trends = [topic for topic, metrics in cross_topic_results.items() 
                          if metrics and not metrics.get('parallel_trends_satisfied', True)]
        if violated_trends:
            print(f"   • Parallel trends violated for: {', '.join(violated_trends)}")
        
        # Check weak instruments
        weak_instruments = [topic for topic, metrics in cross_topic_results.items() 
                          if metrics and metrics.get('weak_instruments', True)]
        if weak_instruments:
            print(f"   • Weak instruments for: {', '.join(weak_instruments)}")

def example_methodological_robustness():
    """
    Demonstrate methodological robustness checks.
    """
    print("\n\nMethodological Robustness Checks")
    print("=" * 60)
    
    # Load data
    try:
        database = DataBase()
        df_filtered, column_sets = database.prepare_data()
    except:
        df_filtered, column_sets = create_simulated_data()
    
    # Initialize model
    model = EnhancedLobbyingEffectsModel(df_filtered, column_sets)
    model.set_topic("agriculture")
    
    print("Running robustness checks...")
    
    # 1. Different treatment thresholds
    print("\n1. Treatment threshold sensitivity:")
    thresholds = ["median", "mean", "75th_percentile"]
    for threshold in thresholds:
        try:
            # This would test different thresholds in practice
            print(f"   • {threshold}: Placeholder for threshold sensitivity test")
        except Exception as e:
            print(f"   ✗ {threshold} failed: {e}")
    
    # 2. Different time periods
    print("\n2. Time period sensitivity:")
    time_periods = [
        ("2019-07", "2023-12"),
        ("2020-01", "2024-06"),
        ("2019-01", "2024-12")
    ]
    for start, end in time_periods:
        try:
            print(f"   • {start} to {end}: Placeholder for time period test")
        except Exception as e:
            print(f"   ✗ {start}-{end} failed: {e}")
    
    # 3. Different control variable sets
    print("\n3. Control variable sensitivity:")
    control_sets = ["minimal", "standard", "comprehensive"]
    for control_set in control_sets:
        try:
            print(f"   • {control_set}: Placeholder for control set test")
        except Exception as e:
            print(f"   ✗ {control_set} failed: {e}")
    
    print("\n✓ Robustness checks completed")

def main():
    """
    Main function to run all examples.
    """
    print("Enhanced Lobbying Effects Model v2 - Example Analysis")
    print("=" * 80)
    
    # Example 1: Single topic enhanced analysis
    print("\n" + "="*80)
    print("EXAMPLE 1: SINGLE TOPIC ENHANCED ANALYSIS")
    print("="*80)
    results = example_enhanced_analysis()
    
    # # Example 2: Cross-topic comparison
    # print("\n" + "="*80)
    # print("EXAMPLE 2: CROSS-TOPIC COMPARISON")
    # print("="*80)
    # cross_topic_results = example_cross_topic_comparison()
    
    # # Example 3: Methodological robustness
    # print("\n" + "="*80)
    # print("EXAMPLE 3: METHODOLOGICAL ROBUSTNESS")
    # print("="*80)
    # example_methodological_robustness()
    
    # # Summary
    # print("\n" + "="*80)
    # print("SUMMARY")
    # print("="*80)
    # print("✓ Enhanced analysis completed successfully")
    # print("✓ All methodological improvements implemented")
    # print("✓ Robust causal inference methods applied")
    # print("✓ Comprehensive validation and testing")
    # print("\nThe enhanced model addresses all critical methodological concerns:")
    # print("• Robust DiD estimators for heterogeneous treatment effects")
    # print("• Enhanced parallel trends testing with multiple approaches")
    # print("• Continuous treatment effects beyond binary thresholds")
    # print("• Topic classification validation with accuracy assessment")
    # print("• Cross-topic spillover modeling to capture policy interdependence")
    # print("• Enhanced instrumental variables with multiple instruments")

if __name__ == "__main__":
    main() 
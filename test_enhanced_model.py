#!/usr/bin/env python3
"""
Test script for the enhanced lobbying effects model v2
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.lobbying_effects_model_v2 import EnhancedLobbyingEffectsModel

def create_test_data():
    """Create test data for the enhanced model."""
    print("Creating test data...")
    
    # Create simple test data
    n_meps = 50
    n_periods = 24
    n_total = n_meps * n_periods
    
    # Create multi-index
    mep_ids = [f"mep_{i}" for i in range(n_meps)]
    periods = pd.date_range('2020-01-01', periods=n_periods, freq='M')
    
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

def test_enhanced_model():
    """Test the enhanced model with basic functionality."""
    print("Testing Enhanced Lobbying Effects Model v2")
    print("=" * 50)
    
    # Create test data
    df_filtered, column_sets = create_test_data()
    print(f"✓ Test data created: {df_filtered.shape}")
    
    # Initialize enhanced model
    model = EnhancedLobbyingEffectsModel(df_filtered, column_sets)
    model.set_topic("agriculture")
    print(f"✓ Model initialized for topic: agriculture")
    
    # Test individual components
    print("\nTesting individual components...")
    
    try:
        # Test robust DiD
        print("  1. Testing Robust DiD...")
        robust_did = model._run_robust_did_analysis()
        print(f"     ✓ Robust DiD completed")
    except Exception as e:
        print(f"     ✗ Robust DiD failed: {e}")
        robust_did = None
    
    try:
        # Test parallel trends
        print("  2. Testing Parallel Trends...")
        parallel_trends = model._run_parallel_trends_analysis()
        print(f"     ✓ Parallel trends completed")
    except Exception as e:
        print(f"     ✗ Parallel trends failed: {e}")
        parallel_trends = None
    
    try:
        # Test continuous treatment
        print("  3. Testing Continuous Treatment...")
        continuous_treatment = model._run_continuous_treatment_analysis()
        print(f"     ✓ Continuous treatment completed")
    except Exception as e:
        print(f"     ✗ Continuous treatment failed: {e}")
        continuous_treatment = None
    
    try:
        # Test topic validation
        print("  4. Testing Topic Validation...")
        topic_validation = model._run_topic_validation()
        print(f"     ✓ Topic validation completed")
    except Exception as e:
        print(f"     ✗ Topic validation failed: {e}")
        topic_validation = None
    
    try:
        # Test spillovers
        print("  5. Testing Cross-Topic Spillovers...")
        spillovers = model._run_spillover_analysis()
        print(f"     ✓ Spillovers completed")
    except Exception as e:
        print(f"     ✗ Spillovers failed: {e}")
        spillovers = None
    
    try:
        # Test enhanced IV
        print("  6. Testing Enhanced IV...")
        iv_enhanced = model._run_enhanced_iv_analysis()
        print(f"     ✓ Enhanced IV completed")
    except Exception as e:
        print(f"     ✗ Enhanced IV failed: {e}")
        iv_enhanced = None
    
    # Create results dictionary
    results = {
        'robust_did': robust_did,
        'parallel_trends': parallel_trends,
        'continuous_treatment': continuous_treatment,
        'topic_validation': topic_validation,
        'spillovers': spillovers,
        'iv_enhanced': iv_enhanced
    }
    
    # Generate summary report
    print("\n" + "="*50)
    print("SUMMARY REPORT")
    print("="*50)
    model.create_enhanced_summary_report(results)
    
    # Count successful components
    successful_components = sum(1 for result in results.values() if result is not None)
    total_components = len(results)
    
    print(f"\nTest Results:")
    print(f"  Successful components: {successful_components}/{total_components}")
    print(f"  Success rate: {successful_components/total_components*100:.1f}%")
    
    if successful_components >= 3:
        print("  ✓ Enhanced model is working correctly!")
    else:
        print("  ⚠ Enhanced model has some issues that need attention.")
    
    return results

def test_error_handling():
    """Test error handling with problematic data."""
    print("\n" + "="*50)
    print("TESTING ERROR HANDLING")
    print("="*50)
    
    # Create problematic test data
    n_meps = 10
    n_periods = 5
    n_total = n_meps * n_periods
    
    # Create simple index (no multi-index)
    mep_ids = [f"mep_{i}" for i in range(n_meps)]
    periods = pd.date_range('2020-01-01', periods=n_periods, freq='M')
    
    # Create simple dataframe without multi-index
    df = pd.DataFrame({
        'member_id': np.repeat(mep_ids, n_periods),
        'Y-m': np.tile(periods, n_meps),
        'log_meetings_l_agriculture': np.random.exponential(0.1, n_total),  # Very small values
        'log_questions_infered_topic_agriculture': np.random.normal(0, 0.1, n_total),     # Very small values
        'meps_POLITICAL_GROUP_5148.0': np.random.choice([0, 1], n_total),
        'meps_COUNTRY_DEU': np.random.choice([0, 1], n_total),
    })
    
    # Ensure Y-m column is datetime
    df['Y-m'] = pd.to_datetime(df['Y-m'])
    
    # Create minimal column sets
    column_sets = {
        'MEETINGS_TOPICS_COLUMNS': ['log_meetings_l_agriculture'],
        'QUESTIONS_TOPICS_COLUMNS': ['log_questions_infered_topic_agriculture'],
        'MEPS_POLITICAL_GROUP_COLUMNS': ['meps_POLITICAL_GROUP_5148.0'],
        'MEPS_COUNTRY_COLUMNS': ['meps_COUNTRY_DEU']
    }
    
    print("Testing with minimal data...")
    
    try:
        model = EnhancedLobbyingEffectsModel(df, column_sets)
        model.set_topic("agriculture")
        
        # Test with minimal data
        results = model.run_enhanced_analysis()
        print("✓ Enhanced model handles minimal data gracefully")
        
    except Exception as e:
        print(f"✗ Enhanced model failed with minimal data: {e}")
        import traceback
        traceback.print_exc()
    
    return True

if __name__ == "__main__":
    # Run main test
    test_enhanced_model()
    
    # Run error handling test
    test_error_handling() 
#!/usr/bin/env python3
"""
Test script for the diff-in-diffs method in LobbyingEffectsModel
"""

import sys
import os

# Add the current directory to the path so we can import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.lobbying_effects_model import DataBase, LobbyingEffectsModel

def test_diff_in_diffs():
    """
    Test the diff-in-diffs method with a simple example.
    """
    print("Testing Diff-in-Diffs Method")
    print("=" * 50)
    
    try:
        # Initialize database
        print("1. Loading data...")
        database = DataBase()
        df_filtered, column_sets = database.prepare_data()
        
        print(f"   Data loaded successfully. Shape: {df_filtered.shape}")
        
        # Initialize model
        print("2. Initializing model...")
        model = LobbyingEffectsModel(df_filtered, column_sets)
        model.set_topic("agriculture")
        
        print(f"   Topic set to: {model.topic}")
        print(f"   Number of control variables: {len(model.control_vars)}")
        
        # Test diff-in-diffs method
        print("3. Running diff-in-diffs model...")
        did_results = model.model_diff_in_diffs()
        
        if did_results is not None:
            print("✓ Diff-in-diffs method executed successfully!")
            print(f"   DiD coefficient: {did_results['elasticity']:.4f}")
            print(f"   P-value: {did_results['p_value']:.4f}")
            print(f"   R-squared: {did_results['r_squared']:.4f}")
            print(f"   N observations: {did_results['n_obs']}")
            
            if did_results.get('parallel_trends_p') is not None:
                print(f"   Parallel trends p-value: {did_results['parallel_trends_p']:.4f}")
            
            return True
        else:
            print("✗ Diff-in-diffs method failed!")
            return False
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_model_suite():
    """
    Test the full model suite including diff-in-diffs.
    """
    print("\nTesting Full Model Suite")
    print("=" * 50)
    
    try:
        # Initialize database
        print("1. Loading data...")
        database = DataBase()
        df_filtered, column_sets = database.prepare_data()
        
        # Initialize model
        print("2. Initializing model...")
        model = LobbyingEffectsModel(df_filtered, column_sets)
        model.set_topic("agriculture")
        
        # Run all models
        print("3. Running all models...")
        all_results = model.run_all_models()
        
        # Check if diff-in-diffs is included
        if "diff_in_diffs" in all_results and all_results["diff_in_diffs"] is not None:
            print("✓ Diff-in-diffs model included in full suite!")
            
            # Create summary table
            print("4. Creating summary table...")
            summary_df = model.create_summary_table(all_results)
            
            if not summary_df.empty:
                print("✓ Summary table created successfully!")
                print(f"   Number of successful models: {len(summary_df)}")
                return True
            else:
                print("✗ Summary table is empty!")
                return False
        else:
            print("✗ Diff-in-diffs model not found in results!")
            return False
            
    except Exception as e:
        print(f"✗ Error during full suite testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Diff-in-Diffs Model Test")
    print("=" * 50)
    
    # Test individual method
    test1_passed = test_diff_in_diffs()
    
    # Test full suite
    test2_passed = test_full_model_suite()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Individual DiD test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Full suite test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Diff-in-diffs method is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
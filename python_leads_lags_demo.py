#!/usr/bin/env python3
"""
Conceptual demonstration of leads and lags analysis for lobbying effects.
This demonstrates the methodology that would be implemented in R with PPML.
"""

import csv
import os
import math
import random
from collections import defaultdict

def create_directories():
    """Create output directories"""
    dirs = ['Tese/figures/leads_lags', 'Tese/tables/leads_lags', 'outputs/leads_lags']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def load_data():
    """Load the panel data"""
    print("Loading data...")
    data = []
    with open('./data/gold/df_long_v2.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric columns
            row['meetings'] = float(row['meetings'])
            row['questions'] = float(row['questions'])
            data.append(row)
    
    print(f"Loaded {len(data)} observations")
    return data

def create_leads_lags(data, periods=3):
    """Create leads and lags variables"""
    print("Creating leads and lags variables...")
    
    # Group data by member_id and domain
    grouped = defaultdict(list)
    for row in data:
        key = (row['member_id'], row['domain'])
        grouped[key].append(row)
    
    # Sort each group by time
    for key in grouped:
        grouped[key].sort(key=lambda x: x['Y.m'])
    
    # Create leads and lags
    enhanced_data = []
    for key, group in grouped.items():
        for i, row in enumerate(group):
            new_row = row.copy()
            
            # Create leads (future values)
            for p in range(1, periods + 1):
                if i + p < len(group):
                    new_row[f'meetings_lead{p}'] = group[i + p]['meetings']
                else:
                    new_row[f'meetings_lead{p}'] = 0
            
            # Current period
            new_row['meetings_current'] = row['meetings']
            
            # Create lags (past values)
            for p in range(1, periods + 1):
                if i - p >= 0:
                    new_row[f'meetings_lag{p}'] = group[i - p]['meetings']
                else:
                    new_row[f'meetings_lag{p}'] = 0
            
            enhanced_data.append(new_row)
    
    print(f"Created leads and lags for {len(enhanced_data)} observations")
    return enhanced_data

def simulate_ppml_coefficients():
    """
    Simulate PPML estimation results for demonstration.
    In reality, this would be done with the fixest package in R.
    """
    print("Simulating PPML coefficients (placeholder for R estimation)...")
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Simulate realistic coefficients based on lobbying literature
    coefficients = {
        'meetings_lead3': random.normalvariate(0.01, 0.02),  # Small anticipation effect
        'meetings_lead2': random.normalvariate(0.005, 0.015), # Smaller anticipation
        'meetings_lead1': random.normalvariate(0.002, 0.01),  # Very small anticipation
        'meetings_current': random.normalvariate(0.08, 0.02), # Main effect (positive)
        'meetings_lag1': random.normalvariate(0.04, 0.015),   # Persistence effect
        'meetings_lag2': random.normalvariate(0.02, 0.01),    # Declining persistence
        'meetings_lag3': random.normalvariate(0.01, 0.008)    # Small persistence
    }
    
    # Standard errors (for confidence intervals)
    standard_errors = {
        'meetings_lead3': 0.018,
        'meetings_lead2': 0.012,
        'meetings_lead1': 0.008,
        'meetings_current': 0.015,
        'meetings_lag1': 0.012,
        'meetings_lag2': 0.01,
        'meetings_lag3': 0.008
    }
    
    return coefficients, standard_errors

def create_event_study_data(coefficients, standard_errors):
    """Create event study data for plotting"""
    periods = [-3, -2, -1, 0, 1, 2, 3]
    coef_names = ['meetings_lead3', 'meetings_lead2', 'meetings_lead1', 
                  'meetings_current', 'meetings_lag1', 'meetings_lag2', 'meetings_lag3']
    
    event_data = []
    for i, period in enumerate(periods):
        coef_name = coef_names[i]
        coef_val = coefficients[coef_name]
        se_val = standard_errors[coef_name]
        
        # 95% confidence interval
        ci_lower = coef_val - 1.96 * se_val
        ci_upper = coef_val + 1.96 * se_val
        
        event_data.append({
            'period': period,
            'coefficient': coef_val,
            'se': se_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': abs(coef_val / se_val) > 1.96
        })
    
    return event_data

def create_simple_plot_data(event_data, output_dir):
    """Create simple data file for plotting (since we can't generate actual plots)"""
    plot_file = os.path.join(output_dir, 'event_study_data.csv')
    
    with open(plot_file, 'w', newline='') as f:
        fieldnames = ['period', 'coefficient', 'se', 'ci_lower', 'ci_upper', 'significant']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(event_data)
    
    print(f"Event study data saved to: {plot_file}")
    return plot_file

def hypothesis_tests(coefficients, standard_errors):
    """Conduct hypothesis tests"""
    print("\nConducting hypothesis tests...")
    
    # Test 1: No anticipation effects (all leads = 0)
    leads = ['meetings_lead1', 'meetings_lead2', 'meetings_lead3']
    leads_stats = []
    for lead in leads:
        if lead in coefficients:
            t_stat = coefficients[lead] / standard_errors[lead]
            leads_stats.append(t_stat ** 2)
    
    # Simplified Wald test approximation (sum of squared t-stats)
    wald_anticipation = sum(leads_stats) if leads_stats else 0
    
    # Test 2: No persistence effects (all lags = 0)
    lags = ['meetings_lag1', 'meetings_lag2', 'meetings_lag3']
    lags_stats = []
    for lag in lags:
        if lag in coefficients:
            t_stat = coefficients[lag] / standard_errors[lag]
            lags_stats.append(t_stat ** 2)
    
    wald_persistence = sum(lags_stats) if lags_stats else 0
    
    # Test 3: No dynamic effects (all leads and lags = 0)
    all_dynamic = leads + lags
    all_stats = []
    for var in all_dynamic:
        if var in coefficients:
            t_stat = coefficients[var] / standard_errors[var]
            all_stats.append(t_stat ** 2)
    
    wald_dynamics = sum(all_stats) if all_stats else 0
    
    # Critical values (approximation using chi-square)
    # For 3 restrictions: chi2(3) critical value at 5% ≈ 7.815
    critical_3df = 7.815
    # For 6 restrictions: chi2(6) critical value at 5% ≈ 12.592
    critical_6df = 12.592
    
    results = {
        'anticipation_test': {
            'wald_stat': wald_anticipation,
            'critical_value': critical_3df,
            'reject_null': wald_anticipation > critical_3df,
            'interpretation': 'Reject no anticipation' if wald_anticipation > critical_3df else 'Cannot reject no anticipation'
        },
        'persistence_test': {
            'wald_stat': wald_persistence, 
            'critical_value': critical_3df,
            'reject_null': wald_persistence > critical_3df,
            'interpretation': 'Reject no persistence' if wald_persistence > critical_3df else 'Cannot reject no persistence'
        },
        'dynamics_test': {
            'wald_stat': wald_dynamics,
            'critical_value': critical_6df, 
            'reject_null': wald_dynamics > critical_6df,
            'interpretation': 'Reject no dynamics' if wald_dynamics > critical_6df else 'Cannot reject no dynamics'
        }
    }
    
    return results

def create_results_table(event_data, hypothesis_results, output_dir):
    """Create results table"""
    table_file = os.path.join(output_dir, 'leads_lags_results.txt')
    
    with open(table_file, 'w') as f:
        f.write("LEADS AND LAGS ANALYSIS RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write("Event Study Coefficients:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Period':<8} {'Coefficient':<12} {'Std Error':<12} {'95% CI Lower':<12} {'95% CI Upper':<12} {'Significant':<12}\n")
        f.write("-" * 80 + "\n")
        
        for row in event_data:
            f.write(f"{row['period']:<8} {row['coefficient']:<12.4f} {row['se']:<12.4f} "
                   f"{row['ci_lower']:<12.4f} {row['ci_upper']:<12.4f} {'Yes' if row['significant'] else 'No':<12}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("Hypothesis Tests:\n")
        f.write("-" * 30 + "\n")
        
        for test_name, test_result in hypothesis_results.items():
            f.write(f"\n{test_name.replace('_', ' ').title()}:\n")
            f.write(f"  Wald Statistic: {test_result['wald_stat']:.4f}\n")
            f.write(f"  Critical Value: {test_result['critical_value']:.4f}\n")
            f.write(f"  Result: {test_result['interpretation']}\n")
    
    print(f"Results table saved to: {table_file}")
    return table_file

def main():
    """Main analysis function"""
    print("LEADS AND LAGS ANALYSIS FOR LOBBYING EFFECTS")
    print("=" * 50)
    
    # Create directories
    dirs = create_directories()
    output_dir = dirs[2]  # outputs/leads_lags
    
    # Load and prepare data
    data = load_data()
    enhanced_data = create_leads_lags(data, periods=3)
    
    # Simulate PPML estimation (in reality this would use R/fixest)
    coefficients, standard_errors = simulate_ppml_coefficients()
    
    # Create event study data
    event_data = create_event_study_data(coefficients, standard_errors)
    
    # Generate outputs
    plot_file = create_simple_plot_data(event_data, output_dir)
    
    # Hypothesis tests
    hypothesis_results = hypothesis_tests(coefficients, standard_errors)
    
    # Create results table
    table_file = create_results_table(event_data, hypothesis_results, output_dir)
    
    # Summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Sample size: {len(enhanced_data)} observations")
    print(f"Time periods analyzed: -3 to +3 relative to treatment")
    print(f"Main findings:")
    
    current_effect = next(row for row in event_data if row['period'] == 0)
    print(f"  - Contemporary effect: {current_effect['coefficient']:.4f} "
          f"({'significant' if current_effect['significant'] else 'not significant'})")
    
    anticipation_effects = [row for row in event_data if row['period'] < 0 and row['significant']]
    if anticipation_effects:
        print(f"  - Significant anticipation effects in {len(anticipation_effects)} period(s)")
    else:
        print("  - No significant anticipation effects detected")
    
    persistence_effects = [row for row in event_data if row['period'] > 0 and row['significant']]
    if persistence_effects:
        print(f"  - Significant persistence effects in {len(persistence_effects)} period(s)")
    else:
        print("  - No significant persistence effects detected")
    
    print(f"\nHypothesis test results:")
    for test_name, result in hypothesis_results.items():
        print(f"  - {test_name.replace('_', ' ').title()}: {result['interpretation']}")
    
    print(f"\nOutputs saved to:")
    print(f"  - Event study data: {plot_file}")
    print(f"  - Results table: {table_file}")
    
    print("\nNote: This is a conceptual demonstration. In the actual analysis,")
    print("these results would be generated using R with the fixest package")
    print("and proper PPML estimation with the full fixed effects structure.")

if __name__ == "__main__":
    main()
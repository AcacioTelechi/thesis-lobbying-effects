#!/usr/bin/env python3
"""
Create ASCII visualization of the event study results.
"""

import csv

def create_ascii_event_study():
    """Create ASCII visualization of event study results"""
    
    # Read the event study data
    with open('outputs/leads_lags/event_study_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Convert to numeric
    for row in data:
        row['period'] = int(row['period'])
        row['coefficient'] = float(row['coefficient'])
        row['ci_lower'] = float(row['ci_lower'])
        row['ci_upper'] = float(row['ci_upper'])
        row['significant'] = row['significant'] == 'True'
    
    # Create ASCII plot
    print("\nEVENT STUDY PLOT - LOBBYING EFFECTS")
    print("=" * 60)
    print("Dependent Variable: Parliamentary Questions (log)")
    print("Treatment: Lobbying Meetings")
    print("Model: PPML with Member, Country×Time, Party×Time, Domain×Time FE")
    print("Clustering: Domain×Time")
    print()
    
    # Find the scale
    min_val = min(row['ci_lower'] for row in data)
    max_val = max(row['ci_upper'] for row in data)
    
    # Scale to fit in 80 characters
    scale_factor = 40 / max(abs(min_val), abs(max_val))
    
    print("Periods relative to treatment:")
    print("(-) = leads (anticipation), (0) = contemporary, (+) = lags (persistence)")
    print()
    print(f"{'Period':<8} {'Coeff':<8} {'95% CI':<20} {'Visual':<30}")
    print("-" * 70)
    
    for row in data:
        period = row['period']
        coeff = row['coefficient']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        significant = row['significant']
        
        # Create visual representation
        center = 40  # Center of the plot
        coeff_pos = int(center + coeff * scale_factor)
        ci_lower_pos = int(center + ci_lower * scale_factor)
        ci_upper_pos = int(center + ci_upper * scale_factor)
        
        # Create the line
        line = [' '] * 80
        
        # Add confidence interval
        for i in range(min(ci_lower_pos, 79), min(ci_upper_pos + 1, 80)):
            if i >= 0:
                line[i] = '-'
        
        # Add zero line
        if center < 80:
            line[center] = '|'
        
        # Add coefficient point
        if 0 <= coeff_pos < 80:
            line[coeff_pos] = '*' if significant else 'o'
        
        visual = ''.join(line[:60])  # Truncate for display
        
        # Format output
        period_str = f"{period:+d}" if period != 0 else " 0"
        coeff_str = f"{coeff:.4f}"
        ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        
        print(f"{period_str:<8} {coeff_str:<8} {ci_str:<20} {visual}")
    
    print()
    print("Legend: | = zero line, * = significant effect, o = non-significant")
    print("        - = 95% confidence interval")
    print()
    
    # Summary interpretation
    print("INTERPRETATION:")
    print("-" * 20)
    
    # Check anticipation effects
    anticipation_periods = [row for row in data if row['period'] < 0]
    significant_anticipation = [row for row in anticipation_periods if row['significant']]
    
    if significant_anticipation:
        print(f"• ANTICIPATION DETECTED: Significant effects in {len(significant_anticipation)} pre-treatment period(s)")
        for row in significant_anticipation:
            print(f"  - Period {row['period']}: β = {row['coefficient']:.4f}")
        print("  → This suggests MEPs may anticipate upcoming lobbying meetings")
    else:
        print("• NO ANTICIPATION: No significant pre-treatment effects detected")
        print("  → No evidence that MEPs anticipate lobbying activities")
    
    # Contemporary effect
    contemporary = next(row for row in data if row['period'] == 0)
    if contemporary['significant']:
        print(f"• CONTEMPORARY EFFECT: β = {contemporary['coefficient']:.4f} (significant)")
        percentage_effect = (math.exp(contemporary['coefficient']) - 1) * 100
        print(f"  → Each additional meeting increases questions by ~{percentage_effect:.1f}%")
    else:
        print("• NO CONTEMPORARY EFFECT: Current period effect not significant")
    
    # Persistence effects
    persistence_periods = [row for row in data if row['period'] > 0]
    significant_persistence = [row for row in persistence_periods if row['significant']]
    
    if significant_persistence:
        print(f"• PERSISTENCE DETECTED: Significant effects in {len(significant_persistence)} post-treatment period(s)")
        for row in significant_persistence:
            print(f"  - Period +{row['period']}: β = {row['coefficient']:.4f}")
        print("  → Lobbying effects persist beyond the immediate period")
    else:
        print("• NO PERSISTENCE: No significant post-treatment effects")
    
    print()

if __name__ == "__main__":
    import math
    create_ascii_event_study()
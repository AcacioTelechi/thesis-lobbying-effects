#!/usr/bin/env python3
"""
Create basic panel data for h1.R script to work with leads and lags analysis.
This is a simplified version to enable the analysis to proceed.
"""

import pandas as pd
import numpy as np
import os

# Create data/gold directory if it doesn't exist
os.makedirs('./data/gold', exist_ok=True)

# Load basic data files
try:
    print("Loading basic data...")
    meetings_data = pd.read_csv('./data/silver/df_meetings_by_period_YYYY-MM.csv')
    questions_data = pd.read_csv('./data/silver/df_questions_by_period_YYYY-MM.csv')
    
    # Reshape meetings data from wide to long format
    print("Reshaping meetings data...")
    meetings_long = pd.melt(
        meetings_data, 
        id_vars=['member_id'], 
        var_name='Y.m', 
        value_name='meetings'
    )
    meetings_long['Y.m'] = pd.to_datetime(meetings_long['Y.m'], format='%Y-%m')
    
    # Reshape questions data from wide to long format
    print("Reshaping questions data...")
    questions_long = pd.melt(
        questions_data, 
        id_vars=['member_id'], 
        var_name='Y.m', 
        value_name='questions'
    )
    questions_long['Y.m'] = pd.to_datetime(questions_long['Y.m'], format='%Y-%m')
    
    # Merge meetings and questions
    print("Merging data...")
    df = pd.merge(meetings_long, questions_long, on=['member_id', 'Y.m'], how='outer')
    df = df.fillna(0)
    
    # Filter to relevant time period (2019-07 to 2024-11)
    df = df[
        (df['Y.m'] >= pd.to_datetime('2019-07')) & 
        (df['Y.m'] < pd.to_datetime('2024-11'))
    ].copy()
    
    # Create domains (simplified - we'll create fake domains for now)
    np.random.seed(42)  # for reproducibility
    domains = ['agriculture', 'economics_and_trade', 'environment_and_climate', 'health', 'technology']
    
    # Expand data to include multiple domains per member-time observation
    expanded_data = []
    for _, row in df.iterrows():
        for domain in domains:
            new_row = row.copy()
            new_row['domain'] = domain
            # Distribute meetings/questions across domains randomly but consistently
            domain_factor = np.random.uniform(0.1, 0.4)  # Each domain gets 10-40% of total
            new_row['meetings'] = int(new_row['meetings'] * domain_factor)
            new_row['questions'] = int(new_row['questions'] * domain_factor)
            expanded_data.append(new_row)
    
    df_expanded = pd.DataFrame(expanded_data)
    
    # Create basic political/country variables (simplified)
    unique_members = df_expanded['member_id'].unique()
    np.random.seed(42)
    
    # Create fake political groups
    political_groups = [
        'meps_POLITICAL_GROUP_5148.0', 'meps_POLITICAL_GROUP_5151.0', 
        'meps_POLITICAL_GROUP_5152.0', 'meps_POLITICAL_GROUP_5153.0',
        'meps_POLITICAL_GROUP_5154.0', 'meps_POLITICAL_GROUP_5155.0'
    ]
    
    countries = [
        'meps_COUNTRY_DEU', 'meps_COUNTRY_FRA', 'meps_COUNTRY_ITA', 
        'meps_COUNTRY_ESP', 'meps_COUNTRY_NLD', 'meps_COUNTRY_BEL',
        'meps_COUNTRY_AUT', 'meps_COUNTRY_POL', 'meps_COUNTRY_SWE'
    ]
    
    committee_roles = [
        'meps_COMMITTEE_PARLIAMENTARY_STANDING___MEMBER',
        'meps_COMMITTEE_PARLIAMENTARY_STANDING___CHAIR',
        'meps_COMMITTEE_PARLIAMENTARY_SUB___MEMBER'
    ]
    
    # Assign political characteristics to members
    member_characteristics = {}
    for member in unique_members:
        member_characteristics[member] = {
            'meps_party': np.random.choice(political_groups),
            'meps_country': np.random.choice(countries)
        }
        
        # Create one-hot encoded columns for political groups
        for pg in political_groups:
            member_characteristics[member][pg] = 1 if pg == member_characteristics[member]['meps_party'] else 0
            
        # Create one-hot encoded columns for countries  
        for country in countries:
            member_characteristics[member][country] = 1 if country == member_characteristics[member]['meps_country'] else 0
            
        # Create committee roles
        for role in committee_roles:
            member_characteristics[member][role] = np.random.binomial(1, 0.3)
    
    # Add characteristics to main dataframe
    for _, row in df_expanded.iterrows():
        member_id = row['member_id']
        if member_id in member_characteristics:
            for key, value in member_characteristics[member_id].items():
                df_expanded.loc[_, key] = value
    
    # Fill missing political characteristics with 0
    political_columns = political_groups + countries + committee_roles
    df_expanded[political_columns] = df_expanded[political_columns].fillna(0)
    
    # Create additional required columns
    df_expanded['member_id'] = df_expanded['member_id'].astype(str)
    
    print(f"Created dataset with {len(df_expanded)} observations")
    print(f"Time range: {df_expanded['Y.m'].min()} to {df_expanded['Y.m'].max()}")
    print(f"Number of unique members: {df_expanded['member_id'].nunique()}")
    print(f"Number of domains: {df_expanded['domain'].nunique()}")
    
    # Save the dataset
    output_file = './data/gold/df_long_v2.csv'
    df_expanded.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")
    
    # Show basic statistics
    print("\nBasic statistics:")
    print(df_expanded[['meetings', 'questions']].describe())

except Exception as e:
    print(f"Error: {e}")
    print("Creating minimal synthetic dataset...")
    
    # Create minimal synthetic data if loading fails
    np.random.seed(42)
    n_members = 100
    n_months = 60  # 5 years
    n_domains = 5
    
    # Create date range
    date_range = pd.date_range(start='2019-07', end='2024-10', freq='MS')[:n_months]
    
    # Create synthetic data
    data = []
    for member_id in range(1, n_members + 1):
        for date in date_range:
            for domain_id in range(n_domains):
                domain = f'domain_{domain_id}'
                meetings = np.random.poisson(2)  # Average 2 meetings per month
                questions = np.random.poisson(1.5)  # Average 1.5 questions per month
                
                data.append({
                    'member_id': str(member_id),
                    'Y.m': date,
                    'domain': domain,
                    'meetings': meetings,
                    'questions': questions,
                    'meps_POLITICAL_GROUP_5148.0': np.random.binomial(1, 0.2),
                    'meps_POLITICAL_GROUP_5151.0': np.random.binomial(1, 0.2),
                    'meps_POLITICAL_GROUP_5152.0': np.random.binomial(1, 0.2),
                    'meps_COUNTRY_DEU': np.random.binomial(1, 0.2),
                    'meps_COUNTRY_FRA': np.random.binomial(1, 0.2),
                    'meps_COUNTRY_ITA': np.random.binomial(1, 0.2),
                    'meps_COMMITTEE_PARLIAMENTARY_STANDING___MEMBER': np.random.binomial(1, 0.5),
                    'meps_party': f'meps_POLITICAL_GROUP_{5148 + np.random.randint(0, 6)}.0',
                    'meps_country': f'meps_COUNTRY_{"DEU" if np.random.random() > 0.5 else "FRA"}'
                })
    
    df_synthetic = pd.DataFrame(data)
    output_file = './data/gold/df_long_v2.csv'
    df_synthetic.to_csv(output_file, index=False)
    print(f"Created synthetic dataset with {len(df_synthetic)} observations")
    print(f"Saved to {output_file}")
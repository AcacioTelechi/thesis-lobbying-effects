#!/usr/bin/env python3
"""
Create basic CSV data for h1.R script using only standard library.
"""

import csv
import random
import os

def main():
    # Create data/gold directory if it doesn't exist
    os.makedirs('./data/gold', exist_ok=True)
    
    print("Creating simplified dataset...")
    
    # Parameters
    random.seed(42)  # for reproducibility
    n_members = 200
    domains = ['agriculture', 'economics_and_trade', 'environment_and_climate', 'health', 'technology']
    
    # Generate simple date list manually
    dates = [
        '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
        '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06',
        '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12',
        '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06',
        '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
        '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06',
        '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
        '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
        '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12',
        '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06',
        '2024-07', '2024-08', '2024-09', '2024-10'
    ]
    
    # Political groups and countries for controls
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
    
    # Create member characteristics (fixed over time)
    member_chars = {}
    for member_id in range(1, n_members + 1):
        # Assign political group
        party = random.choice(political_groups)
        country = random.choice(countries)
        
        member_chars[member_id] = {
            'meps_party': party,
            'meps_country': country
        }
        
        # Create binary variables for political groups
        for pg in political_groups:
            member_chars[member_id][pg] = 1 if pg == party else 0
            
        # Create binary variables for countries
        for c in countries:
            member_chars[member_id][c] = 1 if c == country else 0
            
        # Committee roles (can have multiple)
        for role in committee_roles:
            member_chars[member_id][role] = 1 if random.random() < 0.3 else 0
    
    # Generate data
    output_file = './data/gold/df_long_v2.csv'
    
    with open(output_file, 'w', newline='') as csvfile:
        # Define all column names
        columns = ['member_id', 'Y.m', 'domain', 'meetings', 'questions']
        columns.extend(political_groups)
        columns.extend(countries) 
        columns.extend(committee_roles)
        columns.extend(['meps_party', 'meps_country'])
        
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        
        total_rows = 0
        
        for member_id in range(1, n_members + 1):
            for date in dates:
                for domain in domains:
                    # Generate random meetings and questions with some correlation
                    # Use Poisson-like distribution manually
                    meetings = max(0, int(random.expovariate(0.5)))  # Average ~2 meetings
                    # Questions somewhat correlated with meetings
                    base_questions = max(0, meetings * random.uniform(0.5, 1.5))
                    questions = max(0, int(base_questions + random.expovariate(1.0)))
                    
                    # Create row
                    row = {
                        'member_id': str(member_id),
                        'Y.m': date,
                        'domain': domain,
                        'meetings': meetings,
                        'questions': questions
                    }
                    
                    # Add member characteristics
                    row.update(member_chars[member_id])
                    
                    writer.writerow(row)
                    total_rows += 1
                    
                    if total_rows % 10000 == 0:
                        print(f"Generated {total_rows} rows...")
    
    print(f"Successfully created dataset with {total_rows} observations")
    print(f"Saved to {output_file}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Members: {n_members}")
    print(f"Domains: {len(domains)}")
    
if __name__ == "__main__":
    main()
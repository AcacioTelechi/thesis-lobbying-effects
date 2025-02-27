import re
import unicodedata
from datetime import datetime, timedelta
import calendar
from thefuzz import fuzz
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def month_range(start_date, end_date):
    """Generate start and end dates for each month between start_date and end_date."""
    start_date = datetime.strptime(start_date, "%d/%m/%Y")
    end_date = datetime.strptime(end_date, "%d/%m/%Y")

    current_date = start_date

    while current_date <= end_date:
        month_start = current_date.replace(day=1)
        _, last_day = calendar.monthrange(current_date.year, current_date.month)

        month_end = current_date.replace(day=last_day)

        if month_end > end_date:
            month_end = end_date

        yield (month_start.strftime("%d/%m/%Y"), month_end.strftime("%d/%m/%Y"))

        current_date = month_end + timedelta(days=1)


def treat_string(string: str) -> str:
    """Normalize organization names for better matching."""
    if not string or not isinstance(string, str):
        return ""
    
    # Convert to lowercase
    name = str(string).lower()
    
    # Remove leading/trailing spaces
    name = name.strip()
    
    # Expanded abbreviations dictionary
    abbreviations = {
        "inc.": "incorporated",
        "inc": "incorporated",
        "ltd.": "limited",
        "ltd": "limited",
        "llc": "limited liability company",
        "llc.": "limited liability company",
        "plc": "public limited company",
        "plc.": "public limited company",
        "eu": "european union",
        "intl.": "international",
        "intl": "international",
        "corp.": "corporation",
        "corp": "corporation",
        "&": "and",
        "assn": "association",
        "assoc.": "association",
        "fed.": "federation",
        "fed": "federation",
        "org.": "organization",
        "org": "organization",
    }
    
    # Replace abbreviations (case-insensitive)
    for abbr, full in abbreviations.items():
        name = re.sub(rf"\b{re.escape(abbr)}\b", full, name, flags=re.IGNORECASE)
    
    # Remove legal entity indicators
    legal_suffixes = r'\b(incorporated|corporation|limited|llc|plc|inc|ltd|corp|gmbh|sa|nv|bv|ag)\b'
    name = re.sub(legal_suffixes, '', name)
    
    # Remove special characters and extra symbols but keep hyphens
    name = re.sub(r'[^\w\s-]', '', name)
    
    # Normalize unicode characters (handle accents)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    
    # Collapse multiple spaces and hyphens
    name = re.sub(r'[\s-]+', ' ', name)
    
    return name.strip()


def find_best_match(name: str, possible_matches: list[str], threshold: float = 75.0) -> tuple[str, float]:
    """
    Find the best matching organization name from a list of possibilities.
    Returns tuple of (best_match, confidence_score)
    """
    treated_name = treat_string(name)
    best_match = None
    best_score = 0
    
    for candidate in possible_matches:
        treated_candidate = treat_string(candidate)
        
        # Try different fuzzy matching algorithms
        ratio = fuzz.ratio(treated_name, treated_candidate)
        partial_ratio = fuzz.partial_ratio(treated_name, treated_candidate)
        token_sort_ratio = fuzz.token_sort_ratio(treated_name, treated_candidate)
        token_set_ratio = fuzz.token_set_ratio(treated_name, treated_candidate)
        
        # Use the median score among different matching methods
        score = sorted([ratio, partial_ratio, token_sort_ratio, token_set_ratio])[1:3]
        score = sum(score) / 2
        
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return (best_match, best_score) if best_score >= threshold else (None, 0)


def match_organizations(meetings_df: pd.DataFrame, 
                       register_df: pd.DataFrame,
                       meetings_col: str = 'attendees',
                       register_col: str = 'organization_name',
                       threshold: float = 75.0) -> pd.DataFrame:
    """
    Match organizations between meetings dataset and transparency register.
    Returns DataFrame with matched organizations and confidence scores.
    """
    register_names = register_df[register_col].tolist()
    
    # Create new columns for matched names and confidence scores
    meetings_df['matched_organization'] = None
    meetings_df['match_confidence'] = 0.0
    
    
    for idx, row in tqdm(meetings_df.iterrows(), total=len(meetings_df), desc="Matching organizations"):
        org_name = row[meetings_col]
        matched_name, confidence = find_best_match(org_name, register_names, threshold)
        
        meetings_df.at[idx, 'matched_organization'] = matched_name
        meetings_df.at[idx, 'match_confidence'] = confidence
    return meetings_df


def _process_batch(args):
    """Helper function to process a batch of data for parallel processing"""
    batch_df, register_df, meetings_col, register_col, threshold = args
    return match_organizations(batch_df, register_df, meetings_col, register_col, threshold)


def parallel_match_organizations(meetings_df: pd.DataFrame, 
                               register_df: pd.DataFrame,
                               meetings_col: str = 'attendees',
                               register_col: str = 'organization_name',
                               threshold: float = 75.0,
                               batch_size: int = 100,
                               n_processes: int = None) -> pd.DataFrame:
    """
    Parallel processing version of organization matching using multiprocessing.
    
    Args:
        meetings_df: DataFrame containing meeting records
        register_df: DataFrame containing organization registry
        meetings_col: Column name for organization names in meetings_df
        register_col: Column name for organization names in register_df
        threshold: Minimum confidence score for matching
        batch_size: Number of records to process in each batch
        n_processes: Number of parallel processes (defaults to CPU count - 1)
    """
    # Prepare batches
    batches = []
    for i in range(0, len(meetings_df), batch_size):
        batch_df = meetings_df.iloc[i:i+batch_size]
        batches.append((batch_df, register_df, meetings_col, register_col, threshold))

    # Process batches in parallel
    with Pool(processes=n_processes) as pool:
        # Create progress bar
        pbar = tqdm(total=len(batches), desc="Processing batches")
        
        # Process batches and update progress bar for each completed batch
        results = []
        for result in pool.imap(_process_batch, batches):
            results.append(result)
            pbar.update(1)
        
        pbar.close()
    # Combine results
    return pd.concat(results, ignore_index=True)

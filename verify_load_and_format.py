import pandas as pd
import os

file_path = "data/raw/Incidents.txt"

print(f"Attempting to load: {file_path}")

try:
    # Load using pipe separator
    # 'skipinitialspace=True' helps with spaces after the delimiter
    df = pd.read_csv(file_path, sep='|', skipinitialspace=True)
    
    # 1. Clean column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Clean string data (strip whitespace)
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
        
    # 3. Handle potential empty columns if lines end with '|'
    # Use simpler invalid-column filtering if needed, but 'Unnamed' often appears
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    print("SUCCESS: Data loaded.")
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 2 records:")
    print(df.head(2))
    
    # Convert to list of dicts for LLM context
    records = df.to_dict(orient='records')
    print("\nConverted to records (first 1):")
    print(records[0])
    
except Exception as e:
    print(f"ERROR: {e}")

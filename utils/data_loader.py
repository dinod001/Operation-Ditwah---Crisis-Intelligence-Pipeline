import pandas as pd
import os

def load_incidents(file_path):
    """
    Loads incidents from a pipe-delimited text file and returns a formatted string 
    containing all records.
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    try:
        # Load Data Frame with pipe separator
        df = pd.read_csv(file_path, sep='|', skipinitialspace=True)
        
        # Cleanup column names
        df.columns = [str(c).strip() for c in df.columns]
        
        # Cleanup string values
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].str.strip()
            
        # Remove likely empty columns (common with trailing pipes)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Convert to text format
        records = df.to_dict(orient='records')
        formatted_rows = []
        
        for record in records:
            # Create a clean string representation: "Key: Value, Key: Value..."
            row_str = ", ".join([f"{k}: {v}" for k, v in record.items() if pd.notna(v)])
            formatted_rows.append(row_str)
            
        # Join all rows
        return "\n".join(formatted_rows)
        
    except Exception as e:
        return f"Error reading file: {e}"


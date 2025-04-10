import pandas as pd
import re

def clean_dataframe_columns(df):
    """
    Remove inappropriate columns from a dataframe.
    """
    original_cols = df.columns.tolist()
    columns_to_drop = []
    
    for col in original_cols:
        # Filter out columns with insufficient data
        valid_values = [x for x in df[col] if pd.notna(x) and x != "--"]
        if len(valid_values) <= 1:
            columns_to_drop.append(col)
            continue
            
        # Remove unnamed or index columns
        if "Unnamed:" in col:
            columns_to_drop.append(col)
            
    # Drop identified columns
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print("Excluded columns:", columns_to_drop)
        
    return df

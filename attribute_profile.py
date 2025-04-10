from . import embedding_setup
import pandas as pd
import numpy as np
import re
import random
from dateutil.parser import parse as date_parser
from sentence_transformers import util

# Use the embedder from embedding_setup
language_model = embedding_setup.embedder

# Unit identifiers removed

def analyze_table(file_path):
    """
    Load and analyze a data table from file.
    """
    return pd.read_csv(file_path)

def is_mostly_numeric(values, threshold=0.95):
    """
    Determine if a list of values is mostly numeric.
    """
    count = 0
    for val in values:
        try:
            float(val)
            count += 1
        except (ValueError, TypeError):
            pass
    
    return count >= threshold * len(values)

def contains_numeric_patterns(values, threshold=0.9):
    """
    Check if values contain recognizable numeric patterns.
    """
    matches = 0
    
    for val in values:
        val_str = str(val)
        # Remove formatting characters
        cleaned = val_str.replace(",", "")
        
        # No unit conversion needed here
        
        # Check if there are significant digits
        digit_matches = re.findall(r'\d+', cleaned)
        if digit_matches and sum(len(m) for m in digit_matches) >= 0.5 * len(cleaned):
            matches += 1
    
    return matches >= threshold * len(values)

def get_numeric_values(values):
    """
    Extract numeric values from text with potential formatting.
    """
    try:
        return [float(v) for v in values]
    except (ValueError, TypeError):
        pass
    
    results = []
    multiplier_values = []
    
    for val in values:
        val_str = str(val).replace(",", "")
        # No multiplier handling needed
        
        # Extract numbers using regex
        matches = re.findall(r'([-]?([0-9]*[.])?[0-9]+)', val_str)
        if matches:
            # Use the first match
            results.append(float(matches[0][0]))
        
    
    return results

def calculate_stats(values):
    """
    Calculate statistical properties of a numeric list.
    """
    avg = np.mean(values)
    minimum = np.min(values)
    maximum = np.max(values)
    var = np.var(values)
    coeff_var = var / avg if avg != 0 else 0
    unique_ratio = len(set(values)) / len(values)
    
    return np.array([avg, minimum, maximum, var, coeff_var, unique_ratio])

def detect_web_urls(values, threshold=0.9):
    """
    Detect if values are likely URLs.
    """
    url_count = 0
    url_pattern = r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    
    for val in values:
        if not isinstance(val, str):
            continue
            
        if re.search(url_pattern, val):
            url_count += 1
    
    return url_count >= threshold * len(values)

def detect_date_format(values, threshold=0.9):
    """
    Detect if values are likely dates.
    """
    date_count = 0
    date_markers = ["月", "日", "年", "date", "day", "month", "year"]
    
    for val in values:
        if not isinstance(val, str):
            continue
            
        # Check for date keywords
        if any(marker in val.lower() for marker in date_markers):
            date_count += 1
            continue
            
        # Try parsing as date
        try:
            parsed_date = date_parser(val)
            # Exclude unlikely years
            if 1900 <= parsed_date.year <= 2100:
                date_count += 1
        except:
            pass
    
    return date_count >= threshold * len(values)

def analyze_text_properties(values):
    """
    Analyze text properties of string values.
    """
    punct_chars = [".", ",", ";", ":", "!", "?", "，", "。", "；", "：", "！", "？"]
    special_chars = ["@", "#", "$", "%", "&", "*", "+", "=", "/", "\\", "-", "_", "<", ">", "[", "]", "{", "}", "(", ")", "~", "`"]
    
    space_ratios = []
    punct_ratios = []
    special_ratios = []
    digit_ratios = []
    
    for val_str in values:
        val_length = len(val_str) or 1  # Avoid division by zero
        
        # Calculate ratios
        space_ratio = (val_str.count(" ") + val_str.count("\t") + val_str.count("\n")) / val_length
        punct_ratio = sum(1 for c in val_str if c in punct_chars) / val_length
        special_ratio = sum(1 for c in val_str if c in special_chars) / val_length
        digit_ratio = sum(1 for c in val_str if c.isdigit()) / val_length
        
        space_ratios.append(space_ratio)
        punct_ratios.append(punct_ratio)
        special_ratios.append(special_ratio)
        digit_ratios.append(digit_ratio)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-12
    
    # Calculate statistics for each ratio
    space_arr = np.array(space_ratios) + epsilon
    punct_arr = np.array(punct_ratios) + epsilon
    special_arr = np.array(special_ratios) + epsilon
    digit_arr = np.array(digit_ratios) + epsilon
    
    # Return all metrics
    mean_values = [
        np.mean(space_arr), 
        np.mean(punct_arr), 
        np.mean(special_arr), 
        np.mean(digit_arr)
    ]
    
    variation_values = [
        np.var(space_arr) / np.mean(space_arr),
        np.var(punct_arr) / np.mean(punct_arr),
        np.var(special_arr) / np.mean(special_arr),
        np.var(digit_arr) / np.mean(digit_arr)
    ]
    
    return np.array(mean_values + variation_values)

def compute_text_embeddings(values):
    """
    Generate semantic embeddings for text values.
    """
    # Sample values if there are too many
    if len(values) > 15:
        sample_values = random.sample(values, 15)
    else:
        sample_values = values
    
    # Generate embeddings for each sample
    embeddings = [language_model.encode(str(val)) for val in sample_values]
    
    # Return the average embedding across samples
    return np.mean(embeddings, axis=0)

def generate_column_profile(values):
    """
    Generate a comprehensive profile of a data column.
    """
    # Clean up values
    valid_values = [v for v in values if pd.notna(v) and v != "--"]
    if not valid_values:
        raise ValueError("Column contains no valid values")
    
    # Determine the likely data type
    data_types = ["url", "numeric", "date", "text"]
    
    if detect_web_urls(valid_values):
        column_type = "url"
    elif detect_date_format(valid_values):
        column_type = "date"
    elif is_mostly_numeric(valid_values) or contains_numeric_patterns(valid_values):
        column_type = "numeric"
    else:
        column_type = "text"
    
    # Create type indicator vector
    type_indicator = np.zeros(len(data_types))
    type_indicator[data_types.index(column_type)] = 1
    
    # Calculate numeric features if applicable
    if column_type == "numeric":
        numeric_values = get_numeric_values(valid_values)
        numeric_stats = calculate_stats(numeric_values)
    else:
        numeric_stats = np.array([-1] * 6)
    
    # Calculate length features for all types
    length_stats = calculate_stats([len(str(v)) for v in valid_values])
    
    # Get text features for textual data
    if column_type == "text" or (not is_mostly_numeric(valid_values) and contains_numeric_patterns(valid_values)):
        text_props = analyze_text_properties([str(v) for v in valid_values])
        semantic_vector = compute_text_embeddings([str(v) for v in valid_values])
    else:
        text_props = np.array([-1] * 8)
        semantic_vector = np.array([-0.5] * 768)  # Neutral vector with small values
    
    # Combine all feature groups
    return np.concatenate((type_indicator, numeric_stats, length_stats, text_props, semantic_vector))

def profile_dataframe_columns(df):
    """
    Generate profiles for all columns in a dataframe.
    """
    all_profiles = None
    
    for column_name in df.columns:
        # Skip index or unnamed columns
        if "Unnamed:" in column_name:
            continue
            
        # Generate profile for this column
        column_profile = generate_column_profile(df[column_name])
        column_profile = column_profile.reshape(1, -1)
        
        # Add to result set
        if all_profiles is None:
            all_profiles = column_profile
        else:
            all_profiles = np.concatenate((all_profiles, column_profile), axis=0)
    
    return all_profiles

if __name__ == '__main__':
    sample_profile = profile_dataframe_columns(analyze_table("Sample_Data/example_table.csv"))
    print(sample_profile.shape)

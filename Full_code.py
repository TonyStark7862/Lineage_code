## 1. embedding_setup.py

```python
from sentence_transformers import SentenceTransformer
print("data_lineage|Initializing language representation model...")
# Using a different but comparable embedding model
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')
print("data_lineage|Language representation model loaded successfully")
```

## 2. data_helpers.py

```python
import pandas as pd
import re

def clean_dataframe_columns(df):
    """
    Prepare columns in a dataframe for data lineage analysis.
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
```

## 3. attribute_profile.py

```python
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
    print("This module is for data lineage analysis only")
```

## 4. cross_attribute_analysis.py

```python
from . import embedding_setup
from .attribute_profile import profile_dataframe_columns
from .data_helpers import clean_dataframe_columns
import pandas as pd
import numpy as np
from numpy.linalg import norm
import random
import os
from strsimpy.metric_lcs import MetricLCS
from strsimpy.damerau import Damerau
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from sentence_transformers import util
import re

# Access the embedding model
nlp_model = embedding_setup.embedder

# Initialize text similarity metrics
smooth_func = SmoothingFunction().method4
lcs_metric = MetricLCS()
edit_distance = Damerau()
random_seed = 345
random.seed(random_seed)

def normalize_text(input_text):
    """
    Normalize text by lowercase and splitting on special characters.
    """
    lowered = input_text.lower()
    tokens = re.split(r'[\s\_\.]', lowered)
    return " ".join(tokens).strip()

def compute_semantic_similarity(text1, text2):
    """
    Compute semantic similarity between two texts using language model.
    """
    vector1 = nlp_model.encode(text1)
    vector2 = nlp_model.encode(text2)
    return util.cos_sim(vector1, vector2)

def compute_name_similarity_features(name1, name2, name_embeddings):
    """
    Compute similarity features between column names.
    """
    # Calculate BLEU score
    bleu_score = bleu([name1], name2, smoothing_function=smooth_func)
    
    # Calculate edit distance
    edit_dist = edit_distance.distance(name1, name2)
    
    # Calculate longest common subsequence
    lcs_dist = lcs_metric.distance(name1, name2)
    
    # Calculate transformer-based similarity
    transformer_sim = util.cos_sim(name_embeddings[name1], name_embeddings[name2])
    
    # Check for substring relationship
    substring_relation = int(name1 in name2 or name2 in name1)
    
    return np.array([bleu_score, edit_dist, lcs_dist, transformer_sim, substring_relation])

def compute_content_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between column content embeddings.
    """
    cos_sim = np.inner(embedding1, embedding2) / (norm(embedding1) * norm(embedding2) + 1e-10)
    return np.array([cos_sim])

def analyze_table_columns(df1, df2, mapping_path=None, mode="test"):
    """
    Generate comparison features for data lineage between two tables.
    """
    # For prediction, we don't need mappings
    mappings = set()
    
    # Get column names
    columns1 = list(df1.columns)
    columns2 = list(df2.columns)

    # For prediction, create all possible column pairs
    pair_labels = {}
    for i in range(len(columns1)):
        for j in range(len(columns2)):
            pair_labels[(i, j)] = 0  # Default label, will be predicted
    
    # Generate individual column profiles
    table1_profiles = profile_dataframe_columns(df1)
    table2_profiles = profile_dataframe_columns(df2)

    # Pre-compute name embeddings for all columns
    name_embeddings = {}
    for col in columns1 + columns2:
        normalized = normalize_text(col)
        name_embeddings[normalized] = nlp_model.encode(normalized)

    # Number of additional features beyond profile differences
    additional_features = 6
    
    # Initialize output arrays
    feature_count = table1_profiles.shape[1] - 768 + additional_features
    output_features = np.zeros((len(pair_labels), feature_count), dtype=np.float32)
    output_labels = np.zeros(len(pair_labels), dtype=np.int32)
    
    # Generate comparison features for each pair
    for idx, (pair, label) in enumerate(pair_labels.items()):
        col1_idx, col2_idx = pair
        col1_name = columns1[col1_idx]
        col2_name = columns2[col2_idx]
        
        # Calculate profile differences as percentages
        profile_diffs = np.abs(table1_profiles[col1_idx] - table2_profiles[col2_idx]) / (
            table1_profiles[col1_idx] + table2_profiles[col2_idx] + 1e-8)
        
        # Normalize column names
        norm_name1 = normalize_text(col1_name)
        norm_name2 = normalize_text(col2_name)
        
        # Calculate name similarity features
        name_features = compute_name_similarity_features(norm_name1, norm_name2, name_embeddings)
        
        # Calculate content similarity
        content_sim = compute_content_similarity(
            table1_profiles[col1_idx][-768:], 
            table2_profiles[col2_idx][-768:])
        
        # Combine features and add to output
        output_features[idx, :] = np.concatenate((
            profile_diffs[:-768], 
            name_features, 
            content_sim))
        
        output_labels[idx] = label
    
    return output_features, output_labels

if __name__ == '__main__':
    print("This module is for data lineage discovery only")
```

## 5. attribute_matcher.py

```python
from . import embedding_setup
from .cross_attribute_analysis import analyze_table_columns
from .data_helpers import clean_dataframe_columns
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import argparse
import time
from pathlib import Path

# Get the directory containing this script
base_directory = Path(__file__).parent

# Set up command line arguments
arg_parser = argparse.ArgumentParser(description="Trace data lineage between two data tables")
arg_parser.add_argument("-d", "--directory", help="Directory containing the input tables")
arg_parser.add_argument("-m", "--model_path", help="Path to the data lineage model")
arg_parser.add_argument("-t", "--threshold", help="Confidence threshold for lineage")
arg_parser.add_argument("-s", "--strategy", help="Lineage strategy: many-to-many/one-to-one/one-to-many",
                       default="many-to-many")
args = arg_parser.parse_args()

def build_correspondence_matrix(df1, df2, predictions, prediction_confidence, strategy="many-to-many"):
    """
    Build a correspondence matrix for data lineage between tables.
    """
    identified_pairs = []
    
    # Ensure predictions are numpy arrays
    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)
    
    prediction_confidence = np.array(prediction_confidence)
    prediction_confidence = np.mean(prediction_confidence, axis=0)
    
    # Apply threshold to get binary lineage relations
    binary_matches = np.where(prediction_confidence > 0.5, 1, 0)
    
    # Get column names from both dataframes
    columns1 = df1.columns
    columns2 = df2.columns
    
    # Reshape predictions into matrix form (columns1 × columns2)
    confidence_matrix = np.array(predictions).reshape(len(columns1), len(columns2))
    
    # Initialize the binary lineage matrix based on the specified strategy
    if strategy == "many-to-many":
        # Many-to-many: use threshold directly
        match_matrix = np.array(binary_matches).reshape(len(columns1), len(columns2))
    else:
        # Create empty matrix initially
        match_matrix = np.zeros((len(columns1), len(columns2)))
        
        # Apply additional constraints for one-to-one or one-to-many matching
        for i in range(len(columns1)):
            for j in range(len(columns2)):
                # Check if this pair passed the threshold
                if binary_matches[i * len(columns2) + j] == 1:
                    if strategy == "one-to-one":
                        # One-to-one: must be max in both row and column for lineage
                        row_max = np.max(confidence_matrix[i, :])
                        col_max = np.max(confidence_matrix[:, j])
                        
                        if confidence_matrix[i, j] == row_max and confidence_matrix[i, j] == col_max:
                            match_matrix[i, j] = 1
                            
                    elif strategy == "one-to-many":
                        # One-to-many: must be max in its row for lineage
                        row_max = np.max(confidence_matrix[i, :])
                        
                        if confidence_matrix[i, j] == row_max:
                            match_matrix[i, j] = 1
    
    # Create DataFrame representations of the matrices
    confidence_df = pd.DataFrame(confidence_matrix, index=columns1, columns=columns2)
    match_df = pd.DataFrame(match_matrix, index=columns1, columns=columns2)
    
    # Compile list of lineage relationships with confidence scores
    for i in range(len(match_df.index)):
        for j in range(len(match_df.columns)):
            if match_df.iloc[i, j] == 1:
                identified_pairs.append((
                    match_df.index[i],
                    match_df.columns[j],
                    confidence_df.iloc[i, j]
                ))
                
    return confidence_df, match_df, identified_pairs

def find_data_lineage(table1_path, table2_path, threshold=None, strategy="many-to-many", model_path=None):
    """
    Identify data lineage between two tables.
    """
    # Use default model path if not specified
    if model_path is None:
        model_path = str(base_directory / "models" / "lineage_model_v1")
    
    # Load tables
    table1_df = pd.read_csv(table1_path)
    table2_df = pd.read_csv(table2_path)

    # Clean up columns in both tables
    table1_df = clean_dataframe_columns(table1_df)
    table2_df = clean_dataframe_columns(table2_df)

    # Generate comparison features
    features, _ = analyze_table_columns(table1_df, table2_df, mode="test")

    # Initialize prediction variables
    predictions = []
    confidence_scores = []
    
    # Load and apply each model in the ensemble
    n_models = len([f for f in os.listdir(model_path) if f.endswith('.model')])
    for i in range(n_models):
        # Initialize booster
        booster = xgb.Booster({'nthread': 4})
        
        # Load model file
        model_file = os.path.join(model_path, f"{i}.model")
        booster.load_model(model_file)
        
        # Determine threshold to use
        if threshold is not None:
            current_threshold = float(threshold)
        else:
            # Read threshold from file
            threshold_file = os.path.join(model_path, f"{i}.threshold")
            with open(threshold_file, 'r') as f:
                current_threshold = float(f.read().strip())
        
        # Convert features to DMatrix format
        test_dmatrix = xgb.DMatrix(features)
        
        # Get predictions
        preds = booster.predict(test_dmatrix)
        pred_labels = np.where(preds > current_threshold, 1, 0)
        
        # Store results
        predictions.append(preds)
        confidence_scores.append(pred_labels)
        
        # Clean up
        del booster

    # Build correspondence matrix
    confidence_matrix, match_matrix, lineage_pairs = build_correspondence_matrix(
        table1_df, table2_df, predictions, confidence_scores, strategy=strategy
    )
    
    return confidence_matrix, match_matrix, lineage_pairs

if __name__ == '__main__':
    start_time = time.time()
    
    # Prepare input path
    input_dir = args.directory.rstrip("/")
    
    # Find data lineage
    confidence_matrix, match_matrix, lineage_pairs = find_data_lineage(
        os.path.join(input_dir, "Table1.csv"),
        os.path.join(input_dir, "Table2.csv"),
        threshold=args.threshold,
        strategy=args.strategy,
        model_path=args.model_path
    )
    
    # Save results
    confidence_matrix.to_csv(os.path.join(input_dir, "confidence_scores.csv"), index=True)
    match_matrix.to_csv(os.path.join(input_dir, "lineage_results.csv"), index=True)

    # Print lineage relationships
    for pair in lineage_pairs:
        print(pair)
        
    print(f"data_lineage|Processing time: {time.time() - start_time:.2f} seconds")
```

## requirements.txt

```
numpy==1.21.0
pandas==1.3.4
nltk==3.6.7
python-dateutil==2.8.2
sentence-transformers==2.2.0
xgboost==1.6.1
strsimpy==0.2.1
```

from . import embedding_setup
from .cross_attribute_analysis import analyze_table_columns
from .data_helpers import json_to_dataframe, clean_dataframe_columns
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
arg_parser = argparse.ArgumentParser(description="Match columns between two data tables")
arg_parser.add_argument("-d", "--directory", help="Directory containing the input tables")
arg_parser.add_argument("-m", "--model_path", help="Path to the prediction model")
arg_parser.add_argument("-t", "--threshold", help="Confidence threshold for matches")
arg_parser.add_argument("-s", "--strategy", help="Matching strategy: many-to-many/one-to-one/one-to-many",
                       default="many-to-many")
args = arg_parser.parse_args()

def build_correspondence_matrix(df1, df2, predictions, prediction_confidence, strategy="many-to-many"):
    """
    Build a correspondence matrix between columns of two tables.
    """
    identified_pairs = []
    
    # Ensure predictions are numpy arrays
    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)
    
    prediction_confidence = np.array(prediction_confidence)
    prediction_confidence = np.mean(prediction_confidence, axis=0)
    
    # Apply threshold to get binary matches
    binary_matches = np.where(prediction_confidence > 0.5, 1, 0)
    
    # Get column names from both dataframes
    columns1 = df1.columns
    columns2 = df2.columns
    
    # Reshape predictions into matrix form (columns1 × columns2)
    confidence_matrix = np.array(predictions).reshape(len(columns1), len(columns2))
    
    # Initialize the binary match matrix based on the specified strategy
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
                        # One-to-one: must be max in both row and column
                        row_max = np.max(confidence_matrix[i, :])
                        col_max = np.max(confidence_matrix[:, j])
                        
                        if confidence_matrix[i, j] == row_max and confidence_matrix[i, j] == col_max:
                            match_matrix[i, j] = 1
                            
                    elif strategy == "one-to-many":
                        # One-to-many: must be max in its row
                        row_max = np.max(confidence_matrix[i, :])
                        
                        if confidence_matrix[i, j] == row_max:
                            match_matrix[i, j] = 1
    
    # Create DataFrame representations of the matrices
    confidence_df = pd.DataFrame(confidence_matrix, index=columns1, columns=columns2)
    match_df = pd.DataFrame(match_matrix, index=columns1, columns=columns2)
    
    # Compile list of matched pairs with confidence scores
    for i in range(len(match_df.index)):
        for j in range(len(match_df.columns)):
            if match_df.iloc[i, j] == 1:
                identified_pairs.append((
                    match_df.index[i],
                    match_df.columns[j],
                    confidence_df.iloc[i, j]
                ))
                
    return confidence_df, match_df, identified_pairs

def find_matching_columns(table1_path, table2_path, threshold=None, strategy="many-to-many", model_path=None):
    """
    Identify matching columns between two tables.
    """
    # Use default model path if not specified
    if model_path is None:
        model_path = str(base_directory / "models" / "column_matcher_v1")
    
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
    confidence_matrix, match_matrix, matched_pairs = build_correspondence_matrix(
        table1_df, table2_df, predictions, confidence_scores, strategy=strategy
    )
    
    return confidence_matrix, match_matrix, matched_pairs

if __name__ == '__main__':
    start_time = time.time()
    
    # Prepare input path
    input_dir = args.directory.rstrip("/")
    
    # Find matching columns
    confidence_matrix, match_matrix, matched_pairs = find_matching_columns(
        os.path.join(input_dir, "Table1.csv"),
        os.path.join(input_dir, "Table2.csv"),
        threshold=args.threshold,
        strategy=args.strategy,
        model_path=args.model_path
    )
    
    # Save results
    confidence_matrix.to_csv(os.path.join(input_dir, "confidence_scores.csv"), index=True)
    match_matrix.to_csv(os.path.join(input_dir, "match_results.csv"), index=True)

    # Print matched pairs
    for pair in matched_pairs:
        print(pair)
        
    print(f"attribute_matching|Processing time: {time.time() - start_time:.2f} seconds")

from . import embedding_setup
from .attribute_profile import profile_dataframe_columns
from .data_helpers import clean_dataframe_columns
import pandas as pd
import numpy as np
from numpy.linalg import norm
import random
import os
import subprocess
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

def parse_mappings(mapping_path):
    """
    Parse column mapping file into a set of tuples.
    """
    if not mapping_path or not os.path.exists(mapping_path):
        return set()
        
    with open(mapping_path, 'r') as f:
        lines = f.readlines()
    
    mapping_pairs = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(",")
        parts = [p.strip("< >") for p in parts]
        if len(parts) >= 2:
            mapping_pairs.add(tuple(parts))
    
    return mapping_pairs

def generate_column_pairs(columns1, columns2, mappings, mode="train"):
    """
    Generate pairs of columns with their matching labels.
    """
    pair_labels = {}
    
    # Create all possible combinations with labels
    for i, col1 in enumerate(columns1):
        for j, col2 in enumerate(columns2):
            # Check if this pair is in the mapping
            if (col1, col2) in mappings or (col2, col1) in mappings:
                pair_labels[(i, j)] = 1
            else:
                pair_labels[(i, j)] = 0
    
    # Balance classes for training mode
    if mode == "train":
        positive_count = sum(1 for v in pair_labels.values() if v == 1)
        negative_count = len(pair_labels) - positive_count
        
        # If imbalanced, remove some negative examples
        if positive_count > 0 and positive_count < 0.1 * len(pair_labels):
            pairs_to_remove = []
            for pair, label in pair_labels.items():
                if label == 0 and len(pairs_to_remove) < negative_count - 9 * positive_count:
                    pairs_to_remove.append(pair)
            
            for pair in pairs_to_remove:
                del pair_labels[pair]
    
    return pair_labels

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
    
def analyze_table_columns(df1, df2, mapping_path=None, mode="train"):
    """
    Generate comparison features between columns of two tables.
    """
    # Parse mapping file if provided
    mappings = parse_mappings(mapping_path)
    
    # Get column names
    columns1 = list(df1.columns)
    columns2 = list(df2.columns)

    # Generate column pairs with labels
    pair_labels = generate_column_pairs(columns1, columns2, mappings, mode)
    
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
        
        # Augment data in training mode by occasionally masking column names
        if mode == "train" and idx % 5 == 0:
            # Create a version with masked column name features
            masked_name_features = np.array([0, 12, 0, 0.2, 0])
            
            # Combine into full feature vector
            augmented_features = np.concatenate((
                profile_diffs[:-768],
                masked_name_features,
                content_sim
            ))
            
            # Add to output arrays
            augmented_features = augmented_features.reshape(1, -1)
            output_features = np.concatenate((output_features, augmented_features), axis=0)
            output_labels = np.concatenate((output_labels, np.array([label])))
    
    return output_features, output_labels

if __name__ == '__main__':
    # Setup directory structure
    if os.path.exists("ProcessedData"):
        subprocess.call(["rm", "-r", "ProcessedData"])
    
    os.mkdir("ProcessedData")
    
    # Process all folders in training data
    folders = os.listdir("Training Examples")
    
    train_features_dict = {}
    train_labels_dict = {}
    test_features_dict = {}
    test_labels_dict = {}
    
    for folder in folders:
        print(f"Processing {folder}...")
        data_path = os.path.join("Training Examples", folder)
        
        # Load tables
        table1 = pd.read_csv(os.path.join(data_path, "Table1.csv"))
        table2 = pd.read_csv(os.path.join(data_path, "Table2.csv"))
        
        # Clean tables
        table1 = clean_dataframe_columns(table1)
        table2 = clean_dataframe_columns(table2)
        
        # Get mapping file
        mapping_file = os.path.join(data_path, "mapping.txt")
        
        # Generate features for training
        features, labels = analyze_table_columns(table1, table2, mapping_file, mode="train")
        train_features_dict[folder] = features
        train_labels_dict[folder] = labels
        
        # Generate features for testing
        features, labels = analyze_table_columns(table1, table2, mapping_file, mode="test")
        test_features_dict[folder] = features
        test_labels_dict[folder] = labels
    
    # Save data with cross-validation structure
    for i, test_folder in enumerate(folders):
        # Create directories
        fold_dir = os.path.join("ProcessedData", str(i))
        os.makedirs(os.path.join(fold_dir, "train"))
        os.makedirs(os.path.join(fold_dir, "test"))
        
        # Determine train/test split
        train_folders = [f for f in folders if f != test_folder]
        
        # Save training data
        for folder in train_folders:
            np.save(
                os.path.join(fold_dir, "train", f"{folder}_features.npy"), 
                train_features_dict[folder]
            )
            np.save(
                os.path.join(fold_dir, "train", f"{folder}_labels.npy"), 
                train_labels_dict[folder]
            )
        
        # Save test data
        np.save(
            os.path.join(fold_dir, "test", f"{test_folder}_features.npy"),
            test_features_dict[test_folder]
        )
        np.save(
            os.path.join(fold_dir, "test", f"{test_folder}_labels.npy"),
            test_labels_dict[test_folder]
        )

import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from models import ClientModel
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
torch.manual_seed(3)
RANDOM_SEED = 3


def calculate_group_mi(group, X, y):
    """Calculate mutual information of a group of features with the target."""
    group_data = X[list(group)]
    return mutual_info_regression(group_data, y)

def calculate_feature_interactions(X):
    """Calculate pairwise feature interactions using Spearman correlation."""
    corr_matrix, _ = spearmanr(X)
    return pd.DataFrame(np.abs(corr_matrix), index=X.columns, columns=X.columns)

def random_feature_distribution(X, n_clusters, min_feature_per_client, run_id, use_saved=False, save_dir="reliability_lists"):
    """Randomly distribute features into clusters without duplication."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = f"client_reliability_{run_id}.pkl" if run_id is not None else "client_reliability_default.pkl"
    filepath = os.path.join(save_dir, filename)
    
    if use_saved and os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            client_reliability = pickle.load(f)
        print(f"Loaded reliability list for run {run_id}: {client_reliability}")
    else:
        client_reliability = list(np.random.rand(n_clusters))  # Generate reliability list with n_clusters elements
        print(f"Generated new reliability list for run {run_id}: {client_reliability}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(client_reliability, f)
        print(f"Saved reliability list to {filepath}")
    
    features = list(X.columns)
    print(f"len features = {len(features)}")
    np.random.shuffle(features)
    print(f'Reliability of users = {client_reliability}')
    
    total_features = len(features)
    min_required_features = n_clusters * min_feature_per_client
    
    if total_features < min_required_features:
        raise ValueError(f"Not enough features ({total_features}) to ensure at least {min_feature_per_client} per client")
    
    cluster_sizes = np.full(n_clusters, min_feature_per_client)
    extra_features = total_features - min_required_features
    
    for i in range(extra_features):
        cluster_sizes[i % n_clusters] += 1  # Distribute extra features evenly without out-of-bounds error
    
    clusters = []
    start = 0
    for size in cluster_sizes:
        clusters.append(features[start:start + size])  # Ensure unique feature assignment
        start += size
    
    # Verify no duplicate features across clusters
    seen_features = set()
    for cluster in clusters:
        for feature in cluster:
            if feature in seen_features:
                raise ValueError(f"Feature duplication detected: {feature}")
            seen_features.add(feature)
    
    return clusters, client_reliability



def partition_features(features, feature_importance, n_clients, client_reliability, min_features_per_client=1):
    """
    Partition features among clients based on their importance and reliability scores.
    
    Args:
    features (list): List of feature names.
    feature_importance (pd.Series): Feature importance scores.
    n_clients (int): Number of clients.
    client_reliability (list): List of reliability scores for each client (0 to 1).
    min_features_per_client (int): Minimum number of features per client.
    
    Returns:
    list: List of feature sets for each client.
    """
    # Fix client_reliability length if needed
    if len(client_reliability) != n_clients:
        if len(client_reliability) < n_clients:
            extension = list(np.random.rand(n_clients - len(client_reliability)))
            client_reliability = client_reliability + extension
        else:
            client_reliability = client_reliability[:n_clients]
    
    if min(client_reliability) < 0 or max(client_reliability) > 1:
        client_reliability = [max(0, min(1, r)) for r in client_reliability]
    
    # Sort features by importance
    if isinstance(feature_importance, dict):
        feature_importance_dict = feature_importance
    else:
        feature_importance_dict = dict(zip(features, feature_importance))
        
    sorted_features = sorted(features, key=lambda x: feature_importance_dict.get(x, 0), reverse=True)
    
    # Sort clients by reliability
    client_indices = list(range(n_clients))
    sorted_clients = sorted(client_indices, key=lambda x: client_reliability[x], reverse=True)
    
    # Initialize client feature sets
    client_features = [[] for _ in range(n_clients)]
    
    # Maintain a single source of truth for all assigned features
    all_assigned_features = set()
    
    # Calculate feature counts based on reliability
    total_reliability = sum(client_reliability)
    normalized_reliability = [r/total_reliability for r in client_reliability]
    total_features = len(features)
    
    # Calculate target feature counts
    target_features = [max(min_features_per_client, 
                          int(r * total_features)) 
                      for r in normalized_reliability]
    
    # Adjust target features to match total available
    while sum(target_features) > total_features:
        max_idx = target_features.index(max(target_features))
        if target_features[max_idx] > min_features_per_client:
            target_features[max_idx] -= 1
    
    # Initial distribution based on reliability and importance
    for client_idx in sorted_clients:
        features_to_assign = target_features[client_idx]
        
        for feature in sorted_features:
            if len(client_features[client_idx]) >= features_to_assign:
                break
                
            if feature not in all_assigned_features:
                client_features[client_idx].append(feature)
                all_assigned_features.add(feature)
    
    # Distribute remaining features
    remaining_features = [f for f in features if f not in all_assigned_features]
    for feature in remaining_features:
        # Find client with fewest features
        client_idx = min(range(n_clients), key=lambda x: len(client_features[x]))
        client_features[client_idx].append(feature)
        all_assigned_features.add(feature)
    
    # Verify no duplication across clients after initial distribution
    all_features_check = []
    for client_feature_list in client_features:
        all_features_check.extend(client_feature_list)
    
    if len(all_features_check) != len(set(all_features_check)):
        # This indicates a bug in our logic - let's rebuild client_features from scratch
        # with a more explicit approach that guarantees no duplication
        client_features = [[] for _ in range(n_clients)]
        available_features = sorted_features.copy()
        
        # First ensure minimum per client
        for i in range(n_clients):
            for _ in range(min_features_per_client):
                if available_features:
                    feature = available_features.pop(0)
                    client_features[i].append(feature)
        
        # Then distribute remaining features based on reliability
        client_order = sorted_clients.copy()
        while available_features:
            for client_idx in client_order:
                if not available_features:
                    break
                feature = available_features.pop(0)
                client_features[client_idx].append(feature)
    
    # Final check: ensure no feature appears in multiple clients
    assigned_counts = {}
    for i, feature_list in enumerate(client_features):
        for feature in feature_list:
            if feature in assigned_counts:
                # Found duplicate! Remove from current client
                client_features[i].remove(feature)
            else:
                assigned_counts[feature] = i
    
    # Ensure minimum again after removing duplicates
    for i in range(n_clients):
        if len(client_features[i]) < min_features_per_client:
            # Find unassigned features
            all_current_features = set()
            for feature_list in client_features:
                all_current_features.update(feature_list)
            
            unassigned = [f for f in features if f not in all_current_features]
            
            # Assign unassigned features to meet minimum requirement
            needed = min_features_per_client - len(client_features[i])
            for j in range(min(needed, len(unassigned))):
                client_features[i].append(unassigned[j])
    
    return client_features



def vfl_feature_distribution(X, y,y_label, n_clients, min_features_per_client=5,run_id=None, use_saved=False, save_dir="reliability_lists"):
    """
    Distribute features to clients based on PCA importance for VFL.
    
    Args:
    X (pd.DataFrame): The input features.
    y (pd.Series): The target variable.
    n_clients (int): Number of clients.
    min_features_per_client (int): Minimum number of features per client.
    
    Returns:
    tuple: (client_features, client_scores)
    
    """

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    filename = f"client_reliability_{run_id}.pkl" if run_id is not None else f"client_reliability_default.pkl"
    filepath = os.path.join(save_dir, filename)

    if use_saved and os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            client_reliability = pickle.load(f)
        print(f"Loaded reliability list for run {run_id}: {client_reliability}")
    else:
        # Generate new reliability list
        alpha = 8
        beta = 2
        client_reliability = list(np.random.beta(alpha, beta, size=n_clients))
        print(f"Generated new reliability list for run {run_id}: {client_reliability}")
        
        # Save the generated list
        with open(filepath, 'wb') as f:
            pickle.dump(client_reliability, f)
        print(f"Saved reliability list to {filepath}")
        
    
    # Calculate feature importance
    model = RandomForestRegressor()
    # Assuming y_label is the name of the column we want to exclude
    X_without_y_label = X.drop(columns = y_label)
    features = list(X_without_y_label.columns)
    model.fit(X_without_y_label, y)
    # feature_importance = calculate_group_mi(features, X_without_y_label, y)


    # Get feature importances
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_without_y_label.columns, 'Importance': feature_importance})

    # Add y_label with importance = 0
    importance_df = pd.concat([importance_df, pd.DataFrame({'Feature': y_label, 'Importance': [0]})], ignore_index=True)
    # importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(f"feature _ importance : {importance_df}")
    # Partition features
    print(f'reliability of users = {client_reliability}') 
    client_features = partition_features(features,feature_importance, n_clients,client_reliability, min_features_per_client)
    
    # Calculate client scores (using sum of feature importances)
    
    return client_features, client_reliability, None 



def get_processed_data(filepath='features_nomiss.csv',params = None):
    """
    Retrieve and preprocess the dataset.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        tuple: (processed DataFrame, target variable name)
    """
    # Load data
    try:
        X = pd.read_csv(filepath)
    except:
        print("input a correct path to csv file")
        
    columns_to_drop = params.get("simulation").get("columns_drop")


    X.drop(columns_to_drop, axis=1, inplace=True)
    # print(X.columns)
    # Handle infinite values and missing data
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)

    # Convert all columns to float64
    X = X.astype(np.float64)
    # X = X.head(512)
    # Define target variable
    y_label = "qoe_YinX_v2"
    return X, y_label
    

def _create_features(df):

    all_keywords = set(df.columns)
 
    return df, all_keywords

def min_max_scaling(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)


def get_partitions_and_label(params=None, run_id=0,device=None):
    path = params.get("simulation").get("path")
    X, y_label = get_processed_data(path, params)

    # Generate features and partition data
    processed_df, all_keywords = _create_features(X)
    raw_partitions, reliability = _partition_data(processed_df, all_keywords, params, y_label, run_id)

    # Split data into train and test sets
    partitions, partitions_test = zip(*[train_test_split(partition, test_size=0.1, random_state=RANDOM_SEED)
                                        for partition in raw_partitions])
    
    partitions, partitions_test = list(partitions), list(partitions_test)  # Convert tuples to lists

    print(f"ylabel is {y_label}")

    # Extract y_label from the first partition only (since it's the same across all)
    if y_label in partitions[0].columns:
        train_ground = partitions[0][y_label].values
        test_ground = partitions_test[0][y_label].values
    else:
        raise ValueError(f"y_label '{y_label}' not found in the first partition!")

    # Drop y_label from all partitions
    for partition in partitions:
        partition.drop(columns=[y_label], inplace=True, errors="ignore")
    for partition in partitions_test:
        partition.drop(columns=[y_label], inplace=True, errors="ignore")

    # Verify if y_label is still present
    remaining_columns = [
        i for i, partition in enumerate(partitions) if y_label in partition.columns
    ] + [
        i + len(partitions) for i, partition in enumerate(partitions_test) if y_label in partition.columns
    ]

    if remaining_columns:
        print(f" Warning: y_label '{y_label}' is still present in partitions {remaining_columns}!")



    # Normalize features
    partitions = [normalize_features(partition) for partition in partitions]
    partitions_test = [normalize_features(partition) for partition in partitions_test]

    print(f"partitions shape = {[np.shape(p) for p in partitions]}")

    return partitions, partitions_test, train_ground, test_ground, len(partitions), reliability
    


def _partition_data(df, all_keywords, params, y_label, run_id):
    initial_n_clusters = params.get('simulation').get('num_clients') 
    min_feature_per_client = 4 
    if params.get('simulation').get('Optimized'):
        print("*******************************************Optimized Assignment*******************************************")
        clusters, client_reliability, _  = vfl_feature_distribution(df, df[y_label],y_label, initial_n_clusters, min_feature_per_client, run_id=run_id, use_saved=False, save_dir="reliability_lists")
    else:
        print("*******************************************Random Assignment*******************************************")
        clusters, client_reliability = random_feature_distribution(df, initial_n_clusters, min_feature_per_client, run_id=run_id, use_saved=True, save_dir="reliability_lists")


    partitions = []
    
    keywords_sets = clusters
    
    for keywords in keywords_sets:
        selected_columns = set(keywords)  # Only include assigned features
        if y_label in df.columns:
            selected_columns.add(y_label)  # Ensure y_label is included
        
        partitions.append(df[list(selected_columns)])
  
    
    # Check for intersection of features across partitions (except y_label)
    all_features = [set(part.columns) - {y_label} for part in partitions]
    for i in range(len(all_features)):
        for j in range(i + 1, len(all_features)):
            intersection = all_features[i] & all_features[j]
            if intersection:
                raise ValueError(f"Feature overlap detected between partitions {i} and {j}: {intersection}")
    
    return partitions, client_reliability

def normalize_features(df, epsilon=1e-10):
    """
    Normalize features using min-max normalization for pandas DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame containing the features to normalize
        epsilon (float): Small constant to avoid division by zero
        
    Returns:
        normalized DataFrame
    """
    # Skip normalization if DataFrame is empty
    if df.empty:
        return df
        
    # Compute min and max for each feature
    min_vals = df.min()
    max_vals = df.max()
    
    # Check for constant features
    constant_features = (max_vals - min_vals) < epsilon
    if any(constant_features):
        print(f"Warning: Features {constant_features[constant_features].index.tolist()} are constant")
    
    # Normalize the data
    normalized_df = (df - min_vals) / (max_vals - min_vals + epsilon)
    
    return normalized_df

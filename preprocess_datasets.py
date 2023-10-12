import requests
from tqdm import tqdm
from pathlib import Path
import os
import zipfile
import io
import pickle
import scipy.io as scio
import scipy.sparse as sp
from scipy.stats import rankdata
import numpy as np
import networkx as nx
from community import community_louvain
import copy
import argparse
import json


def convert_ratings_to_sparse(ratings, trustnetwork):
    max_values_ratings = np.amax(ratings, axis=0)
    max_values_trustnetwork = np.amax(trustnetwork, axis=0)
    max_user_id = max(max_values_ratings[0], max_values_trustnetwork[0], max_values_trustnetwork[1])
    print(f"{max_values_ratings[0]} {max_values_trustnetwork[0]} {max_values_trustnetwork[1]}")
    rows, cols = max_user_id + 1, max_values_ratings[1] + 1

    row_indices = ratings[:, 0]
    col_indices = ratings[:, 1]
    values = ratings[:, 3]

    sparse_matrix = sp.csr_matrix((values, (row_indices, col_indices)), shape=(rows, cols))
    # # Find non-zero columns
    # non_zero_columns = sparse_matrix.sum(axis=0).nonzero()[1]
    #
    # # Retain only non-zero columns
    # filtered_sparse_matrix = sparse_matrix[:, non_zero_columns]

    return sparse_matrix

def convert_trustnetwork_to_sparse(ratings, trustnetwork):
    max_values_ratings = np.amax(ratings, axis=0)
    max_values_trustnetwork = np.amax(trustnetwork, axis=0)
    max_user_id = max(max_values_ratings[0], max_values_trustnetwork[0], max_values_trustnetwork[1])
    rows, cols = max_user_id + 1, max_user_id + 1

    row_indices = trustnetwork[:, 0]
    col_indices = trustnetwork[:, 1]
    values = np.ones(len(row_indices))

    sparse_matrix = sp.csr_matrix((values, (row_indices, col_indices)), shape=(rows, cols))
    return sparse_matrix

def convert_trustnetwork_to_undirected(sparse_trustnetwork):
    # Transpose the matrix
    matrix_transpose = sparse_trustnetwork.transpose()

    # Perform element-wise OR operation on the original and transposed matrices
    symmetric_matrix = (sparse_trustnetwork + matrix_transpose).astype(bool).astype(int)

    return symmetric_matrix

def convert_nonzero_to_ones(matrix):
    # Find the non-zero indices
    row_indices, col_indices = matrix.nonzero()
    # Set all non-zero values to ones
    values = [1] * len(row_indices)
    # Create a new sparse matrix with the same shape as the original matrix
    binary_matrix = sp.coo_matrix((values, (row_indices, col_indices)), shape=matrix.shape)

    return binary_matrix

def get_users_to_remove_from_ratings(ratings_mat, min_user_num_ratings):
    row_sums = np.squeeze(np.asarray(ratings_mat.sum(axis=1)))
    empty_row_ids = np.where(row_sums == 0)[0]

    implicit_matrix = convert_nonzero_to_ones(ratings_mat)
    row_sums = np.squeeze(np.asarray(implicit_matrix.sum(axis=1)))
    less_than_threshold_row_ids = np.where(row_sums < min_user_num_ratings)[0]

    rows_to_remove = np.unique(np.concatenate((empty_row_ids, less_than_threshold_row_ids)))
    return rows_to_remove

def get_users_to_remove_from_trustnetwork(trustnetwork_mat):
    row_sums = np.squeeze(np.asarray(trustnetwork_mat.sum(axis=1)))
    col_sums = np.squeeze(np.asarray(trustnetwork_mat.sum(axis=0)))

    empty_row_ids = np.where(row_sums == 0)[0]
    empty_col_ids = np.where(col_sums == 0)[0]

    return np.unique(np.concatenate((empty_row_ids, empty_col_ids)))

def get_items_to_remove_from_ratings(ratings_mat, min_item_num_ratings):
    col_sums = np.squeeze(np.asarray(ratings_mat.sum(axis=0)))
    empty_col_ids = np.where(col_sums == 0)[0]

    implicit_matrix = convert_nonzero_to_ones(ratings_mat)
    col_sums = np.squeeze(np.asarray(implicit_matrix.sum(axis=0)))
    less_than_threshold_col_ids = np.where(col_sums < min_item_num_ratings)[0]

    cols_to_remove = np.unique(np.concatenate((empty_col_ids, less_than_threshold_col_ids)))
    return cols_to_remove

def remove_rows(matrix, rows_to_remove):
    rows_to_remove = np.sort(rows_to_remove)
    all_rows = np.arange(matrix.shape[0])
    rows_to_keep = np.setdiff1d(all_rows, rows_to_remove)
    return matrix[rows_to_keep, :]

def remove_columns(matrix, columns_to_remove):
    columns_to_remove = np.sort(columns_to_remove)
    all_columns = np.arange(matrix.shape[1])
    columns_to_keep = np.setdiff1d(all_columns, columns_to_remove)
    return matrix[:, columns_to_keep]

def remove_zero_ratings(ratings):
    ratings_filtered = ratings[ratings[:, 3] != 0]
    return ratings_filtered

def remove_duplicates(ratings):
    user_item_ratings = {}
    ratings_list = ratings.tolist()

    for entry in ratings_list:
        user_item_ratings[(entry[0], entry[1])] = entry

    filtered_ratings = np.array(list(user_item_ratings.values()))

    return filtered_ratings

def create_graph_from_sparse_matrix(sparse_trustnetwork):
    G = nx.Graph()

    # Get the nonzero elements of the sparse matrix
    nonzero_rows, nonzero_cols = sparse_trustnetwork.nonzero()

    # Iterate over the nonzero elements and add edges to the graph
    for row, col in zip(nonzero_rows, nonzero_cols):
        G.add_edge(row, col)

    return G

def get_component_sizes(G):
    # Use the built-in function `nx.connected_components` to get a list of all connected components in G
    # Each connected component is represented as a set of nodes
    components = nx.connected_components(G)

    # Use a list comprehension to get the length of each connected component, and store the lengths in a list
    component_sizes = [len(c) for c in components]

    # Sort the list of component sizes in descending order (largest to smallest)
    component_sizes.sort(reverse=True)

    # Return the list of component sizes
    return component_sizes

def get_largest_component_subgraph(G):
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    largest_subgraph = G.subgraph(largest_component)
    return largest_subgraph

def get_communities(G):
    # Initialize an empty dictionary to store the communities
    communities = {}

    # Use the Louvain community detection algorithm to partition the graph into communities
    graph_communities = community_louvain.best_partition(G)

    # Loop through each node in the graph
    for user_id in graph_communities:
        # Find the ID of the community that the node belongs to
        community_id = graph_communities[user_id]

        # If the community ID is not in the `communities` dictionary, add it as a key with an empty list as the value
        if community_id not in communities:
            communities[community_id] = []

        # Append the node to the list of nodes in the appropriate community
        communities[community_id].append(user_id)

    # Return the dictionary of communities
    return communities

def calculate_rating_density(sparse_ratings):
    # Get the shape of the sparse matrix
    num_rows, num_cols = sparse_ratings.shape

    # Calculate the total number of elements in the matrix
    total_elements = num_rows * num_cols

    # Calculate the number of nonzero elements in the matrix
    nonzero_elements = sparse_ratings.nnz

    # Calculate the rating density
    rating_density = nonzero_elements / total_elements

    return rating_density

def get_densities(sparse_trustnetwork, sparse_ratings):
    G = create_graph_from_sparse_matrix(sparse_trustnetwork)
    # Calculate the social density of the graph
    social_density = nx.density(G)

    # Calculate the rating density of the user-item matrix using the calculate_rating_density function
    rating_density = calculate_rating_density(sparse_ratings)

    # Return a tuple of the social and rating densities
    return social_density, rating_density

def clean_empty_rows_and_cols(sparse_ratings, sparse_trustnetwork):
    # Check for empty rows
    row_sums = np.array(sparse_ratings.sum(axis=1)).flatten()
    empty_rows = np.where(row_sums == 0)[0]

    # Check for empty columns
    col_sums = np.array(sparse_ratings.sum(axis=0)).flatten()
    empty_cols = np.where(col_sums == 0)[0]

    while empty_rows.size > 0 or empty_cols.size > 0:
        sparse_ratings = remove_rows(sparse_ratings, empty_rows)
        sparse_ratings = remove_columns(sparse_ratings, empty_cols)
        sparse_trustnetwork = remove_rows(sparse_trustnetwork, empty_rows)
        sparse_trustnetwork = remove_columns(sparse_trustnetwork, empty_rows)

        # Check for empty rows
        row_sums = np.array(sparse_ratings.sum(axis=1)).flatten()
        empty_rows = np.where(row_sums == 0)[0]

        # Check for empty columns
        col_sums = np.array(sparse_ratings.sum(axis=0)).flatten()
        empty_cols = np.where(col_sums == 0)[0]

    return sparse_ratings, sparse_trustnetwork

def find_the_most_appropriate_community(G, sparse_ratings, sparse_trustnetwork):
    communities = get_communities(G)

    social_density_full, rating_density_full = get_densities(sparse_trustnetwork, sparse_ratings)

    social_densities_comms = {}
    rating_densities_comms = {}
    social_density_maes = {}
    rating_density_maes = {}
    community_dictionaries = {}

    for comm in tqdm(communities):
        if len(communities[comm]) < 0.05 * len(G.nodes):
            continue

        users_to_keep = communities[comm]
        keep_indices = np.array(users_to_keep)

        sparse_ratings_comm = sparse_ratings.tocsr()[keep_indices, :]
        sparse_trustnetwork_comm = sparse_trustnetwork.tocsr()[keep_indices, :][:, keep_indices]
        sparse_ratings_comm, sparse_trustnetwork_comm = clean_empty_rows_and_cols(copy.deepcopy(sparse_ratings_comm),
                                                                                  copy.deepcopy(
                                                                                      sparse_trustnetwork_comm))

        assert sparse_ratings_comm.shape[0] == sparse_trustnetwork_comm.shape[0] == sparse_trustnetwork_comm.shape[
            1], "Matrices shape mismatch!"
        print(f"{sparse_ratings_comm.nnz}, {sparse_trustnetwork_comm.nnz}, {sparse_ratings_comm.shape}, {sparse_trustnetwork_comm.shape}")

        social_density_comm, rating_density_comm = get_densities(sparse_trustnetwork_comm, sparse_ratings_comm)
        social_densities_comms[comm] = social_density_comm
        rating_densities_comms[comm] = rating_density_comm
        social_density_maes[comm] = abs(social_density_full - social_density_comm)
        rating_density_maes[comm] = abs(rating_density_full - rating_density_comm)

        community_dictionaries[comm] = {
            "num_users": sparse_ratings_comm.shape[0],
            "num_items": sparse_ratings_comm.shape[1],
            "num_ratings": sparse_ratings_comm.nnz,
            "social_density_comm": social_density_comm,
            "rating_density_comm": rating_density_comm,
            "social_density_mae": social_density_maes[comm],
            "rating_density_mae": rating_density_maes[comm],
            "sparse_ratings": sparse_ratings_comm,
            "sparse_trustnetwork": sparse_trustnetwork_comm
        }

    social_density_mae_values = list(social_density_maes.values())
    rating_density_mae_values = list(rating_density_maes.values())
    social_density_mae_ranked = rankdata(social_density_mae_values, method='dense')
    rating_density_mae_ranked = rankdata(rating_density_mae_values, method='dense')
    social_density_mae_ranks = dict(zip(social_density_maes.keys(), social_density_mae_ranked))
    rating_density_mae_ranks = dict(zip(rating_density_maes.keys(), rating_density_mae_ranked))

    average_ranks = {}
    for comm in community_dictionaries:
        average_ranks[comm] = (social_density_mae_ranks[comm] + rating_density_mae_ranks[comm]) / 2
        community_dictionaries[comm]["social_density_mae_rank"] = social_density_mae_ranks[comm]
        community_dictionaries[comm]["rating_density_mae_rank"] = rating_density_mae_ranks[comm]
        community_dictionaries[comm]["average_rank"] = average_ranks[comm]

    most_appropriate_community_id = min(average_ranks, key=average_ranks.get)

    return community_dictionaries, most_appropriate_community_id, social_density_full, rating_density_full

def get_users_with_zero_connections(sparse_trustnetwork):
    # Convert to CSC format for column operations
    sparse_trustnetwork_csc = sparse_trustnetwork.tocsc()

    users_with_zero_connections = []
    for i in range(sparse_trustnetwork.shape[0]):
        # Check if the row has no non-zero elements
        if sparse_trustnetwork.indptr[i] == sparse_trustnetwork.indptr[i + 1]:
            # Check if the column has no non-zero elements
            if sparse_trustnetwork_csc.indptr[i] == sparse_trustnetwork_csc.indptr[i + 1]:
                users_with_zero_connections.append(i)
    return np.array(users_with_zero_connections)

def get_users_with_zero_ratings(sparse_ratings):
    # Find users (rows) with no non-zero ratings
    users_with_zero_ratings = np.where(np.diff(sparse_ratings.indptr) == 0)[0]
    return users_with_zero_ratings

def save_stats(dataset_name, prefix, stats):
    parent_directory = f"./data/processed/graphrec/{dataset_name}/filtered"
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    with open(f"{parent_directory}/{prefix}_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

def compute_stats_from_numpy_arrays(ratings, trustnetwork):
    stats = {}
    stats["users_in_ratings"] = np.unique(ratings[:, 0]).shape[0]
    stats["items_in_ratings"] = np.unique(ratings[:, 1]).shape[0]
    stats["num_ratings"] = ratings.shape[0]
    stats["users_in_trustnetwork"] = np.unique(np.concatenate((trustnetwork[:, 0], trustnetwork[:, 1]))).shape[0]
    stats["social_connections"] = trustnetwork.shape[0]
    stats["social_density"] = stats["social_connections"] / (stats["users_in_trustnetwork"] ** 2)
    stats["rating_density"] = stats["num_ratings"] / (stats["users_in_ratings"] * stats["items_in_ratings"])
    stats["users_with_ratings_but_zero_connections"] = np.setdiff1d(np.unique(ratings[:, 0]), np.concatenate((trustnetwork[:, 0], trustnetwork[:, 1]))).shape[0]
    stats["users_with_zero_ratings_but_connections"] = np.setdiff1d(np.unique(trustnetwork[:, 0]), np.unique(ratings[:, 0])).shape[0]
    return stats

def compute_stats_from_sparse_matrices(sparse_ratings, sparse_trustnetwork):
    stats = {}
    stats["users_in_ratings"] = int(np.sum(sparse_ratings.getnnz(axis=1) > 0))
    stats["items_in_ratings"] = int(np.sum(sparse_ratings.getnnz(axis=0) > 0))
    stats["num_ratings"] = sparse_ratings.nnz
    users_with_outgoing_connections = list(set(sparse_trustnetwork.nonzero()[0]))
    users_with_incoming_connections = list(set(sparse_trustnetwork.nonzero()[1]))
    users_with_connections = list(set(users_with_outgoing_connections).union(users_with_incoming_connections))
    stats["users_in_trustnetwork"] = len(users_with_connections)
    stats["social_connections"] = sparse_trustnetwork.nnz
    stats["social_density"] = stats["social_connections"] / (stats["users_in_trustnetwork"] ** 2)
    stats["rating_density"] = stats["num_ratings"] / (stats["users_in_ratings"] * stats["items_in_ratings"])
    stats["users_with_ratings_but_zero_connections"] = np.setdiff1d(sparse_ratings.nonzero()[0], users_with_connections).shape[0]
    stats["users_with_zero_ratings_but_connections"] = np.setdiff1d(users_with_connections, sparse_ratings.nonzero()[0]).shape[0]
    return stats

def remove_users_with_no_ratings(ratings, trustnetwork):
    # Extract user ids from ratings
    user_ids_ratings = set(ratings[:, 0])
    # Extract user ids from trustnetwork (both columns)
    user_ids_trustnetwork = set(np.concatenate((trustnetwork[:, 0], trustnetwork[:, 1])))
    # Find user ids that are in trustnetwork but not in ratings
    difference = user_ids_trustnetwork - user_ids_ratings
    # Filter out the undesired entries from trustnetwork
    filtered_trustnetwork = trustnetwork[~((np.isin(trustnetwork[:, 0], list(difference))) | (np.isin(trustnetwork[:, 1], list(difference))))]
    return ratings, filtered_trustnetwork

def remove_users_with_no_connections(ratings, trustnetwork):
    # Extract unique user ids from trustnetwork
    user_ids_trustnetwork = set(np.concatenate([trustnetwork[:, 0], trustnetwork[:, 1]]))
    # Extract user ids from ratings
    user_ids_ratings = set(ratings[:, 0])
    # Identify user ids that are in ratings but not in trustnetwork
    difference = user_ids_ratings - user_ids_trustnetwork
    # Remove rows from ratings that have the user ids in difference
    filtered_ratings = ratings[~np.isin(ratings[:, 0], list(difference))]
    return filtered_ratings, trustnetwork

def remove_all_zero_users(sparse_ratings, sparse_trustnetwork):
    # Find all-zero rows in sparse_ratings
    zero_rows_ratings = np.where(sparse_ratings.sum(axis=1).A1 == 0)[0]

    # Find all-zero rows in sparse_trustnetwork
    zero_rows_trustnetwork = np.where(sparse_trustnetwork.sum(axis=1).A1 == 0)[0]

    # Find all-zero columns in sparse_trustnetwork
    zero_cols_trustnetwork = np.where(sparse_trustnetwork.sum(axis=0).A1 == 0)[0]

    # Find the intersection of the indices
    to_remove = np.intersect1d(zero_rows_ratings, zero_rows_trustnetwork)
    to_remove = np.intersect1d(to_remove, zero_cols_trustnetwork)

    # Remove rows from sparse_ratings
    mask_ratings = np.ones(sparse_ratings.shape[0], dtype=bool)
    mask_ratings[to_remove] = False
    sparse_ratings_filtered = sparse_ratings[mask_ratings]

    # Remove rows and columns from sparse_trustnetwork
    mask_trustnetwork = np.ones(sparse_trustnetwork.shape[0], dtype=bool)
    mask_trustnetwork[to_remove] = False
    sparse_trustnetwork_filtered = sparse_trustnetwork[mask_trustnetwork][:, mask_trustnetwork]

    return sparse_ratings_filtered, sparse_trustnetwork_filtered

def clean_and_evaluate_matrices(sparse_ratings, sparse_trustnetwork, item_categories):
    filtered_sparse_ratings, filtered_sparse_trustnetwork = remove_all_zero_users(sparse_ratings, sparse_trustnetwork)
    # Remove zero item columns
    # Find non-zero columns
    non_zero_item_columns = filtered_sparse_ratings.sum(axis=0).nonzero()[1]
    # Retain only non-zero columns
    # Find all-zero columns
    zero_cols = np.where(filtered_sparse_ratings.sum(axis=0).A1 == 0)[0]
    # Create a mask where non-zero columns are set to True
    mask = np.ones(filtered_sparse_ratings.shape[1], dtype=bool)
    mask[zero_cols] = False
    # Use the mask to remove the all-zero columns
    filtered_sparse_ratings = filtered_sparse_ratings[:, mask]
    # Remove the corresponding columns from item_categories
    filtered_item_categories = item_categories[:, mask]
    # Assert that filtered_sparse_matrix and item_categories has no zero columns
    assert not any(filtered_sparse_ratings.sum(axis=0).A1 == 0), "filtered_sparse_ratings has one or more all-zero columns"
    assert filtered_sparse_ratings.shape[1] == filtered_item_categories.shape[1], "Mismatch in number of columns between filtered_sparse_ratings and item_categories"
    # Assert that sparse_ratings has the same number of rows as sparse_trustnetwork has rows and columns
    assert filtered_sparse_ratings.shape[0] == filtered_sparse_trustnetwork.shape[0], "Mismatch in number of rows between sparse_ratings and sparse_trustnetwork"
    assert filtered_sparse_ratings.shape[0] == filtered_sparse_trustnetwork.shape[1], "Mismatch in number of rows of sparse_ratings and columns of sparse_trustnetwork"
    return filtered_sparse_ratings, filtered_sparse_trustnetwork, filtered_item_categories

def create_item_category_matrix(ratings):
    item_category_mapping = {}
    unique_rows = np.unique(ratings[:, [1, 2]], axis=0)
    sorted_unique_rows = unique_rows[unique_rows[:, 0].argsort()]
    for entry in sorted_unique_rows:
        item_category_mapping[entry[0]] = entry[1]
    item_categories = []
    max_item_id = np.max(sorted_unique_rows[:, 0])
    for i in range(max_item_id + 1):
        if i in item_category_mapping:
            item_categories.append(item_category_mapping[i])
        else:
            item_categories.append(0)
    # Convert item_category_mapping to numpy array of size (1, max_item_id )
    item_categories = np.array(item_categories).reshape(1, -1)
    return item_categories

def preprocess_datasets(ratings_mat,
                        trustnetwork_mat,
                        dataset_name,
                        is_remove_zero_ratings=True,
                        is_remove_duplicates=True,
                        is_remove_users_with_no_ratings=True,
                        is_remove_users_with_no_connections=True,
                        convert_to_undirected=True,
                        # users_from_ratings_must_be_in_trustnetwork=True,
                        min_user_num_ratings=0,
                        min_item_num_ratings=0,
                        keep_only_lcc=False,
                        keep_only_best_community=False):
    ratings = ratings_mat["rating"]
    trustnetwork = trustnetwork_mat["trustnetwork"]
    item_categories = create_item_category_matrix(ratings)

    # Compute stats from numpy arrays
    stats = compute_stats_from_numpy_arrays(ratings, trustnetwork)
    save_stats(dataset_name, f"{dataset_name}_original", stats)

    # Step 1: Remove zero ratings
    if is_remove_zero_ratings == True:
        # Remove zero ratings
        ratings = remove_zero_ratings(ratings)
        stats = compute_stats_from_numpy_arrays(ratings, trustnetwork)
        save_stats(dataset_name, f"{dataset_name}_"
                                 f"r0-{is_remove_zero_ratings}_"
                                 f"rd-False_"
                                 f"runr-False_"
                                 f"runconn-False_"
                                 f"undir-False_"
                                 f"minur-0_"
                                 f"minir-0"
                                 f"_lcc_False"
                                 f"_bcomm_False", stats)

    # Step 2: Remove duplicates
    if is_remove_duplicates == True:
        # Keep only the last rating of each user for each item
        ratings = remove_duplicates(ratings)
        stats = compute_stats_from_numpy_arrays(ratings, trustnetwork)
        save_stats(dataset_name, f"{dataset_name}_"
                                 f"r0-{is_remove_zero_ratings}_"
                                 f"rd-{is_remove_duplicates}_"
                                 f"runr-False_"
                                 f"runconn-False_"
                                 f"undir-False_"
                                 f"minur-0_"
                                 f"minir-0"
                                 f"_lcc_False"
                                 f"_bcomm_False", stats)

    # Step 3: Remove users with no ratings
    if is_remove_users_with_no_ratings == True:
        ratings, trustnetwork = remove_users_with_no_ratings(ratings, trustnetwork)
        stats = compute_stats_from_numpy_arrays(ratings, trustnetwork)
        save_stats(dataset_name, f"{dataset_name}_"
                                 f"r0-{is_remove_zero_ratings}_"
                                 f"rd-{is_remove_duplicates}_"
                                 f"runr-{is_remove_users_with_no_ratings}_"
                                 f"runconn-False_"
                                 f"undir-False_"
                                 f"minur-0_"
                                 f"minir-0"
                                 f"_lcc_False"
                                 f"_bcomm_False", stats)

    # Step 4: Remove users with no connections
    if is_remove_users_with_no_connections == True:
        ratings, trustnetwork = remove_users_with_no_connections(ratings, trustnetwork)
        stats = compute_stats_from_numpy_arrays(ratings, trustnetwork)
        save_stats(dataset_name, f"{dataset_name}_"
                                 f"r0-{is_remove_zero_ratings}_"
                                 f"rd-{is_remove_duplicates}_"
                                 f"runr-{is_remove_users_with_no_ratings}_"
                                 f"runconn-{is_remove_users_with_no_connections}_"
                                 f"undir-False_"
                                 f"minur-0_"
                                 f"minir-0"
                                 f"_lcc_False"
                                 f"_bcomm_False", stats)

    # Take max user and item ids from ratings and trustnetwork to be able to define the shape of the sparse matrix
    sparse_ratings = convert_ratings_to_sparse(ratings, trustnetwork)
    # Take max user id from ratings and trustnetwork to define the shape of the sparse matrix
    sparse_trustnetwork = convert_trustnetwork_to_sparse(ratings, trustnetwork)
    sparse_ratings, sparse_trustnetwork, item_categories = clean_and_evaluate_matrices(sparse_ratings, sparse_trustnetwork, item_categories)

    # Step 5: Convert trustnetwork to undirected
    if convert_to_undirected == True:
        sparse_trustnetwork = convert_trustnetwork_to_undirected(sparse_trustnetwork)
        stats = compute_stats_from_sparse_matrices(sparse_ratings, sparse_trustnetwork)
        save_stats(dataset_name, f"{dataset_name}_"
                                 f"r0-{is_remove_zero_ratings}_"
                                 f"rd-{is_remove_duplicates}_"
                                 f"runr-{is_remove_users_with_no_ratings}_"
                                 f"runconn-{is_remove_users_with_no_connections}_"
                                 f"undir-{convert_to_undirected}_"
                                 f"minur-0_"
                                 f"minir-0"
                                 f"_lcc_False"
                                 f"_bcomm_False", stats)
    sparse_ratings, sparse_trustnetwork, item_categories = clean_and_evaluate_matrices(sparse_ratings, sparse_trustnetwork, item_categories)

    # Step 6: Remove users with less than min_user_num_ratings ratings
    if min_user_num_ratings > 0:
        users_to_remove = get_users_to_remove_from_ratings(sparse_ratings, min_user_num_ratings)
        sparse_ratings = remove_rows(sparse_ratings, users_to_remove)
        sparse_trustnetwork = remove_rows(sparse_trustnetwork, users_to_remove)
        sparse_trustnetwork = remove_columns(sparse_trustnetwork, users_to_remove)
        stats = compute_stats_from_sparse_matrices(sparse_ratings, sparse_trustnetwork)
        save_stats(dataset_name, f"{dataset_name}_"
                                 f"r0-{is_remove_zero_ratings}_"
                                 f"rd-{is_remove_duplicates}_"
                                 f"runr-{is_remove_users_with_no_ratings}_"
                                 f"runconn-{is_remove_users_with_no_connections}_"
                                 f"undir-{convert_to_undirected}_"
                                 f"minur-{min_user_num_ratings}_"
                                 f"minir-0"
                                 f"_lcc_False"
                                 f"_bcomm_False", stats)
    sparse_ratings, sparse_trustnetwork, item_categories = clean_and_evaluate_matrices(sparse_ratings, sparse_trustnetwork, item_categories)

    # Step 7: Remove items with less than min_item_num_ratings ratings
    if min_item_num_ratings > 0:
        items_to_remove = get_items_to_remove_from_ratings(sparse_ratings, min_item_num_ratings)
        sparse_ratings = remove_columns(sparse_ratings, items_to_remove)
        item_categories = remove_columns(item_categories, items_to_remove)
        stats = compute_stats_from_sparse_matrices(sparse_ratings, sparse_trustnetwork)
        save_stats(dataset_name, f"{dataset_name}_"
                                 f"r0-{is_remove_zero_ratings}_"
                                 f"rd-{is_remove_duplicates}_"
                                 f"runr-{is_remove_users_with_no_ratings}_"
                                 f"runconn-{is_remove_users_with_no_connections}_"
                                 f"undir-{convert_to_undirected}_"
                                 f"minur-{min_user_num_ratings}_"
                                 f"minir-{min_item_num_ratings}"
                                 f"_lcc_False"
                                 f"_bcomm_False", stats)
    sparse_ratings, sparse_trustnetwork, item_categories = clean_and_evaluate_matrices(sparse_ratings, sparse_trustnetwork, item_categories)

    # users_with_zero_connections = get_users_with_zero_connections(sparse_trustnetwork)
    # users_with_zero_ratings = get_users_with_zero_ratings(sparse_ratings)

    # # Step 6: Get users that are in ratings but not in trustnetwork and vice versa
    # if users_from_ratings_must_be_in_trustnetwork == True:
    #     # Remove users from ratings that are not in trustnetwork
    #     users_to_remove = np.unique(np.union1d(users_with_zero_connections, users_with_zero_ratings))
    #     sparse_ratings = remove_rows(sparse_ratings, users_to_remove)
    #     sparse_trustnetwork = remove_rows(sparse_trustnetwork, users_to_remove)
    #     sparse_trustnetwork = remove_columns(sparse_trustnetwork, users_to_remove)
    #     stats = compute_stats_from_sparse_matrices(sparse_ratings, sparse_trustnetwork)
    #     save_stats(dataset_name, f"{dataset_name}"
    #                f"_is_remove_zero_ratings_{is_remove_zero_ratings}"
    #                f"_is_remove_duplicates_{is_remove_duplicates}"
    #                f"_convert_to_undirected_{convert_to_undirected}"
    #                f"_min_user_num_ratings_{min_user_num_ratings}"
    #                f"_min_item_num_ratings_{min_item_num_ratings}"
    #                f"_users_from_ratings_must_be_in_trustnetwork_{users_from_ratings_must_be_in_trustnetwork}", stats)
    # else:
    #     users_to_remove = np.intersect1d(users_with_zero_connections, users_with_zero_ratings)
    #     sparse_ratings = remove_rows(sparse_ratings, users_to_remove)
    #     sparse_trustnetwork = remove_rows(sparse_trustnetwork, users_to_remove)
    #     sparse_trustnetwork = remove_columns(sparse_trustnetwork, users_to_remove)
    #
    # sparse_ratings, sparse_trustnetwork = clean_empty_rows_and_cols(copy.deepcopy(sparse_ratings),
    #                                                                 copy.deepcopy(sparse_trustnetwork))

    # Remove users with less than min_user_num_ratings ratings
    # users_to_remove = np.unique(np.concatenate((get_users_to_remove_from_ratings(sparse_ratings, min_user_num_ratings), get_users_to_remove_from_trustnetwork(sparse_trustnetwork))))

    # Remove items with less than min_item_num_ratings ratings
    # Side note: in this process all users that are part of ratings but not trustnetwork are removed and vice versa (from both matrices)

    # sparse_ratings = remove_rows(sparse_ratings, users_to_remove)
    # sparse_ratings = remove_columns(sparse_ratings, items_to_remove)
    #
    # sparse_trustnetwork = remove_rows(sparse_trustnetwork, users_to_remove)
    # sparse_trustnetwork = remove_columns(sparse_trustnetwork, users_to_remove)
    #
    # sparse_ratings, sparse_trustnetwork = clean_empty_rows_and_cols(copy.deepcopy(sparse_ratings),
    #                                                                 copy.deepcopy(sparse_trustnetwork))

    G = create_graph_from_sparse_matrix(sparse_trustnetwork)
    if keep_only_lcc == True:
        # Get the largest component of the graph
        largest_component = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_component)

        users_to_keep = list(G_lcc.nodes)
        keep_indices = np.array(users_to_keep)

        print(f"{sparse_ratings.nnz}, {sparse_trustnetwork.nnz}, {sparse_ratings.shape}, {sparse_trustnetwork.shape}")
        sparse_ratings = sparse_ratings[keep_indices, :]
        sparse_trustnetwork = sparse_trustnetwork.tocsr()[keep_indices, :][:, keep_indices]
        sparse_ratings, sparse_trustnetwork = clean_empty_rows_and_cols(copy.deepcopy(sparse_ratings),
                                                                        copy.deepcopy(sparse_trustnetwork))
        print(f"{sparse_ratings.nnz}, {sparse_trustnetwork.nnz}, {sparse_ratings.shape}, {sparse_trustnetwork.shape}")
        G = create_graph_from_sparse_matrix(sparse_trustnetwork)

    if keep_only_best_community == True:
        community_dictionaries, most_appropriate_community_id, social_density_full, rating_density_full = find_the_most_appropriate_community(
            G, sparse_ratings, sparse_trustnetwork)
        sparse_ratings = community_dictionaries[most_appropriate_community_id]["sparse_ratings"]
        sparse_trustnetwork = community_dictionaries[most_appropriate_community_id]["sparse_trustnetwork"]

        for community_id in community_dictionaries:
            print(
                f"community_id: {community_id}, social_density_comm: {community_dictionaries[community_id]['social_density_comm']}, rating_density_comm: {community_dictionaries[community_id]['rating_density_mae']}, social_density_mae: {community_dictionaries[community_id]['social_density_mae']}, rating_density_mae: {community_dictionaries[community_id]['rating_density_mae']}, social_density_mae_rank: {community_dictionaries[community_id]['social_density_mae_rank']}, rating_density_mae_rank: {community_dictionaries[community_id]['rating_density_mae_rank']}, average_rank: {community_dictionaries[community_id]['average_rank']}")

    social_density, rating_density = get_densities(sparse_trustnetwork, sparse_ratings)

    print(f"{sparse_ratings.nnz}, {sparse_trustnetwork.nnz}, {sparse_ratings.shape}, {sparse_trustnetwork.shape}, {social_density}, {rating_density}")

    return sparse_ratings, sparse_trustnetwork, item_categories

def train_test_validation_split_ratings(sparse_ratings, test_ratio=0.2, validation_ratio=0.5, users_from_test_must_be_in_train=True):
    # Find the nonzero elements
    nonzero_rows, nonzero_cols = sparse_ratings.nonzero()
    nonzero_elements = list(zip(nonzero_rows, nonzero_cols))

    # Shuffle the nonzero elements
    np.random.shuffle(nonzero_elements)

    # Calculate the number of test and validation elements
    num_test_and_validation_elements = int(len(nonzero_elements) * test_ratio)
    num_test_elements = int(num_test_and_validation_elements * validation_ratio)
    # Initialize the train, test, and validation matrices
    train_matrix = sparse_ratings.copy()
    test_matrix = sp.lil_matrix(sparse_ratings.shape, dtype=sparse_ratings.dtype)
    validation_matrix = sp.lil_matrix(sparse_ratings.shape, dtype=sparse_ratings.dtype)

    test_and_validation_count = 0
    test_count = 0
    for row, col in tqdm(nonzero_elements):
        if test_and_validation_count >= num_test_and_validation_elements:
            break

        if users_from_test_must_be_in_train == True:
            # Check if the element is not the last nonzero in its row and column
            row_nonzero = np.count_nonzero(train_matrix.getrow(row).toarray())
            col_nonzero = np.count_nonzero(train_matrix.getcol(col).toarray())
            if row_nonzero > 1 and col_nonzero > 1:
                if test_count < num_test_elements:
                    # Move the element to the test set
                    test_matrix[row, col] = train_matrix[row, col]
                    test_count += 1
                else:
                    # Move the element to the validation set
                    validation_matrix[row, col] = train_matrix[row, col]

                train_matrix[row, col] = 0
                test_and_validation_count += 1
        else:
            if test_count < num_test_elements:
                # Move the element to the test set
                test_matrix[row, col] = train_matrix[row, col]
                test_count += 1
            else:
                # Move the element to the validation set
                validation_matrix[row, col] = train_matrix[row, col]

            train_matrix[row, col] = 0
            test_and_validation_count += 1

    # Eliminate zero entries in train_matrix
    train_matrix.eliminate_zeros()

    # Get the last nonzero column ID for each matrix
    train_last_col_id = last_nonzero_col_id(train_matrix)
    test_last_col_id = last_nonzero_col_id(test_matrix)
    validation_last_col_id = last_nonzero_col_id(validation_matrix)
    largest_col_id = max(train_last_col_id, test_last_col_id, validation_last_col_id)

    if largest_col_id != train_last_col_id and (largest_col_id == test_last_col_id or largest_col_id == validation_last_col_id):
        largest_col_id = max(test_last_col_id, validation_last_col_id)
        if largest_col_id == test_last_col_id:
            row_id = test_matrix[:, test_last_col_id].nonzero()[0][0]
            train_matrix[row_id, test_last_col_id] = test_matrix[row_id, test_last_col_id]
            test_matrix[row_id, test_last_col_id] = 0
        else:
            row_id = validation_matrix[:, validation_last_col_id].nonzero()[0][0]
            train_matrix[row_id, validation_last_col_id] = validation_matrix[row_id, validation_last_col_id]
            validation_matrix[row_id, validation_last_col_id] = 0

    # Convert the test_matrix and validation_matrix to csr_matrix format
    test_matrix = test_matrix.tocsr()
    validation_matrix = validation_matrix.tocsr()
    test_matrix.eliminate_zeros()
    validation_matrix.eliminate_zeros()

    return train_matrix, test_matrix, validation_matrix


def last_nonzero_col_id(matrix):
    # Sum each column to get a count of nonzeros in each column
    col_sum = np.squeeze(np.asarray(matrix.sum(axis=0)))

    # Find the index of the last nonzero column
    last_nonzero_col = np.where(col_sum != 0)[0][-1]

    return last_nonzero_col

def sparse_matrix_to_list_of_lists(sparse_matrix):
    result = []
    for i in range(sparse_matrix.shape[0]):
        # Get the start and end indices of the rows in the data array
        start, end = sparse_matrix.indptr[i], sparse_matrix.indptr[i + 1]
        # Extract the column indices and values for the row
        cols = sparse_matrix.indices[start:end]
        vals = sparse_matrix.data[start:end]
        for col, val in zip(cols, vals):
            result.append([i, col, val])
    return result

def create_final_lists(train_matrix, test_matrix, validation_matrix, sparse_trustnetwork):
    # Initialize dictionaries and lists
    history_u_lists = {}
    history_ur_lists = {}
    history_v_lists = {}
    history_vr_lists = {}
    social_adj_lists = {}
    ratings_list = {}
    unique_ratings = set()

    # Loop through training data and fill dictionaries
    for user_id, item_id in zip(*train_matrix.nonzero()):
        rating = train_matrix[user_id, item_id]
        if user_id not in history_u_lists:
            history_u_lists[user_id] = []
        history_u_lists[user_id].append(item_id)

        if user_id not in history_ur_lists:
            history_ur_lists[user_id] = []
        history_ur_lists[user_id].append(rating)

        if item_id not in history_v_lists:
            history_v_lists[item_id] = []
        history_v_lists[item_id].append(user_id)

        if item_id not in history_vr_lists:
            history_vr_lists[item_id] = []
        history_vr_lists[item_id].append(rating)

        unique_ratings.add(rating)

    # Loop through social network and fill dictionary
    for source in range(sparse_trustnetwork.shape[0]):
        social_adj_lists[source] = list(sparse_trustnetwork.getrow(source).nonzero()[1])

    # Assign an index to each rating
    i = 0
    for rating in unique_ratings:
        ratings_list[rating] = i
        i += 1

    # Convert data to list of lists
    traindata = [list(t) for t in sparse_matrix_to_list_of_lists(train_matrix)]
    validdata = [list(t) for t in sparse_matrix_to_list_of_lists(validation_matrix)]
    testdata = [list(t) for t in sparse_matrix_to_list_of_lists(test_matrix)]

    # item_adj_lists = build_item_adj_lists(history_v_lists)
    # item_adj_lists = {}

    history_ur_lists_remapped = {}
    history_vr_lists_remapped = {}

    for user_id in history_ur_lists:
        history_ur_lists_remapped[user_id] = [ratings_list[rating] for rating in history_ur_lists[user_id]]
    for item_id in history_vr_lists:
        history_vr_lists_remapped[item_id] = [ratings_list[rating] for rating in history_vr_lists[item_id]]

    traindata = np.array(traindata)
    validdata = np.array(validdata)
    testdata = np.array(testdata)
    train_u = traindata[:, 0]
    train_v = traindata[:, 1]
    train_r = traindata[:, 2]
    valid_u = validdata[:, 0]
    valid_v = validdata[:, 1]
    valid_r = validdata[:, 2]
    test_u = testdata[:, 0]
    test_v = testdata[:, 1]
    test_r = testdata[:, 2]

    return history_u_lists, history_ur_lists_remapped, history_v_lists, history_vr_lists_remapped, train_u, train_v, train_r, test_u, test_v, test_r, valid_u, valid_v, valid_r, social_adj_lists, ratings_list

def create_pickle_file(filepath, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, valid_u, valid_v, valid_r, social_adj_lists, ratings_list, item_categories_dict):
    pickle_data = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, valid_u, valid_v, valid_r, social_adj_lists, ratings_list, item_categories_dict]
    # Create a Path object
    path_obj = Path(filepath)
    # Create parent directories if they don't exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open('wb') as f:
        pickle.dump(pickle_data, f)

def store_data_for_elliot(social_adj_lists, traindata, validdata, testdata, parent_folder, social_filename,
                          train_filename, validation_filename, test_filename, store_validation=True):
    # Check if the folder already exists
    if not os.path.exists(parent_folder):
        # Create the new folder
        os.makedirs(parent_folder)

    with open(parent_folder + '/' + train_filename, 'w+') as f:
        # Write the training data to a TSV file
        for (user_id, item_id, rating) in traindata:
            f.write(str(user_id) + "\t" + str(item_id) + "\t" + str(rating) + "\n")

    if store_validation == True:
        with open(parent_folder + '/' + validation_filename, 'w+') as f:
            # Write the validation data to a TSV file
            for (user_id, item_id, rating) in validdata:
                f.write(str(user_id) + "\t" + str(item_id) + "\t" + str(rating) + "\n")

    with open(parent_folder + '/' + test_filename, 'w+') as f:
        # Write the test data to a TSV file
        for (user_id, item_id, rating) in testdata:
            f.write(str(user_id) + "\t" + str(item_id) + "\t" + str(rating) + "\n")
        if store_validation == False:
            # If store_validation is False, write the validation data to the same TSV file as the test data
            for (user_id, item_id, rating) in validdata:
                f.write(str(user_id) + "\t" + str(item_id) + "\t" + str(rating) + "\n")

    with open(parent_folder + '/' + social_filename, 'w+') as f:
        # Write the social data to a TSV file
        for source in social_adj_lists:
            for target in social_adj_lists[source]:
                f.write(str(source) + "\t" + str(target) + "\n")

def process_dataset(dataset_name, args):
    ratings_mat = scio.loadmat(f"./data/raw/{dataset_name}/rating.mat")
    trustnetwork_mat = scio.loadmat(f"./data/raw/{dataset_name}/trustnetwork.mat")

    output_name = (f"{dataset_name}_"
                   f"r0-{args.is_remove_zero_ratings}_"
                   f"rd-{args.is_remove_duplicates}_"
                   f"runr-{args.is_remove_users_with_no_ratings}_"
                   f"runconn-{args.is_remove_users_with_no_connections}_"
                   f"undir-{args.convert_to_undirected}_"
                   f"testtrain-{args.users_from_test_must_be_in_train}_"
                   f"minur-{args.min_user_num_ratings}_"
                   f"minir-{args.min_item_num_ratings}"
                   f"_lcc_{args.keep_only_lcc}"
                   f"_bcomm_{args.keep_only_best_community}")

    sparse_ratings, sparse_trustnetwork, item_categories = preprocess_datasets(
        ratings_mat,
        trustnetwork_mat,
        dataset_name,

        args.is_remove_zero_ratings,
        args.is_remove_duplicates,
        args.is_remove_users_with_no_ratings,
        args.is_remove_users_with_no_connections,
        args.convert_to_undirected,

        # args.users_from_ratings_must_be_in_trustnetwork,

        args.min_user_num_ratings,
        args.min_item_num_ratings,

        args.keep_only_lcc,
        args.keep_only_best_community
    )

    train_matrix, test_matrix, validation_matrix = train_test_validation_split_ratings(sparse_ratings, args.test_ratio,
                                                                                       args.validation_ratio,
                                                                                       args.users_from_test_must_be_in_train)

    item_categories_dict = {i: val for i, val in enumerate(item_categories[0, :].tolist())}
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, valid_u, valid_v, valid_r, social_adj_lists, ratings_list = create_final_lists(train_matrix, test_matrix, validation_matrix, sparse_trustnetwork)
    create_pickle_file(f"./data/processed/graphrec/{dataset_name}/{output_name}.pickle", history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, valid_u, valid_v, valid_r, social_adj_lists, ratings_list, item_categories_dict)

    parent_folder = './data/processed/elliot/' + dataset_name + '/filtered_full_train_test'
    social_filename = f"{output_name}_social_connections.tsv"
    train_filename = f"{output_name}_ratings_train.tsv"
    test_filename = f"{output_name}_ratings_test.tsv"
    validation_filename = f"{output_name}_ratings_validation.tsv"

    # Combine the individual arrays back into traindata, validdata, and testdata
    traindata = np.column_stack((train_u, train_v, train_r))
    validdata = np.column_stack((valid_u, valid_v, valid_r))
    testdata = np.column_stack((test_u, test_v, test_r))

    # Convert the individual NumPy arrays to lists of lists
    traindata = traindata.tolist()
    validdata = validdata.tolist()
    testdata = testdata.tolist()

    store_data_for_elliot(social_adj_lists, traindata, validdata, testdata, parent_folder, social_filename, train_filename, validation_filename, test_filename, args.store_validation)

def main(args):
    for dataset_name in args.dataset_names.split():
        print(dataset_name)
        process_dataset(dataset_name, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_names', type=str, nargs='+', default='ciao epinions', help='Dataset name.')

    parser.add_argument('--is_remove_zero_ratings', type=bool, default=True, help='Remove zero ratings from the dataset.')
    parser.add_argument('--is_remove_duplicates', type=bool, default=True, help='Remove duplicate ratings from the dataset.')
    # parsed.add_argument('--users_from_ratings_must_be_in_trustnetwork', type=bool, default=True, help='Users from ratings must be in trust network.')
    parser.add_argument('--is_remove_users_with_no_ratings', type=bool, default=True, help='Remove users with no ratings.')
    parser.add_argument('--is_remove_users_with_no_connections', type=bool, default=True, help='Remove users with no connections.')
    parser.add_argument('--convert_to_undirected', type=bool, default=True, help='Convert trust network to undirected.')
    parser.add_argument('--users_from_test_must_be_in_train', type=bool, default=True, help='Users from test or validation set must be in train set.')

    parser.add_argument('--min_user_num_ratings', type=int, default=1, help='Minimum number of ratings per user.')
    parser.add_argument('--min_item_num_ratings', type=int, default=1, help='Minimum number of ratings per item.')
    parser.add_argument('--keep_only_lcc', type=bool, default=False, help='Keep only the largest connected component.')

    parser.add_argument('--keep_only_best_community', type=bool, default=False, help='Keep only the best community.')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test ratio.')
    parser.add_argument('--validation_ratio', type=float, default=0.5, help='Validation ratio.')
    parser.add_argument('--store_validation', type=bool, default=False, help='Store the validation set.')

    args = parser.parse_args()
    main(args)

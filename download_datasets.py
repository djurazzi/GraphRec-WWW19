import os
import io
import zipfile
import requests
import numpy as np
from tqdm import tqdm
import scipy.io as scio
import pickle
import rarfile

def download_and_unzip_dataset(url):
    # create the necessary directories if they don't exist
    if not os.path.exists(f"./data/raw"):
        os.makedirs(f"./data/raw")
        print(f"Created directory: ./data/raw")

    # download the zip file
    response = requests.get(url)
    print(f"Downloading from {url}...")

    # extract the zip file contents and store in the specified folder
    # extract the contents of the epinions folder only and store in the specified folder
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # extract the contents to the specified folder
        z.extractall(f"./data/raw")
    print(f"Extracted contents to ./data/raw")

def download_unrar_and_convert_to_mat_lastfm():
    if not os.path.exists(f"./data/raw/lastfm"):
        os.makedirs(f"./data/raw/lastfm")
        print(f"Created directory: ./data/raw/lastfm")

    playcounts_url = "https://github.com/tommantonela/umap2022-mrecuri/raw/main/data/dataset.rar"
    edges_url = "https://github.com/tommantonela/umap2022-mrecuri/raw/main/lastfm_sn/lastfm.edges"
    nodes_url = "https://github.com/tommantonela/umap2022-mrecuri/raw/main/lastfm_sn/lastfm.nodes"

    response = requests.get(playcounts_url)
    print(f"Downloading from {playcounts_url}...")

    # Save the RAR file to disk
    with open('./data/raw/lastfm/dataset.rar', 'wb') as f:
        f.write(response.content)

    response = requests.get(edges_url)
    print(f"Downloading from {edges_url}...")

    # Save the RAR file to disk
    with open('./data/raw/lastfm/lastfm.edges', 'wb') as f:
        f.write(response.content)

    response = requests.get(nodes_url)
    print(f"Downloading from {nodes_url}...")

    # Save the RAR file to disk
    with open('./data/raw/lastfm/lastfm.nodes', 'wb') as f:
        f.write(response.content)

    # Extract the contents of the RAR file
    with rarfile.RarFile('./data/raw/lastfm/dataset.rar') as rar:
        rar.extractall('./data/raw/lastfm')

    print(f"Extracted contents to ./data/raw/lastfm")


    # Delete the RAR file
    os.remove('./data/raw/lastfm/dataset.rar')

    lastfm_dataset = pickle.load(open("./data/raw/lastfm/dataset.pickle",'rb'))

    users = invert_users_dict(lastfm_dataset)
    tracks = build_tracks_dict(lastfm_dataset)
    interaction_vectors = build_interaction_vectors(lastfm_dataset, users, tracks)
    processed_interaction_vectors = process_interaction_vectors(interaction_vectors)

    user_mappings, item_mappings = create_user_item_mappings(processed_interaction_vectors)
    user_item_ratings = create_user_item_ratings(interaction_vectors, user_mappings, item_mappings)
    original_graph_nodes = create_original_graph_nodes('./data/raw/lastfm/lastfm.nodes')
    final_social_edges = create_final_social_edges('./data/raw/lastfm/lastfm.edges', original_graph_nodes, lastfm_dataset, user_mappings)
    trustnetwork = np.array(final_social_edges, dtype=int)
    rating = create_rating_array(user_item_ratings)
    save_to_mat_files(trustnetwork, rating)

def get_discrete_cdf(values):
    # values = (values - np.min(values)) / (np.max(values) - np.min(values))
    values_sort = np.sort(values)
    values_sum = np.sum(values)

    values_sums = []
    cur_sum = 0
    for it in values_sort:
        cur_sum += it
        values_sums.append(cur_sum)

    cdf = [values_sums[np.searchsorted(values_sort, it)]/values_sum for it in values]
    return values_sort, np.sort(cdf)

def get_playcount_cdfs(values, cdfs):
    playcount_cdfs = {}

    for i in range(len(values)):
        playcount_value = values[i]
        playcount_cdf = cdfs[i]
        if playcount_value not in playcount_cdfs:
            playcount_cdfs[playcount_value] = []
        playcount_cdfs[playcount_value].append(playcount_cdf)

    return playcount_cdfs

def convert_playcount_cdfs_to_ratings(playcount_cdfs):
    playcount_ratings = {}
    for playcount_value in playcount_cdfs:
        playcount_ratings[playcount_value] = round(4 * (np.mean(playcount_cdfs[playcount_value]))) + 1
    return playcount_ratings

def invert_users_dict(dataset):
    return {v: k for k, v in dataset['users'].items()}

def build_tracks_dict(dataset):
    tracks = {}
    for artist in dataset['artist-tracks']:
        for track in dataset['artist-tracks'][artist]:
            track_id = dataset['artist-tracks'][artist][track]
            if track_id not in tracks:
                tracks[track_id] = artist + ' - ' + track
    return tracks

def build_interaction_vectors(dataset, users, tracks):
    interaction_vectors = {}
    for edge in tqdm(dataset['full'].edges):
        source = edge[0]
        target = edge[1]
        if source in users:
            user_id = source
        if source in tracks:
            track_id = source
        if target in users:
            user_id = target
        if target in tracks:
            track_id = target
        playcount = dataset['full'].edges[edge]['scrobbles']

        if user_id not in interaction_vectors:
            interaction_vectors[user_id] = {'track_ids': [], 'playcounts': []}
        interaction_vectors[user_id]['track_ids'].append(track_id)
        interaction_vectors[user_id]['playcounts'].append(playcount)

    return interaction_vectors

def process_interaction_vectors(interaction_vectors):
    for user_id in tqdm(interaction_vectors):
        user_playcounts = np.array(interaction_vectors[user_id]['playcounts'])
        sorted_playcounts, sorted_cdf_values = get_discrete_cdf(user_playcounts)
        playcount_cdfs = get_playcount_cdfs(sorted_playcounts, sorted_cdf_values)
        playcount_ratings = convert_playcount_cdfs_to_ratings(playcount_cdfs)
        interaction_vectors[user_id]['playcount_ratings'] = playcount_ratings
        interaction_vectors[user_id]['ratings'] = []
        for playcount_value in user_playcounts:
            interaction_vectors[user_id]['ratings'].append(playcount_ratings[playcount_value])
    return interaction_vectors

def create_user_item_mappings(interaction_vectors):
    user_mappings = {}
    item_mappings = {}

    user_id_counter = 0
    item_id_counter = 0

    for user_id in tqdm(interaction_vectors):
        if user_id not in user_mappings:
            user_mappings[user_id] = user_id_counter
            user_id_counter += 1

        for i in range(len(interaction_vectors[user_id]['track_ids'])):
            track_id = interaction_vectors[user_id]['track_ids'][i]
            if track_id not in item_mappings:
                item_mappings[track_id] = item_id_counter
                item_id_counter += 1

    return user_mappings, item_mappings

def create_user_item_ratings(interaction_vectors, user_mappings, item_mappings):
    user_item_ratings = []

    for user_id in tqdm(interaction_vectors):
        for i in range(len(interaction_vectors[user_id]['track_ids'])):
            track_id = interaction_vectors[user_id]['track_ids'][i]

            user_id_mapping = user_mappings[user_id]
            item_id_mapping = item_mappings[track_id]
            rating = interaction_vectors[user_id]['ratings'][i]

            user_item_ratings.append((user_id_mapping, item_id_mapping, rating))

    return user_item_ratings

def create_original_graph_nodes(filepath):
    original_graph_nodes = {}
    for line in open(filepath, 'r').readlines():
        tokens = line.split()
        original_user_mapping = int(tokens[0])
        username = tokens[1]
        original_graph_nodes[original_user_mapping] = username

    return original_graph_nodes

def create_final_social_edges(filepath, original_graph_nodes, dataset, user_mappings):
    final_social_edges = []
    for line in open(filepath, 'r').readlines():
        tokens = line.split()
        original_user_mapping_source = int(tokens[0])
        original_user_mapping_target = int(tokens[1])

        username_source = original_graph_nodes[original_user_mapping_source]
        username_target = original_graph_nodes[original_user_mapping_target]

        if username_source in dataset['users'] and username_target in dataset['users']:
            source_user_id = dataset['users'][username_source]
            target_user_id = dataset['users'][username_target]

            source_user_mapping = user_mappings[source_user_id]
            target_user_mapping = user_mappings[target_user_id]

            final_social_edges.append((source_user_mapping, target_user_mapping))
            final_social_edges.append((target_user_mapping, source_user_mapping))

    return final_social_edges

def create_rating_array(user_item_ratings):
    rating = []
    category_id = -1
    helpfulness = -1

    for user_item_rating in user_item_ratings:
        user_id = user_item_rating[0]
        item_id = user_item_rating[1]
        r = user_item_rating[2]
        rating.append([user_id, item_id, category_id, r, helpfulness])

    return np.array(rating, dtype=int)

def save_to_mat_files(trustnetwork, rating):
    scio.savemat('./data/raw/lastfm/trustnetwork.mat', {'trustnetwork': trustnetwork})
    scio.savemat('./data/raw/lastfm/rating.mat', {'rating': rating})

def main():
    epinions_url = "http://www.cse.msu.edu/~tangjili/datasetcode/epinions.zip"
    ciao_url = "http://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip"
    download_and_unzip_dataset(epinions_url)
    download_and_unzip_dataset(ciao_url)
    # download_unrar_and_convert_to_mat_lastfm()

if __name__ == '__main__':
    main()
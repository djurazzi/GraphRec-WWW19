import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import math
from collections import Counter
import json
from pathlib import Path

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_ndcg):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best ndcg: %.6f.' % (
                epoch, i, running_loss / 100, best_ndcg))
            running_loss = 0.0
    return 0

def select_and_pad(lst, k):
    # If the list is shorter than k, pad with -1
    if len(lst) < k:
        return lst + [-1] * (k - len(lst))
    # If the list is longer than k, truncate to first k items
    else:
        return lst[:k]


# def dcg_at_k(r, k):
#     """Compute DCG@k for a list of rankings."""
#     r = r[:k]
#     return sum([(rel / math.log(i + 2, 2)) for i, rel in enumerate(r)])
#
#
# def compute_ndcg_at_k(predictions, ground_truth, k):
#     ndcg_scores = {}
#     for user_id, predicted_items in predictions.items():
#         # Convert list of predicted items to binary relevance scores
#         # 1 if the item is in the ground truth, 0 otherwise
#         rel_list = [1 if item in ground_truth[user_id] and item != -1 else 0 for item in predicted_items]
#
#         # Get ideal DCG (where all relevant items are placed at the top)
#         idcg = dcg_at_k([1] * len(ground_truth[user_id]), k)
#
#         # If IDCG is 0, then NDCG is undefined; we'll set it to 0 here.
#         if idcg == 0:
#             ndcg_scores[user_id] = 0.0
#         else:
#             actual_dcg = dcg_at_k(rel_list, k)
#             ndcg_scores[user_id] = actual_dcg / idcg
#
#     # Return the average NDCG@k for all users
#     return ndcg_scores

def compute_ndcg_at_k(predictions, ground_truth, k):
    ndcg_scores = {}

    for user, predicted_items in predictions.items():
        dcg = 0.0
        idcg = 0.0
        relevant_items = set(ground_truth[user])
        num_relevant = sum([1 for item in relevant_items if item != -1])

        for i in range(k):
            if i < len(predicted_items) and predicted_items[i] in relevant_items:
                dcg += 1 / (math.log2(i + 2))  # i+2 because log2(1) is 0 and index starts from 0

        # For IDCG, assume that all the relevant items are ranked at the top
        for i in range(num_relevant):
            idcg += 1 / (math.log2(i + 2))

        if idcg == 0:
            ndcg_scores[user] = 0.0
        else:
            ndcg_scores[user] = dcg / idcg

    return ndcg_scores


def process_batch(model, device, batch_user_list, user_ids_tensor_list, items_tensor_list, test_u_lists, test_u_lists_padded, num_items_to_sample, top_k):
    # Stack the tensors for the batch
    user_ids_final = torch.cat(user_ids_tensor_list, dim=0)
    items_final = torch.cat(items_tensor_list, dim=0)

    # Forward pass to get scores
    # Assuming the model returns scores with shape [b * n]
    scores = torch.sigmoid(model.forward(user_ids_final, items_final))

    # Reshape scores to [b, n]
    b = len(user_ids_tensor_list) # number of users in the batch
    n = num_items_to_sample # number of items you're scoring for each user
    scores_reshaped = scores.view(b, n)

    # Get top_k scores and their indices for each user
    top_scores, top_indices = scores_reshaped.topk(top_k, dim=1)
    items_final = items_final.to(device)
    top_item_ids = torch.gather(items_final.repeat(b, 1), 1, top_indices)

    # Convert to dictionary
    batch_user_recommendations = {user_id: top_item_ids[i].tolist() for i, user_id in enumerate(batch_user_list)}

    # Batch ndcg scores
    batch_ndcg_scores = compute_ndcg_at_k(batch_user_recommendations, test_u_lists_padded, top_k)

    # Clear the lists for the next batch
    user_ids_tensor_list.clear()
    items_tensor_list.clear()
    batch_user_list.clear()

    return batch_user_recommendations, batch_ndcg_scores


def compute_gini_coefficient(recommendations, num_items):
    item_count = {}
    free_norm = 0

    def user_gini(user_recommendations):
        nonlocal free_norm
        user_norm = len(user_recommendations)
        free_norm += user_norm
        for i in user_recommendations:
            item_count[i] = item_count.get(i, 0) + 1

    # Computing user-wise item count
    for _, u_r in recommendations.items():
        user_gini(u_r)

    n_recommended_items = len(item_count)

    gini = sum([(2 * (j + (num_items - n_recommended_items) + 1) - num_items - 1) * (cs / free_norm) for j, cs in enumerate(sorted(item_count.values()))])
    gini /= (num_items - 1)
    gini = 1 - gini

    return gini

def compute_item_coverage(user_recommendations, num_items):
    """
    Compute the item coverage based on user recommendations.

    :param user_recommendations: A dictionary where key is user id and value is a list of recommended item ids.
    :param num_items: (Optional) Total number of distinct items that could be recommended.
                        If not provided, it's assumed to be the set of all items recommended to any user.
    :return: Item coverage.
    """
    # Flatten the list of all recommended items
    all_recommendations = [item for sublist in user_recommendations.values() for item in sublist]

    # Get the unique set of recommended items
    unique_recommended_items = set(all_recommendations)

    coverage = len(unique_recommended_items) / num_items
    return coverage

def category_entropy_per_user(user_recommendations, item_to_category):
    """
    Compute the category entropy for each user based on the items recommended to them.

    :param user_recommendations: A dictionary where key is user id and value is a list of recommended item ids.
    :param item_to_category: A dictionary mapping each item id to its category.
    :return: A dictionary where key is user id and value is the category entropy for that user.
    """

    user_entropy = {}

    for user, items in user_recommendations.items():
        # Count the number of recommendations per category for this user
        category_counts = {}
        for item in items:
            category = item_to_category[item]
            category_counts[category] = category_counts.get(category, 0) + 1

        # Convert counts to probabilities
        total_items = len(items)
        category_probs = {cat: count / total_items for cat, count in category_counts.items()}

        # Compute entropy for this user
        entropy = -sum(p * math.log(p, 2) for p in category_probs.values())
        user_entropy[user] = entropy

    return user_entropy

def test(model, device, test_u_lists, test_u_lists_padded, all_items, test_batch_size, top_k, num_items, num_items_to_sample=100, item_categories=None):
    model.eval()
    batch_size = test_batch_size // num_items_to_sample
    user_ids_tensor_list = []
    items_tensor_list = []
    batch_user_list = []
    ndcg_scores = {}
    user_recommendations = {}

    with torch.no_grad():
        for idx, u in enumerate(test_u_lists):
            batch_user_list.append(u)
            target_items_u = set(test_u_lists[u])
            available_items = list(all_items - target_items_u)
            available_items = random.sample(available_items, num_items_to_sample)

            user_ids_tensor = torch.full((num_items_to_sample,), u, dtype=torch.long)
            items_tensor = torch.Tensor(available_items).to(torch.int64)

            user_ids_tensor_list.append(user_ids_tensor)
            items_tensor_list.append(items_tensor)

            # Check if we've accumulated enough for a batch
            if (idx + 1) % batch_size == 0:
                batch_user_recommendations, batch_ndcg_scores = process_batch(model, device, batch_user_list, user_ids_tensor_list, items_tensor_list, test_u_lists, test_u_lists_padded, num_items_to_sample, top_k)
                user_recommendations.update(batch_user_recommendations)
                ndcg_scores.update(batch_ndcg_scores)
        # Handle any remaining items after the loop
        if user_ids_tensor_list and items_tensor_list:
            batch_user_recommendations, batch_ndcg_scores = process_batch(model, device, batch_user_list, user_ids_tensor_list, items_tensor_list, test_u_lists, test_u_lists_padded, num_items_to_sample, top_k)
            user_recommendations.update(batch_user_recommendations)
            ndcg_scores.update(batch_ndcg_scores)

    mean_ndcg = np.mean(list(ndcg_scores.values()))
    gini_coefficient = compute_gini_coefficient(user_recommendations, num_items)
    coverage = compute_item_coverage(user_recommendations, num_items)
    if item_categories is not None:
        category_entropy = category_entropy_per_user(user_recommendations, item_categories)
        mean_category_entropy = np.mean(list(category_entropy.values()))
    else:
        category_entropy = 9999.0
    filtered_users = [user for user, items in test_u_lists_padded.items() if -1 not in items]
    filtered_users_ndcg = {user: ndcg for user, ndcg in ndcg_scores.items() if user in filtered_users}
    mean_filtered_ndcg = np.mean(list(filtered_users_ndcg.values()))
    return user_recommendations, mean_filtered_ndcg, mean_ndcg, gini_coefficient, coverage, mean_category_entropy

def add_negative_samples(train_u, train_v, train_r, history_u_lists, history_v_lists, num_negative):
    train_u_extended = []
    train_v_extended = []
    train_r_extended = []
    all_item_ids = list(history_v_lists.keys())
    for i in range(len(train_u)):
        train_u_extended.append(train_u[i])
        train_v_extended.append(train_v[i])
        train_r_extended.append(train_r[i])
        for t in range(num_negative):
            j = random.choice(all_item_ids)
            while j in history_u_lists[train_u[i]]:
                j = random.choice(all_item_ids)
            train_u_extended.append(train_u[i])
            train_v_extended.append(j)
            train_r_extended.append(0)
    return train_u_extended, train_v_extended, train_r_extended

def create_u_lists(test_u, test_v, test_r):
    test_u_lists = {}
    test_u_ratings = {}
    for i in range(len(test_u)):
        u = test_u[i]
        if u not in test_u_lists:
            test_u_lists[u] = []
            test_u_ratings[u] = []
        test_u_lists[u].append(test_v[i])
        test_u_ratings[u].append(test_r[i])
    # Sort descending each test_u_lists[u] by rating
    for u in test_u_lists:
        test_u_lists[u] = [x for _, x in sorted(zip(test_u_ratings[u], test_u_lists[u]), reverse=True)]
    return test_u_lists

def create_interaction_histories(train_u, train_v, train_r):
    # Initialize the dictionaries
    history_u_lists = {}
    history_ur_lists = {}
    history_v_lists = {}
    history_vr_lists = {}

    # Populate the dictionaries
    for u, v, r in zip(train_u, train_v, train_r):
        # For history_u_lists
        if u not in history_u_lists:
            history_u_lists[u] = []
        history_u_lists[u].append(v)

        # For history_ur_lists
        if u not in history_ur_lists:
            history_ur_lists[u] = []
        history_ur_lists[u].append(r)

        # For history_v_lists
        if v not in history_v_lists:
            history_v_lists[v] = []
        history_v_lists[v].append(u)

        # For history_vr_lists
        if v not in history_vr_lists:
            history_vr_lists[v] = []
        history_vr_lists[v].append(r)

    return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--num_negative', type=int, default=1, metavar='N', help='number of negative samples')
    parser.add_argument('--dataset', type=str, default='epinions', metavar='N', help='dataset')
    parser.add_argument('--top_k', type=int, default=10, metavar='N', help='top k for evaluation')
    parser.add_argument('--num_items_to_sample', type=int, default=100, metavar='N', help='number of items to sample for evaluation')
    parser.add_argument('--convert_histories_to_implicit', type=bool, default=False, metavar='N', help='convert histories to implicit feedback')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id.')
    parser.add_argument('--hyperparam_search', type=bool, default=False, help='Hyperparameter search.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    embed_dim = args.embed_dim

    if args.dataset == 'toy_dataset':
        dir_data = './data/processed/graphrec/toy_dataset/toy_dataset'

        path_data = dir_data + ".pickle"
        data_file = open(path_data, 'rb')
        history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(data_file)
    else:
        dir_data = './data/processed/graphrec/' + args.dataset + '/' + args.dataset + '_r0-True_rd-True_runr-True_runconn-True_undir-False_testtrain-True_minur-1_minir-1_lcc_False_bcomm_False'
        path_data = dir_data + ".pickle"
        data_file = open(path_data, 'rb')
        history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, valid_u, valid_v, valid_r, social_adj_lists, ratings_list, item_categories = pickle.load(data_file)
        if args.hyperparam_search == False:
            test_u = np.concatenate((test_u, valid_u))
            test_v = np.concatenate((test_v, valid_v))
            test_r = np.concatenate((test_r, valid_r))

    # Convert train/test data to implicit feedback
    train_r = [1 for i in range(len(train_r))]

    # Add negative samples
    train_u, train_v, train_r = add_negative_samples(train_u, train_v, train_r, history_u_lists, history_v_lists, args.num_negative)
    if args.convert_histories_to_implicit:
        history_u_lists, history_ur_lists, history_v_lists, history_vr_lists = create_interaction_histories(train_u, train_v, train_r)
        ratings_list = {0:0, 1:1}

    # Create test set for top-k evaluation
    test_u_lists = create_u_lists(test_u, test_v, test_r)
    test_u_lists_padded = {u: select_and_pad(lst, args.top_k) for u, lst in test_u_lists.items()}
    all_items = set(test_v)

    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)

    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)

    # please add the validation set

    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v), torch.FloatTensor(train_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    num_users = history_u_lists.__len__()
    # num_items = history_v_lists.__len__()
    num_items = max(history_v_lists.keys()) + 1
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # model
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_ndcg = -1
    endure_count = 0
    if args.dataset == 'toy_dataset':
        results_dir = f'./data/processed/graphrec/{args.dataset}/results'
        results_filename = f'top_k_{args.dataset}_results.json'
    else:
        results_dir = f'./data/processed/graphrec/{args.dataset}/results'
        results_filename = f'top_k_{args.dataset}_r0-True_rd-True_runr-True_runconn-True_undir-False_testtrain-True_minur-1_minir-1_lcc_False_bcomm_False_results_implicit_{args.convert_histories_to_implicit}.json'
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "best_ndcg": best_ndcg,
        "epoch": 0
    }

    with open(f'{results_dir}/{results_filename}', 'w') as json_file:
        json.dump(results, json_file)

    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_ndcg)
        user_recommendations, filtered_ndcg, ndcg, gini_coefficient, coverage, category_entropy = test(graphrec, device, test_u_lists, test_u_lists_padded, all_items, args.test_batch_size, args.top_k, len(all_items), args.num_items_to_sample, item_categories)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            results = {
                "best_ndcg": str(round(best_ndcg, 6)),
                "gini": str(round(gini_coefficient, 6)),
                "coverage": str(round(coverage, 6)),
                "category_entropy": str(round(category_entropy, 6)),
                "epoch": epoch
            }
            with open(f'{results_dir}/{results_filename}', 'w') as json_file:
                json.dump(results, json_file)
            path_obj = Path(f'{results_dir}/recommendations_{args.dataset}_r0-True_rd-True_runr-True_runconn-True_undir-False_testtrain-True_minur-1_minir-1_lcc_False_bcomm_False_results_implicit_{args.convert_histories_to_implicit}.pickle')
            with path_obj.open('wb') as f:
                pickle.dump(user_recommendations, f)
            endure_count = 0
        else:
            endure_count += 1
        print(f"Epoch {epoch}: Filtered NDCG: {round(filtered_ndcg, 4)} NDCG: {round(ndcg, 4)}, Gini: {round(gini_coefficient, 4)}, Coverage: {round(coverage, 4)}, Category Entropy: {round(category_entropy, 4)}")

        if endure_count > 5:
            break


if __name__ == "__main__":
    main()
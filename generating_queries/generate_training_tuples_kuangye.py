import os
import pickle
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KDTree

##########################################
# spilt the positive and negative samples of train data
# save in the training_queries_baseline.pickle
# for kuangye data
##########################################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "/media/cyw/CYW-ZX2/kuangye_data"

trainsubmaps_file = 'train_submaps_pcd' #train_submaps_bin
filename = "/train_submaps_poses.txt"
save_path = "./data_split/training_queries.pickle"
seqs = ['188loop']

def construct_query_dict(df_centroids, seq_index, queries_len):
    tree = KDTree(df_centroids[['x','y', 'z']])
    ind_nn = tree.query_radius(df_centroids[['x', 'y', 'z']], r=3)
    ind_r = tree.query_radius(df_centroids[['x', 'y', 'z']], r=10)
    queries = {}
    for i in range(len(ind_nn)):
        data_id = int(df_centroids.iloc[i]["data_id"])
        scan_file = df_centroids.iloc[i]["data_file"]
        all_positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        near_ids = []
        other_ids = []

        # for id in all_positives:
        #     if abs(id - data_id) < 10:
        #         near_ids.append(id)
        #     else:
        #         other_ids.append(id)
        # if len(true_ids)>=50:
        #     choose_near_ids = random.choices(near_ids, k=int(1.5 * len(true_ids)))
        # else:
        #     choose_near_ids = near_ids
        # choose_near_ids = near_ids
        # positives = other_ids + choose_near_ids、
        positives = all_positives
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        if seq_index != 0:
            data_id += queries_len
            positives = [p + queries_len for p in positives]
            negatives = [n + queries_len for n in negatives]
        queries[data_id] = {"data_id": data_id, "positives": positives, "negatives": negatives, "scan_file": scan_file}
    return queries

# Initialize pandas DataFrame
train_seqs= {}
# train_seqs = []
queries_len = 0
for seq_i, seq in enumerate(tqdm(seqs, position=0)):
    df_train = pd.DataFrame(columns=['data_id', 'x', 'y', 'z'])
    poses_path = base_path + '/' + seq + filename
    df_locations = pd.read_table(poses_path, sep=' ',
                                 names=['times', 'r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33', 'z'])
    df_locations = df_locations.reset_index()
    df_locations.columns.values[0] = 'data_id'
    df_locations = df_locations.loc[:, ['data_id', 'x', 'y', 'z']]
    df_locations['data_file'] = df_locations['data_id'].apply(lambda x: str(x).zfill(6))
    df_locations['data_file'] = seq + '/' + trainsubmaps_file + '/' + df_locations['data_file'].astype(str) + '.pcd'

    for index, row in df_locations.iterrows():
        df_train = df_train.append(row, ignore_index=True)
    # df_train['seq'] = seq
    print("Number of training submaps: " + str(len(df_train['data_id'])))
    queries = construct_query_dict(df_train, seq_i, queries_len)
    train_seqs.update(queries)
    queries_len += len(queries)

with open(save_path, 'wb') as handle:
    pickle.dump(train_seqs, handle, protocol=pickle.HIGHEST_PROTOCOL)
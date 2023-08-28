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
##########################################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = '/home/cyw/CYW/Datasets/mmWave-Radar-Relocalization-main/dataset/'

# runs_folder = "1-1"
trainsubmaps_file = 'train_submaps_cluster' #train_submaps_bin
filename = "/train_submaps_poses.txt"
save_path = "./1843_tuples/training_queries_clusters.pickle"
seqs = ['seq1/1-1', 'seq2/2-1', 'seq5/5-1', 'seq7/7-1']

def construct_query_dict(df_centroids, seq_index, queries_len):
    tree = KDTree(df_centroids[['x','y']])
    ind_nn = tree.query_radius(df_centroids[['x','y']],r=3)
    ind_r = tree.query_radius(df_centroids[['x','y']], r=10)
    queries = {}
    for i in range(len(ind_nn)):
        data_id = int(df_centroids.iloc[i]["data_id"])
        scan_file = df_centroids.iloc[i]["data_file"]
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
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
    df_train = pd.DataFrame(columns=['data_id', 'x', 'y'])
    poses_path = base_path + seq + filename
    df_locations = pd.read_table(poses_path, sep=' ',
                                 names=['r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33', 'z'])
    df_locations = df_locations.reset_index()
    df_locations.columns.values[0] = 'data_id'
    df_locations = df_locations.loc[:, ['data_id', 'x', 'y']]
    df_locations['data_file'] = df_locations['data_id'].apply(lambda x: str(x).zfill(6))
    df_locations['data_file'] = seq + '/' + trainsubmaps_file + '/' + df_locations['data_file'].astype(str) + '.bin'

    for index, row in df_locations.iterrows():
        df_train = df_train.append(row, ignore_index=True)
    # df_train['seq'] = seq
    print("Number of training submaps: " + str(len(df_train['data_id'])))
    queries = construct_query_dict(df_train, seq_i, queries_len)
    train_seqs.update(queries)
    queries_len += len(queries)

with open(save_path, 'wb') as handle:
    pickle.dump(train_seqs, handle, protocol=pickle.HIGHEST_PROTOCOL)



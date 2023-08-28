import os
import pickle
import numpy as np
import pandas as pd
import tqdm
from sklearn.neighbors import KDTree

##########################################
# split query and database data
# save in evaluation_database.pickle / evaluation_query.pickle
##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, seq_forders, querysubmaps_file, databasesubmaps_file, pos_distence, query_posefile, database_posefile,save_path):
    # global database_tree
    database_trees = []
    test_trees = []
    database_sets = {}
    test_sets = {}
    for seq_id, seq in enumerate(tqdm.tqdm(seq_forders)):
        # print(seq)
        tra_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, seq)))
        tra_sets =[]
        for tra_folder in tra_folders:
            tra_folder = str(tra_folder)
            database = {}
            test = {}
            # print(tra_folder)
            tra_id = tra_folder.split('-')[-1]
            if tra_id == '1':
                posename = database_posefile
                data_file = databasesubmaps_file
            else:
                posename = query_posefile
                data_file = querysubmaps_file

            df_locations = pd.read_table(os.path.join(base_path, seq, tra_folder, posename), sep=' ',
                                         names=['r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33', 'z'])
            df_locations = df_locations.reset_index()
            df_locations.columns.values[0] = 'data_id'
            df_locations = df_locations.loc[:, ['data_id', 'x', 'y']]
            df_locations['data_id'] = df_locations['data_id'].apply(lambda x: str(x).zfill(6))
            df_locations['data_id'] = seq + '/' + tra_folder + data_file + '/' + df_locations['data_id'].astype(str) + '.bin'
            df_locations = df_locations.rename(columns={'data_id': 'file'})
            if tra_id == '1':
                df_database = df_locations
                database_tree = KDTree(df_database[['x', 'y']])
                database_trees.append(database_tree)
                for index, row in df_locations.iterrows():
                    database[len(database.keys())] = {
                        'query': row['file'], 'x': row['x'], 'y': row['y']}
                # database_sets.append(database)
                database_sets[seq] = database
            else:
                df_test = df_locations
                test_tree = KDTree(df_test[['x', 'y']])
                test_trees.append(test_tree)
                for index, row in df_locations.iterrows():
                    test[len(test.keys())] = {
                        'query': row['file'], 'x': row['x'], 'y': row['y']}
                # test_sets.append(test)
                tra_sets.append(test)
        test_sets[seq] = tra_sets

        for i in range(len(test_sets[seq])):
            for key in range(len(test_sets[seq][i].keys())):
                coor = np.array(
                    [[test_sets[seq][i][key]["x"], test_sets[seq][i][key]["y"]]])
                index = database_trees[seq_id].query_radius(coor, r=pos_distence)
                test_sets[seq][i][key][seq] = index[0].tolist()


    output_to_file(database_sets, save_path + 'evaluation_database_clusters.pickle')
    output_to_file(test_sets, save_path + 'evaluation_query_clusters_'+ str(pos_distence) +'m.pickle')


# Building database and query files for evaluation
if __name__ == '__main__':

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = '/home/cyw/CYW/Datasets/mmWave-Radar-Relocalization-main/dataset/'
    query_posefile = "query_submaps_poses.txt"
    database_posefile = "database_submaps_poses.txt"
    querysubmaps_file = "/query_submaps_cluster/"  # "/query_submaps_bin/"
    databasesubmaps_file = "/database_submaps_cluster/"  # "/database_submaps_bin/"
    save_path = "./1843_tuples/"
    pos_distence = 3
    all_folders = (os.listdir(os.path.join(BASE_DIR, base_path)))
    all_folders.sort(key=lambda x: int(x[3:]))
    index_list = range(len(all_folders))
    print("Number of runs: " + str(len(index_list)))
    seqs = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8', 'seq9']  #
    # for index in index_list:
    #     seqs.append(all_folders[index])
    # print(seqs)

    construct_query_and_database_sets(base_path, seqs, querysubmaps_file, databasesubmaps_file, pos_distence,
                                      query_posefile, database_posefile, save_path)







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
        posenames = [query_posefile, database_posefile]
        for posename in posenames:
            database = {}
            test = {}
            if posename == query_posefile:
                data_file = querysubmaps_file
            elif posename == database_posefile:
                data_file = databasesubmaps_file

            df_locations = pd.read_table(os.path.join(base_path, seq, posename), sep=' ',
                                         names=['times', 'r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33', 'z'])
            df_locations = df_locations.reset_index()
            df_locations.columns.values[0] = 'data_id'
            df_locations = df_locations.loc[:, ['data_id', 'x', 'y', 'z']]
            df_locations['data_id'] = df_locations['data_id'].apply(lambda x: str(x).zfill(6))
            df_locations['data_id'] = seq + data_file + df_locations['data_id'].astype(str) + '.pcd'
            df_locations = df_locations.rename(columns={'data_id': 'file'})
            if posename == database_posefile:
                df_database = df_locations
                database_tree = KDTree(df_database[['x', 'y', 'z']])
                database_trees.append(database_tree)
                for index, row in df_locations.iterrows():
                    database[len(database.keys())] = {
                        'query': row['file'], 'x': row['x'], 'y': row['y'], 'z': row['z']}
                # database_sets.append(database)
                database_sets[seq] = database
            else:
                df_test = df_locations
                test_tree = KDTree(df_test[['x', 'y', 'z']])
                test_trees.append(test_tree)
                for index, row in df_locations.iterrows():
                    test[len(test.keys())] = {
                        'query': row['file'], 'x': row['x'], 'y': row['y'], 'z': row['z']}
                test_sets[seq] = test


        for i in range(len(test_sets[seq])):
            for key in range(len(test_sets[seq][i].keys())):
                coor = np.array(
                    [[test_sets[seq][i]["x"], test_sets[seq][i]["y"], test_sets[seq][i]["z"]]])
                index = database_trees[seq_id].query_radius(coor, r=pos_distence)
                test_sets[seq][i][seq] = index[0].tolist()


    output_to_file(database_sets, save_path + 'evaluation_database.pickle')
    output_to_file(test_sets, save_path + 'evaluation_query_'+ str(pos_distence) +'m.pickle')


# Building database and query files for evaluation
if __name__ == '__main__':

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = '/media/cyw/CYW-ZX2/kuangye_data/'
    query_posefile = "query_submaps_poses.txt"
    database_posefile = "database_submaps_poses.txt"
    querysubmaps_file = "/query_submaps_pcd/"
    databasesubmaps_file = "/database_submaps_pcd/"
    save_path = "./data_split/"
    pos_distence = 3
    seqs = ['238jfsloop']  # 238jfsloop， 188loop

    construct_query_and_database_sets(base_path, seqs, querysubmaps_file, databasesubmaps_file, pos_distence,
                                      query_posefile, database_posefile, save_path)







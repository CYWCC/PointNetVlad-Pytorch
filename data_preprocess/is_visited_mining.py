# Based on: https://github.com/kxhit/pointnetvlad/blob/master/submap_generation/KITTI/gen_gt.py

import sys
import os
import json
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from configs.kitti_config import get_config
from utils.kitti_dataset import load_poses_from_txt, load_timestamps
cfg = get_config()


def p_dist(pose1, pose2, threshold=3):
    # dist = np.linalg.norm(pose1 - pose2)
    # a = pose1[0::2]
    # b = pose2[0::2]
    dist = np.linalg.norm(pose1 - pose2)
    if abs(dist) <= threshold:
        return True
    else:
        return False


def t_dist(t1, t2, threshold=10):
    if abs(t1-t2) > threshold:
        return True
    else:
        return False

def overlap_dist(t1, t2, threshold=10):
    if (t1 - t2) > threshold:
        return True
    else:
        return False

def get_positive_dict(basedir,  sequences, output_dir, d_thresh, t_thresh):
    positive_dict = {}
    print('d_thresh: ', d_thresh)
    print('output_dir: ', output_dir)
    print('')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sequence in sequences:
        print(sequence)
        _, scan_positions = load_poses_from_txt(
            basedir +'/sequences/'+ sequence + '/poses.txt')
        scan_timestamps = load_timestamps(
            basedir + '/sequences/' + sequence + '/times.txt')

        sequence_id = str(int(sequence))
        if sequence not in positive_dict:
            positive_dict[sequence_id] = {}

        for t1 in tqdm(range(len(scan_timestamps))):
            for t2 in range(len(scan_timestamps)):
                # 满足距离阈值和时间阈值
                if p_dist(scan_positions[t1], scan_positions[t2], d_thresh) & t_dist(scan_timestamps[t1], scan_timestamps[t2], t_thresh):
                    if t1 not in positive_dict[sequence_id]:
                        positive_dict[sequence_id][t1] = []
                    positive_dict[sequence_id][t1].append(t2)
                # 找每个样本的回环帧
                # if p_dist(scan_positions[t1], scan_positions[t2], d_thresh) & overlap_dist(scan_timestamps[t1], scan_timestamps[t2], 30):
                #     if t1 not in positive_dict[sequence_id]:
                #         positive_dict[sequence_id][t1] = []
                #     positive_dict[sequence_id][t1].append(t2)
                # else:
                #     if t1 not in positive_dict[sequence_id]:
                #         positive_dict[sequence_id][t1] = []

    # save_file_name = '{}/visited_D-{}_T-{}.json'.format(output_dir, d_thresh, 30)

    save_file_name = '{}/positive_all_sequences_D-{}_T-{}.json'.format(output_dir, d_thresh, t_thresh)
    with open(save_file_name, 'w') as f:
        json.dump(positive_dict, f)
    print('Saved: ', save_file_name)

    return positive_dict


#####################################################################################
if __name__ == "__main__":
    basedir = "/home/cyw/CYW/Datasets/mmWave-Radar-Relocalization-main/dataset"
    pose_folder = 'SemanticKitti/'
    pose_filename = "pose.txt"  #
    pointcloud_fols = "/scan_cells_NDT_onelabel/"
    semlabel_fols = "/scan_cells_onelabel/"

    sequences = ['08', '00', '01','02','03', '04','05', '06', '07',  '09', '10'] # '00','02','05', '06', '07', '08'
    output_dir = os.path.join(os.path.dirname(
        __file__), '../configs/kitti_tuples/')

    t_thresh = 0

    # pose_basedir = os.path.join(basedir, pose_folder)
    get_positive_dict(basedir, sequences, output_dir, 3, t_thresh)
    # get_positive_dict(basedir,  sequences, output_dir, 20, t_thresh)

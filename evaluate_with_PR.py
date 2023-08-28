import argparse
import os
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import torch.nn as nn
from torch.backends import cudnn
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from loading_pointclouds import *
import models.PointNetVlad as PNV
import numpy as np
import config as cfg

cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.getLogger().setLevel(logging.INFO)

def save_pickle(data_variable, file_name):
    dbfile2 = open(file_name, 'ab')
    pickle.dump(data_variable, dbfile2)
    dbfile2.close()
    print('Finished saving: ', file_name)

def evaluate():
    model = PNV.PointNetVlad(global_feat=True, feature_transform=False, max_pool=False, inputdim=cfg.INPUT_DIM,
                             output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)
    model = model.to(device)

    resume_filename = cfg.LOG_DIR + cfg.MODEL_FILENAME
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    model = nn.DataParallel(model)
    evaluate_model(model)

def evaluate_model(model):
    global F1_TN, F1_FP, F1_TP, F1_FN, F1_thresh_id
    eval_seq = str(cfg.EVAL_SEQ)
    save_descriptors = False
    plot_pr_curve = True
    DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)

    database_set = DATABASE_SETS[eval_seq]
    query_sets = QUERY_SETS[eval_seq]
    is_revisit_list = []
    for tra in query_sets:
        for i in tra:
            num_positive = len(tra[i][eval_seq])
            if num_positive !=0:
                is_revisit_list.append(int(1))
            else:
                is_revisit_list.append(int(0))
    thresholds = np.linspace(cfg.thresh_min, cfg.thresh_max, int(500))
    # thresholds = np.linspace(0.01, 2.0, int(1000))

    num_thresholds = len(thresholds)

    # Store results of evaluation.
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    min_min_dist = 1.0
    max_min_dist = 0.0
    num_revisits = 0  # 回环数量
    num_correct_loc = 0  # 正确定位数量
    percentage_correct_loc = 0

    DATABASE_VECTORS = get_latent_vectors(model, database_set)
    query_len = 0
    querysets = {}
    for set in query_sets:
        set = {k + query_len: v for k, v in set.items()}
        querysets.update(set)
        query_len = query_len + len(set)
    QUERY_VECTORS=get_latent_vectors(model, querysets)

    min_min_dist_list = np.zeros(len(QUERY_VECTORS))

    for query_idx, query in enumerate(QUERY_VECTORS):
        query = np.expand_dims(query, 0)
        feat_dists = cdist(query, DATABASE_VECTORS, metric=cfg.eval_feature_distance).reshape(-1)  # euclidean
        min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)  # 最近特征距离,索引
        min_min_dist_list[query_idx] = min_dist # 0408
        true_positives = querysets[query_idx][eval_seq]
        # place_candidate_pose = seen_poses[nearest_idx]
        # p_dist = np.linalg.norm(query_pose - place_candidate_pose)  # 位置距离

        is_revisit = is_revisit_list[query_idx]
        is_correct_loc = 0
        if is_revisit:
            num_revisits += 1
            if nearest_idx in true_positives:
                num_correct_loc += 1
                is_correct_loc = 1

        logging.info(
            f'id: {query_idx} n_id: {nearest_idx} is_rev: {is_revisit} is_correct_loc: {is_correct_loc} min_dist: {min_dist}')

        if min_dist < min_min_dist:
            min_min_dist = min_dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist

        # Evaluate top-1 candidate.
        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]

            if (min_dist < threshold):  # Positive Prediction
                if nearest_idx in true_positives:
                    num_true_positive[thres_idx] += 1

                else:
                    num_false_positive[thres_idx] += 1

            else:  # Negative Prediction
                if (is_revisit == 0):
                    num_true_negative[thres_idx] += 1
                else:
                    num_false_negative[thres_idx] += 1

    F1max = 0.0
    RP100 = 0.0
    RP100_thers = 0.0
    Precisions, Recalls = [], []
    Accuracies = []
    #
    if not save_descriptors:
        for ithThres in range(num_thresholds):
            nTrueNegative = num_true_negative[ithThres]
            nFalsePositive = num_false_positive[ithThres]
            nTruePositive = num_true_positive[ithThres]
            nFalseNegative = num_false_negative[ithThres]
            nTotalTestPlaces = nTrueNegative + nFalsePositive + nTruePositive + nFalseNegative

            Precision = 0.0
            Recall = 0.0
            F1 = 0.0
            Acc = (nTruePositive + nTrueNegative) / nTotalTestPlaces

            if nTruePositive > 0.0:
                Precision = nTruePositive / (nTruePositive + nFalsePositive)
                Recall = nTruePositive / (nTruePositive + nFalseNegative)

                F1 = 2 * Precision * Recall * (1 / (Precision + Recall))

            Precisions.append(Precision)
            Recalls.append(Recall)
            Accuracies.append(Acc)

            if F1 > F1max:
                F1max = F1
                F1_TN = nTrueNegative
                F1_FP = nFalsePositive
                F1_TP = nTruePositive
                F1_FN = nFalseNegative
                F1_thresh_id = ithThres

            if int(Precision) == 1: # precision 100%
                RP100 = Recall
                RP100_thers = thresholds[ithThres]
                rp100_tn = nTrueNegative
                rp100_fp = nFalsePositive
                rp100_tp = nTruePositive
                rp100_fn = nFalseNegative

        if RP100 == 0.0:
            EP = Precisions[1] / 2.0

        else:
            EP = 0.5 + (RP100 / 2.0)

        # RP100 Record
        # recall_list_path = './tools/'+ eval_seq + 'recall_list.txt'
        # with open(recall_list_path, 'w') as fp:
        #     [fp.write(str(item) + '\n') for item in min_min_dist_list]
        #     fp.close()

        percentage_correct_loc = num_correct_loc * 100.0 / num_revisits
        logging.info(f'num_revisits: {num_revisits}')
        logging.info(f'num_correct_loc: {num_correct_loc}')
        logging.info(
            f'percentage_correct_loc: {percentage_correct_loc}')
        logging.info(
            f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}')
        logging.info(
            f'F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}')
        logging.info(f'F1_thresh_id: {F1_thresh_id}')
        logging.info(f'F1max: {F1max}')
        logging.info(f'EP: {EP}')
        logging.info(f'RP100_thers: {RP100_thers} ')
        logging.info(f'rp100_tn: {rp100_tn} rp100_fp:{rp100_fp} rp100_tp: {rp100_tp} rp100_fn:{rp100_fn} ')

        # # PR data record
        save_dir = os.path.join(os.path.dirname(__file__), 'AWR1843_pr_data_0811', eval_seq)  # ARS548_pr_data_5m
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        results_file = save_dir + '/' + eval_seq + cfg.eval_feature_distance + '_result.txt' # _cosine
        with open(results_file, 'w') as f:
            f.write(f'num_revisits: {num_revisits}\n'
                    f'num_correct_loc: {num_correct_loc}\n'
                    f'percentage_correct_loc: {percentage_correct_loc}\n'
                    f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}\n'
                    f'F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}\n'
                    f'F1_thresh_id: {F1_thresh_id} \n'
                    f'F1max: {F1max} EP: {EP}\n'
                    f'RP100_thers: {RP100_thers}\n'
                    f'rp100_tn: {rp100_tn} rp100_fp:{rp100_fp} rp100_tp: {rp100_tp} rp100_fn:{rp100_fn}')

        PR_file = save_dir + '/' + eval_seq + cfg.eval_feature_distance + '.txt' # _cosine
        pr = [Precisions, Recalls]
        with open(PR_file, 'w') as f:
            for i in zip(*pr):
                f.write("{0}\t{1}\n".format(*i))

        if plot_pr_curve:
            plt.title(eval_seq + ' F1Max: ' + "%.4f" % (F1max))
            plt.plot(Recalls, Precisions, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.axis([0, 1, 0, 1.1])
            plt.xticks(np.arange(0, 1.01, step=0.1))
            plt.grid(True)
            # save_dir = os.path.join(os.path.dirname(__file__), 'pr_curves')
            plt.savefig(save_dir + '/' + eval_seq + cfg.eval_feature_distance + '.png') # _cosine

    if save_descriptors:
        eval_seq = str(eval_seq).split('/')[-1]
        save_dir = os.path.join(os.path.dirname(__file__), 'pr_curves') + '/' + eval_seq
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('Saving pickles: ', eval_seq)
        save_pickle(num_true_positive, save_dir + '/num_true_positive.pickle')
        save_pickle(num_false_positive, save_dir + '/num_false_positive.pickle')
        save_pickle(num_true_negative, save_dir + '/num_true_negative.pickle')
        save_pickle(num_false_negative, save_dir + '/num_false_negative.pickle')

    return F1max, percentage_correct_loc

def get_latent_vectors(model, set):

    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(set))

    # batch_num = cfg.EVAL_BATCH_SIZE * \
    #     (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY)
    batch_num = cfg.EVAL_BATCH_SIZE
    q_output = []
    # for q_index in range(len(train_file_idxs)//batch_num):
    #     file_indices = train_file_idxs[q_index *
    #                                    batch_num:(q_index+1)*(batch_num)]
    for q_index in range(len(train_file_idxs) // batch_num):
        file_indices = train_file_idxs[q_index * batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(set[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        #out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(set):
        file_indices = train_file_idxs[index_edge:len(set)]
        file_names = []
        for index in file_indices:
            file_names.append(set[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1 = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    return q_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--positives_per_query', type=int, default=4,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--negatives_per_query', type=int, default=12,
                        help='Number of definite negatives in each training tuple [default: 20]')
    parser.add_argument('--eval_batch_size', type=int, default=12,
                        help='Batch Size during training [default: 1]')
    parser.add_argument('--dimension', type=int, default=256)
    parser.add_argument('--decay_step', type=int, default=200000,
                        help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--results_dir', default='results/',
                        help='results dir [default: results]')
    parser.add_argument('--dataset_folder', default='/home/cyw/CYW/Datasets/mmWave-Radar-Relocalization-main/dataset/',
                        # /home/cyw/CYW/Datasets/ARS548/
                        help='PointNetVlad Dataset Folder')
    parser.add_argument('--eval_feature_distance', type=str, default='euclidean')  # cosine # euclidean
    parser.add_argument("--thresh_min", default=0.01, type=float, help="Thresholds on distance to top-1.")
    parser.add_argument("--thresh_max", default=1.5, type=float, help="Thresholds on distance to top-1.")
    FLAGS = parser.parse_args()

    # BATCH_SIZE = FLAGS.batch_size
    cfg.EVAL_BATCH_SIZE = FLAGS.eval_batch_size
    # cfg.NUM_POINTS = 1280  # 1024
    cfg.FEATURE_OUTPUT_DIM = 256
    cfg.EVAL_POSITIVES_PER_QUERY = FLAGS.positives_per_query
    cfg.EVAL_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
    cfg.DECAY_STEP = FLAGS.decay_step
    cfg.DECAY_RATE = FLAGS.decay_rate

    cfg.RESULTS_FOLDER = FLAGS.results_dir

    cfg.EVAL_DATABASE_FILE = './generating_queries/1843_tuples/evaluation_database_9seqs.pickle'
    cfg.EVAL_QUERY_FILE = './generating_queries/1843_tuples/evaluation_query_9seqs_3m.pickle'

    # cfg.EVAL_DATABASE_FILE = './generating_queries/ARS548_tuples/evaluation_database_1seq.pickle'
    # cfg.EVAL_QUERY_FILE = './generating_queries/ARS548_tuples/evaluation_query_1seq_5m.pickle'

    cfg.LOG_DIR = 'log/'
    cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'AWR1843+i_results_9seqs_3m'  # ARS548_results_5m.txt
    cfg.MODEL_FILENAME = 'best_checkpoint.pth.tar'

    cfg.DATASET_FOLDER = FLAGS.dataset_folder
    cfg.eval_feature_distance = FLAGS.eval_feature_distance
    cfg.thresh_min = FLAGS.thresh_min
    cfg.thresh_max = FLAGS.thresh_max
    cfg.EVAL_SEQ = 'seq9'


    evaluate()
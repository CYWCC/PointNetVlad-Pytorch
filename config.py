# GLOBAL
NUM_POINTS = 1280   # 1280 1024
FEATURE_OUTPUT_DIM = 256
RESULTS_FOLDER = "results/"
OUTPUT_FILE = "results/results.txt"
INPUT_DIM = 4

LOG_DIR = 'log/'
MODEL_FILENAME = "model.ckpt"

DATASET_FOLDER = '/home/cyw/CYW/Datasets/mmWave-Radar-Relocalization-main/dataset'

# TRAIN
BATCH_NUM_QUERIES = 2
TRAIN_POSITIVES_PER_QUERY = 2
TRAIN_NEGATIVES_PER_QUERY = 18
DECAY_STEP = 200000
DECAY_RATE = 0.7
BASE_LEARNING_RATE = 0.000005
MOMENTUM = 0.9
OPTIMIZER = 'ADAM'
MAX_EPOCH = 20

MARGIN_1 = 0.5
MARGIN_2 = 0.2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_CLIP = 0.99

RESUME = False

TRAIN_FILE = './generating_queries/1843_tuples/training_queries_baseline.pickle'
TEST_FILE = './generating_queries/1843_tuples/evaluation_query_9seqs_3m.pickle'


# LOSS
LOSS_FUNCTION = 'quadruplet'
LOSS_LAZY = True
TRIPLET_USE_BEST_POSITIVES = False
LOSS_IGNORE_ZERO_BATCH = False

# EVAL6
EVAL_BATCH_SIZE = 1
EVAL_POSITIVES_PER_QUERY = 4
EVAL_NEGATIVES_PER_QUERY = 12
EVAL_SEQ = 'seq1'

EVAL_DATABASE_FILE = './generating_queries/1843_tuples/evaluation_database_9seqs.pickle'
EVAL_QUERY_FILE = './generating_queries/1843_tuples/evaluation_query_9seqs_3m.pickle'
# EVAL_DATABASE_FILE = './generating_queries/1843_tuples/evaluation_database_clusters.pickle'
# EVAL_QUERY_FILE = './generating_queries/1843_tuples/evaluation_query_clusters_3m.pickle'

def cfg_str():
    out_string = ""
    for name in globals():
        if not name.startswith("__") and not name.__contains__("cfg_str"):
            #print(name, "=", globals()[name])
            out_string = out_string + "cfg." + name + \
                "=" + str(globals()[name]) + "\n"
    return out_string

""" Precision-Recall curves plus introspection tools. """

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import yaml
from sklearn import  metrics

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# from utils.misc_utils import *
font = {'family': 'serif',
        # 'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
# cfg_file = open('config.yml', 'r')
# cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

data_dir = '../AWR1843_pr_data/'  #

macros = {
    0: ['seq1/seq1euclidean.txt', 'seq1', 'green'],
    1: ['seq2/seq2euclidean.txt', 'seq2', 'purple'],
    2: ['seq3/seq3euclidean.txt', 'seq3', 'black'],
    3: ['seq4/seq4euclidean.txt', 'seq4', 'grey'],
    4: ['seq5/seq5euclidean.txt', 'seq5', 'pink'],
    5: ['seq6/seq6euclidean.txt', 'seq6', 'blue'],
    6: ['seq7/seq7euclidean.txt', 'seq7', 'orange'],
    7: ['seq8/seq8euclidean.txt', 'seq8', 'red'],
    8: ['seq9/seq9euclidean.txt', 'seq9', 'yellow']
}


for i in range(len(macros)):
    folder = macros[i][0]
    label = macros[i][1]
    colour = macros[i][2]
    _dir = data_dir + folder
    Precisions = []
    Recalls = []

    with open(_dir, 'r') as f:
        data = f.readlines()
        for line in data:
            line_list = line.split()
            if np.float(line_list[0]) == 0.0 and np.float(line_list[1]) == 0.0:
                continue
            Precision = np.float(line_list[0])
            Recall = np.float(line_list[1])
            Precisions.append(Precision)
            Recalls.append(Recall)
    # x = np.array(Recalls)
    # y = np.array(Precisions)
    # xs = np.linspace(x.min(), x.max(), 1000)
    # ys = make_interp_spline(x, y)(xs)

    plt.plot(Recalls, Precisions, marker='.', color=colour, label=label, linewidth=2,markersize='2')
    # plt.legend(prop='lower left')
    # plt.xlabel('Recall', fontdict=font)
    # plt.ylabel('Precision', fontdict=font)
    # plt.axis([0, 1, 0, 1.1])
    # plt.xticks(np.arange(0, 1.01, step=0.1))
    # plt.grid(True)
    # plt.show()

##########################################################################################################################
""" Plot Precision-Recall curves """

# plt.legend(prop=font_legend)
plt.legend(loc=3,prop = {'size':10})
# plt.legend(prop = {'size':12})
plt.title('performance on AWR1843')

plt.xlabel('Recall', fontdict=font)
plt.ylabel('Precision', fontdict=font)
plt.axis([0, 1.01, 0, 1.05])
plt.xticks(np.arange(0, 1.01, step=0.1))
plt.grid(True)
# plt.show()
plt.savefig(data_dir + 'pr_AWR1843.png')



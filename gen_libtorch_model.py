# -*-coding:utf-8-*-
#!/usr/bin/env python3
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import torch
import config as cfg
import models.PointNetVlad as PNV
import yaml
import numpy as np

# load config ================================================================
resume_filename = cfg.LOG_DIR + "0811-2/best_checkpoint.pth.tar"
database_path = "/home/cyw/CYW/TPAMI/PointNetVlad-Pytorch-master/log/databases/seq1_database.bin"
filepath = '/home/cyw/CYW/Datasets/mmWave-Radar-Relocalization-main/dataset/seq1/1-2/query_submaps_bin/000000.bin'

# pc = np.fromfile(filepath, dtype=np.float64)
# pc = np.float32(pc)
# pc = np.reshape(pc,(pc.shape[0]//cfg.INPUT_DIM, cfg.INPUT_DIM))
# pc_tensor = (torch.tensor(pc)).unsqueeze(dim = 0)
# pc_tensor = pc_tensor.unsqueeze(dim = 1)

checkpoint = torch.load(resume_filename)
saved_state_dict = checkpoint['state_dict']
print(resume_filename)

amodel = PNV.PointNetVlad(global_feat=True, feature_transform=False, max_pool=False, inputdim=cfg.INPUT_DIM,
                                      output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)

amodel.load_state_dict(saved_state_dict)

# amodel.cuda()
amodel.eval()

# o1 = amodel(pc_tensor)
# print(o1)
# 64 for KITTI
example = torch.rand(1, 1, 1280, 4)

# example = example.cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(amodel, example)
traced_script_module.save("./log/PointNetVlad.pt")

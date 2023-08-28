# PointNetVlad-Pytorch
Unofficial PyTorch implementation of PointNetVlad (https://github.com/mikacuy/pointnetvlad)
refer to https://github.com/cattaneod/PointNetVlad-Pytorch

The main differences are:
* data_preprocess folder: Data preprocessing or Tsinghua data and ARS548 data.
* generating_queries folder: Query and database partitioning for Tsinghua data and ARS548 data. 
* Some parameters in configs added.
* add gen_libtorch_model.py.
* tools folder: Some scripts for processing data or post-processing results.

### Pre-Requisites
* PyTorch 2.0.1

### Generate pickle files
```
cd generating_queries/

# For training tuples
python generate_training_tuples_baseline.py

# For network evaluation
python generate_test_sets.py
```

### Train
```
python train_pointnetvlad.py
```

### Evaluate
```
python evaluate_radar.py
```

### Evaluate with pr curve 
```
python evaluate_radar_PR.py
```

Take a look at train_pointnetvlad.py and evaluate.py for more parameters

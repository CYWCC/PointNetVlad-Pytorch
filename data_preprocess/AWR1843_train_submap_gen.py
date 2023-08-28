import numpy as np
import os
from txt2bin import txt2bin
from tqdm import tqdm

##########################################
# generate train submaps and poses for train
##########################################
def normalization(data):
    _range = np.max(data, axis=0) - np.min(data, axis=0)
    pc_xyz = data[:, :3] / _range[:3]
    i_min =np.min(data, axis=0)[-1]
    pc_i = (data[:, -1] - np.min(data, axis=0)[-1]) / _range[:3]
    pc_normalized = np.concatenate((pc_xyz, pc_i), axis=1)
    return pc_normalized

def standardization(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized

def load_qinghua_poses(xyz_path, yaw_path):
    trans = []
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            xyz = np.fromstring(line, dtype=np.float32, sep=' ')
            xyz = xyz.reshape(3, 1)
            trans.append(xyz)

    poses = []
    with open(yaw_path, 'r') as f:
        lines_yaw = f.readlines()
        for i in range(len(lines_yaw)):
            yaw = np.fromstring(lines_yaw[i], dtype=np.float32, sep=' ')[0]
            yaw = - (np.pi / 180) * yaw  # 将yaw角度转为弧度
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw, 0],
                                        [sin_yaw, cos_yaw, 0],
                                        [0, 0, 1]])  # 旋转矩阵
            xyz = trans[i]
            pose = np.hstack((rotation_matrix, xyz))  # 将xyz和rotation矩阵按列拼接
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
    return poses


def load_scans(scan_path):
  """ Load 3D points of a scan. The fileformat is the .bin format used in
    the KITTI dataset.
    Args:
      scan_path: the (full) filename of the scan file
    Returns:
      A nx4 numpy array of homogeneous points (x, y, z, 1).
  """
  current_vertex = np.fromfile(scan_path, dtype=np.float32)
  current_vertex = current_vertex.reshape((-1, 4))
  current_points = current_vertex[:, 0:3]
  current_intensity = current_vertex[:, 3]
  current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
  current_vertex[:, :-1] = current_points
  return current_vertex, current_intensity


if __name__ == '__main__':
    data_path = "/home/cyw/CYW/Datasets/mmWave-Radar-Relocalization-main/dataset"
    seqs = ['seq1','seq2','seq5','seq7']
    tra_id = '-1'
    traget_num_points = 1024
    for seq in tqdm(seqs):
        tra_forder = seq[-1] + tra_id
        data_dir = data_path + "/" + seq +"/" + tra_forder
        submaps_path = data_dir + "/train_submaps_bin/"
        xyz_path = data_dir + "/data_label.txt"
        yaw_path = data_dir + "/yaw.txt"
        submap_size = int(2)
        poses = load_qinghua_poses(xyz_path, yaw_path)
        scans_path = txt2bin(data_path, seq, tra_forder)
        file_names = sorted(os.listdir(scans_path))
        submap_poses = []
        norm = 0  # 0: false 1: true
        count = 0
        # for i, scans_name in enumerate(tqdm(file_names, position=0)):
        for i in tqdm(range(len(file_names)),position=0):
            if i - submap_size < 0 or i + submap_size >= len(file_names):
                continue
            currscan_pc, currscan_intensity= load_scans(os.path.join(scans_path, file_names[i]))
            submap_pc = currscan_pc
            submap_pc[:, -1] = currscan_intensity  # x,y,x,i
            submap_pose = poses[i]
            for j in range(-submap_size, submap_size + 1):
                if j != 0:
                    neiscan_pc, neiscan_intensity= load_scans(
                        os.path.join(scans_path, file_names[i + j]))  # near neighbour scan point clouds
                    relative_pose = np.linalg.inv(poses[i]).dot(poses[i + j])
                    neiscan_pc_in_current = np.linalg.inv(poses[i]).dot(poses[i + j]).dot(neiscan_pc.T).T
                    neiscan_pc_in_current[:, -1] = neiscan_intensity
                    submap_pc = np.vstack((submap_pc, neiscan_pc_in_current))
            if norm == 1:
                submap_pc = standardization(submap_pc) # standardization normalization
            if not os.path.exists(submaps_path):
                os.makedirs(submaps_path)
            with open(os.path.join(submaps_path, file_names[count]), 'wb') as f:
                submap_pc.tofile(f)
            submap_poses.append(submap_pose)
            count += 1

        submaps_poses_path = data_dir + "/train_submaps_poses.txt"
        with open(submaps_poses_path, 'w', encoding='utf-8') as f:
            for pose in submap_poses:
                pose_reshape = pose[:3, :4].reshape(1, 12)
                f.write(' '.join(str(x) for x in pose_reshape.flatten()) + '\n')
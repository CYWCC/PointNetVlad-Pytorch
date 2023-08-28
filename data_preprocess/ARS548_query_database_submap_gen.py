import numpy as np
import os
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import random

##########################################
# generate query and database submaps for evaluation or test
##########################################

def load_ARS548_poses(poses_path):
    poses = []
    with open(poses_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = np.fromstring(line, dtype=np.float32, sep=' ')
            # temp = temp.reshape(8, 1)
            xyz = temp[1:4].reshape(3, 1)
            quaternion = temp[4:8]
            r = Rotation.from_quat(quaternion)
            rotation_matrix = r.as_matrix()
            pose = np.hstack((rotation_matrix, xyz))  # 将xyz和rotation矩阵按列拼接
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
    return poses


def load_pcds(scan_path):
  """ Load 3D points of a scan. The fileformat is the .bin format used in
    the KITTI dataset.
    Args:
      scan_path: the (full) filename of the scan file
    Returns:
      A nx4 numpy array of homogeneous points (x, y, z, 1).
  """
  pcd = o3d.io.read_point_cloud(scan_path)
  xyz = np.asarray(pcd.points)
  current_vertex = np.ones((xyz.shape[0], xyz.shape[1] + 1))
  current_vertex[:, :-1] = xyz
  return current_vertex

def random_down_sample(pc, sample_points):
    submap_pcd = o3d.geometry.PointCloud()
    submap_pcd.points = o3d.utility.Vector3dVector(submap_pc[:, :3])
    sampleA = random.sample(range(pc.shape[0]), sample_points)
    sampled_cloud = submap_pcd.select_by_index(sampleA)
    sampled_pc = np.asarray(sampled_cloud.points)
    return sampled_pc

def normalize(pc):
    centroid = np.mean(pc, axis=0)  # 求取点云的中心
    pc = pc - centroid  # 将点云中心置于原点 (0, 0, 0)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  # 求取长轴的的长度
    pc_normalized = pc / m  # 依据长轴将点云归一化到 (-1, 1)
    return pc_normalized

if __name__ == '__main__':
    data_path = "/home/cyw/CYW/TPAMI/radar_reloc/src/data_db"
    output_path = "/home/cyw/CYW/Datasets/ARS548/pointnetvlad"
    mode = "query" # query database
    submaps_path = output_path + "/" + mode + "_submaps_bin/"
    submaps_poses_path = output_path + "/" + mode + "_submaps_poses.txt"
    database_folder = "testscans"
    database_posesfile = "testtruth.txt"
    queries_folder = "testscans4"
    query_posesfile = "testtruth4.txt"
    norm = 1
    sub_size = 10
    target_points = 1024

    if mode == 'database':
        gap = sub_size - 5
        scans_folder = os.path.join(data_path, database_folder)
        poses_path = os.path.join(data_path, database_posesfile)
    else:
        gap = sub_size
        scans_folder = os.path.join(data_path, queries_folder)
        poses_path = os.path.join(data_path, query_posesfile)

    poses = load_ARS548_poses(poses_path)
    file_names = sorted(os.listdir(scans_folder))

    submap_poses = []
    start = 0
    count = 0
    for i in tqdm(range(len(file_names))):
        if i == start:
            end = i + sub_size
            currscan_pc = load_pcds(os.path.join(scans_folder, file_names[i]))
            submap_pc = currscan_pc
            submap_pose = poses[i]
            for j in range(start + 1, end):
                if j < len(file_names):
                    neiscan_pc = load_pcds(os.path.join(scans_folder, file_names[j])) # near neighbour scan point clouds
                    neiscan_pc_in_current = np.linalg.inv(poses[i]).dot(poses[j]).dot(neiscan_pc.T).T
                    submap_pc = np.vstack((submap_pc, neiscan_pc_in_current))
            if len(submap_pc) < target_points:
                continue
            target_submap = random_down_sample(submap_pc[:,:3], target_points)
            if norm == 1:
                submap_pc = normalize(target_submap)
            if not os.path.exists(submaps_path):
                os.makedirs(submaps_path)
            submap_name = file_names[count].split(".")[0] + ".bin"
            with open(os.path.join(submaps_path, submap_name), 'wb') as f:
                submap_pc[:, :3].tofile(f)
            submap_poses.append(submap_pose)
            start = start + gap
            count += 1

    with open(submaps_poses_path, 'w', encoding='utf-8') as f:
        for pose in submap_poses:
            pose_reshape = pose[:3, :4].reshape(1, 12)
            f.write(' '.join(str(x) for x in pose_reshape.flatten()) + '\n')


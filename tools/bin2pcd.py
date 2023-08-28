# -*-coding:utf-8-*-
import os
import numpy as np
import open3d as o3d

data_dir = '/home/cyw/CYW/Datasets/mmWave-Radar-Relocalization-main/dataset/seq4/4-1/'
bin_folder = data_dir + 'database_submaps_bin/' #
pcd_folder = data_dir + 'pcds_all/'
poses_file = '/home/cyw/CYW/Datasets/mmWave-Radar-Relocalization-main/dataset/seq4/4-1/database_submaps_poses.txt'
xyz_path = data_dir + "/data_label.txt"
yaw_path = data_dir + "/yaw.txt"

def load_scans(scan_path):
  current_vertex = np.fromfile(scan_path, dtype=np.float64)
  current_vertex = current_vertex.reshape((-1, 4))
  return current_vertex[:,:3]

def load_poses(pose_path):
    poses= []
    with open(pose_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pose = np.fromstring(line, dtype=np.float32, sep=' ')
            pose = pose.reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
    return poses

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
            yaw = -(np.pi / 180) * yaw  # 将yaw角度转为弧度
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

if not os.path.exists(pcd_folder):
    os.makedirs(pcd_folder)

# poses = load_qinghua_poses(xyz_path, yaw_path)
poses = load_poses(poses_file)

file_names = sorted(os.listdir(bin_folder))
for i, bin_name in enumerate(file_names):
    bin_path = bin_folder + bin_name
    xyz = load_scans(bin_path)
    transform = poses[i]
    xyz_homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1)
    xyz_transformed = xyz_homo.dot(transform.T)
    # 去除添加的维度
    xyz1 = xyz_transformed[:, :3]

    pcd=o3d.geometry.PointCloud()  # 传入3d点云格式
    pcd.points=o3d.utility.Vector3dVector(xyz1)#转换格式
    pcd_path = pcd_folder + bin_name.split('.')[0] + '.pcd'
    o3d.io.write_point_cloud(pcd_path, pcd)

    




import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import open3d as o3d
import random

##########################################
# generate train submaps and poses for train / kuangye data
##########################################

def computeW_T_Bs(pose):
    gravity_direction = np.array([0, 0, 1])
    g_s = np.dot(np.linalg.inv(pose[:3,:3]), gravity_direction)

    rotation_axis = np.cross(g_s, gravity_direction)
    rotation_angle = np.arccos(np.dot(g_s, gravity_direction))

    rotation_matric_aligned = Rotation.from_rotvec(rotation_axis * rotation_angle).as_matrix()

    Bs_T_B = np.eye(4)
    Bs_T_B[:3,:3] = rotation_matric_aligned
    Bs_T_B[:3, 3] = [0, 0, 0]
    W_T_Bs = np.dot(pose, np.linalg.inv(Bs_T_B))

    return W_T_Bs, Bs_T_B

def random_down_sample(pcd, sample_points):
    sampleA = random.sample(range(len(pcd.points)), sample_points)
    sampled_cloud = pcd.select_by_index(sampleA)
    return sampled_cloud

def standardization(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized

def load_poses(pose_path):
    poses = []
    timestamps = []
    with open(pose_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            timestamp = np.fromstring(line, dtype=np.float64, sep=' ')[0]
            temp = np.fromstring(line, dtype=np.float32, sep=' ')
            a = (temp[1:4]).shape
            if a[0] <3:
                print(a)

            xyz = temp[1:4].reshape(3, 1)
            quaternion = temp[4:8]
            r = Rotation.from_quat(quaternion)
            rotation_matrix = r.as_matrix()
            pose = np.hstack((rotation_matrix, xyz))
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
            timestamps.append(timestamp)
    return poses, timestamps

if __name__ == '__main__':
    data_path = "/media/cyw/CYW-ZX2/Kuangye/kuangye_data"
    seqs = ['238jfsloop']  # 238jfsloop 188loop'330loop',,'blss_loo''410loop'
    traget_num_points = 4096
    for seq in tqdm(seqs):
        seq_dir = data_path + "/" + seq
        data_dir = seq_dir + "/PCD"
        pose_path = seq_dir + "/pos_log.txt"
        submaps_path = seq_dir + "/database_submaps_pcd_Bs/"
        submaps_poses_Bs_path = seq_dir + "/database_submaps_poses_Bs.txt"
        submap_size = 6

        poses, timestamps = load_poses(pose_path)
        pcd_files = os.listdir(data_dir)
        pcd_files = sorted(pcd_files, key=lambda x: int((x.split('.')[0]).split('_')[-1]))[:len(poses)]

        if len(poses) != len(pcd_files):
            raise ValueError("The number of PCD files and poses is inconsistent")

        submap_poses = []
        submap_poses_Bs = []
        submap_times =[]
        norm = 0  # 0: false 1: true
        count = 0
        gap = 10 # 6, 10, 20 for train, database, query

        for i in tqdm(range(0, len(poses), gap)):
            if i - submap_size < 0 or i + submap_size >= len(poses):
                continue
            curr_pcd = o3d.io.read_point_cloud(os.path.join(data_dir, pcd_files[i]))
            curr_pc = np.asarray(curr_pcd.points)
            submap_pc = curr_pc
            submap_pose = poses[i]
            submap_time = timestamps[i]
            for j in range(-submap_size, submap_size + 1):
                if j != 0:
                    nei_pcd = o3d.io.read_point_cloud(os.path.join(data_dir, pcd_files[i + j]))
                    nei_pc = np.asarray(nei_pcd.points)
                    nei_pc = np.hstack((nei_pc, np.ones((nei_pc.shape[0], 1))))  # n×3-->n*4
                    relative_pose = np.linalg.inv(poses[i]).dot(poses[i + j])  # poses[i + j]
                    nei_pc_in_current = relative_pose.dot(nei_pc.T).T
                    submap_pc = np.vstack((submap_pc, nei_pc_in_current[:,:3]))
            if norm == 1:
                submap_pc = standardization(submap_pc) # standardization normalization
            filtered_pc = submap_pc[(submap_pc[:, 0] ** 2 + submap_pc[:, 1] ** 2) > 0.5]

            # Pose compensation
            points_g = np.mean(filtered_pc, axis=0)
            points_g = points_g.reshape(1, 3)
            B_T_B1= np.eye(4)
            B_T_B1[:3, 3] = -points_g
            B_T_B1 = np.linalg.inv(B_T_B1)
            submap_pose = submap_pose@B_T_B1
            filtered_pc = np.hstack((filtered_pc, np.ones((filtered_pc.shape[0], 1))))
            submap_pc_new = np.linalg.inv(B_T_B1).dot(filtered_pc.T).T

            submap_pcd_new = o3d.geometry.PointCloud()
            submap_pcd_new.points = o3d.utility.Vector3dVector(submap_pc_new[:, :3])

            # Bs_T_B
            W_T_Bs, Bs_T_B = computeW_T_Bs(submap_pose)
            submap_pc_Bs = Bs_T_B.dot(submap_pc_new.T).T

            submap_pcd_Bs = o3d.geometry.PointCloud()
            submap_pcd_Bs.points = o3d.utility.Vector3dVector(submap_pc_Bs[:, :3])

            clean_submap, ind = submap_pcd_Bs.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.0)
            target_submap = random_down_sample(clean_submap, traget_num_points)

            # save W_T_Bs and submap_pc_Bs
            if not os.path.exists(submaps_path):
                os.makedirs(submaps_path)
            save_name = str(count).zfill(6) + '.pcd'
            path = os.path.join(submaps_path, save_name)
            o3d.io.write_point_cloud(path, target_submap)

            submap_times.append(submap_time)
            submap_poses_Bs.append(W_T_Bs)
            count += 1

        with open(submaps_poses_Bs_path, 'w', encoding='utf-8') as f:
            for i, pose in enumerate(submap_poses_Bs):
                pose_reshape = pose[:3, :4].reshape(1, 12)
                time = timestamps[i].reshape(-1, 1)
                save_info = np.hstack((time, pose_reshape))
                f.write(' '.join(str(x) for x in save_info.flatten()) + '\n')

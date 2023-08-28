# -*-coding:utf-8-*-
import numpy as np
import os

def txt2bin(data_path, seq, tra):
    x_path = data_path + "/" + seq + "/" + tra + "/" + "data_x.txt"
    y_path = data_path + "/" + seq + "/" + tra + "/" + "data_y.txt"
    z_path = data_path + "/" + seq + "/" + tra + "/" + "data_z.txt"
    i_path = data_path + "/" + seq + "/" + tra + "/" + "data_i.txt"

    # Inserted code
    with open(x_path, 'r') as x_file:
        x_data = x_file.readlines()

    with open(y_path, 'r') as y_file:
        y_data = y_file.readlines()

    with open(z_path, 'r') as z_file:
        z_data = z_file.readlines()

    with open(i_path, 'r') as i_file:
        i_data = i_file.readlines()

    for scan_id in range(len(x_data)):
        scan_xs = x_data[scan_id]
        scan_ys = y_data[scan_id]
        scan_zs = z_data[scan_id]
        scan_is = i_data[scan_id]

        scan_xs = scan_xs.split()
        scan_ys = scan_ys.split()
        scan_zs = scan_zs.split()
        scan_is = scan_is.split()

        scan_xs = [float(x) for x in scan_xs]
        scan_ys = [float(y) for y in scan_ys]
        scan_zs = [float(z) for z in scan_zs]
        scan_is = [float(i) for i in scan_is]

        scan_xs = np.array(scan_xs, dtype=np.float32)
        scan_ys = np.array(scan_ys, dtype=np.float32)
        scan_zs = np.array(scan_zs, dtype=np.float32)
        scan_is = np.array(scan_is, dtype=np.float32)
        scan_data = np.column_stack((scan_xs, scan_ys, scan_zs, scan_is))

        scan_name = str(scan_id).zfill(6)
        scan_forder = data_path + "/" + seq + "/" + tra + "/bins/"
        if not os.path.exists(scan_forder):
            os.makedirs(scan_forder)

        scan_path = scan_forder + scan_name + ".bin"
        with open(scan_path, 'wb') as bin_file:
            bin_file.write(scan_data)

    return scan_forder

# def farthest_point_sample(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     # 初始化一个centroids矩阵，用于存储npoint个采样点的索引位置，大小为B×npoint
#     # 其中B为BatchSize的个数
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#     # distance矩阵(B×N)记录某个batch中所有点到某一个点的距离，初始化的值很大，后面会迭代更新
#     distance = torch.ones(B, N).to(device) * 1e10
#     # farthest表示当前最远的点，也是随机初始化，范围为0~N，初始化B个；每个batch都随机有一个初始最远点
#     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
#     # batch_indices初始化为0~(B-1)的数组
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)
#     # 直到采样点达到npoint，否则进行如下迭代：
#     for i in range(npoint):
#         # 设当前的采样点centroids为当前的最远点farthest
#         centroids[:, i] = farthest
#         # 取出该中心点centroid的坐标
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
#         # 求出所有点到该centroid点的欧式距离，存在dist矩阵中
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值
#         # 随着迭代的继续，distance矩阵中的值会慢慢变小，
#         # 其相当于记录着某个Batch中每个点距离所有已出现的采样点的最小距离
#         mask = dist < distance#确保拿到的是距离所有已选中心点最大的距离。比如已经是中心的点，其dist始终保持为	 #0，二在它附近的点，也始终保持与这个中心点的距离
#         distance[mask] = dist[mask]
#         # 从distance矩阵取出最远的点为farthest，继续下一轮迭代
#         farthest = torch.max(distance, -1)[1]
#     return centroids










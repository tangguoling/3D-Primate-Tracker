# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:12:22 2026

@author: 11748
"""

import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dimention = 3
num_animal = 1
min_cluster_size = 100
num_keypoints = 21
umapdim = 2
random_int = 0
save_path = 'C:/m3/result/result_s_' + str(random_int) + '_' + str(min_cluster_size) + '/'
data_path = 'C:/m3/data_single/*.csv'
# In[]
def normalize_and_align_flattened_pose(data, center_idx, ref_pt1_idx, ref_pt2_idx):
    """
    对 shape=(n, t, k*2) 的关键点序列进行中心化 + 旋转矫正

    :param data: 原始数据, shape = (n_animals, t_frames, k*2)
    :param center_idx: 中心点索引（基于关键点数量）
    :param ref_pt1_idx: 用于旋转的点1索引（基于关键点数量）
    :param ref_pt2_idx: 用于旋转的点2索引（基于关键点数量）
    :return: 标准化后的数据，shape = (n, t, k*2)
    """
    if dimention == 2:
        print(data.shape)
        n, t, k2 = data.shape
        k = k2 // 2  # 关键点个数
        normalized_data = np.zeros_like(data)
    
        for i in range(n):
            for j in range(t):
                flat_frame = data[i, j]  # shape: (k*2,)
                points = flat_frame.reshape(k, 2)  # shape: (k, 2)
    
                # Step 1: 中心点归一化
                center = points[center_idx]
                centered = points - center  # 所有点减去中心点
    
                # Step 2: 旋转校准（将 ref_pt1 → ref_pt2 对齐到 x 轴）
                p1 = centered[ref_pt1_idx]
                p2 = centered[ref_pt2_idx]
                vec = p2 - p1
    
                angle = np.arctan2(vec[1], vec[0])
                rotation_matrix = np.array([
                    [np.cos(-angle), -np.sin(-angle)],
                    [np.sin(-angle),  np.cos(-angle)]
                ])
    
                rotated = centered @ rotation_matrix.T  # shape: (k, 2)
    
                # Step 3: reshape 回平铺形式
                normalized_data[i, j] = rotated.flatten()
    else:
        
        print(data.shape)
        n, t, k2 = data.shape
        k = k2 // 3  # 关键点个数
        normalized_data = np.zeros_like(data)
    
        for i in range(n):
            for j in range(t):
                flat_frame = data[i, j]  # shape: (k*2,)
                points = flat_frame.reshape(k, 3)  # shape: (k, 2)
                
                center = points[center_idx]
                centered = points - center
            
                # Step 2: 构造身体局部坐标轴
                z_axis = centered[ref_pt2_idx] - centered[ref_pt1_idx]  # 身体前后方向
                z_axis /= np.linalg.norm(z_axis)
            
                # 示例 x_axis：假设关键点 3 和 4 是左/右肩
                x_axis = centered[5] - centered[6]
                x_axis -= np.dot(x_axis, z_axis) * z_axis  # 去除 z 轴投影，保证垂直
                x_axis /= np.linalg.norm(x_axis)
            
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
            
                # Step 3: 构造旋转矩阵
                R = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape: (3, 3)
            
                # Step 4: 应用旋转
                rotated = centered @ R  # (k, 3)
                
                # Step 3: reshape 回平铺形式
                normalized_data[i, j] = rotated.flatten()
        
        
    return normalized_data

# 1. 加载数据（行为特征）——每一行是一个时间片的特征
def load_behavior_features(file_path, dimention= 3, velocity_window=5):
    # 这里以 CSV 为例，每行是一个时间窗口的速度、角度等特征
    df = pd.read_csv(file_path)
    if dimention == 2:
        data = df.values[3:, 1:]
        # if dimention == 2:
        data_xy = np.zeros((data.shape[0], int(2 * data.shape[1] / 3)))
        data_xy[:,::2] = data[:,::3]
        data_xy[:,1::2] = data[:,1::3]
        data = data_xy
    else:
        data = df.values[1:, :63]
        # if dimention == 2:
        data_xyz = np.zeros((data.shape[0], data.shape[1]))
        data_xyz[:,::3] = data[:,::3]
        data_xyz[:,1::3] = data[:,1::3]
        data_xyz[:,2::3] = data[:,2::3]
        data = data_xyz
    t, feat = data.shape
# 计算 5 帧平均速度：v[t] = (pos[t] - pos[t-velocity_window]) / velocity_window
    vel = np.zeros_like(data)
    vel[velocity_window:] = (data[velocity_window:] - data[:-velocity_window]) / velocity_window

# 拼接位置 + 速度 特征
    features = np.hstack([data, vel])
    return features
    # return data  # shape: (num_samples, num_features)
# In[]
# 2. 标准化
def normalize_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# In[]3. 降维（UMAP）
def reduce_dim_umap(X, n_neighbors=30, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state = 42 + random_int)
    X_umap = reducer.fit_transform(X)
    return X_umap

# In[]4. 聚类（HDBSCAN）
def cluster_behavior(X_embedded, min_cluster_size = 100):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
    cluster_selection_epsilon=0.5)
    labels = clusterer.fit_predict(X_embedded)
    return labels

# In[]5. 可视化

def plot_behavior_m3_2d(X_embedded, labels, save_path):
    length = int(len(X_embedded) / num_animal)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    sc1 = ax.scatter(X_embedded[:length, 0], X_embedded[:length, 1], c=labels[:length], marker = '*', cmap='Spectral', s=20)
    # plt.scatter(X_embedded[length::100, 0], X_embedded[length::100, 1], c=labels[length::100], marker = 'o', cmap='Spectral', s=20)
    mappable = sc1
    cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.7)
    cbar.set_label("Cluster ID")
    title = 'A'
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    # plt.set_xticks([])
    # plt.set_yticks([])
    # ax.axis('off')
    ax.grid(False)
    # ax.tight_layout()
    # plt.savefig(save_path + '_A' + '_cluster.png')
    
    
    # ax = fig.add_subplot(122)
    # # plt.scatter(X_embedded[:length:100, 0], X_embedded[:length:100, 1], c=labels[:length:100], marker = '*', cmap='Spectral', s=20)
    # sc2 = ax.scatter(X_embedded[length:, 0], X_embedded[length:, 1], c=labels[length:], marker = 'o', cmap='Spectral', s=20)
    # mappable = sc2
    # cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.7)
    # cbar.set_label("Cluster ID")
    # title = 'B'
    # ax.set_title(title)
    # ax.set_xlabel("UMAP-1")
    # ax.set_ylabel("UMAP-2")
    # # plt.set_xticks([])
    # # plt.set_yticks([])
    # # ax.axis('off')
    # ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    # In[]
# 6. 主流程
def build_behavior_umap(file_path,umapdim):
    print("Loading data...")
    X= []
    V= []
    for f in file_path:
        tempx = load_behavior_features(f)
        print(tempx.shape)
        X.append(tempx[:, :num_animal * num_keypoints * dimention].reshape(-1, tempx[:,
        :num_animal * num_keypoints * dimention].shape[-1] // num_animal))
        
        V.append(tempx[:, num_animal * num_keypoints * dimention:].reshape(-1, tempx[:,
        num_animal * num_keypoints * dimention:].shape[-1] // num_animal))
        # print(X[0][98:100],V[0][98:100])
    # X = X.reshape(-1, X.shape[-1])
    X = np.array(X)
    V = np.array(V)
    X = normalize_and_align_flattened_pose(X, center_idx=-4, ref_pt1_idx=-4, ref_pt2_idx=4)
    
    
    print("Normalizing features...")
    
    V = V.reshape(-1, V.shape[-1])
    X = X.reshape(-1, X.shape[-1])
    X = np.hstack([X,V])
    X_norm = normalize_features(X)

    print("Reducing dimensionality with UMAP...")
    X_embedded = reduce_dim_umap(X_norm, n_components = umapdim)

    print("Clustering with HDBSCAN...")
    # labels = cluster_behavior(X_embedded)

    print("Plotting behavior atlas...")

    return X_embedded#, labels
import glob
import os
# 🧪 示例调用
if __name__ == '__main__':
    # 示例 CSV: 每行是一个帧窗口的行为特征（例如速度、角度、体长、关键点距离）
    example_paths = glob.glob(data_path)
    # for example in example_paths:
    example_path = example_paths
    embedding = build_behavior_umap(example_path, umapdim)
    embedding_re = embedding.reshape(len(example_paths), embedding.shape[0] // len(example_paths), -1)
    # cluster_labels_re = cluster_labels.reshape(len(example_paths), cluster_labels.shape[0] // len(example_paths))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # min_cluster_size = 20
    for idx, exp in enumerate(example_path):
        exp_name = exp.split('\\')[-1]
        
        embedding = embedding_re[idx]
        
        cluster_labels = cluster_behavior(embedding, min_cluster_size)
        
        # plot_behavior_m3_2d(embedding, cluster_labels, save_path  + exp_name + "_" + str(min_cluster_size) + '_cluster.png')
        
        DF = pd.DataFrame(embedding.reshape(embedding.shape[0] // num_animal, -1))
    
        DF.to_csv(save_path + exp_name + '_embedding2.csv')
        
        DF = pd.DataFrame(cluster_labels.reshape( cluster_labels.shape[0] // num_animal, -1))
    
        DF.to_csv(save_path + exp_name + "_" + str(min_cluster_size) + '_' + '_cluster_labels2.csv')
# In[] plot
    for idx, exp in enumerate(example_path):
        exp_name = exp.split('\\')[-1]
        
        embedding = pd.read_csv(save_path + exp_name + '_embedding2.csv').values[:,1:].reshape(-1, umapdim)
        
        cluster_labels = pd.read_csv(save_path + exp_name + "_" + str(min_cluster_size) + '_' + '_cluster_labels2.csv').values[:,1:].reshape(-1, 1)
        
        plot_behavior_m3_2d(embedding, cluster_labels, save_path  + exp_name + "_" + str(min_cluster_size) + '_cluster.png')
    # In[]
    file_list_control = glob.glob(save_path + '*orma*' + "_" + str(min_cluster_size) + '__cluster_labels2.csv')
    file_list_DXXX = glob.glob(save_path + '*DISC1*' + "_" + str(min_cluster_size) + '__cluster_labels2.csv')
    from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu
# 2. 遍历每个文件，统计两只猴子的 motif 数量
    motif_counts_control = []  # 每行 [猴子0的motif数, 猴子1的motif数]
    for fp in sorted(file_list_control):
        df = pd.read_csv(fp)
        # 假设 CSV 有两列，分别对应猴子0、猴子1
        cols = df.columns[1]
        counts = [df[col].nunique() for col in cols]
        motif_counts_control.append(counts)
    
    motif_counts_control = np.array(motif_counts_control)
    
    motif_counts_DXXX = []  # 每行 [猴子0的motif数, 猴子1的motif数]
    for fp in sorted(file_list_DXXX):
        df = pd.read_csv(fp)
        # 假设 CSV 有两列，分别对应猴子0、猴子1
        cols = df.columns[1]
        counts = [df[col].nunique() for col in cols]
        motif_counts_DXXX.append(counts)
    
    motif_counts_DXXX = np.array(motif_counts_DXXX)
    # m0 = motif_counts[:, 0]
    # m1 = motif_counts[:, 1]
    
    # 3. 配对 t-test
    t_stat, p_t = ttest_ind(motif_counts_control, motif_counts_DXXX)
    
    # 4. 配对 Wilcoxon 符号秩检验（非参数）
    w_stat, p_w = mannwhitneyu(motif_counts_control, motif_counts_DXXX, alternative='two-sided')
    
    # 5. 输出结果
    # print("每个实验两只猴子的 motif 数量：")
    # for idx, (c0, c1) in enumerate(motif_counts, 1):
        # print(f"  实验{idx:02d}: 猴子0 = {c0}, 猴子1 = {c1}")
    
    print(f"\n配对 t-test: t = {t_stat[0]:.3f}, p = {p_t[0]:.4f}")
    print(f"Mann-Whitney U:    W = {w_stat[0]:.3f}, p = {p_w[0]:.4f}")
    t_stat, p_value = mannwhitneyu(motif_counts_DXXX[:,0], motif_counts_control[:,0], alternative='two-sided')

    # Determine significance annotation
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'n.s.'

    # Create boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [motif_counts_DXXX[:,0], motif_counts_control[:,0]]
    box = ax.boxplot(data, labels=['DISC1', 'Control',], patch_artist=True,
        boxprops={'facecolor':'lightblue','zorder':1},
        whiskerprops={'zorder':1},
        capprops={'zorder':1},
        medianprops={'zorder':2})

    # Customize box colors
    colors = ['#C00000', '#002060']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    # Annotate significance
    # y_max = max(motif_counts_DXXX.max(), motif_counts_control.max())
    # y, h, col = y_max + 10, 5, 'k'
    # ax.plot([1, 1, 2, 2], [y, y+h, y+h, y], lw=1.5, c=col)
    # ax.text(1.5, y+h+2, significance, ha='center', va='bottom', color=col, fontsize=12)
    markers = ['^', 'o']
    # 绘制单点（散点抖动）
    for i, d in enumerate(data, start=1):
        x = np.random.normal(i, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.7, color='black', facecolors='none', edgecolors = 'black', linewidths=2 ,marker = markers[i - 1], zorder=3)
        
    # Labels and title
    ax.set_ylabel('Number of Behavior Motifs')
    ax.set_title('Comparison of Motif Counts between Partner Monkey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    data = [p_t[0], p_w[0]]
    np.savetxt(save_path + 'p_value.txt', data)
    plt.tight_layout()
    plt.show()
    
# In[]

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载所有 embedding 数据
def load_embeddings(file_list):
    coords_list = []
    frame_indices_list = []
    for fp in file_list:
        df = pd.read_csv(fp)
        # 假定列名为 'umap_1', 'umap_2'
        coords = df[['0', '1']].values
        coords_list.append(coords)
        frame_indices_list.append(np.arange(len(df)))
    all_coords = np.vstack(coords_list)
    return all_coords, coords_list, frame_indices_list

# 文件模式
control_files = sorted(glob.glob('C:/m3/result/result_s_0_100/*ormal*_embedding2.csv'))
dxxx_files    = sorted(glob.glob('C:/m3/result/result_s_0_100/*DISC1*_embedding2.csv'))

# 加载数据
ctrl_all, ctrl_list, ctrl_inds = load_embeddings(control_files)
dxxx_all, dxxx_list, dxxx_inds = load_embeddings(dxxx_files)

# 2. 计算 2D 密度直方图
bins = 50
xmin = min(ctrl_all[:,0].min(), dxxx_all[:,0].min())
xmax = max(ctrl_all[:,0].max(), dxxx_all[:,0].max())
ymin = min(ctrl_all[:,1].min(), dxxx_all[:,1].min())
ymax = max(ctrl_all[:,1].max(), dxxx_all[:,1].max())
xedges = np.linspace(xmin, xmax, bins+1)
yedges = np.linspace(ymin, ymax, bins+1)

ctrl_hist, _, _ = np.histogram2d(ctrl_all[:,0], ctrl_all[:,1], bins=[xedges, yedges], density=True)
dxxx_hist, _, _ = np.histogram2d(dxxx_all[:,0], dxxx_all[:,1], bins=[xedges, yedges], density=True)

from scipy.ndimage import gaussian_filter

dxxx_hist = gaussian_filter(dxxx_hist, sigma=2)
ctrl_hist = gaussian_filter(ctrl_hist, sigma=2)
# density_map = dxxx_hist
# 3. 可视化热图
fig, axs = plt.subplots(1, 3, figsize=(30, 10))
sns.heatmap(ctrl_hist.T, ax=axs[0], cmap='viridis', cbar=False)
axs[0].set_title('Feature Density (Control)')
sns.heatmap(dxxx_hist.T, ax=axs[1], cmap='viridis', cbar=False)
axs[1].set_title('Feature Density (DISC1)')
diff = ctrl_hist - dxxx_hist
sns.heatmap(diff.T, ax=axs[2], cmap='coolwarm', center=0)
axs[2].set_title('Feature Density Difference (Control - DISC1)')

# circle1 = plt.Circle(
#     (5.221786022186279+25, 0.3188595175743103+25),  # 圆心
#     17.460119247436523,  # 半径
#     color='green',
#     alpha=0.1
# )

# circle2 = plt.Circle(
#     (5.531402587+25, 0.02870178222572+25),  # 圆心
#     17.559631347656,  # 半径
#     color='red',
#     alpha=0.1
# )
# # axs[2].add_artist(circle1)
# axs[2].add_artist(circle2)
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()
# In[]
import cv2
from scipy.ndimage import gaussian_filter
density_map = dxxx_hist.T[::-1,:]

smoothed_density = density_map

# 找出密度最高的点
max_density = np.max(smoothed_density)
max_positions = np.argwhere(smoothed_density > 0.20 * max_density)

# 如果没有找到高密度点，使用所有点
if len(max_positions) == 0:
    max_positions = np.argwhere(smoothed_density > 0)

# 转换为 (x, y) 格式的点集
points = max_positions[:, [1, 0]].astype(np.float32)  # 注意：OpenCV使用(x,y)格式

# 计算最小外接圆
if len(points) > 0:
    if len(points) == 1:
        center = tuple(points[0])
        radius = 1.0  # 单个点默认半径
    elif len(points) == 2:
        center = tuple(np.mean(points, axis=0))
        radius = np.linalg.norm(points[0] - points[1]) / 2
    else:
        center, radius = cv2.minEnclosingCircle(points)
else:
    center = (25, 25)  # 默认中心
    radius = 5  # 默认半径

# 可视化
plt.figure(figsize=(30, 10))


plt.subplot(132)
plt.axis('off')

plt.imshow(smoothed_density, cmap='viridis', origin='lower')

# 绘制高密度点
plt.scatter(points[:, 0], points[:, 1], 
            s=10, c='red', alpha=0.7, label='High Density Points')

# 绘制最小外接圆
circle = plt.Circle(center, radius, 
                   edgecolor='red', 
                   facecolor='none', 
                   linewidth=2, 
                   linestyle='--',
                   label=f'Center: ({center[0]:.1f}, {center[1]:.1f})\nRadius: {radius:.1f}')
plt.gca().add_patch(circle)

# 绘制圆心
plt.scatter([center[0]], [center[1]], 
            s=100, c='red', marker='x', 
            label='Center')

plt.title('Feature Density (DISC1) with Enclosing Circle')
plt.legend()
plt.tight_layout()

# 显示中心坐标和半径
print(f"高密度区域中心: ({center[0]:.2f}, {center[1]:.2f})")
print(f"半径: {radius:.2f}")

plt.subplot(133)

# plt.colorbar()

# 绘制高密度点
plt.scatter(points[:, 0], points[:, 1], 
            s=10, c='red', alpha=0.7, label='High Density Points')

# 绘制最小外接圆
circle = plt.Circle(center, radius, 
                   edgecolor='red', 
                   facecolor='none', 
                   linewidth=2, 
                   linestyle='--',
                   label=f'Center: ({center[0]:.1f}, {center[1]:.1f})\nRadius: {radius:.1f}')
plt.gca().add_patch(circle)

# 绘制圆心
plt.scatter([center[0]], [center[1]], 
            s=100, c='red', marker='x', 
            label='Center')

plt.title('Smoothed Density with Enclosing Circle')
plt.legend()
plt.tight_layout()
plt.axis('off')
density_map = ctrl_hist.T[::-1,:]

smoothed_density = density_map

# 找出密度最高的点
max_density = np.max(smoothed_density)
max_positions = np.argwhere(smoothed_density > 0.20 * max_density)

# 如果没有找到高密度点，使用所有点
if len(max_positions) == 0:
    max_positions = np.argwhere(smoothed_density > 0)

# 转换为 (x, y) 格式的点集
points = max_positions[:, [1, 0]].astype(np.float32)  # 注意：OpenCV使用(x,y)格式

# 计算最小外接圆
if len(points) > 0:
    if len(points) == 1:
        center = tuple(points[0])
        radius = 1.0  # 单个点默认半径
    elif len(points) == 2:
        center = tuple(np.mean(points, axis=0))
        radius = np.linalg.norm(points[0] - points[1]) / 2
    else:
        center, radius = cv2.minEnclosingCircle(points)
else:
    center = (25, 25)  # 默认中心
    radius = 5  # 默认半径

# 可视化
# plt.figure(figsize=(10, 8))
plt.subplot(131)
plt.axis('off')

plt.imshow(smoothed_density, cmap='viridis', origin='lower')
# plt.colorbar()

# 绘制高密度点
plt.scatter(points[:, 0], points[:, 1], 
            s=10, c='green', alpha=0.7, label='High Density Points')

# 绘制最小外接圆
circle = plt.Circle(center, radius, 
                   edgecolor='green', 
                   facecolor='none', 
                   linewidth=2, 
                   linestyle='--',
                   label=f'Center: ({center[0]:.1f}, {center[1]:.1f})\nRadius: {radius:.1f}')
plt.gca().add_patch(circle)

# 绘制圆心
plt.scatter([center[0]], [center[1]], 
            s=100, c='green', marker='x', 
            label='Center')

plt.title('Feature Density (Control) with Enclosing Circle')
plt.legend()
plt.tight_layout()

plt.subplot(133)

plt.colorbar()
plt.scatter(points[:, 0], points[:, 1], 
            s=10, c='green', alpha=0.7, label='High Density Points')

# 绘制最小外接圆
circle = plt.Circle(center, radius, 
                   edgecolor='green', 
                   facecolor='none', 
                   linewidth=2, 
                   linestyle='--',
                   label=f'Center: ({center[0]:.1f}, {center[1]:.1f})\nRadius: {radius:.1f}')
plt.gca().add_patch(circle)

# 绘制圆心
plt.scatter([center[0]], [center[1]], 
            s=100, c='green', marker='x', 
            label='Center')

plt.title('Density Comparison with Enclosing Circle')
plt.legend()
plt.tight_layout()
# 显示中心坐标和半径
print(f"高密度区域中心: ({center[0]:.2f}, {center[1]:.2f})")
print(f"半径: {radius:.2f}")

# for ax in axs:
#     ax.axis('off')
# plt.tight_layout()
plt.show()# In[]4. 定位显著减少区域

threshold = 0.01  # 下5%区域
sig_bins = np.argwhere(abs(diff) > threshold)

records = []
for (i, j) in sig_bins:
    x_low, x_high = xedges[i], xedges[i+1]
    y_low, y_high = yedges[j], yedges[j+1]
    for coords, inds, fp in zip(dxxx_list, dxxx_inds, dxxx_files):
        mask = (
            (coords[:,0] >= x_low) & (coords[:,0] < x_high) &
            (coords[:,1] >= y_low) & (coords[:,1] < y_high)
        )
        if mask.any():
            frames = inds[mask]
            records.append({
                'file': fp.split('\\')[-1],
                'bin_coord': f'({i},{j})',
                'frame_indices': frames.tolist(),
                'density': diff[i,j]
            })

    for coords, inds, fp in zip(ctrl_list, ctrl_inds, control_files):
        mask = (
            (coords[:,0] >= x_low) & (coords[:,0] < x_high) &
            (coords[:,1] >= y_low) & (coords[:,1] < y_high)
        )
        if mask.any():
            frames = inds[mask]
            records.append({
                'file': fp.split('\\')[-1],
                'bin_coord': f'({i},{j})',
                'frame_indices': frames.tolist(),
                'density': diff[i,j]
            })
# frame_df = pd.DataFrame(records)
# import ace_tools as tools; tools.display_dataframe_to_user(name="Significant Reduced Regions", dataframe=frame_df)

# records = []
# for (x, y), fp, fr in zip(all_coords, file_inds, frame_inds):
#     # bin 索引
#     i = np.searchsorted(xedges, x) - 1
#     j = np.searchsorted(yedges, y) - 1
#     if 0 <= i < hist.shape[0] and 0 <= j < hist.shape[1]:
#         if hist[i,j] > 0.005:
#             records.append({
#                 'file': fp.split('\\')[-1],
#                 'frame': fr,
#                 'umap_x': x,
#                 'umap_y': y,
#                 'density': hist[i,j]
#             })

# # 5. 保存结果到 CSV
result_df = pd.DataFrame(records)
result_df.to_csv('C:/wjh/singelmonkey/result/Control - DXXX.csv', index=False)


# In[]
# import glob
# from scipy.ndimage import gaussian_filter
control_ = []
dxxx_ = []
control_x = []
dxxx_x= []
control_y = []
dxxx_y = []
def load_embeddings(file_list):
    coords_list = []
    frame_indices_list = []
    for fp in file_list:
        df = pd.read_csv(fp)
        # 假定列名为 'umap_1', 'umap_2'
        coords = df[['0', '1']].values
        coords_list.append(coords)
        frame_indices_list.append(np.arange(len(df)))
    all_coords = np.vstack(coords_list)
    return all_coords, coords_list, frame_indices_list

# 文件模式
control_files = sorted(glob.glob('C:/m3/result/result_s_0_100/*ormal*_embedding2.csv'))
dxxx_files    = sorted(glob.glob('C:/m3/result/result_s_0_100/*DISC1*_embedding2.csv'))
for i in range(len(control_files)):
    ctrl_all, ctrl_list, ctrl_inds = load_embeddings([control_files[i]])
    dxxx_all, dxxx_list, dxxx_inds = load_embeddings([dxxx_files[i]])
    
    dxxx_hist = gaussian_filter(dxxx_hist, sigma=2)
    ctrl_hist = gaussian_filter(ctrl_hist, sigma=2)
    
    bins = 50
    xmin = min(ctrl_all[:,0].min(), dxxx_all[:,0].min())
    xmax = max(ctrl_all[:,0].max(), dxxx_all[:,0].max())
    ymin = min(ctrl_all[:,1].min(), dxxx_all[:,1].min())
    ymax = max(ctrl_all[:,1].max(), dxxx_all[:,1].max())
    xedges = np.linspace(xmin, xmax, bins+1)
    yedges = np.linspace(ymin, ymax, bins+1)
    
    ctrl_hist, _, _ = np.histogram2d(ctrl_all[:,0], ctrl_all[:,1], bins=[xedges, yedges], density=True)
    dxxx_hist, _, _ = np.histogram2d(dxxx_all[:,0], dxxx_all[:,1], bins=[xedges, yedges], density=True)
    
    
    dxxx_hist = gaussian_filter(dxxx_hist, sigma=2)
    ctrl_hist = gaussian_filter(ctrl_hist, sigma=2)
    
    # from scipy.ndimage import gaussian_filter
    density_map = dxxx_hist.T[::-1,:]
    
    smoothed_density = density_map
    
    # 找出密度最高的点
    max_density = np.max(smoothed_density)
    max_positions = np.argwhere(smoothed_density > 0.20 * max_density)
    
    # 如果没有找到高密度点，使用所有点
    if len(max_positions) == 0:
        max_positions = np.argwhere(smoothed_density > 0)
    
    # 转换为 (x, y) 格式的点集
    points = max_positions[:, [1, 0]].astype(np.float32)  # 注意：OpenCV使用(x,y)格式
    
    # 计算最小外接圆
    if len(points) > 0:
        if len(points) == 1:
            center = tuple(points[0])
            radius = 1.0  # 单个点默认半径
        elif len(points) == 2:
            center = tuple(np.mean(points, axis=0))
            radius = np.linalg.norm(points[0] - points[1]) / 2
        else:
            center, radius = cv2.minEnclosingCircle(points)
    else:
        center = (25, 25)  # 默认中心
        radius = 5  # 默认半径
    
    # 可视化
    plt.figure(figsize=(10, 10))
    
    
    # plt.subplot(132)
    plt.axis('off')
    
    plt.imshow(smoothed_density, cmap='viridis', origin='lower')
    
    # 绘制高密度点
    plt.scatter(points[:, 0], points[:, 1], 
                s=10, c='red', alpha=0.7, label='High Density Points')
    
    # 绘制最小外接圆
    circle = plt.Circle(center, radius, 
                       edgecolor='red', 
                       facecolor='none', 
                       linewidth=2, 
                       linestyle='--',
                       label=f'Center: ({center[0]:.1f}, {center[1]:.1f})\nRadius: {radius:.1f}')
    plt.gca().add_patch(circle)
    
    # 绘制圆心
    plt.scatter([center[0]], [center[1]], 
                s=100, c='red', marker='x', 
                label='Center')
    
    plt.title('Feature Density (DISC1) with Enclosing Circle')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path + 'DISC1_' + str(i) + '.png')
    # 显示中心坐标和半径
    print(f"高密度区域中心: ({center[0]:.2f}, {center[1]:.2f})")
    print(f"半径: {radius:.2f}")
    dxxx_.append(radius)
    dxxx_x.append(center[0])
    dxxx_y.append(center[1])
    density_map = ctrl_hist.T[::-1,:]
    
    smoothed_density = density_map
    
    # 找出密度最高的点
    max_density = np.max(smoothed_density)
    max_positions = np.argwhere(smoothed_density > 0.20 * max_density)
    
    # 如果没有找到高密度点，使用所有点
    if len(max_positions) == 0:
        max_positions = np.argwhere(smoothed_density > 0)
    
    # 转换为 (x, y) 格式的点集
    points = max_positions[:, [1, 0]].astype(np.float32)  # 注意：OpenCV使用(x,y)格式
    
    # 计算最小外接圆
    if len(points) > 0:
        if len(points) == 1:
            center = tuple(points[0])
            radius = 1.0  # 单个点默认半径
        elif len(points) == 2:
            center = tuple(np.mean(points, axis=0))
            radius = np.linalg.norm(points[0] - points[1]) / 2
        else:
            center, radius = cv2.minEnclosingCircle(points)
    else:
        center = (25, 25)  # 默认中心
        radius = 5  # 默认半径
        plt.show()
    
    # 可视化
    plt.figure(figsize=(10, 10))
    # plt.subplot(131)
    plt.axis('off')
    
    plt.imshow(smoothed_density, cmap='viridis', origin='lower')
    # plt.colorbar()
    
    # 绘制高密度点
    plt.scatter(points[:, 0], points[:, 1], 
                s=10, c='green', alpha=0.7, label='High Density Points')
    
    # 绘制最小外接圆
    circle = plt.Circle(center, radius, 
                       edgecolor='green', 
                       facecolor='none', 
                       linewidth=2, 
                       linestyle='--',
                       label=f'Center: ({center[0]:.1f}, {center[1]:.1f})\nRadius: {radius:.1f}')
    plt.gca().add_patch(circle)
    
    # 绘制圆心
    plt.scatter([center[0]], [center[1]], 
                s=100, c='green', marker='x', 
                label='Center')
    
    plt.title('Feature Density (Control) with Enclosing Circle')
    plt.legend()
    plt.tight_layout()
    
    print(f"高密度区域中心: ({center[0]:.2f}, {center[1]:.2f})")
    print(f"半径: {radius:.2f}")
    
    control_.append(radius)
    control_x.append(center[0])
    control_y.append(center[1])
    plt.show()
    plt.savefig(save_path + 'Con_' + str(i) + '.png')
print(np.mean(dxxx_),np.mean(control_),np.std(dxxx_),np.std(control_))
print(np.mean(dxxx_x),np.mean(control_x),np.std(dxxx_x),np.std(control_x))
print(np.mean(dxxx_y),np.mean(control_y),np.std(dxxx_y),np.std(control_y))
print(ttest_ind(dxxx_, control_))
print(ttest_ind(dxxx_x, control_x))
print(ttest_ind(dxxx_y, control_y))

# In[]
t_stat, p_value = mannwhitneyu(dxxx_, control_, alternative='two-sided')

# Determine significance annotation
if p_value < 0.001:
    significance = '***'
elif p_value < 0.01:
    significance = '**'
elif p_value < 0.05:
    significance = '*'
else:
    significance = 'n.s.'

# Create boxplot
fig, ax = plt.subplots(figsize=(8, 5))
data = [dxxx_, control_]
box = ax.boxplot(data, labels=['DISC1', 'Control',], patch_artist=True, showfliers = False,
    boxprops={'facecolor':'lightblue','zorder':1},
    whiskerprops={'zorder':1},
    capprops={'zorder':1},
    medianprops={'zorder':2})

# Customize box colors
colors = ['#C00000', '#002060']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# Annotate significance
# y_max = max(motif_counts_DXXX.max(), motif_counts_control.max())
# y, h, col = y_max + 10, 5, 'k'
# ax.plot([1, 1, 2, 2], [y, y+h, y+h, y], lw=1.5, c=col)
# ax.text(1.5, y+h+2, significance, ha='center', va='bottom', color=col, fontsize=12)
markers = ['^', 'o']
# 绘制单点（散点抖动）
for i, d in enumerate(data, start=1):
    x = np.random.normal(i, 0.04, size=len(d))
    ax.scatter(x, d, alpha=0.7, color='black', facecolors='none', edgecolors = 'black', linewidths=2 ,marker = markers[i - 1], zorder=3)
    
# Labels and title
ax.set_ylabel('Radius')
ax.set_title('Comparison of Principal Radius')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

data = [p_t[0], p_w[0]]
# np.savetxt(save_path + 'p_value.txt', data)
plt.tight_layout()
plt.show()
# In[]
t_stat, p_value = mannwhitneyu(dxxx_x, control_x, alternative='two-sided')

# Determine significance annotation
if p_value < 0.001:
    significance = '***'
elif p_value < 0.01:
    significance = '**'
elif p_value < 0.05:
    significance = '*'
else:
    significance = 'n.s.'

# Create boxplot
fig, ax = plt.subplots(figsize=(8, 5))
data = [dxxx_x, control_x]
box = ax.boxplot(data, labels=['DISC1', 'Control',], patch_artist=True, showfliers = False,
    boxprops={'facecolor':'lightblue','zorder':1},
    whiskerprops={'zorder':1},
    capprops={'zorder':1},
    medianprops={'zorder':2})

# Customize box colors
colors = ['#C00000', '#002060']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
# Annotate significance
# y_max = max(motif_counts_DXXX.max(), motif_counts_control.max())
# y, h, col = y_max + 10, 5, 'k'
# ax.plot([1, 1, 2, 2], [y, y+h, y+h, y], lw=1.5, c=col)
# ax.text(1.5, y+h+2, significance, ha='center', va='bottom', color=col, fontsize=12)
markers = ['^', 'o']
# 绘制单点（散点抖动）
for i, d in enumerate(data, start=1):
    x = np.random.normal(i, 0.04, size=len(d))
    ax.scatter(x, d, alpha=0.7, color='black', facecolors='none', edgecolors = 'black', linewidths=2 ,marker = markers[i - 1], zorder=3)
    
# Labels and title
ax.set_ylabel('Y, Coordination')
ax.set_title('Comparison of Centroids')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

data = [p_t[0], p_w[0]]
# np.savetxt(save_path + 'p_value.txt', data)
plt.tight_layout()
plt.show()
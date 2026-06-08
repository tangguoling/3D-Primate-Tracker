# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:32:22 2026

@author: 11748
"""


import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dimention = 3
num_animal = 2
min_cluster_size = 100
num_keypoints = 21
umapdim = 2
random_int = 4
save_path = 'C:/m3/result/result_d_' + str(random_int) + '_' + str(min_cluster_size) + '/'
data_path = 'C:/m3/data_pair/*.csv'
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
        data = df.values[1:, :-1]
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
    
    fig = plt.figure(figsize=(21, 9))
    ax = fig.add_subplot(121)
    sc1 = ax.scatter(X_embedded[:length:100, 0], X_embedded[:length:100, 1], c=labels[:length:100], marker = '*', cmap='Spectral', s=20)
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
    
    
    ax = fig.add_subplot(122)
    # plt.scatter(X_embedded[:length:100, 0], X_embedded[:length:100, 1], c=labels[:length:100], marker = '*', cmap='Spectral', s=20)
    sc2 = ax.scatter(X_embedded[length::100, 0], X_embedded[length::100, 1], c=labels[length::100], marker = 'o', cmap='Spectral', s=20)
    mappable = sc2
    cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.7)
    cbar.set_label("Cluster ID")
    title = 'B'
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    # plt.set_xticks([])
    # plt.set_yticks([])
    # ax.axis('off')
    ax.grid(False)
    
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
    # In[]
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
    
    # pd.save(save_path + 'cluster_labels.csv', embedding)
    # In[]
    file_list_control = glob.glob(save_path + '*ontrol*' + "_" + str(100) + '__cluster_labels2.csv')
    file_list_DXXX = glob.glob(save_path + '*DXXX*' + "_" + str(100) + '__cluster_labels2.csv')
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
    
    motif_counts_control_partner = []  # 每行 [猴子0的motif数, 猴子1的motif数]
    for fp in sorted(file_list_control):
        df = pd.read_csv(fp)
        # 假设 CSV 有两列，分别对应猴子0、猴子1
        cols = df.columns[2]
        counts = [df[col].nunique() for col in cols]
        motif_counts_control_partner.append(counts)
    
    motif_counts_control_partner = np.array(motif_counts_control_partner)
    
    motif_counts_DXXX_partner = []  # 每行 [猴子0的motif数, 猴子1的motif数]
    for fp in sorted(file_list_DXXX):
        df = pd.read_csv(fp)
        # 假设 CSV 有两列，分别对应猴子0、猴子1
        cols = df.columns[2]
        counts = [df[col].nunique() for col in cols]
        motif_counts_DXXX_partner.append(counts)
    
    motif_counts_DXXX_partner = np.array(motif_counts_DXXX_partner)
    
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
    data = [motif_counts_DXXX[:,0], motif_counts_DXXX_partner[:,0], motif_counts_control[:,0], motif_counts_control_partner[:,0]]
    box = ax.boxplot(data, labels=['DISC1','DISC1, partner', 'Control','Control, partner'], patch_artist=True,
        boxprops={'facecolor':'lightblue','zorder':1},
        whiskerprops={'zorder':1},
        capprops={'zorder':1},
        medianprops={'zorder':2})

    # Customize box colors
    colors = ['#C00000', '#385723', '#002060', '#002060']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    # Annotate significance
    # y_max = max(motif_counts_DXXX.max(), motif_counts_control.max())
    # y, h, col = y_max + 10, 5, 'k'
    # ax.plot([1, 1, 2, 2], [y, y+h, y+h, y], lw=1.5, c=col)
    # ax.text(1.5, y+h+2, significance, ha='center', va='bottom', color=col, fontsize=12)
    markers = ['^', 'o', 's', 'o']
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
    
    def compute_synchrony_rate_windowed(A, B, window=200, step=50):
        """
        A, B: 两只猴的motif序列（长度相同的一维array）
        window: 时间窗长度（frames）
        step: 滑动步长
    
        返回：
        - synchrony_rate: 每个window中匹配比例的平均值
        - per_window_rates: 每个window的匹配比例（可用于画分布）
        """
    
        assert len(A) == len(B), "A and B must have the same length"
        n = len(A)
    
        rates = []
    
        # 滑动窗口
        for start in range(0, n - window + 1, step):
            end = start + window
    
            A_win = A[start:end]
            B_win = B[start:end]
    
            # 这个window内的匹配比例
            match_rate = np.mean(A_win == B_win)
            rates.append(match_rate)
    
        rates = np.array(rates)
    
        # session-level同步率（你论文用的量）
        synchrony_rate = np.mean(rates)
    
        return synchrony_rate, rates
    
    file_list_control = glob.glob(save_path + '*ontrol*' + "_" + str(min_cluster_size) + '__cluster_labels2.csv')
    file_list_DXXX = glob.glob(save_path + '*DXXX*' + "_" + str(min_cluster_size) + '__cluster_labels2.csv')
    
    records = []
    
    for group, file_list in [('Control', file_list_control), ('DXXX', file_list_DXXX)]:
        for fp in file_list:
            df = pd.read_csv(fp)
            A = df.iloc[:, 1].values
            B = df.iloc[:, 2].values
            
            sync_rate, _ = compute_synchrony_rate_windowed(A, B)
            
            records.append({
                'group': group,
                'file': fp.split('/')[-1],
                'sync_rate': sync_rate,
            })

    metrics_df = pd.DataFrame(records)
    
    # Paired tests
    for metric in ['sync_rate']:
        ctrl = metrics_df[metrics_df.group=='Control'][metric].values
        dxxx = metrics_df[metrics_df.group=='DXXX'][metric].values
        t_stat, p_value = mannwhitneyu(ctrl, dxxx, alternative='two-sided')
        w_stat, p_w = wilcoxon(ctrl, dxxx)
        # print(f"{metric}: paired t-test p={p_t:.4f}, Wilcoxon p={p_w:.4f}")
    
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'n.s.'
        
    # Boxplot for sync_rate
    plt.figure(figsize=(6, 4))
    metrics_df.boxplot(column='sync_rate', by='group')
    plt.title('Synchrony Rate by Group')
    plt.suptitle('')
    plt.xlabel('')
    plt.ylabel('Synchrony Rate')
    plt.tight_layout()
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(6, 6))
    box = ax.boxplot([dxxx, ctrl], labels=['DXXX', 'Control'], patch_artist=True)
    
    # 设置箱体颜色
    colors = ['#C00000', '#002060']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    markers = ['^', 's']
    # 绘制单点（散点抖动）
    for i, d in enumerate([dxxx, ctrl], start=1):
        x = np.random.normal(i, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.7, color='black', facecolors='none', edgecolors = 'black', linewidths=2 ,marker = markers[i - 1], zorder=3)
    
    # 注释显著性连线
    y_max = max(dxxx.max(), ctrl.max())
    h = (dxxx.max() - dxxx.min()) * 0.1
    y = y_max + h
    ax.plot([1, 1, 2, 2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text(1.5, y+1.2*h, significance, ha='center', va='bottom', color='k', fontsize=12)
    
    # 去除上、右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 坐标和标题
    ax.set_ylabel('Number of Behavior Motifs')
    ax.set_title(f'Synchrony Rate by Group (p = {p_value:.3f})')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path + 'syn_rate.png')
# In[]
    file_list_control = glob.glob(save_path + '*ontrol*' + "_" + str(min_cluster_size) + '__cluster_labels2.csv')
    file_list_DXXX = glob.glob(save_path + '*DXXX*' + "_" + str(min_cluster_size) + '__cluster_labels2.csv')
    def compute_synchrony_with_shuffle(A, B, window=200, step=50, n_iter=1000):
        obs, _ = compute_synchrony_rate_windowed(A, B, window, step)
    
        null_vals = []
    
        for _ in range(n_iter):
            B_shuff = np.random.permutation(B)
            val, _ = compute_synchrony_rate_windowed(A, B_shuff, window, step)
            null_vals.append(val)
    
        null_vals = np.array(null_vals)
    
        p_value = np.mean(null_vals >= obs)
    
        return obs, null_vals, p_value
    
    for group, file_list in [('Control', file_list_control), ('DXXX', file_list_DXXX)]:
        for fp in file_list:
            df = pd.read_csv(fp)
            A = df.iloc[:, 1].values
            B = df.iloc[:, 2].values
            # print(group,fp)
            print(group, compute_synchrony_with_shuffle(A, B)[0], compute_synchrony_with_shuffle(A, B)[2])
            
# In[]

    import glob
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from scipy.spatial import ConvexHull
    from sklearn.neighbors import NearestNeighbors
    
    save_path = 'C:/m3/result/result_d_' + str(0) + '_' + str(100) + '/'
    # =========================
    # 1. 参数设置
    # =========================
    
    control_files = sorted(glob.glob(save_path + '*ontrol*_embedding2.csv'))
    disc1_files   = sorted(glob.glob(save_path + '*DXXX*_embedding2.csv'))
    
    save_dir = save_path + '2d_umap_results'
    os.makedirs(save_dir, exist_ok=True)
    
    CORE_PERCENTILE = 80      # 保留密度最高20%的点
    K_NEIGHBORS = 20
    MIN_POINTS_FOR_HULL = 5
    
    
    # =========================
    # 2. 数据读取
    # =========================
    
    def load_2d_embeddings(file_list):
        """
        每个csv:
        columns 0,1 = subject monkey
        columns 2,3 = partner monkey
        """
        subject_list = []
        partner_list = []
        file_names = []
    
        for fp in file_list:
            df = pd.read_csv(fp)
    
            subject_xy = df[['0', '1']].values
            partner_xy = df[['2', '3']].values
    
            subject_list.append(subject_xy)
            partner_list.append(partner_xy)
            file_names.append(os.path.basename(fp))
    
        return subject_list, partner_list, file_names
    
    
    ctrl_subject_list, ctrl_partner_list, ctrl_names = load_2d_embeddings(control_files)
    disc_subject_list, disc_partner_list, disc_names = load_2d_embeddings(disc1_files)
    
    
    # =========================
    # 3. 核心分析函数
    # =========================
    
    def estimate_density_knn(points, k=20):
        """
        使用KNN距离估计局部密度。
        距离越小，密度越高。
        """
        points = np.asarray(points)
    
        if len(points) <= k:
            k = max(2, len(points) - 1)
    
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(points)
    
        distances, _ = nbrs.kneighbors(points)
        kth_dist = distances[:, -1]
    
        density = 1.0 / (kth_dist + 1e-8)
    
        return density
    
    
    def extract_core_points(points, percentile=80, k=20):
        """
        提取高密度核心点。
        percentile=80 表示保留密度最高20%的点。
        """
        density = estimate_density_knn(points, k=k)
        threshold = np.percentile(density, percentile)
        core_points = points[density >= threshold]
    
        return core_points, density, threshold
    
    
    def compute_2d_hull_metrics(points):
        """
        计算2D convex hull area, perimeter, centroid。
        """
        points = np.asarray(points)
        centroid = np.mean(points, axis=0)
    
        if len(points) < MIN_POINTS_FOR_HULL:
            return {
                "area": np.nan,
                "perimeter": np.nan,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "n_points": len(points)
            }
    
        try:
            hull = ConvexHull(points)
            area = hull.volume      # 2D中 hull.volume = 面积
            perimeter = hull.area   # 2D中 hull.area = 周长
        except Exception:
            area = np.nan
            perimeter = np.nan
    
        return {
            "area": area,
            "perimeter": perimeter,
            "centroid_x": centroid[0],
            "centroid_y": centroid[1],
            "n_points": len(points)
        }
    
    
    def analyze_one_session_2d(points, group, role, file_name):
        """
        对一个session的一只猴子做：
        1. core extraction
        2. convex hull area
        3. centroid
        """
        core_points, density, threshold = extract_core_points(
            points,
            percentile=CORE_PERCENTILE,
            k=K_NEIGHBORS
        )
    
        metrics_all = compute_2d_hull_metrics(points)
        metrics_core = compute_2d_hull_metrics(core_points)
    
        result = {
            "group": group,
            "role": role,
            "file": file_name,
    
            "all_area": metrics_all["area"],
            "all_perimeter": metrics_all["perimeter"],
            "all_centroid_x": metrics_all["centroid_x"],
            "all_centroid_y": metrics_all["centroid_y"],
            "all_n_points": metrics_all["n_points"],
    
            "core_area": metrics_core["area"],
            "core_perimeter": metrics_core["perimeter"],
            "core_centroid_x": metrics_core["centroid_x"],
            "core_centroid_y": metrics_core["centroid_y"],
            "core_n_points": metrics_core["n_points"],
    
            "density_threshold": threshold
        }
    
        return result, core_points, density
    
    
    # =========================
    # 4. 2D可视化函数
    # =========================
    
    def plot_2d_points_with_hull(points, core_points, title, color, save_path):
        """
        绘制：
        - 全部点：浅色
        - core点：深色
        - core convex hull
        - centroid
        """
        points = np.asarray(points)
        core_points = np.asarray(core_points)
    
        fig, ax = plt.subplots(figsize=(8, 8))
    
        # all points
        ax.scatter(
            points[:, 0], points[:, 1],
            s=2, alpha=0.08, color=color
        )
    
        # core points
        ax.scatter(
            core_points[:, 0], core_points[:, 1],
            s=8, alpha=0.45, color=color
        )
    
        # centroid
        centroid = np.mean(core_points, axis=0)
        ax.scatter(
            centroid[0], centroid[1],
            s=120, color='black', marker='x'
        )
    
        # convex hull boundary
        if len(core_points) >= MIN_POINTS_FOR_HULL:
            try:
                hull = ConvexHull(core_points)
                hull_points = core_points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
    
                ax.plot(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    color=color,
                    linewidth=2.5,
                    alpha=0.9
                )
            except Exception:
                pass
    
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(False)
        ax.axis("equal")
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    
    # =========================
    # 5. 批量分析
    # =========================
    
    results = []
    colors = ['#C00000', '#385723', '#002060', '#002060']
    # Control subject
    for points, name in zip(ctrl_subject_list, ctrl_names):
        res, core, density = analyze_one_session_2d(
            points,
            group="Control",
            role="Subject",
            file_name=name
        )
        results.append(res)
    
        plot_2d_points_with_hull(
            points, core,
            title=f"Control Subject: {name}",
            color=colors[2],
            save_path=os.path.join(save_dir, f"Control_Subject_{name}.png")
        )
    
    # Control partner
    for points, name in zip(ctrl_partner_list, ctrl_names):
        res, core, density = analyze_one_session_2d(
            points,
            group="Control",
            role="Partner",
            file_name=name
        )
        results.append(res)
    
        plot_2d_points_with_hull(
            points, core,
            title=f"Control Partner: {name}",
            color=colors[3],
            save_path=os.path.join(save_dir, f"Control_Partner_{name}.png")
        )
    
    # DISC1 subject
    for points, name in zip(disc_subject_list, disc_names):
        res, core, density = analyze_one_session_2d(
            points,
            group="DISC1",
            role="Subject",
            file_name=name
        )
        results.append(res)
    
        plot_2d_points_with_hull(
            points, core,
            title=f"DISC1 Subject: {name}",
            color=colors[0],
            save_path=os.path.join(save_dir, f"DISC1_Subject_{name}.png")
        )
    
    # DISC1 partner
    for points, name in zip(disc_partner_list, disc_names):
        res, core, density = analyze_one_session_2d(
            points,
            group="DISC1",
            role="Partner",
            file_name=name
        )
        results.append(res)
    
        plot_2d_points_with_hull(
            points, core,
            title=f"DISC1 Partner: {name}",
            color=colors[1],
            save_path=os.path.join(save_dir, f"DISC1_Partner_{name}.png")
        )
    
    
    # =========================
    # 6. 保存结果表
    # =========================
    
    result_df = pd.DataFrame(results)
    
    csv_path = os.path.join(save_dir, "2D_UMAP_core_hull_metrics.csv")
    result_df.to_csv(csv_path, index=False)
    
    print("Saved metrics to:", csv_path)
    print(result_df.head())
    
    
    # =========================
 # In[] # 7. 分组统计可视化
    # =========================
    markers = ['^','o', 's','o']    
    def plot_group_metric(df, metric, save_path, pltleg =False):
        """
        每组每个session的分布。
        """
        df = df.copy()
        df["group_role"] = df["group"] + "_" + df["role"]
    
        order = [
            "DISC1_Subject",
            "DISC1_Partner",
            "Control_Subject",
            "Control_Partner"
        ]
    
        plt.figure(figsize=(8, 6))
    
        label = [
            "DISC1",
            "P-DISC1",
            "Con",
            "P-Con"
        ]
        data = [df[df["group_role"] == g][metric].dropna().values for g in order]
    
        plt.boxplot(data, labels=label, showfliers=False)
    
        for i, vals in enumerate(data, start=1):
            x = np.random.normal(i, 0.04, size=len(vals))
            plt.scatter(x, vals, label = label[i-1], c = colors[i-1], edgecolor='black', facecolors='none', s = 100,marker = markers[i-1], alpha=0.8)
    
        plt.ylabel(metric)
        # plt.xticks(ha="right")
        plt.title(metric)
    
        ax = plt.gca()
        
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if pltleg:
            print(1)
            plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    
    plot_group_metric(
        result_df,
        metric="core_area",
        save_path=os.path.join(save_dir, "group_core_area.png")
    )
    
    plot_group_metric(
        result_df,
        metric="all_area",
        save_path=os.path.join(save_dir, "group_all_area.png")
    )
    
    plot_group_metric(
        result_df,
        metric="core_perimeter",
        save_path=os.path.join(save_dir, "group_core_perimeter.png")
    )
    
    plot_group_metric(
        result_df,
        metric="all_centroid_x",
        save_path=os.path.join(save_dir, "all_centroid_x.png"),
                               pltleg = True
    )
    
    # =========================
    # 8. 汇总统计
    # =========================
    
    summary = result_df.groupby(["group", "role"])[
        ["core_area", "all_area", "core_perimeter", "all_perimeter"]
    ].agg(["mean", "sem", "count"])
    
    summary_path = os.path.join(save_dir, "2D_UMAP_group_summary.csv")
    summary.to_csv(summary_path)
    
    print(summary)
    print("Saved summary to:", summary_path)
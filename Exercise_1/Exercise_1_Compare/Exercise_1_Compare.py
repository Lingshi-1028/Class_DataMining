# ----------------------------------------------------------------
# ------------ 层次聚类与高斯混合模型（GMM）可视化对比项目 -----------
# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 数据加载函数
def load_data(file_path,n_samples):
    data_All = np.loadtxt(file_path, delimiter=None)
    data = data_All[0:int(n_samples*data_All.shape[0]), 0:2]
    return data

# ---------------------- 层次聚类（单链接法）----------------------
class HierarchicalClustering:
    def __init__(self, n_clusters=2):   # 目标聚类数量，如果没设置默认为2
        self.n_clusters = n_clusters
        self.clusters = None

    def _distance(self, x1, x2):
        # 欧氏距离
        return np.sqrt(np.sum((x1 - x2)**2))

    def _build_distance_matrix(self, data):
        n = len(data)
        dist_matrix = np.zeros((n, n))
        for i in range(n):  # 对角元素距离为零
            for j in range(i+1, n):
                dist_matrix[i, j] = self._distance(data[i], data[j])
                dist_matrix[j, i] = dist_matrix[i, j]  # 因为距离矩阵为对称阵，减少计算量
        return dist_matrix

    def fit(self, data):
        # 初始化：每个样本为一个聚类，目标为[0~n-1]
        self.clusters = [[i] for i in range(len(data))]
        dist_matrix = self._build_distance_matrix(data)

        # 合并聚类直到达到目标数量
        while len(self.clusters) > self.n_clusters:
            # 找到距离最近的两个聚类
            min_dist = float('inf')   # 初始化最小距离为 “正无穷大”，避免数据不真实
            merge_idx1, merge_idx2 = 0, 1

            for i in range(len(self.clusters)):
                for j in range(i+1, len(self.clusters)):
                    # 单链接：取两个聚类间样本的最小距离
                    cluster1 = self.clusters[i]
                    cluster2 = self.clusters[j]
                    current_dist = min(dist_matrix[p][q] for p in cluster1 for q in cluster2)
                    
                    if current_dist < min_dist:
                        min_dist = current_dist
                        merge_idx1, merge_idx2 = i, j

            # 合并聚类
            self.clusters[merge_idx1].extend(self.clusters[merge_idx2])
            del self.clusters[merge_idx2]

        # 生成聚类标签
        labels = np.zeros(len(data), dtype=int)
        for cluster_id, cluster in enumerate(self.clusters):
            for idx in cluster:
                labels[idx] = cluster_id
        return labels

# ---------------------- 高斯混合模型（GMM）----------------------
class GMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means = None  # 均值
        self.covs = None  # 协方差矩阵
        self.weights = None  # 混合系数

    def _gaussian_prob(self, x, mean, cov):
        # 多维高斯概率密度函数
        dim = len(x)
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        diff = x - mean
        exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        return (1 / np.sqrt((2*np.pi)**dim * cov_det)) * np.exp(exponent)

    def fit(self, data):
        n_samples, dim = data.shape

        # 初始化参数
        self.weights = np.ones(self.n_components) / self.n_components  # 混合系数均匀分布
        self.means = data[np.random.choice(n_samples, self.n_components, replace=False)]  # 随机选择均值
        self.covs = [np.cov(data.T) for _ in range(self.n_components)]  # 协方差矩阵初始化为数据整体协方差

        log_likelihood_prev = -float('inf')

        for _ in range(self.max_iter):
            # E步：计算后验概率（责任矩阵）
            responsibilities = np.zeros((n_samples, self.n_components))
            for i in range(n_samples):
                total_prob = sum(self.weights[k] * self._gaussian_prob(data[i], self.means[k], self.covs[k]) 
                                for k in range(self.n_components))
                for k in range(self.n_components):
                    responsibilities[i, k] = (self.weights[k] * self._gaussian_prob(data[i], self.means[k], self.covs[k])) / total_prob

            # M步：更新参数
            N_k = np.sum(responsibilities, axis=0)  # 每个聚类的有效样本数
            self.weights = N_k / n_samples  # 更新混合系数

            # 更新均值
            self.means = np.dot(responsibilities.T, data) / N_k.reshape(-1, 1)

            # 更新协方差矩阵
            self.covs = []
            for k in range(self.n_components):
                diff = data - self.means[k]
                cov = np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k]
                # 添加微小对角项避免奇异矩阵
                cov += np.eye(dim) * 1e-6
                self.covs.append(cov)

            # 计算对数似然性
            log_likelihood = sum(np.log(sum(self.weights[k] * self._gaussian_prob(data[i], self.means[k], self.covs[k]) 
                                         for k in range(self.n_components))) for i in range(n_samples))

            # 收敛判断
            if abs(log_likelihood - log_likelihood_prev) < self.tol:
                break
            log_likelihood_prev = log_likelihood

        # 生成聚类标签（取后验概率最大的聚类）
        labels = np.argmax(responsibilities, axis=1)
        return labels

# ---------------------- 可视化函数 ----------------------
def plot_clusters(data, labels_hc, labels_gmm, title_hc, title_gmm):
    plt.figure(figsize=(12, 5))

    # 层次聚类结果
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=labels_hc, cmap='tab20', s=20)
    plt.title(title_hc)
    plt.xlabel('x')
    plt.ylabel('y')

    # GMM结果
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c=labels_gmm, cmap='tab20', s=20)
    plt.title(title_gmm)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()

# ------------- 基于轮廓系数的聚类评价函数 -------------
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def calculate_silhouette_score(data, labels):
    n_samples = len(data)
    silhouette_scores = np.zeros(n_samples)
    
    # 遍历每个样本计算单个轮廓系数
    for i in range(n_samples):
        current_label = labels[i]
        current_sample = data[i]
        
        # 计算a(i)：同簇内其他样本的平均距离
        same_cluster_samples = data[labels == current_label]
        # 计算当前样本到同簇所有样本的距离
        a_distances = [euclidean_distance(current_sample, sample) for sample in same_cluster_samples]
        a_i = np.mean(a_distances)  # 直接求均值，无异常处理
        
        # 计算b(i)：最近异簇的平均距离
        unique_labels = np.unique(labels)
        b_i = float('inf')
        
        for label in unique_labels:
            if label != current_label:
                # 计算当前样本到该异簇所有样本的距离
                other_cluster_samples = data[labels == label]
                b_distances = [euclidean_distance(current_sample, sample) for sample in other_cluster_samples]
                other_cluster_mean_dist = np.mean(b_distances)
                # 更新最小异簇平均距离
                if other_cluster_mean_dist < b_i:
                    b_i = other_cluster_mean_dist
        
        # 计算单个样本的轮廓系数
        s_i = (b_i - a_i) / max(a_i, b_i)
        silhouette_scores[i] = s_i
    
    # 计算整体轮廓系数（所有样本的平均值）
    return np.mean(silhouette_scores)

def evaluate_silhouette(data, labels_hc, labels_gmm):
    # 计算两种算法的轮廓系数
    silhouette_hc = calculate_silhouette_score(data, labels_hc)
    silhouette_gmm = calculate_silhouette_score(data, labels_gmm)
    
    # 输出评价结果
    print(f"{'层次聚类：'} {silhouette_hc:.6f}")
    print(f"{'GMM：'} {silhouette_gmm:.6f}")

# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    # 加载数据（考虑到所用计算机算力有限，此处选择其中一半（50%）数据进行分析，全部数据跑了5小时还没出一次结果，还请老师您谅解 T^T ）
    data = load_data('Data.txt',0.5)
    # 设定聚类数量
    n_clusters = 5

    # 层次聚类
    hc = HierarchicalClustering(n_clusters=n_clusters)
    labels_hc = hc.fit(data)

    # GMM
    gmm = GMM(n_components=n_clusters, max_iter=100)
    labels_gmm = gmm.fit(data)

    # 可视化对比
    plot_clusters(data, labels_hc, labels_gmm, 
                 f'Hierarchical Clustering (k={n_clusters})', 
                 f'GMM (k={n_clusters})')
    
    # 轮廓系数评价函数
    evaluate_silhouette(data, labels_hc, labels_gmm)




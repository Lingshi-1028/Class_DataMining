# ----------------------------------------------------------------
# ------------------ 高斯混合模型（GMM）最优值讨论 ------------------
# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 数据加载函数
def load_data(file_path,n_samples):
    data_All = np.loadtxt(file_path, delimiter=None)
    data = data_All[0:int(n_samples*data_All.shape[0]), 0:2]
    return data

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
def plot_clusters(data, labels_gmm, title_gmm):
    plt.figure(figsize=(6, 5))

    # GMM结果
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

def evaluate_silhouette(data, labels_gmm):
    # 计算两种算法的轮廓系数
    silhouette_gmm = calculate_silhouette_score(data, labels_gmm)
    
    # 输出评价结果
    print(f"{'GMM：'} {silhouette_gmm:.6f}")

# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    # 加载数据（考虑到所用计算机算力有限，此处选择其中一半（50%）数据进行分析，还请谅解 T^T ）
    data = load_data('Data.txt',0.5)
    # 设定聚类数量（可自行调整）
    n_clusters = 14

    # GMM
    gmm = GMM(n_components=n_clusters, max_iter=100)
    labels_gmm = gmm.fit(data)

    # 可视化对比
    plot_clusters(data, labels_gmm, 
                 f'GMM (k={n_clusters})')
    
    # 轮廓系数评价函数
    evaluate_silhouette(data, labels_gmm)
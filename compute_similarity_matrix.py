import numpy as np
from scipy.stats import pearsonr

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrix(X_train):
    """
    计算特征间 Pearson 相关系数矩阵，并进行 softmax 归一化。

    参数：
        X_train (numpy.ndarray): 训练数据，形状为 (n_samples, n_features)。

    返回：
        numpy.ndarray: 归一化的 Pearson 相似性矩阵，形状为 (n_features, n_features)。
    """
    # 计算标准化数据（z-score）
    X_train_standardized = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

    # 计算 Pearson 相关系数矩阵
    similarity_matrix = np.corrcoef(X_train_standardized, rowvar=False)

    # --- 修改开始 ---
    # 1. 取绝对值，确保冗余度在 [0, 1] 之间
    similarity_matrix = np.abs(similarity_matrix)
    # 2. 填充对角线为0，避免自相关
    np.fill_diagonal(similarity_matrix, 0)

    return similarity_matrix
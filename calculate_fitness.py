
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from joblib import Parallel, delayed

def calculate_fitness(chromosomes, X, y, similarity_matrix, n_jobs=-1, cv_folds=3):
    """
    参数：
    - chromosomes: 染色体列表 [n_chromosomes, n_features]
    - X: 已标准化的特征矩阵 [n_samples, n_features]
    - y: 目标标签 [n_samples]
    - similarity_matrix: 预计算的全局特征相似性矩阵 [n_features, n_features]
    - n_jobs: 并行任务数（-1使用全部核心）
    - cv_folds: 交叉验证折数（默认3折，平衡速度与稳健性）
  
    返回：
    - fitness_values: 适应度值列表 [n_chromosomes]
    """

    # 定义单个染色体处理函数
    def _process_chromosome(chromosome):
        chromosome_arr = np.array(chromosome)
        selected_mask = chromosome_arr.astype(bool)
        selected_features = np.where(selected_mask)[0]
        
        # 如果没有选中任何特征，适应度为0
        if len(selected_features) == 0:
            return 0.0
      
        X_sub = X[:, selected_features]
      
        # 使用分层交叉验证替代单次划分
        knn = KNeighborsClassifier(n_neighbors=5)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        try:
            # 使用交叉验证的平均准确率
            cv_scores = cross_val_score(knn, X_sub, y, cv=cv, scoring='accuracy')
            ca = cv_scores.mean()
        except Exception:
            # 如果交叉验证失败（如某折中只有一个类别），回退到简单评估
            ca = 0.0

        # 从全局相似性矩阵中提取仅包含当前染色体选中特征的子矩阵
        sub_sim = similarity_matrix[selected_features, :][:, selected_features]
        # 生成上三角索引
        triu_idx = np.triu_indices_from(sub_sim, k=1)
        total_sim = sub_sim[triu_idx].sum()
      
        n = len(selected_features)
        if n > 1:
            denominator = (2 * total_sim) / (n * (n - 1))
        else:
            denominator = 0.0
            
        fitness = ca / (denominator + 1e-9)
        return fitness

    # 使用并行处理替代原有循环
    return Parallel(n_jobs=n_jobs)(delayed(_process_chromosome)(chr) for chr in chromosomes)

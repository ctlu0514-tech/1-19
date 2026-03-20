import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


# def calculate_fitness(chromosomes, X, y, similarity_matrix, n_jobs=-1, cv_folds=3):
#     """
#     计算种群适应度 (Standard KNN Mode)。
#     Fitness = KNN_CV_Accuracy / (Redundancy_Denominator + 1e-9)
#     """

#     def _process_chromosome(chromosome):
#         chromosome_arr = np.array(chromosome)
#         selected_mask = chromosome_arr.astype(bool)
#         selected_features = np.where(selected_mask)[0]
        
#         if len(selected_features) == 0:
#             return 0.0, 0.0, 0.0
      
#         # 1. 计算 Redundancy (Denominator)
#         sub_sim = similarity_matrix[selected_features, :][:, selected_features]
#         triu_idx = np.triu_indices_from(sub_sim, k=1)
        
#         n = len(selected_features)
#         if n > 1:
#             total_sim = np.sum(np.abs(sub_sim[triu_idx])) 
#             denominator = (2 * total_sim) / (n * (n - 1))
#         else:
#             denominator = 0.0
            
#         # 2. 计算 Performance (CA)
#         # 开关控制：
#         # 'knn' - 快速，距离敏感，适合早期探索
#         # 'rf'  - 较慢，非线性强，适合寻找复杂交互特征
#         classifier_type = 'knn'  # 可选: 'knn', 'rf'
        
#         X_sub = X[:, selected_features]
#         cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
#         try:
#             if classifier_type == 'knn':
#                 from sklearn.neighbors import KNeighborsClassifier
#                 clf = KNeighborsClassifier(n_neighbors=5)
#                 # KNN 改用 roc_auc 评估
#                 cv_scores = cross_val_score(clf, X_sub, y, cv=cv, scoring='roc_auc')
#                 ca = cv_scores.mean()
                
#             elif classifier_type == 'rf':
#                 from sklearn.ensemble import RandomForestClassifier
#                 clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
#                 # RF 用 ROC_AUC 评估更准
#                 cv_scores = cross_val_score(clf, X_sub, y, cv=cv, scoring='roc_auc')
#                 ca = cv_scores.mean()
                
#             else: 
#                 ca = 0.5
                
#         except Exception:
#             ca = 0.5 if classifier_type == 'rf' else 0.0
        
#         # 3. 计算 Final Fitness
#         # 公式: Fitness = CA / Den
#         fitness = ca / (denominator + 1e-9)
            
#         return (fitness, ca, denominator)

#     results = Parallel(n_jobs=n_jobs)(delayed(_process_chromosome)(chr) for chr in chromosomes)
    
#     fitness_values = [res[0] for res in results]
#     ca_values = [res[1] for res in results]
#     den_values = [res[2] for res in results]
    
#     return fitness_values, ca_values, den_values

def calculate_fitness(chromosomes, X, y, similarity_matrix, n_jobs=-1):
    """
    参数：
    - chromosomes: 染色体列表 [n_chromosomes, n_features]
    - X: 已标准化的特征矩阵 [n_samples, n_features]
    - y: 目标标签 [n_samples]
    - similarity_matrix: 预计算的全局特征相似性矩阵 [n_features, n_features]
    - n_jobs: 并行任务数（-1使用全部核心）
  
    返回：
    - fitness_values: 适应度值列表 [n_chromosomes]
    - ca_values: 模型准确率列表 [n_chromosomes]
    - den_values: 特征冗余度列表 [n_chromosomes]
    """

    def _process_chromosome(chromosome):
        chromosome_arr = np.array(chromosome)
        selected_mask = chromosome_arr.astype(bool)
        selected_features = np.where(selected_mask)[0]
        
        n = len(selected_features)
        
        # 【安全检查】：如果未选择任何特征，直接返回极差的三项指标 0.0
        if n == 0:
            return 0.0, 0.0, 0.0
            
        # 1. 计算 Redundancy (Denominator) 冗余度
        if n > 1:
            sub_sim = similarity_matrix[selected_features, :][:, selected_features]
            triu_idx = np.triu_indices_from(sub_sim, k=1)
            total_sim = sub_sim[triu_idx].sum()
            denominator = (2 * total_sim) / (n * (n - 1))
        else:
            # 只有一个特征时，没有特征间冗余
            denominator = 0.0
            
        # 2. 计算 Performance (CA) 准确率
        X_sub = X[:, selected_features]
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_sub, y, 
                test_size=0.3, 
                stratify=y, 
                random_state=42  # 保持固定随机种子保证可重复性
            )
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)
            ca = knn.score(X_val, y_val)
        except Exception as e:
            # 打印真实报错信息，不要静默吞掉！
            print(f"致命错误: {e}") 
            ca = 0.0

        # 3. 计算 Final Fitness 适应度
        fitness = ca / (denominator + 1e-9)
        
        # 返回三项指标的元组
        return fitness, ca, denominator

    # 使用并行处理获取所有染色体的结果列表（列表里是元组）
    results = Parallel(n_jobs=n_jobs)(delayed(_process_chromosome)(chr) for chr in chromosomes)
    
    # 将元组列表解包拆分成三个独立的列表
    fitness_values = [res[0] for res in results]
    ca_values = [res[1] for res in results]
    den_values = [res[2] for res in results]
    
    return fitness_values, ca_values, den_values
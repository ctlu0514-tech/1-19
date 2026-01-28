import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from joblib import Parallel, delayed

def calculate_fitness(chromosomes, X, y, similarity_matrix, n_jobs=-1, cv_folds=3):
    """
    计算种群适应度 (Standard KNN Mode)。
    Fitness = KNN_CV_Accuracy / (Redundancy_Denominator + 1e-9)
    """

    def _process_chromosome(chromosome):
        chromosome_arr = np.array(chromosome)
        selected_mask = chromosome_arr.astype(bool)
        selected_features = np.where(selected_mask)[0]
        
        if len(selected_features) == 0:
            return 0.0, 0.0, 0.0
      
        # 1. 计算 Redundancy (Denominator)
        sub_sim = similarity_matrix[selected_features, :][:, selected_features]
        triu_idx = np.triu_indices_from(sub_sim, k=1)
        
        n = len(selected_features)
        if n > 1:
            total_sim = np.sum(np.abs(sub_sim[triu_idx])) 
            denominator = (2 * total_sim) / (n * (n - 1))
        else:
            denominator = 0.0
            
        # 2. 计算 Performance (CA) - RF CV AUC
        # 使用 AUC 替代 Accuracy，更适合非平衡数据，且更能反映模型排序能力。
        X_sub = X[:, selected_features]
        # knn = KNeighborsClassifier(n_neighbors=5)
        # 既然最后关注 RF/LR 等，这里用 RF 能兼顾非线性和集成优势，
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        try:
            # scoring='roc_auc' 计算 AUC
            cv_scores = cross_val_score(clf, X_sub, y, cv=cv, scoring='roc_auc')
            ca = cv_scores.mean()
        except Exception:
            ca = 0.5 # AUC 的基线是 0.5
        
        # 3. 计算 Final Fitness
        # 公式: Fitness = AUC / Den (或者 AUC - alpha * Den)
        # 这里先改回除法，让你看下 AUC 的效果
        fitness = ca / (denominator + 1e-9)
            
        return (fitness, ca, denominator)

    results = Parallel(n_jobs=n_jobs)(delayed(_process_chromosome)(chr) for chr in chromosomes)
    
    fitness_values = [res[0] for res in results]
    ca_values = [res[1] for res in results]
    den_values = [res[2] for res in results]
    
    return fitness_values, ca_values, den_values

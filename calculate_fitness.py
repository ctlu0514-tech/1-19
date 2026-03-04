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
            
        # 2. 计算 Performance (CA)
        # 开关控制：
        # 'knn' - 快速，距离敏感，适合早期探索
        # 'rf'  - 较慢，非线性强，适合寻找复杂交互特征
        classifier_type = 'knn'  # 可选: 'knn', 'rf'
        
        X_sub = X[:, selected_features]
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        try:
            if classifier_type == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                clf = KNeighborsClassifier(n_neighbors=5)
                # KNN 改用 roc_auc 评估
                cv_scores = cross_val_score(clf, X_sub, y, cv=cv, scoring='roc_auc')
                ca = cv_scores.mean()
                
            elif classifier_type == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
                # RF 用 ROC_AUC 评估更准
                cv_scores = cross_val_score(clf, X_sub, y, cv=cv, scoring='roc_auc')
                ca = cv_scores.mean()
                
            else: 
                ca = 0.5
                
        except Exception:
            ca = 0.5 if classifier_type == 'rf' else 0.0
        
        # 3. 计算 Final Fitness
        # 公式: Fitness = CA / Den
        fitness = ca / (denominator + 1e-9)
            
        return (fitness, ca, denominator)

    results = Parallel(n_jobs=n_jobs)(delayed(_process_chromosome)(chr) for chr in chromosomes)
    
    fitness_values = [res[0] for res in results]
    ca_values = [res[1] for res in results]
    den_values = [res[2] for res in results]
    
    return fitness_values, ca_values, den_values

import pandas as pd
import numpy as np
from collections import defaultdict
import random
from fisher_score import compute_fisher_score
from normalize_scores import normalize_scores
from construct_feature_graph import construct_pearson_only_graph
from compute_similarity_matrix import compute_similarity_matrix
from community_detection import iscd_algorithm_auto_k
from calculate_fitness import calculate_fitness 
from genetic_algorithm_utils import initialize_population, genetic_algorithm
from final_subset_selection import final_subset_selection
from genetic_algorithm_utils import set_random_seed

def cdgafs_feature_selection(X, y, feature_list, theta, omega, population_size):
    """
    (已简化)
    执行特征选择流程（建图、社团检测、GA）。
    仅使用 Pearson 相关性构建特征图。

    输入:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 标签
        feature_list (list): 特征名称列表
        theta (float): 建图阈值
        omega (float): 社团内特征选择比例
        population_size (int): 种群大小
        
    返回: (元组)
        selected_original_indices (list): GA选出的最终特征索引
        X (np.ndarray): 特征矩阵 (原样返回)
        original_indices (np.ndarray): 原始索引 (原样返回)
        feature_list_subset (list): 特征名称列表 (原样返回)
        normalized_fisher_scores (dict/np.ndarray): 归一化的FS
    """

    set_random_seed(42)

    # Step 1: 计算 Fisher Scores
    print("Step 1: 计算 Fisher Scores...")
    fisher_scores = compute_fisher_score(X, y)
    
    original_indices = np.arange(X.shape[1])
    
    # 不执行预筛选，直接使用全部特征
    print("使用了全部特征（未进行预筛选）。")
    X_subset = X
    fisher_scores_subset = fisher_scores
    feature_list_subset = feature_list
    
    normalized_fisher_scores = normalize_scores(fisher_scores_subset)
    print("归一化后的 Fisher Scores (前5个):", normalized_fisher_scores[:5]) 

    # Step 2: 构造特征图 (Pearson Only)
    print(f"\nStep 2: 构造特征图 (Pearson Only)...")
    feature_graph = construct_pearson_only_graph(X_subset, theta)

    similarity_matrix = compute_similarity_matrix(X_subset)

    # Step 3: 社区检测
    print("\nStep 3: 进行社区检测...")
    partition = iscd_algorithm_auto_k(feature_graph)

    clusters = defaultdict(list)
    for node, community in partition.items():
        clusters[community].append(node)
    clusters = [cluster for cluster in clusters.values()]
    print(f"检测到 {len(clusters)} 个社区。")
    
    # Step 4: 初始化种群并执行遗传算法
    print("\nStep 4: 执行遗传算法...")
    num_features_subset = X_subset.shape[1]
    population = initialize_population(num_features_subset, clusters, omega, population_size)
    print(f"初始化种群大小: {len(population)}")
    
    fitness_values = calculate_fitness(population, X_subset, y, similarity_matrix, n_jobs=10)
    print(f"初始适应度值示例: {fitness_values[:5]}")

    population, fitness_values = genetic_algorithm(
        population, fitness_values, X_subset, y, clusters, omega, 
        similarity_matrix, population_size, num_features_subset, normalized_fisher_scores
    )
    
    # Step 5: 选择最终特征子集
    print("\nStep 5: 选择最终特征子集...")
    best_chromosome, selected_subset_indices = final_subset_selection(population, fitness_values)
    print(f"从子集中选出了 {len(selected_subset_indices)} 个特征。")
    
    # 对于全量数据，Original Indices 就是 Subset Indices
    selected_original_indices_ga = selected_subset_indices
    
    print(f"\n最终最佳染色体: {best_chromosome}")
    print(f"最终特征索引: {selected_original_indices_ga[:10]} ...") 

    return (
        selected_original_indices_ga, 
        X_subset, 
        original_indices, 
        feature_list_subset, 
        normalized_fisher_scores
    )
    # =================================================================
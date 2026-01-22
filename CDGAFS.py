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
from genetic_algorithm_utils import initialize_population, genetic_algorithm, allocate_quota_by_quality
from final_subset_selection import final_subset_selection
from genetic_algorithm_utils import set_random_seed


def semantic_pre_clustering(feature_names, max_cluster_size=200):
    """
    根据特征命名的语义结构进行预聚类。
    支持多种命名格式:
      - 序列_滤波器_特征类型 (如 MRIT2_wavelet-LLH_glcm_Correlation)
      - 滤波器_特征类型 (如 wavelet-LLH_glcm_Correlation)
      - cluster1_滤波器_特征类型 (如 cluster1_original_shape_Elongation)
    
    Args:
        feature_names: 特征名称列表
        max_cluster_size: 单个聚类的最大特征数，超过则进一步细分
    
    Returns:
        clusters: 聚类列表，每个元素是特征索引列表
    """
    clusters = defaultdict(list)
    
    for i, name in enumerate(feature_names):
        parts = name.split('_')
        if len(parts) >= 2:
            # 使用 前缀_滤波器 作为聚类键
            key = f"{parts[0]}_{parts[1]}"
        else:
            key = parts[0] if parts else "unknown"
        clusters[key].append(i)
    
    # 如果某个组过大，进一步按特征类型细分
    result = []
    for key, indices in clusters.items():
        if len(indices) > max_cluster_size:
            # 细分：按第三部分（通常是特征类别如 glcm, glrlm）
            sub_clusters = defaultdict(list)
            for idx in indices:
                parts = feature_names[idx].split('_')
                sub_key = parts[2] if len(parts) >= 3 else 'other'
                sub_clusters[sub_key].append(idx)
            result.extend(sub_clusters.values())
        else:
            result.append(indices)
    
    return result


def cdgafs_feature_selection(X, y, feature_list, theta, omega, population_size,
                              use_semantic_clustering=False, max_cluster_size=200,
                              use_quality_quota=False, target_k=50, 
                              top_cluster_ratio=0.5, temperature=10.0):
    """
    执行特征选择流程（建图、社团检测/语义聚类、GA）。

    输入:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 标签
        feature_list (list): 特征名称列表
        theta (float): 建图阈值 (仅在 use_semantic_clustering=False 时使用)
        omega (float): 社团内特征选择比例（当 use_quality_quota=False 时使用）
        population_size (int): 种群大小
        use_semantic_clustering (bool): 是否使用语义预聚类替代 ISCD
        max_cluster_size (int): 语义聚类时单个组的最大特征数
        use_quality_quota (bool): 是否使用方案C质量加权配额分配
        target_k (int): 目标选择特征数（仅在 use_quality_quota=True 时使用）
        top_cluster_ratio (float): 筛选的高质量社区比例（如0.5表示选Top 50%）
        temperature (float): Softmax 温度参数，越大权重差异越大
        
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
    
    print("使用了全部特征（未进行预筛选）。")
    X_subset = X
    fisher_scores_subset = fisher_scores
    feature_list_subset = feature_list
    
    normalized_fisher_scores = normalize_scores(fisher_scores_subset)
    print("归一化后的 Fisher Scores (前5个):", normalized_fisher_scores[:5]) 

    # Step 2 & 3: 聚类（语义聚类或ISCD）
    if use_semantic_clustering:
        print(f"\nStep 2: 使用语义预聚类 (max_cluster_size={max_cluster_size})...")
        clusters = semantic_pre_clustering(feature_list_subset, max_cluster_size)
        print(f"语义聚类完成，共 {len(clusters)} 个聚类。")
        
        # 打印聚类分布
        cluster_sizes = [len(c) for c in clusters]
        print(f"聚类大小分布: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
              f"mean={np.mean(cluster_sizes):.1f}")
        
        # 计算相似度矩阵（仍然需要用于 GA 适应度计算）
        print("\n计算相似度矩阵...")
        similarity_matrix = compute_similarity_matrix(X_subset)
    else:
        # 原始 ISCD 流程
        print(f"\nStep 2: 构造特征图 (Pearson Only)...")
        feature_graph = construct_pearson_only_graph(X_subset, theta)
        
        similarity_matrix = compute_similarity_matrix(X_subset)

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
    
    # 计算配额（方案C质量加权或传统omega模式）
    cluster_quotas = None
    if use_quality_quota:
        print(f"  [方案C质量加权配额] target_k={target_k}, top_ratio={top_cluster_ratio}, temp={temperature}")
        cluster_quotas = allocate_quota_by_quality(
            clusters, fisher_scores_subset, target_k, 
            top_cluster_ratio=top_cluster_ratio, temperature=temperature
        )
    
    population = initialize_population(
        num_features_subset, clusters, omega, population_size, 
        cluster_quotas=cluster_quotas
    )
    print(f"初始化种群大小: {len(population)}")
    
    # 统计初始种群特征数
    avg_features = np.mean([np.sum(c) for c in population])
    print(f"初始种群平均特征数: {avg_features:.1f}")
    
    fitness_values = calculate_fitness(population, X_subset, y, similarity_matrix, n_jobs=10)
    print(f"初始适应度值示例: {fitness_values[:5]}")

    population, fitness_values = genetic_algorithm(
        population, fitness_values, X_subset, y, clusters, omega, 
        similarity_matrix, population_size, num_features_subset, normalized_fisher_scores,
        cluster_quotas=cluster_quotas
    )
    
    # Step 5: 选择最终特征子集
    print("\nStep 5: 选择最终特征子集...")
    best_chromosome, selected_subset_indices = final_subset_selection(population, fitness_values)
    print(f"从子集中选出了 {len(selected_subset_indices)} 个特征。")
    
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
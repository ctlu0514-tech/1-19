import random
from collections import defaultdict
import numpy as np
from calculate_fitness import calculate_fitness  

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def allocate_quota_by_quality(clusters, fisher_scores, total_k, 
                               top_cluster_ratio=0.5, temperature=10.0,
                               cluster_names=None, similarity_matrix=None):
    """
    方案C: 两阶段配额分配（筛选 + 加权采样）
    
    1. 阶段1: 筛选 Top N 个高质量社区
    2. 阶段2: 在筛选后的社区内使用加权采样分配配额
    
    高质量社区可获得多个配额，低质量社区获得 0 配额。
    使用原始 Fisher Score（非归一化）以保留最大差异。
    
    Args:
        clusters (list): 社区列表，每个元素是特征索引列表
        fisher_scores (np.ndarray): 原始 Fisher Scores（非归一化）
        total_k (int): 总共需要选择的特征数
        top_cluster_ratio (float): 筛选的高质量社区比例（如0.5表示选Top 50%）
        temperature (float): Softmax 温度参数，越大权重差异越大
    
    Returns:
        list: 每个社区的配额列表（长度与 clusters 相同）
    """
    n_clusters = len(clusters)
    
    # 计算每个社区的平均 Fisher Score 和 内部冗余度
    cluster_quality_scores = []
    cluster_sizes = []
    raw_avg_scores = []
    
    for cluster in clusters:
        cluster_sizes.append(len(cluster))
        if len(cluster) > 0:
            # 1. 基础质量: Top 20% Fisher Score 均值 + 0.2 * 标准差 (sqrt(var))
            cluster_fs_values = np.array([fisher_scores[f] for f in cluster])
            # 排序并取 Top 20%
            top_k_count = max(1, int(len(cluster) * 0.2))
            top_fs_values = np.sort(cluster_fs_values)[-top_k_count:]
            top_mean = np.mean(top_fs_values)
            fs_std = np.std(cluster_fs_values)
            
            base_score = top_mean + 0.2 * fs_std
            
            # 2. 冗余惩罚: 平均内部相似度
            redundancy_factor = 1.0
            if similarity_matrix is not None and len(cluster) > 1:
                # 提取子矩阵
                sub_sim = similarity_matrix[np.ix_(cluster, cluster)]
                # 计算上三角部分的平均值 (不包括对角线)
                tri_u_indices = np.triu_indices_from(sub_sim, k=1)
                if len(tri_u_indices[0]) > 0:
                    avg_corr = np.mean(np.abs(sub_sim[tri_u_indices]))
                    redundancy_factor = 1.0 + avg_corr
            
            # 最终质量分数
            final_score = base_score / redundancy_factor
            raw_avg_scores.append(np.mean(cluster_fs_values))
            
            # 计算详细统计信息
            cluster_fs_min = np.min(cluster_fs_values)
            cluster_fs_max = np.max(cluster_fs_values)
            cluster_fs_var = np.var(cluster_fs_values)
            cluster_fs_mean = np.mean(cluster_fs_values)
        else:
            final_score = 0.0
            raw_avg_scores.append(0.0)
            cluster_fs_min = 0.0
            cluster_fs_max = 0.0
            cluster_fs_var = 0.0
            cluster_fs_mean = 0.0
            top_mean = 0.0 # 记录 top_mean
            
        cluster_quality_scores.append(final_score)
        
        # 存储统计信息供打印 (增加 top_mean)
        if 'fs_stats_list' not in locals():
            fs_stats_list = []
        fs_stats_list.append((cluster_fs_min, cluster_fs_max, cluster_fs_var, cluster_fs_mean, top_mean))
    
    cluster_quality_scores = np.array(cluster_quality_scores)
    raw_avg_scores = np.array(raw_avg_scores)
    cluster_sizes = np.array(cluster_sizes)
    
    print(f"\n{'='*60}")
    print(f"[方案C: 两阶段配额分配] 目标特征数: {total_k}")
    print(f"{'='*60}")
    print(f"  社区数量: {n_clusters}")
    print(f"  社区数量: {n_clusters}")
    print(f"  社区调整后质量: min={cluster_quality_scores.min():.6f}, "
          f"max={cluster_quality_scores.max():.6f}, mean={cluster_quality_scores.mean():.6f}")
    
    # ========== 阶段1: 筛选高质量社区 ==========
    # 计算需要激活的社区数（至少等于 total_k，确保有足够的社区）
    n_active = max(total_k, int(n_clusters * top_cluster_ratio))
    n_active = min(n_active, n_clusters)  # 不能超过总社区数
    
    # 按质量排序，选择 Top N 个社区
    sorted_indices = np.argsort(cluster_quality_scores)[::-1]
    active_indices = sorted_indices[:n_active]
    inactive_indices = sorted_indices[n_active:]
    
    print(f"\n  阶段1: 社区筛选")
    print(f"    激活社区数: {n_active}/{n_clusters} (top {100*n_active/n_clusters:.1f}%)")
    print(f"    激活社区质量范围: {cluster_quality_scores[active_indices[-1]]:.6f} ~ "
          f"{cluster_quality_scores[active_indices[0]]:.6f}")
    if len(inactive_indices) > 0:
        print(f"    被排除社区质量范围: {cluster_quality_scores[inactive_indices[-1]]:.6f} ~ "
              f"{cluster_quality_scores[inactive_indices[0]]:.6f}")
    
    # ========== 阶段2: 加权采样分配配额 ==========
    # 计算激活社区的 Softmax 权重
    active_scores = cluster_quality_scores[active_indices]
    
    # Softmax with temperature (避免数值溢出)
    scaled_scores = active_scores * temperature
    scaled_scores = scaled_scores - np.max(scaled_scores)  # 数值稳定性
    exp_scores = np.exp(scaled_scores)
    weights = exp_scores / exp_scores.sum()
    
    print(f"\n  阶段2: 加权采样分配")
    print(f"    温度参数: {temperature}")
    print(f"    权重范围: min={weights.min():.4f}, max={weights.max():.4f}")
    
    # 初始化配额
    quotas = np.zeros(n_clusters, dtype=int)
    remaining_capacity = cluster_sizes.copy() # 原容量
    
    # 确定性分配逻辑 (Deterministic Quota Allocation)
    # 期望配额
    expected_quotas = np.zeros(n_clusters)
    expected_quotas[active_indices] = weights * total_k
    
    # 1. 整数部分分配
    int_quotas = np.floor(expected_quotas).astype(int)
    
    # 处理容量溢出 (Capacity Overflow) - 初步截断并收集溢出
    # 注意：这里只截断，溢出的量稍后作为“剩余名额”一起处理
    # 但为了计算方便，我们先算出每个人实际上能拿多少“整数”配额
    actual_int_quotas = np.minimum(int_quotas, remaining_capacity)
    remainder_quotas = expected_quotas - int_quotas # 小数部分
    
    # 计算当前已分配的总数
    current_allocated = np.sum(actual_int_quotas)
    to_allocate = total_k - current_allocated
    
    quotas = actual_int_quotas.copy()
    
    # 2. 剩余名额分配 (按小数部分排序 + 溢出再分配)
    # 我们需要一个循环来处理，直到分配完或者没有空间
    
    if to_allocate > 0:
        # 参与分配的候选者逻辑：
        # 必须是 active_indices 中的
        # 必须还有剩余空间
        
        # 为了实现确定性再分配，我们将所有待分配的“权重”视为优先级
        # 这里使用 remainder_quotas 作为初始优先级。
        # 如果一个社区满了，它的优先级就失效。
        
        # 迭代分配，每次分1个给“当前优先级最高且未满”的社区
        # 为了效率，可以批量分配，但为了绝对严格，逐个分配也没问题 (K=50,循环50次很快)
        
        # 将小数部分构造为优先级队列 (index, priority)
        # 注意：对于因容量限制而被砍掉整数配额的社区，它们其实“亏欠”了更多，
        # 但按照逻辑，满了就是满了，不能再分。
        # 所以我们只关心那些 **还没满** 的社区。
        
        # 初始优先级 = 小数部分
        # (对于那些因为容量不够连整数部分都没拿全的，它们已经满了，不需要考虑)
        
        priorities = np.zeros(n_clusters)
        priorities[active_indices] = remainder_quotas[active_indices]
        
        # 如果有因为容量限制导致 expected_quotas > capacity 的情况，
        # 说明这个社区的 "权重" 溢出了，溢出的权重其实应该重新归一化分给别人，
        # 这里的简单做法是：继续在剩下的社区里按优先级分。
        
        for _ in range(int(to_allocate)):
            # 找到还没满的社区
            not_full_mask = quotas < remaining_capacity
            
            # 只在 active 且 未满 的社区中找
            # (注意：非 active 社区 priorities 已经是 0)
            candidates_mask = not_full_mask
            
            if not np.any(candidates_mask):
                print(f"    ⚠️ 警告：所有激活社区已满，无法分配剩余 {to_allocate} 个配额。")
                break
                
            # 在候选者中找优先级最高的
            # 屏蔽掉已满的
            current_priorities = priorities.copy()
            current_priorities[~candidates_mask] = -1.0 # 设为负数确保不被选中
            
            best_idx = np.argmax(current_priorities)
            
            if current_priorities[best_idx] < 0:
                 # 理论上不应发生，除非 mask 逻辑有问题
                 break
                 
            quotas[best_idx] += 1
            priorities[best_idx] = -2.0 # 分过一次小数配额后，该社区的小数优先级归零(或极小)，避免重复拿
            # 注意：这里简单将优先级设为极小，意味着每个社区靠“小数部分”最多只能多拿1个配额。
            # 如果还有虽然满了但没被选中的，或者剩余名额实在太多（weights极不均匀时），
            # 可能需要更复杂的逻辑（如按原始权重继续轮询）。
            # 鉴于 K=50 且 n_active=50+，通常小数分配一轮就够了。
            # 
            # *修正逻辑*：标准最大余数法是不可重复的。但如果我们需要分配的量很大（因溢出回收），
            # 可能需要多轮。
            # 更加鲁棒的方法：
            # 每次选择 argmax(weights / (quotas + 1)) ? (D'Hondt法)
            # 或者简单点：如果一轮分完还有剩，重置优先级为原始 weights。
            
    # 二次检查：如果还有配额没分完（极少见），使用 D'Hondt 思想填补
    real_allocated = np.sum(quotas)
    if real_allocated < total_k:
        gap = int(total_k - real_allocated)
        # print(f"    提示：进入溢出回收再分配阶段，剩余 {gap} 个...")
        for _ in range(gap):
            not_full_mask = quotas < remaining_capacity
            if not np.any(not_full_mask): 
                break
            
            # 在未满社区中，按 weights（原始重要性）分配
            # 为了公平，可以除以已获得配额数+1 (D'Hondt) 或者是直接按 weight
            # 这里简单按 weight
            curr_w = np.zeros(n_clusters)
            # 只考虑 active 且未满
            valid_mask = not_full_mask & np.isin(np.arange(n_clusters), active_indices)
            if not np.any(valid_mask):
                # 假如 active 都满了，考虑非 active? (本逻辑不考虑，仅在 active 中分)
                break
            
            curr_w[valid_mask] = cluster_quality_scores[valid_mask] # 用质量分作为权重
            
            best = np.argmax(curr_w)
            quotas[best] += 1
    
    # ========== 打印结果 ==========
    nonzero_quotas = quotas[quotas > 0]
    print(f"\n  配额分配结果:")
    print(f"    非零配额社区: {len(nonzero_quotas)}/{n_clusters}")
    print(f"    总配额分配: {np.sum(quotas)}/{total_k}")
    if len(nonzero_quotas) > 0:
        print(f"    配额分布: min={nonzero_quotas.min()}, max={nonzero_quotas.max()}, "
              f"mean={nonzero_quotas.mean():.1f}")
    
    # 打印所有有配额的社区（按配额从高到低排序）
    sorted_quota_indices = np.argsort(quotas)[::-1]
    nonzero_count = np.sum(quotas > 0)
    print(f"\n  所有有配额社区 ({nonzero_count}个):")
    for rank, idx in enumerate(sorted_quota_indices, 1):
        if quotas[idx] > 0:
            c_name = cluster_names[idx] if cluster_names and idx < len(cluster_names) else f"社区{idx}"
            stats = fs_stats_list[idx]
            print(f"    #{rank}: {c_name} | 配额={quotas[idx]} | 大小={cluster_sizes[idx]} | "
                  f"FS_Stats(min={stats[0]:.4f}, max={stats[1]:.4f}, var={stats[2]:.4f}, avg={stats[3]:.4f}, top20={stats[4]:.4f})")
    
    # 打印被跳过的社区
    zero_quota_mask = quotas == 0
    n_skipped = np.sum(zero_quota_mask)
    if n_skipped > 0:
        skipped_fs = raw_avg_scores[zero_quota_mask]
        print(f"\n  被跳过的社区 (配额=0):")
        print(f"    数量: {n_skipped}")
        print(f"    FS均值范围: {skipped_fs.min():.4f} ~ {skipped_fs.max():.4f}")
        print(f"    FS_Stats示例(avg): {skipped_fs[:5]}")
    
    print(f"{'='*60}\n")
    
    return quotas.tolist()

def initialize_population(num_features, clusters, omega, population_size, cluster_quotas=None):
    """
    初始化种群。
    Args:
        num_features (int): 原始特征总数。
        clusters (list): 每个簇包含的特征索引列表。
        omega (float): 每个簇中选取的特征数百分比（当 cluster_quotas 为 None 时使用）。
        population_size (int): 种群大小。
        cluster_quotas (list, optional): 每个簇的配额列表。如果提供，则忽略 omega。
    Returns:
        list: 初始化的种群，每个个体是一个染色体（特征选择向量）。
    """
    population = []
    
    for _ in range(population_size):
        chromosome = np.zeros(num_features, dtype=int)  # 初始化染色体为全0
        for i, cluster_features in enumerate(clusters):  # 遍历每个簇
            if cluster_quotas is not None:
                # 使用质量加权配额
                quota = cluster_quotas[i]
            else:
                # 使用传统 omega 比例
                quota = int(omega * len(cluster_features))
            
            # 确保配额不超过簇大小
            quota = min(quota, len(cluster_features))
            
            if quota > 0:
                # 在该簇中随机选择特征
                selected_features = np.random.choice(
                    cluster_features, size=quota, replace=False
                )
                chromosome[selected_features] = 1  # 将选中的特征置为1
        population.append(chromosome)
    
    return population

# 交叉操作函数
def crossover(parent1, parent2, num_features, clusters):
    """
    按社区（Cluster-wise）交叉操纵。
    不再随机切断染色体，而是以“社区”为单位，随机决定子代继承父代1还是父代2的该社区片段。
    这样可以最大程度保证子代继承到合法的配额（Quota），将 Repair 需求降到最低。
    
    :param parent1: 父染色体1（列表）
    :param parent2: 父染色体2（列表）
    :param num_features: 数据集的特征总数
    :param clusters: 簇列表
    :return: 两个子代染色体
    """
    child1 = np.zeros(num_features, dtype=int)
    child2 = np.zeros(num_features, dtype=int)
    
    # 转换为 numpy 以利用索引广播
    p1 = np.array(parent1)
    p2 = np.array(parent2)

    for cluster in clusters:
        cluster_indices = np.array(cluster)
        
        # 50% 概率 Child1 继承 Parent1 的该社区，Child2 继承 Parent2
        # 50% 概率 Child1 继承 Parent2 的该社区，Child2 继承 Parent1
        if random.random() < 0.5:
            child1[cluster_indices] = p1[cluster_indices]
            child2[cluster_indices] = p2[cluster_indices]
        else:
            child1[cluster_indices] = p2[cluster_indices]
            child2[cluster_indices] = p1[cluster_indices]

    return child1, child2

# 变异操作函数
def mutation(chromosome, cluster_indices, min_swaps=1, max_swaps=3):
    """
    变异操作：在同一簇内做 1<->0 交换，保持每个簇的选中特征数不变。
    :param chromosome: 染色体（numpy 数组）
    :param cluster_indices: 每个簇的特征索引列表（numpy 数组列表）
    :param min_swaps: 每个染色体最少交换次数
    :param max_swaps: 每个染色体最多交换次数
    :return: 变异后的染色体
    """
    if not cluster_indices:
        return chromosome

    if max_swaps < min_swaps:
        max_swaps = min_swaps

    n_swaps = random.randint(min_swaps, max_swaps)
    max_tries = max(10, len(cluster_indices) * 2)

    for _ in range(n_swaps):
        swapped = False
        for _ in range(max_tries):
            cluster_idx = cluster_indices[random.randrange(len(cluster_indices))]
            if cluster_idx.size < 2:
                continue

            selected_mask = chromosome[cluster_idx] == 1
            if not np.any(selected_mask) or np.all(selected_mask):
                continue

            selected_positions = cluster_idx[selected_mask]
            unselected_positions = cluster_idx[~selected_mask]
            i_sel = int(np.random.choice(selected_positions))
            i_unsel = int(np.random.choice(unselected_positions))
            chromosome[i_sel] = 0
            chromosome[i_unsel] = 1
            swapped = True
            break

        if not swapped:
            break

    return chromosome

# 修复操作函数
# def repair(chromosome, clusters, omega, normalized_fisher_scores):
    """
    修复操作：确保染色体中每个簇的特征数量不超过omega，基于Fisher Score概率决定特征的添加或移除。
    :param chromosome: 二进制编码的染色体（列表，0表示未选，1表示选中）
    :param clusters: 簇信息，每个簇为一个特征索引的列表
    :param omega: 每个簇中允许的最大特征数
    :param normalized_fisher_scores: 归一化后的Fisher Score，字典形式 {feature_index: score}
    :return: 修复后的染色体
    """
    repaired_chromosome = chromosome.copy()  # 创建染色体副本

    for cluster in clusters:
        cluster_size = len(cluster)
        # 计算当前簇允许的最大特征数（至少1个）
        omega_cluster = max(1, int(cluster_size * omega))  # 四舍五入取整
      
        # 找到当前簇中已选中的特征
        selected_features = [i for i, gene in enumerate(chromosome) if gene == 1 and i in cluster]
        
        # 如果选中特征过多，基于逆Fisher Score概率移除特征
        if len(selected_features) > omega_cluster:
            inverse_scores = [1 / normalized_fisher_scores[feature] for feature in selected_features]
            total_inverse = sum(inverse_scores)
            remove_probabilities = [score / total_inverse for score in inverse_scores]

            # 根据概率移除特征，直到满足数量约束
            num_to_remove = len(selected_features) - omega_cluster
            features_to_remove = np.random.choice(selected_features, size=num_to_remove, replace=False, p=remove_probabilities)
            for feature in features_to_remove:
                repaired_chromosome[feature] = 0  # 将选中的特征标记为未选中（0）
        
        # 如果选中特征不足，基于Fisher Score概率添加特征
        elif len(selected_features) < omega_cluster:
            remaining_features = [feature for feature in cluster if chromosome[feature] == 0]
            num_to_add = omega_cluster - len(selected_features)

            # 正常按概率添加特征
            fisher_scores = [normalized_fisher_scores[feature] for feature in remaining_features]
            total_score = sum(fisher_scores) # 计算所有候选特征的 Fisher Score 总和，用于归一化概率
            add_probabilities = [score / total_score for score in fisher_scores] # 计算特征选择概率
            features_to_add = random.choices(remaining_features, weights=add_probabilities, k=num_to_add)

            for feature in features_to_add:
                repaired_chromosome[feature] = 1  # 将未选中特征标记为选中（1）

    return repaired_chromosome

def repair(chromosome, clusters, omega, normalized_fisher_scores, cluster_quotas=None, temperature=5.0):
    """
    修复操作：概率性修复 (Probabilistic Repair)。
    
    为了避免“确定性修复”导致的贪婪收敛（只选分最高的），我们引入 Softmax 概率选择。
    温度参数 (Temperature) 用于控制选择压力：
    - T 越大，概率分布越“尖”，越倾向于只选分数最高的 (接近确定性)。
    - T 越小，概率分布越“平”，越接近随机选择。
    
    :param chromosome: 二进制编码的染色体
    :param clusters: 簇信息
    :param omega: 传统占比 (当 quotas 为 None 时用)
    :param normalized_fisher_scores: 归一化 Fisher Score (0~1)
    :param cluster_quotas: 每个簇的硬性配额
    :param temperature: Softmax 温度，建议 5.0 ~ 10.0 以拉开差距
    :return: 修复后的染色体
    """
    repaired_chromosome = chromosome.copy()

    for i, cluster in enumerate(clusters):
        cluster_size = len(cluster)
        
        if cluster_quotas is not None:
            omega_cluster = cluster_quotas[i]
        else:
            omega_cluster = max(1, int(cluster_size * omega))

        # 找出该簇中当前被选中的特征
        selected_features = [f for f in cluster if repaired_chromosome[f] == 1]
        n_selected = len(selected_features)
        
        # --- 情况 A: 选多了，需要移除 ---
        if n_selected > omega_cluster:
            num_to_remove = n_selected - omega_cluster
            
            # 移除逻辑：Score 越低，被移除概率越大
            # Prob(Remove) ~ exp( - Score * T ) 
            # 或者是：保留概率 ~ exp( Score * T )，那么移除概率就是 1 - 保留概率 (不严谨)
            # 更简单的做法：计算“被移除的权重” = 1 / (Score + 1e-6) 或者 exp(-Score)
            # 这里我们用负 Softmax：W = exp( - score * T )
            
            scores = np.array([normalized_fisher_scores[f] for f in selected_features])
            # 为了数值稳定性，减去最大值
            # 注意：我们要让分数低的权重 大
            # weight = exp( (max_score - score) * T )
            
            max_s = np.max(scores)
            weights = np.exp((max_s - scores) * temperature)
            probs = weights / np.sum(weights)
            
            # 按概率选择要移除的特征
            features_to_remove = np.random.choice(
                selected_features, size=num_to_remove, replace=False, p=probs
            )
            for feat in features_to_remove:
                repaired_chromosome[feat] = 0

        # --- 情况 B: 选少了，需要添加 ---
        elif n_selected < omega_cluster:
            remaining_features = [f for f in cluster if repaired_chromosome[f] == 0]
            num_to_add = omega_cluster - n_selected
            
            if len(remaining_features) <= num_to_add:
                # 剩下的都不够凑，全部加上
                for feat in remaining_features:
                    repaired_chromosome[feat] = 1
            else:
                # 添加逻辑：Score 越高，被添加概率越大
                # Prob(Add) ~ exp( Score * T )
                
                scores = np.array([normalized_fisher_scores[f] for f in remaining_features])
                # 数值稳定性: exp( (score - max_score) * T )
                max_s = np.max(scores)
                weights = np.exp((scores - max_s) * temperature)
                probs = weights / np.sum(weights)
                
                features_to_add = np.random.choice(
                    remaining_features, size=num_to_add, replace=False, p=probs
                )
                for feat in features_to_add:
                    repaired_chromosome[feat] = 1
                    
    return repaired_chromosome

# 基于适应度选择最高的染色体函数
def select_top_individuals(population, fitness_values, population_size, ca_values=None, den_values=None):
    """
    基于适应度选择最高的指定数量的染色体。
    
    参数：
        population (list): 当前种群，每个元素表示一个个体。
        fitness_values (list): 每个个体的适应度值。
        population_size (int): 要选取的染色体数量。

    返回：
        selected_population (list): 选中的个体列表。
        selected_fitness (list): 选中个体对应的适应度值。
    """
    # 获取适应度排序的索引，从高到低
    sorted_indices = np.argsort(fitness_values)[::-1]

    # 选择前 population_size 个个体
    selected_indices = sorted_indices[:population_size]
    
    selected_population = [population[i] for i in selected_indices]
    selected_fitness = [fitness_values[i] for i in selected_indices]
    
    ret_ca = [ca_values[i] for i in selected_indices] if ca_values is not None else None
    ret_den = [den_values[i] for i in selected_indices] if den_values is not None else None

    if ret_ca is not None and ret_den is not None:
        return selected_population, selected_fitness, ret_ca, ret_den
    return selected_population, selected_fitness
    
# 轮盘赌选择函数
def roulette_wheel_selection(population, fitness_values, select_count):
    """
    基于轮盘赌选择法，从种群中选择指定数量的个体，选择概率与适应度成正比。
    
    参数：
        population (list): 当前种群，每个元素表示一个个体。
        fitness_values (list): 每个个体的适应度值，表示其优劣程度。
        select_count (int): 要从种群中选择的个体数量。
    
    返回：
        selected_population (list): 选中的个体列表。
        selected_fitness (list): 选中个体对应的适应度值。
    """
    # 计算总适应度，用于归一化选择概率
    total_fitness = sum(fitness_values)
    
    # 计算每个个体被选中的概率
    # 此时 selection_probs 中每个值表示该个体被选中的概率，所有概率之和为 1
    selection_probs = [fitness / total_fitness for fitness in fitness_values]  

    # 使用 np.random.choice 选择个体的索引
    selected_indices = np.random.choice(len(population), size=select_count, p=selection_probs)
    
    # 获取被选中的个体及其适应度
    selected_population = [population[i] for i in selected_indices]
    selected_fitness = [fitness_values[i] for i in selected_indices]
    
    return selected_population, selected_fitness

# 交叉变异过程：生成新种群
def perform_crossover_mutation(population, clusters, omega, num_features, normalized_fisher_scores, cluster_quotas=None):
    new_population = []
    repair_change_count = 0  # 统计这一代有多少个体被 repair 修改了
    total_children = 0
    repair_gene_changes = 0  # 统计 repair 导致的基因翻转次数
    cluster_delta_sum = np.zeros(len(clusters), dtype=int)
    cluster_abs_delta_sum = np.zeros(len(clusters), dtype=int)
    cluster_indices = [np.asarray(cluster, dtype=int) for cluster in clusters]
    
    for i in range(0, len(population), 2):  # 每次取两条染色体进行交叉
        parent1 = population[i]
        parent2 = population[i + 1]
        
        # 执行交叉操作 (传入 clusters)
        child1, child2 = crossover(parent1, parent2, num_features, clusters)

        # 执行变异操作
        child1 = mutation(child1, cluster_indices, min_swaps=1, max_swaps=3)
        child2 = mutation(child2, cluster_indices, min_swaps=1, max_swaps=3)

        # 记录 repair 前的状态 (用于统计)
        c1_before = child1.copy()
        c2_before = child2.copy()

        # 修复操作（传递配额）
        child1 = repair(child1, clusters, omega, normalized_fisher_scores, cluster_quotas=cluster_quotas)
        child2 = repair(child2, clusters, omega, normalized_fisher_scores, cluster_quotas=cluster_quotas)
        
        # 检查是否发生改变
        if not np.array_equal(c1_before, child1):
            repair_change_count += 1
        if not np.array_equal(c2_before, child2):
            repair_change_count += 1

        # 统计 repair 导致的基因翻转次数
        repair_gene_changes += int(np.sum(c1_before != child1))
        repair_gene_changes += int(np.sum(c2_before != child2))

        # 统计每个簇修复前后计数差异（after - before）
        for idx, cluster_idx in enumerate(cluster_indices):
            before_cnt = int(np.sum(c1_before[cluster_idx]))
            after_cnt = int(np.sum(child1[cluster_idx]))
            delta = after_cnt - before_cnt
            cluster_delta_sum[idx] += delta
            cluster_abs_delta_sum[idx] += abs(delta)

            before_cnt = int(np.sum(c2_before[cluster_idx]))
            after_cnt = int(np.sum(child2[cluster_idx]))
            delta = after_cnt - before_cnt
            cluster_delta_sum[idx] += delta
            cluster_abs_delta_sum[idx] += abs(delta)

        total_children += 2
        new_population.extend([child1, child2])

    repair_ratio = repair_change_count / total_children if total_children > 0 else 0.0
    repair_stats = {
        "gene_changes": repair_gene_changes,
        "total_children": total_children,
        "cluster_delta_sum": cluster_delta_sum,
        "cluster_abs_delta_sum": cluster_abs_delta_sum,
    }
    return new_population, repair_ratio, repair_stats

# 主遗传算法流程
def genetic_algorithm(population, fitness_values, X_train, y_train, clusters, omega, similarity_matrix, population_size, 
                      num_features, normalized_fisher_scores, cluster_quotas=None,
                      ca_values=None, den_values=None,
                      cv_folds=3, stability_repeats=1, stability_fraction=0.8, stability_seed=42):
    """
    遗传算法主函数
    """
    generations = 50  
    if ca_values is None:
        # 兼容性处理，如果没传，先算一次
        fitness_values, ca_values, den_values = calculate_fitness(
            population, X_train, y_train, similarity_matrix, n_jobs=10, cv_folds=cv_folds
        )

    for generation in range(generations):
        
        # Step 2: 进行交叉、变异和修复，生成新种群
        new_population, repair_ratio, repair_stats = perform_crossover_mutation(
            population, clusters, omega, num_features, normalized_fisher_scores,
            cluster_quotas=cluster_quotas
        )
        
        # Step 3: 计算新种群的适应度值
        new_fitness_values, new_ca_values, new_den_values = calculate_fitness(
            new_population, X_train, y_train, similarity_matrix, n_jobs=10, cv_folds=cv_folds
        )
        
        # Step 4: 合并新旧种群
        combined_population = population + new_population
        combined_fitness = fitness_values + new_fitness_values
        combined_ca = ca_values + new_ca_values
        combined_den = den_values + new_den_values

        # Step 5: 根据适应度选择最高的个体
        population, fitness_values, ca_values, den_values = select_top_individuals(
            combined_population, combined_fitness, population_size,
            ca_values=combined_ca, den_values=combined_den
        )
        
        # --- 监控指标计算 ---
        pop_set = set(ind.tobytes() for ind in population)
        unique_count = len(pop_set)
        
        pop_matrix = np.array(population)
        p_means = np.mean(pop_matrix, axis=0)
        diversity_score = np.sum(2 * p_means * (1 - p_means)) * (population_size / (population_size - 1))

        # 统计数据
        fit_min = min(fitness_values)
        fit_max = max(fitness_values)
        fit_mean = np.mean(fitness_values)
        fit_std = np.std(fitness_values)
        
        # 获取 Best 个体的指标
        best_idx = np.argmax(fitness_values)
        best_ca = ca_values[best_idx]
        best_den = den_values[best_idx]
        
        # 当代 denominator 统计
        den_min = min(den_values)
        den_mean = np.mean(den_values)
        
        print(f"Gen {generation + 1}: Best Fitness = {fit_max:.4f} (CA={best_ca:.4f}, Den={best_den:.6f}) | "
              f"Min/Mean/Std Fit = {fit_min:.4f}/{fit_mean:.4f}/{fit_std:.4f} | "
              f"Den Min/Mean = {den_min:.6f}/{den_mean:.6f} | "
              f"Unique = {unique_count}/{population_size} | "
              f"Avg Hamming Dist = {diversity_score:.2f} | "
              f"Repair Ratio = {repair_ratio:.1%}")

        if repair_stats:
            total_children = repair_stats["total_children"]
            avg_gene_changes = (
                repair_stats["gene_changes"] / total_children
                if total_children > 0 else 0.0
            )
            avg_gene_pct = (avg_gene_changes / num_features * 100.0) if num_features > 0 else 0.0
            total_abs = int(np.sum(repair_stats["cluster_abs_delta_sum"]))
            net_total = int(np.sum(repair_stats["cluster_delta_sum"]))
            avg_abs_per_child = (total_abs / total_children) if total_children > 0 else 0.0

            top_k = min(5, len(repair_stats["cluster_abs_delta_sum"]))
            if top_k > 0:
                top_idx = np.argsort(repair_stats["cluster_abs_delta_sum"])[-top_k:][::-1]
                top_parts = [
                    f"{int(idx)}:net={int(repair_stats['cluster_delta_sum'][idx])},abs={int(repair_stats['cluster_abs_delta_sum'][idx])}"
                    for idx in top_idx
                ]
                top_str = "; ".join(top_parts)
            else:
                top_str = "n/a"

            print(
                f"  Repair Gene Flips: total={repair_stats['gene_changes']} | "
                f"avg/child={avg_gene_changes:.2f} ({avg_gene_pct:.1f}%)"
            )
            print(
                f"  Repair Cluster Delta: total_abs={total_abs} | "
                f"avg_abs/child={avg_abs_per_child:.2f} | net_total={net_total} | "
                f"top_abs={top_str}"
            )

    return population, fitness_values

# 文件名: run_radiomics_multi_grid_search.py
# 描述: 批量网格搜索4个数据集的最佳参数 (Anti-Leakage Mode)

import pandas as pd
import numpy as np
import warnings
import time
import os
import sys
import itertools

# 机器学习
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# --- 导入核心功能 ---
try:
    from CDGAFS import cdgafs_feature_selection
    from fisher_score import compute_fisher_score
except ImportError:
    print("警告: 缺少 CDGAFS 或 fisher_score 模块。请确保它们在同一目录下。")
    sys.exit(1)

# 忽略所有警告
warnings.filterwarnings('ignore')

# ===================================================================
# 1. 数据加载 (改为抛出异常而不是直接退出，以便批量处理)
# ===================================================================
def load_and_parse_data(csv_path, label_col_name):
    print(f"--- 正在读取: {os.path.basename(csv_path)} ---")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"CSV 读取失败: {e}")

    if label_col_name not in data.columns:
        raise ValueError(f"找不到标签列 '{label_col_name}'")

    # 处理标签
    y_raw = data[label_col_name].values
    unique_labels = np.unique(y_raw)
    if len(unique_labels) == 2:
        class_0_label = np.min(unique_labels)
        y = np.where(y_raw == class_0_label, 0, 1)
    else:
        raise ValueError(f"标签必须包含2个类别 (检测到: {unique_labels})")
    
    # 剔除无关列
    id_cols = [col for col in data.columns if 'ID' in col or 'id' in col] 
    diag_cols = [col for col in data.columns if col.startswith('diagnostics_')]
    cols_to_drop = id_cols + [label_col_name] + diag_cols
    
    X_df = data.drop(columns=cols_to_drop, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number])
    feature_names = X_df.columns.tolist()
    X_raw = X_df.values 
    
    return X_raw, y, feature_names

# ===================================================================
# 2. 安全预处理 (防泄漏)
# ===================================================================
def preprocess_securely(X_train, X_test, feature_names_raw):
    # A. 缺失值填补 (Fit on Train)
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # B. 方差筛选 (Fit on Train)
    stds = np.std(X_train_imp, axis=0)
    good_indices = np.where(stds > 1e-6)[0]
    
    X_train_filt = X_train_imp[:, good_indices]
    X_test_filt = X_test_imp[:, good_indices]
    feature_names_new = [feature_names_raw[i] for i in good_indices]
    
    # C. 标准化 (Fit on Train)
    scaler = StandardScaler()
    scaler.fit(X_train_filt)
    X_train_scaled = scaler.transform(X_train_filt)
    X_test_scaled = scaler.transform(X_test_filt)

    return X_train_scaled, X_test_scaled, feature_names_new

# ===================================================================
# 3. 特征选择与评估 (保持不变)
# ===================================================================
def run_cdgafs_with_params(X_train, y_train, feature_names, k, pop, omega, theta):
    if X_train.shape[1] <= k:
        return list(range(X_train.shape[1]))

    (selected_indices, _, _, _, _) = cdgafs_feature_selection(
        X=X_train, y=y_train, gene_list=feature_names, 
        theta=theta, omega=omega, population_size=pop, 
        w_bio_boost=0.0, pre_filter_top_n=None, graph_type='pearson_only'
    )
    
    selected_indices = np.array(selected_indices)
    if len(selected_indices) == 0:
        return []

    # 简单的 RFE 剪枝
    if len(selected_indices) > k:
        X_ga_selected = X_train[:, selected_indices]
        # 为了速度，step 设大一点
        estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        selector = RFE(estimator, n_features_to_select=k, step=50) 
        selector.fit(X_ga_selected, y_train)
        selected_indices = selected_indices[selector.support_]

    return selected_indices

def evaluate_performance(X_train, y_train, X_test, y_test, selected_indices):
    if len(selected_indices) == 0:
        return 0.0, 0.0

    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    clf.fit(X_train_sel, y_train)

    y_pred = clf.predict(X_test_sel)
    try:
        y_prob = clf.predict_proba(X_test_sel)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
    acc = accuracy_score(y_test, y_pred)
    return auc, acc

# ===================================================================
# 4. 单数据集处理逻辑
# ===================================================================
def process_single_dataset(name, path, label_col, param_grid):
    """
    处理单个数据集的完整网格搜索流程
    返回: (name, best_auc, best_params_dict)
    """
    print(f"\n{'#'*60}")
    print(f"开始处理数据集: [{name}]")
    print(f"{'#'*60}")

    # 1. 加载数据
    try:
        X_raw, y_raw, feats_raw = load_and_parse_data(path, label_col)
    except Exception as e:
        print(f"!!! 跳过数据集 [{name}]: {e}")
        return name, 0.0, {"Error": str(e)}

    # 2. 划分与预处理
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw
    )
    X_train, X_test, feature_names = preprocess_securely(X_train_raw, X_test_raw, feats_raw)
    
    print(f"预处理完成. 训练集: {X_train.shape}, 验证集: {X_test.shape}")

    # 3. 准备参数
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results_list = []
    best_auc = 0.0
    best_params = {}

    # 4. 循环参数
    total = len(combinations)
    for i, params in enumerate(combinations):
        print(f"\r[{name}] 进度: {i+1}/{total} | Params: {params} ...", end="")
        
        start_t = time.time()
        sel_idx = run_cdgafs_with_params(
            X_train, y_train, feature_names,
            k=params['K_FEATURES'], pop=params['pop'], 
            omega=params['omega'], theta=params['theta']
        )
        auc, acc = evaluate_performance(X_train, y_train, X_test, y_test, sel_idx)
        elapsed = time.time() - start_t
        
        res = params.copy()
        res.update({'AUC': auc, 'Accuracy': acc, 'Num_Feats': len(sel_idx), 'Time': elapsed})
        results_list.append(res)
        
        if auc > best_auc:
            best_auc = auc
            best_params = params.copy()

    print(f"\n[{name}] 完成! 最佳 AUC: {best_auc:.4f}")

    # 5. 保存结果
    res_df = pd.DataFrame(results_list).sort_values(by='AUC', ascending=False)
    out_file = f'grid_search_result_{name}.csv'
    res_df.to_csv(out_file, index=False)
    print(f"[{name}] 详细结果已保存至: {out_file}")

    return name, best_auc, best_params

# ===================================================================
# 5. 主程序入口 (配置区域)
# ===================================================================
def main():
    # --- 配置 A: 数据集列表 ---
    # 请在此处填入您的四个文件路径
    # "name" 用于生成结果文件名，"path" 是绝对路径，"label" 是标签列名
    DATASETS = [
        {
            "name": "Dataset_1",
            "path": "/data/qh_20T_share_file/lct/CT67/附二.csv",
            "label": "label"
        },
        {
            "name": "Dataset_2", 
            "path": "/data/qh_20T_share_file/lct/CT67/附一.csv", # 修改这里
            "label": "label"
        },
        {
            "name": "Dataset_3",
            "path": "/data/qh_20T_share_file/lct/CT67/市中心.csv", # 修改这里
            "label": "label"
        },
    ]

    # --- 配置 B: 参数搜索范围 (对所有数据通用) ---
    PARAM_GRID = {
        'K_FEATURES': [100, 500, 1000, 1500, 2000],       
        'pop': [50, 100],                
        'omega': [0.5, 0.6, 0.7, 0.8],              
        'theta': [0.5, 0.6, 0.7, 0.8, 0.9]       
    }

    print("=== 开始批量网格搜索任务 ===")
    
    global_summary = []

    for ds in DATASETS:
        name, auc, params = process_single_dataset(
            ds['name'], ds['path'], ds['label'], PARAM_GRID
        )
        global_summary.append({
            'Dataset': name,
            'Best_AUC': auc,
            'Best_Params': str(params)
        })

    # --- 输出最终汇总 ---
    print(f"\n\n{'='*60}")
    print(f"所有任务执行完毕 - 最终汇总")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(global_summary)
    print(summary_df)
    summary_df.to_csv('grid_search_summary_all.csv', index=False)
    print(f"{'='*60}")
    print("汇总文件已保存: grid_search_summary_all.csv")

if __name__ == "__main__":
    main()
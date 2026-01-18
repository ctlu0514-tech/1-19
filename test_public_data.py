# 文件名: run_radiomics_comparison_strict.py
# 描述: 严格防泄露版 (Split -> Impute -> Filter -> Scale -> Select)
#       包含 mRMR, LASSO, RFE, CDGAFS, 以及针对性配置的批量测试

import pandas as pd
import numpy as np
import warnings
import time
import os
import sys

# 机器学习
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

# --- 导入核心功能 (直接导入) ---
# 如果未安装 pymrmr，请注释掉下面这一行
try:
    import pymrmr
    HAS_MRMR = True
except ImportError:
    HAS_MRMR = False
    print("提示: 未安装 pymrmr，将跳过 mRMR。")

from CDGAFS import cdgafs_feature_selection
from fisher_score import compute_fisher_score
from run_evaluation import print_summary_table

# 忽略所有警告
warnings.filterwarnings('ignore')

# ===================================================================
# 1. 数据加载 (仅做结构性清洗，不做统计性清洗)
# ===================================================================
def load_data_structure_only(csv_path, label_col_name):
    """
    仅进行“物理层面”的清洗：
    1. 删掉非特征列 (ID, Diagnostics)
    2. 转换 Label
    3. Inf -> NaN
    !!! 注意：不在此处做填充或方差筛选，因为那是统计操作，必须在 Split 后做 !!!
    """
    print(f"--- 加载数据: {csv_path} ---")
    try:
        data = pd.read_csv(csv_path, compression='gzip' if csv_path.endswith('.gz') else 'infer')
    except Exception as e:
        print(f"读取失败: {e}")
        sys.exit(1)

    if label_col_name not in data.columns:
        print(f"!!! 致命错误: 找不到标签列 '{label_col_name}'。")
        sys.exit(1)

    # 1. 处理标签
    y_raw = data[label_col_name].values
    unique_labels = np.unique(y_raw)
    if len(unique_labels) == 2:
        class_0_label = np.min(unique_labels)
        y = np.where(y_raw == class_0_label, 0, 1)
    else:
        print(f"!!! 错误: 标签必须包含2个类别。找到: {unique_labels}")
        sys.exit(1)
    
    # 2. 移除无关列
    id_cols = [col for col in data.columns if 'ID' in col or 'id' in col] 
    diag_cols = [col for col in data.columns if col.startswith('diagnostics_')]
    cols_to_drop = id_cols + [label_col_name] + diag_cols
    
    X_df = data.drop(columns=cols_to_drop, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number]) 
    feature_names = X_df.columns.tolist()
    
    # 3. 转为 numpy 并处理 Inf
    X_raw = X_df.values.astype(np.float64)
    if np.isinf(X_raw).any():
        print("    - [预处理] 发现 Inf，已替换为 NaN (将在 Split 后填充)")
        X_raw[np.isinf(X_raw)] = np.nan
        
    print(f"    - 原始数据加载完成: {X_raw.shape} (含 NaN)")
    return X_raw, y, feature_names

# ===================================================================
# 2. 特征选择器模块
# ===================================================================
def select_features_cdafs(X, y, feature_names, K_FEATURES, GA_POPULATION_SIZE, GA_OMEGA, THETA, pruning_method='RFE'):
    print(f"\n    [CDGAFS] 开始运行 (剪枝: {pruning_method})...")
    
    (selected_indices, _, _, _, _) = cdgafs_feature_selection(
        X=X, y=y, gene_list=feature_names, theta=THETA, omega=GA_OMEGA, 
        population_size=GA_POPULATION_SIZE, w_bio_boost=0.0, 
        pre_filter_top_n=None, graph_type='pearson_only'
    )
    
    # 剪枝逻辑
    if len(selected_indices) > K_FEATURES:
        if pruning_method == 'RFE':
            print(f"    [CDGAFS] RFE 剪枝: {len(selected_indices)} -> {K_FEATURES}")
            estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
            selector = RFE(estimator, n_features_to_select=K_FEATURES, step=1)
            X_ga_selected = X[:, selected_indices]
            selector.fit(X_ga_selected, y)
            selected_indices = np.array(selected_indices)[selector.support_]
        elif pruning_method == 'FISHER':
            print(f"    [CDGAFS] Fisher 剪枝: {len(selected_indices)} -> {K_FEATURES}")
            X_ga_selected = X[:, selected_indices]
            scores = compute_fisher_score(X_ga_selected, y)
            top_indices = np.argsort(scores)[-K_FEATURES:]
            selected_indices = np.array(selected_indices)[top_indices]

    return selected_indices if len(selected_indices) > 0 else []

def select_features_mrmr(X, y, feature_names, K_FEATURES):
    if not HAS_MRMR:
        return []
    print("\n    [mRMR] 开始运行...")
    # 构造 DataFrame 供 pymrmr 使用
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    # 离散化 (pymrmr 需要)
    for col in feature_names:
        try:
            df[col] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
        except:
            df[col] = 0 # 处理无法分箱的情况
            
    try:
        selected_names = pymrmr.mRMR(df, 'MIQ', K_FEATURES)
        # 映射回索引
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        selected_indices = [name_to_idx[n] for n in selected_names if n in name_to_idx]
        print(f"    [mRMR] 选中 {len(selected_indices)} 个特征")
        return selected_indices
    except Exception as e:
        print(f"    [mRMR] 出错: {e}")
        return []

def select_features_lasso_cv(X, y):
    """
    使用 CV 自动选择特征，不强制 K。
    注意：为了公平对比，我们记录它选了多少个。
    """
    print("\n    [LASSO-CV] 开始运行...")
    model = LogisticRegressionCV(
        cv=5, penalty='l1', solver='liblinear', class_weight='balanced', 
        random_state=42, max_iter=3000, scoring='roc_auc'
    )
    model.fit(X, y)
    indices = np.where(np.abs(model.coef_[0]) > 1e-6)[0]
    print(f"    [LASSO-CV] 最佳 C: {model.C_[0]:.4f}, 选中特征数: {len(indices)}")
    return indices.tolist()

def select_features_rfe(X, y, K_FEATURES):
    """
    纯 RFE 方法，作为基准对比。
    """
    print(f"\n    [RFE] 开始运行 (Target K={K_FEATURES})...")
    # 基模型使用 L2 逻辑回归
    estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    selector = RFE(estimator, n_features_to_select=K_FEATURES, step=1)
    selector.fit(X, y)
    
    selected_indices = np.where(selector.support_)[0]
    print(f"    [RFE] 选中特征数: {len(selected_indices)}")
    return selected_indices

# ===================================================================
# 3. 评估函数
# ===================================================================
def evaluate_on_independent_test(X_train, y_train, X_test, y_test, selected_indices):
    # 如果没有选中特征，返回全0
    if len(selected_indices) == 0:
        return {
            'AUC': 0, 'Train_AUC': 0, 
            'Accuracy': 0, 'Train_Accuracy': 0, 
            'Sensitivity': 0, 'Specificity': 0, 
            'F1-Macro': 0, '#Feat': 0
        }

    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    clf.fit(X_train_sel, y_train)

    # --- 1. 测试集 (Test/Validation) 指标 ---
    y_pred = clf.predict(X_test_sel)
    try:
        y_prob = clf.predict_proba(X_test_sel)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro') # 注意这里用 macro 以匹配 print_summary_table
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0

    # --- 2. 训练集 (Train) 指标 ---
    y_train_pred = clf.predict(X_train_sel)
    try:
        y_train_prob = clf.predict_proba(X_train_sel)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = 0.5
    train_acc = accuracy_score(y_train, y_train_pred)

    # --- 3. 返回字典 (键名必须严格匹配 run_evaluation.py) ---
    return {
        'AUC': round(auc, 4),
        'Train_AUC': round(train_auc, 4),
        'Accuracy': round(acc, 4),
        'Train_Accuracy': round(train_acc, 4),
        'Sensitivity': round(sens, 4),
        'Specificity': round(spec, 4),
        'F1-Macro': round(f1, 4),
        '#Feat': len(selected_indices)
    }

# ===================================================================
# 4. 主流程 (严格防泄露)
# ===================================================================
def run_strict_analysis(X_raw, y_raw, feature_names_raw, K_FEATURES, params, dataset_title="Dataset"):
    """
    Strict Pipeline: Split -> Impute -> Filter -> Scale -> Select -> Evaluate
    """
    print(f"\n{'-'*30} 正在分析: {dataset_title} (防泄露模式) {'-'*30}")
    print(f"参数设置: K={K_FEATURES}, Pop={params['pop']}, Omega={params['omega']}, Theta={params['theta']}")

    # --- 1. 数据切分 (Split) ---
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    print(f"    - 数据切分完成: Train={X_train_raw.shape}, Test={X_test_raw.shape}")

    # --- 2. 缺失值填充 (Impute - Fit on Train ONLY) ---
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train_raw)
    
    X_train_imp = imputer.transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)
    
    # --- 3. 方差筛选 (Variance Filter - Fit on Train ONLY) ---
    vars = np.var(X_train_imp, axis=0)
    good_indices = np.where(vars > 1e-10)[0]
    
    if len(good_indices) == 0:
        print("!!! 错误: 训练集所有特征均为常量，无法继续。")
        return

    X_train_filt = X_train_imp[:, good_indices]
    X_test_filt = X_test_imp[:, good_indices]
    feature_names = [feature_names_raw[i] for i in good_indices]
    print(f"    - 方差筛选完成，剩余特征数: {len(feature_names)}")

    # --- 4. 标准化 (Scale - Fit on Train ONLY) ---
    scaler = StandardScaler()
    scaler.fit(X_train_filt)
    X_train = scaler.transform(X_train_filt)
    X_test = scaler.transform(X_test_filt)
    
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # --- 5. 运行特征选择 ---
    all_selected_indices = {}
    execution_times = {}

    # 5.1 CDGAFS
    start = time.time()
    all_selected_indices['CDGAFS'] = select_features_cdafs(
        X_train, y_train, feature_names, K_FEATURES, 
        params['pop'], params['omega'], params['theta'],
        pruning_method='RFE'
    )
    execution_times['CDGAFS'] = time.time() - start

    # 5.2 LASSO (作为对比)
    # 注意：LASSO CV 并不受 K_FEATURES 控制，它自己决定最佳数量。
    # 这里我们只是运行它一次作为基准线。
    start = time.time()
    all_selected_indices['LASSO-CV'] = select_features_lasso_cv(X_train, y_train)
    execution_times['LASSO-CV'] = time.time() - start
    
    # 5.3 RFE (作为对比，使用相同的 K)
    start = time.time()
    all_selected_indices['RFE'] = select_features_rfe(X_train, y_train, K_FEATURES)
    execution_times['RFE'] = time.time() - start
    
    # 5.4 mRMR (如果安装了)
    if HAS_MRMR:
        start = time.time()
        all_selected_indices['mRMR'] = select_features_mrmr(
            X_train, y_train, feature_names, K_FEATURES
        )
        execution_times['mRMR'] = time.time() - start

    # --- 6. 统一评估 & 打印结果 ---
    all_results = {}
    print(f"\n>>> {dataset_title} 的最终评估结果 (独立测试集) <<<")
    
    for method, indices in all_selected_indices.items():
        res = evaluate_on_independent_test(X_train, y_train, X_test, y_test, indices)
        all_results[method] = res
        print(f"    [{method}] AUC: {res['AUC']} | Feats: {res['#Feat']} | Time: {execution_times[method]:.2f}s")

    # 打印完整汇总表
    print_summary_table(all_results, all_selected_indices, execution_times)


# ===================================================================
# 5. 配置入口 (核心修改部分)
# ===================================================================
def main():
    PUBLIC_DATASET_DIR = '/data/qh_20T_share_file/lct/CT67/dataset'

    # 1. 定义要测试的 K 值列表 (控制变量法)
    # 我们将依次测试这些特征数量，观察模型性能曲线
    K_VALUES_LIST = [30, 50, 100, 200, 500] 

    # 2. 定义数据集配置 (去掉了固定的 target_k，只保留针对性的 params)
    datasets_config = [
        {
            'name': 'LGG-1p19qDeletion',
            'params': {'pop': 100, 'omega': 0.5, 'theta': 0.9}
        },
        {
            'name': 'BraTS-2021',
            'params': {'pop': 100, 'omega': 0.5, 'theta': 0.9} 
        },
        {
            'name': 'Zhang2023',
            'params': {'pop': 100, 'omega': 0.5, 'theta': 0.9} 
        },
        {
            'name': 'UPENN-GBM',
            'params': {'pop': 100, 'omega': 0.5, 'theta': 0.9}
        }
    ]

    print(f"\n{'='*60}")
    print(f"开始批量运行第一梯队数据测试 (遍历 K={K_VALUES_LIST})")
    print(f"{'='*60}")

    for ds_conf in datasets_config:
        ds_name = ds_conf['name']
        params_val = ds_conf['params']
        
        file_path = os.path.join(PUBLIC_DATASET_DIR, f"{ds_name}.gz")
        
        if os.path.exists(file_path):
            print(f"\n\n{'#'*40}")
            print(f">>> 启动数据集任务: {ds_name} <<<")
            print(f"{'#'*40}")
            
            # 加载数据 (只加载一次，节省IO时间)
            X_raw, y, feats = load_data_structure_only(file_path, 'Target')
            
            # 遍历 K 值
            for k_val in K_VALUES_LIST:
                print(f"\n   >>> 测试 K={k_val} ...")
                # 运行分析
                # 传入 ds_name 时带上 k 值后缀，方便看日志
                run_strict_analysis(X_raw, y, feats, k_val, params_val, f"{ds_name} (K={k_val})")
        else:
            print(f"\n[跳过] 未找到文件: {file_path}")

if __name__ == "__main__":
    main()
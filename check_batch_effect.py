# 文件名: check_batch_effect_fixed.py
# 描述: 修复了常量特征导致的崩溃问题，并将绘图标签改为英文以适应 Linux 服务器

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import sys

# 设置绘图风格
sns.set(style="whitegrid")
# 移除强制中文字体设置，改用默认，避免警告
# plt.rcParams['font.sans-serif'] = ... 

def load_data(csv_path):
    print(f"--- Loading Data: {os.path.basename(csv_path)} ---")
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        sys.exit(1)
    return pd.read_csv(csv_path)

def perform_pca_analysis(X_scaled, centers, output_name="batch_effect_pca_zoomed.png"):
    """
    执行 PCA 并绘制散点图 (支持自定义坐标范围)
    """
    print("\n[1/2] Running PCA Visualization (Zoomed)...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    evr = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=centers, 
                    palette="Set1", s=60, alpha=0.7, edgecolor="w")
    
    plt.title("Multi-Center PCA (Zoomed View)", fontsize=15)
    plt.xlabel(f"PC1 ({evr[0]:.1%} var)", fontsize=12)
    plt.ylabel(f"PC2 ({evr[1]:.1%} var)", fontsize=12)
    plt.legend(title="Center ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # --- 【在这里修改坐标范围】 ---
    # 建议先尝试对称范围，避开那些 500 多的离群点
    plt.xlim(-20, 20)  
    plt.ylim(-20, 20)
    
    # 如果您一定要 0-10，请注释掉上面两行，使用下面这两行：
    # plt.xlim(0, 10)
    # plt.ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"   -> PCA Plot saved to: {output_name}")
    plt.close()

def perform_statistical_test(df, feature_cols, center_col):
    """
    对每个特征执行 Kruskal-Wallis H 检验 (含防崩溃机制)
    """
    print("\n[2/2] Running Statistical Tests (Kruskal-Wallis)...")
    
    centers = df[center_col].unique()
    if len(centers) < 2:
        print("Error: Only one center found. Cannot perform comparison.")
        return 0.0

    p_values = []
    significant_count = 0
    skipped_count = 0
    
    total_feats = len(feature_cols)
    
    for feat in feature_cols:
        # 获取每个中心该特征的数据列表
        # dropna() 是为了防止原始数据里有 NaN 导致报错
        groups = [df[df[center_col] == c][feat].dropna().values for c in centers]
        
        # --- 【修复核心】检查数据是否全为常量 ---
        # 将所有组的数据拼起来看是否只有一个唯一值
        all_values = np.concatenate(groups)
        
        # 1. 检查是否为空
        if len(all_values) == 0:
            p = 1.0
            skipped_count += 1
        # 2. 检查方差是否为0 (所有数都一样)
        elif np.unique(all_values).size <= 1:
            # 如果所有数值都一样，说明没有差异，P值设为 1.0
            p = 1.0 
            skipped_count += 1
        else:
            try:
                stat, p = stats.kruskal(*groups)
            except ValueError as e:
                # 万一还有其他边缘情况，捕获异常设为 1.0
                p = 1.0
                skipped_count += 1
        
        # 处理可能的 NaN p值
        if np.isnan(p):
            p = 1.0
            
        p_values.append(p)
        if p < 0.05:
            significant_count += 1

    # 统计结果
    sig_ratio = significant_count / total_feats
    
    print(f"   -> Total Features: {total_feats}")
    print(f"   -> Constant/Skipped Features: {skipped_count}")
    print(f"   -> Significant Batch Effect (P<0.05): {significant_count}")
    print(f"   -> Batch Effect Ratio: {sig_ratio:.2%}")
    
    # 绘制 P值分布直方图 (英文标签)
    plt.figure(figsize=(8, 5))
    sns.histplot(p_values, bins=50, kde=False, color="skyblue")
    plt.axvline(0.05, color='red', linestyle='--', label='P=0.05 Threshold')
    plt.title("Distribution of P-values across Centers", fontsize=14)
    plt.xlabel("P-value (Kruskal-Wallis)")
    plt.ylabel("Count of Features")
    plt.legend()
    
    save_path = "batch_effect_p_values.png"
    plt.savefig(save_path, dpi=300)
    print(f"   -> P-value Histogram saved to: {save_path}")

    return sig_ratio

def main():
    INPUT_CSV = '/data/qh_20T_share_file/lct/CT67/Merged_All_Centers_CovBat_Python.csv'
    CENTER_COL = 'Center_Source'
    ID_COL = 'Sample_ID'
    LABEL_COL = 'label'

    if not os.path.exists(INPUT_CSV):
        print(f"File not found: {INPUT_CSV}")
        return

    # 加载
    df = load_data(INPUT_CSV)
    
    # 提取纯特征矩阵 X
    metadata_cols = [CENTER_COL, ID_COL, LABEL_COL]
    cols_to_drop = [c for c in metadata_cols if c in df.columns]
    diag_cols = [c for c in df.columns if c.startswith('diagnostics_')]
    cols_to_drop += diag_cols
    
    X_df = df.drop(columns=cols_to_drop, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number])
    feature_names = X_df.columns.tolist()
    
    print(f"   -> Analyzing {len(feature_names)} features.")
    
    # 预处理 for PCA
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X_df)
    
    # PCA 需要移除方差为0的列，否则也会报错或产生无意义结果
    # 这里做一个简单的方差筛选
    selector_std = np.std(X, axis=0)
    valid_idx = np.where(selector_std > 1e-6)[0]
    X = X[:, valid_idx]
    print(f"   -> Features after removing constants for PCA: {X.shape[1]}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    centers = df[CENTER_COL].values

    # 运行分析
    perform_pca_analysis(X_scaled, centers)
    # 统计检验使用原始 df (函数内部处理了常量)
    ratio = perform_statistical_test(df, feature_names, CENTER_COL)
    
    print(f"\n{'='*40}")
    print("FINAL CONCLUSION")
    print(f"{'='*40}")
    if ratio > 0.5:
        print(f"!!! HIGH RISK: {ratio:.1%} features have significant batch effects.")
        print("Recommendation: Apply ComBat harmonization immediately.")
    elif ratio > 0.2:
        print(f"WARNING: {ratio:.1%} features are affected.")
        print("Recommendation: Consider ComBat or strict feature selection.")
    else:
        print(f"GOOD: Only {ratio:.1%} features affected.")
        print("Recommendation: Standard normalization might be sufficient.")

if __name__ == "__main__":
    main()
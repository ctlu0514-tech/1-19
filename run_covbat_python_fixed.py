# 文件名: run_covbat_python.py
# 描述: CovBat 多中心特征协调（预测任务安全版：不使用 label 作为协变量）
#
# 关键修复：
#   1) 不再把 label 作为 model / numerical_covariates 传给 CovBat（避免 label leakage、也更可部署）
#   2) 支持可选协变量列（必须是部署时可得的变量：如 Age/PSA 等；不要包含 label）
#   3) 对特征列强制转为数值（不可转换 -> NaN），并在填补前提示
#   4) 明确提示：本脚本默认对“全体样本”做 harmonization；ref_batch 仅限制 PCA 拟合子集，不等价于 train-only fit

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# 导入修改过的 covbat 模块（同目录下 covbat.py）
try:
    import covbat
except ImportError:
    print("错误: 未找到 covbat.py 文件（请确保 run_covbat_python.py 与 covbat.py 在同一目录）。")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Run CovBat harmonization on multi-center radiomics features.")
    p.add_argument("--input_csv", type=str, default="/data/qh_20T_share_file/lct/CT67/ovarian_All_Centers_with_label.csv")
    p.add_argument("--output_csv", type=str, default="/data/qh_20T_share_file/lct/CT67/ovarian_covbat_FuYi.csv")

    p.add_argument("--center_col", type=str, default="Center")
    p.add_argument("--label_col", type=str, default="type")      # 仅作为元数据保留；不会参与 CovBat
    p.add_argument("--id_col", type=str, default="ID")

    # 可选：部署时可得的协变量（不要包含 label）
    p.add_argument("--covariate_cols", type=str, default="",
                   help="Comma-separated covariate columns to preserve (e.g., 'Age,PSA'). Must be available at deployment. DO NOT include label.")

    # ref_batch：仅用于 CovBat 内部 PCA 的拟合子集（你当前 covbat.py 的实现）
    p.add_argument("--ref_batches", type=str, default="FuYi",
                   help="Comma-separated reference batch names for PCA fitting (e.g., 脑学'FuYi,FuEr,ShiZhongXin,NingBo'；卵巢癌).'FuYi,NingBo'")

    # CovBat 参数
    p.add_argument("--pct_var", type=float, default=0.95)
    p.add_argument("--n_pc", type=int, default=0)

    # 常量/低方差阈值
    p.add_argument("--var_thresh", type=float, default=1e-6)

    return p.parse_args()


def main():
    args = parse_args()

    INPUT_CSV = args.input_csv
    OUTPUT_CSV = args.output_csv

    CENTER_COL = args.center_col
    LABEL_COL = args.label_col
    ID_COL = args.id_col

    REF_BATCHES = [s.strip() for s in args.ref_batches.split(",") if s.strip()]
    COV_COLS = [s.strip() for s in args.covariate_cols.split(",") if s.strip()]

    print(f"--- [1] 读取数据: {os.path.basename(INPUT_CSV)} ---")
    if not os.path.exists(INPUT_CSV):
        print(f"文件不存在: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"    原始 shape: {df.shape}")

    # 基本列存在性检查
    for col in [CENTER_COL, LABEL_COL, ID_COL]:
        if col not in df.columns:
            print(f"!!! 错误: 缺少必要列 '{col}'")
            sys.exit(1)

    for c in COV_COLS:
        if c not in df.columns:
            print(f"!!! 错误: covariate 列 '{c}' 不存在于 CSV 中")
            sys.exit(1)

    if LABEL_COL in COV_COLS:
        print("!!! 错误: covariate_cols 里包含 label，这是预测任务不允许的（label leakage / 不可部署）。")
        sys.exit(1)

    # 检查参考中心是否存在
    unique_centers = set(df[CENTER_COL].astype(str).unique().tolist())
    for ref in REF_BATCHES:
        if ref not in unique_centers:
            print(f"!!! 错误: 参考中心 '{ref}' 未找到！现有中心: {sorted(list(unique_centers))[:10]} ...")
            sys.exit(1)

    print("\n--- [2] 数据预处理 (填补 & 去除常量) ---")

    # 元数据列：这些列会原样保留到输出
    meta_cols = [ID_COL, CENTER_COL, LABEL_COL] + COV_COLS
    meta_cols = [c for c in meta_cols if c in df.columns]

    # 特征列：排除 meta_cols + diagnostics_*
    feature_cols = [c for c in df.columns if c not in meta_cols and not str(c).startswith("diagnostics_")]
    print(f"    元数据列数: {len(meta_cols)} ({meta_cols})")
    print(f"    原始特征数: {len(feature_cols)}")

    # 强制特征列转数值，避免整列 object 被静默丢弃/或导致 covbat 崩溃
    X_df = df[feature_cols].copy()
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X_df[c])]
    if non_numeric:
        print(f"    [提示] 发现 {len(non_numeric)} 列为非数值 dtype，将强制转数值(不可转换->NaN)。示例: {non_numeric[:5]}")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    n_all_nan = int(X_df.isna().all(axis=0).sum())
    if n_all_nan > 0:
        print(f"    [提示] 发现 {n_all_nan} 列全为 NaN，将在后续通过方差筛选/常量剔除一并移除。")

    X = X_df.values

    # 1) 缺失值填补（CovBat 不接受 NaN）
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # 2) 去除常量/低方差列（避免内部标准化除以 0）
    selector = VarianceThreshold(threshold=args.var_thresh)
    X_cleaned = selector.fit_transform(X_imputed)

    valid_indices = selector.get_support(indices=True)
    kept_feature_names = [feature_cols[i] for i in valid_indices]

    removed_count = len(feature_cols) - len(kept_feature_names)
    print(f"    移除常量/低方差列: {removed_count}")
    print(f"    剩余特征数: {len(kept_feature_names)}")

    if len(kept_feature_names) == 0:
        print("!!! 错误: 所有特征都被移除了！请检查输入数据。")
        sys.exit(1)

    # CovBat 需要 (n_features, n_samples)
    data_for_covbat = pd.DataFrame(X_cleaned.T, index=kept_feature_names, columns=df.index)

    batch_series = df[CENTER_COL].astype(str)

    # ========= 预测任务安全设置：不使用 label =========
    # model_df 只包含 batch + 可选协变量（Age/PSA 等），绝不包含 label
    if len(COV_COLS) > 0:
        model_df = df[COV_COLS].copy()
        numerical_covariates = COV_COLS
        print(f"    将保留协变量影响(不被 harmonization 抹除): {COV_COLS}")
    else:
        model_df = None
        numerical_covariates = None
        print("    未提供协变量：将仅校正中心效应（不使用 label）。")

    print("\n--- [3] 开始运行 CovBat ---")
    print(f"    ref_batches(PCA拟合子集): {REF_BATCHES}")
    print("    注意：本脚本会对全体样本输出 harmonized 特征；ref_batch 仅限制 PCA 拟合子集，不等价于严格的 train-only fit。")

    try:
        data_harmonized = covbat.covbat(
            data=data_for_covbat,
            batch=batch_series,
            model=model_df,
            numerical_covariates=numerical_covariates,
            pct_var=args.pct_var,
            n_pc=args.n_pc,
            ref_batch=REF_BATCHES,
        )
        print("    >>> 协调完成！")
    except Exception as e:
        print(f"\n!!! CovBat 运行崩溃: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========= 保存结果 =========
    print("\n--- [4] 保存结果 ---")

    meta_df = df[meta_cols].copy()
    harmonized_features = pd.DataFrame(np.asarray(data_harmonized).T, columns=kept_feature_names)

    df_final = pd.concat([meta_df.reset_index(drop=True), harmonized_features.reset_index(drop=True)], axis=1)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    print(f"成功保存至: {OUTPUT_CSV}")
    print(f"输出 shape: {df_final.shape}")
    print(f"harmonized 特征数: {len(kept_feature_names)}")


if __name__ == "__main__":
    main()
